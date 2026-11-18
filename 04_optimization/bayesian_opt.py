"""
PathSnap - Bayesian Hyperparameter Optimization
Optimizes the top-performing Random Forest model using Optuna (TPE Bayesian Optimization)
with a composite objective: 0.7 × Top-1 Observation Accuracy + 0.3 × Route Continuity.
Enforces Group-aware Cross-Validation (GroupKFold by trip_id) to strictly prevent data leakage.
Supports MedianPruner for early stopping of unpromising trials.
Logs all optimization trials per-trial for live progress, refits on all 80 training trips,
compares tuned vs baseline, exports feature importances, and saves final artifacts.
"""

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'perp_dist',
    'dist_start',
    'dist_end',
    't_ratio',
    'arc_len',
    'gps_speed',
    'time_delta',
    'heading_diff',
    'cos_heading_diff',
    'spatial_prior',
    'dist_rank',
    'dist_ratio_to_best',
    'dist_diff_to_best'
]


def load_graph_arcs_dict(partition="train"):
    """Builds lookup of arc (from_node, to_node) tuples for each trip in partition."""
    folder = os.path.join(DATA_DIR, partition)
    trips = sorted(os.listdir(folder))
    graph_dict = {}
    for t in trips:
        arcs_p = os.path.join(folder, t, f"{t}.arcs")
        with open(arcs_p) as f:
            arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
        graph_dict[t] = arcs
    return graph_dict


def compute_top1_and_continuity(df, probs, graph_dict):
    """
    Compute both Observation-level Top-1 Accuracy and Route Continuity.

    Top-1 Accuracy: For each (trip_id, obs_idx), the candidate with highest
    predicted probability is selected; accuracy is whether it matches ground truth.

    Route Continuity: Proportion of consecutive top-1 predicted arcs that are
    connected in the road graph (same arc, or share a node endpoint).
    """
    eval_df = df[['trip_id', 'obs_idx', 'arc_id', 'label']].copy()
    eval_df['pred_prob'] = probs

    top1 = eval_df.sort_values(
        ['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False]
    )
    top1_chosen = top1.groupby(['trip_id', 'obs_idx']).first().reset_index()
    top1_acc = (top1_chosen['label'] == 1).mean()

    # Route continuity
    total_transitions = 0
    connected_transitions = 0

    for trip_id, trip_group in top1_chosen.groupby('trip_id'):
        if trip_id not in graph_dict:
            continue
        arcs = graph_dict[trip_id]
        sorted_arcs = trip_group.sort_values('obs_idx')['arc_id'].tolist()

        for i in range(len(sorted_arcs) - 1):
            a1, a2 = sorted_arcs[i], sorted_arcs[i + 1]
            if a1 == a2:
                connected_transitions += 1
            else:
                u1, v1 = arcs[a1]
                u2, v2 = arcs[a2]
                if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                    connected_transitions += 1
            total_transitions += 1

    continuity = connected_transitions / total_transitions if total_transitions > 0 else 0.0
    return top1_acc, continuity


def compute_top1_acc(df, probs):
    """Legacy helper: computes only Top-1 observation accuracy (no continuity)."""
    eval_df = df[['trip_id', 'obs_idx', 'label']].copy()
    eval_df['pred_prob'] = probs
    top1 = eval_df.sort_values(['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False])
    top1_chosen = top1.groupby(['trip_id', 'obs_idx']).first().reset_index()
    return (top1_chosen['label'] == 1).mean()


class TrialCSVCallback:
    """Optuna callback that appends each completed trial to CSV for live progress monitoring."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Remove stale file from previous runs so header is written fresh
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": str(trial.state.name),
                "duration_s": round(trial.duration.total_seconds(), 1),
            }
            row.update(trial.params)
            df = pd.DataFrame([row])
            write_header = not os.path.exists(self.csv_path)
            df.to_csv(self.csv_path, mode='a', header=write_header, index=False)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            row = {
                "trial_number": trial.number,
                "value": trial.value if trial.value is not None else "PRUNED",
                "state": "PRUNED",
                "duration_s": round(trial.duration.total_seconds(), 1) if trial.duration else 0,
            }
            row.update(trial.params)
            df = pd.DataFrame([row])
            write_header = not os.path.exists(self.csv_path)
            df.to_csv(self.csv_path, mode='a', header=write_header, index=False)


def run_bayesian_optimization(model_type="RandomForest", n_trials=15, objective_type="composite"):
    print("=" * 70)
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE) - {model_type}")
    print(f"Objective: {objective_type} | Trials: {n_trials}")
    print("=" * 70)

    # Load artifacts and data
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)
    scaler = prep_data["scaler"]

    train_df = pd.read_pickle(os.path.join(DATA_DIR, "train_features.pkl"))
    test_df  = pd.read_pickle(os.path.join(DATA_DIR, "student_test_features.pkl"))

    X_train = scaler.transform(train_df[FEATURE_COLUMNS])
    y_train = train_df['label'].values
    trip_ids_train = train_df['trip_id'].values

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df['label'].values

    # Load graph arc dictionaries for continuity computation
    train_graphs = load_graph_arcs_dict("train")
    test_graphs  = load_graph_arcs_dict("student_test")

    gkf = GroupKFold(n_splits=5)

    use_composite = (objective_type == "composite")

    print(f"\nLoaded Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    print(f"Using {'COMPOSITE (0.7×Top-1 + 0.3×Continuity)' if use_composite else 'TOP-1 ONLY'} objective.")
    print(f"Graph dicts loaded: {len(train_graphs)} train trips, {len(test_graphs)} test trips.\n")

    def objective(trial):
        if model_type == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 260, step=20),
                'max_depth': trial.suggest_int('max_depth', 10, 24),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            clf = RandomForestClassifier(**params)

        elif model_type == "HistGradientBoosting":
            params = {
                'max_iter': trial.suggest_int('max_iter', 80, 250, step=20),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.25, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 14),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
                'random_state': 42
            }
            clf = HistGradientBoostingClassifier(**params)

        elif model_type == "XGBoost" and HAS_XGB:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 250, step=20),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.25, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1
            }
            clf = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown or unsupported model type: {model_type}")

        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=trip_ids_train)):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_v, y_v = X_train[val_idx], y_train[val_idx]
            val_df_fold = train_df.iloc[val_idx]

            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_v)[:, 1]

            if use_composite:
                top1, cont = compute_top1_and_continuity(val_df_fold, probs, train_graphs)
                composite_score = 0.7 * top1 + 0.3 * cont
                fold_scores.append(composite_score)
            else:
                top1 = compute_top1_acc(val_df_fold, probs)
                fold_scores.append(top1)

            # Report fold-level score for pruning
            trial.report(fold_scores[-1], fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_scores)

    # Set up Optuna study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Per-trial CSV callback for live monitoring
    trials_csv = os.path.join(RESULTS_DIR, f"optuna_trials_{model_type}.csv")
    csv_callback = TrialCSVCallback(trials_csv)

    t_opt_start = time.time()
    print(f"Starting {n_trials}-trial optimization...\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[csv_callback])

    opt_duration = time.time() - t_opt_start

    print(f"\n{'=' * 70}")
    print(f"Optimization Finished in {opt_duration:.1f}s across {len(study.trials)} trials.")
    print(f"Best Trial #{study.best_trial.number} Score ({'Composite' if use_composite else 'Top-1'}): {study.best_value:.4f}")
    print(f"Best Hyperparameters: {json.dumps(study.best_params, indent=2)}")

    # Also save the full Optuna trials dataframe (overwrites the per-trial CSV with richer data)
    full_trials_df = study.trials_dataframe()
    full_trials_csv = os.path.join(RESULTS_DIR, f"optuna_trials_{model_type}_full.csv")
    full_trials_df.to_csv(full_trials_csv, index=False)
    print(f"Saved full optimization trials log to: {full_trials_csv}")

    # =========================================================================
    # Refit tuned model on entire 80-trip training set
    # =========================================================================
    print("\nRefitting final tuned model on all 80 training trips...")
    best_params = study.best_params

    if model_type == "RandomForest":
        best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif model_type == "HistGradientBoosting":
        best_clf = HistGradientBoostingClassifier(**best_params, random_state=42)
    elif model_type == "XGBoost" and HAS_XGB:
        best_clf = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, n_jobs=-1)

    best_clf.fit(X_train, y_train)

    # =========================================================================
    # Evaluate tuned model on student-test set
    # =========================================================================
    print("Evaluating tuned model on student-test set...")
    test_probs = best_clf.predict_proba(X_test)[:, 1]
    test_preds = best_clf.predict(X_test)

    tuned_test_top1, tuned_test_cont = compute_top1_and_continuity(test_df, test_probs, test_graphs)
    tuned_test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    tuned_test_acc = accuracy_score(y_test, test_preds)
    tuned_test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    tuned_test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
    tuned_test_logloss = log_loss(y_test, best_clf.predict_proba(X_test))

    print(f"\nTuned Model Student-Test Performance:")
    print(f"  Top-1 Observation Accuracy: {tuned_test_top1:.4f}")
    print(f"  Route Continuity:           {tuned_test_cont:.4f}")
    print(f"  Composite Score:            {0.7 * tuned_test_top1 + 0.3 * tuned_test_cont:.4f}")
    print(f"  Macro F1-score:             {tuned_test_f1:.4f}")
    print(f"  Binary Accuracy:            {tuned_test_acc:.4f}")
    print(f"  Precision (macro):          {tuned_test_prec:.4f}")
    print(f"  Recall (macro):             {tuned_test_rec:.4f}")
    print(f"  Log Loss:                   {tuned_test_logloss:.4f}")

    # =========================================================================
    # Baseline vs Tuned Comparison
    # =========================================================================
    print("\nComparing against baseline model...")
    baseline_path = os.path.join(ARTIFACTS_DIR, "baseline_best_model.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            baseline_data = pickle.load(f)
        if isinstance(baseline_data, dict):
            baseline_clf = baseline_data["model"]
        else:
            baseline_clf = baseline_data

        baseline_probs = baseline_clf.predict_proba(X_test)[:, 1]
        baseline_preds = baseline_clf.predict(X_test)

        baseline_top1, baseline_cont = compute_top1_and_continuity(test_df, baseline_probs, test_graphs)
        baseline_f1 = f1_score(y_test, baseline_preds, average="macro", zero_division=0)
        baseline_acc = accuracy_score(y_test, baseline_preds)
        baseline_logloss = log_loss(y_test, baseline_clf.predict_proba(X_test))

        comparison = {
            "baseline": {
                "model_type": model_type,
                "top1_acc": round(baseline_top1, 4),
                "continuity": round(baseline_cont, 4),
                "composite": round(0.7 * baseline_top1 + 0.3 * baseline_cont, 4),
                "macro_f1": round(baseline_f1, 4),
                "binary_accuracy": round(baseline_acc, 4),
                "log_loss": round(baseline_logloss, 4)
            },
            "tuned": {
                "model_type": model_type,
                "best_params": best_params,
                "top1_acc": round(tuned_test_top1, 4),
                "continuity": round(tuned_test_cont, 4),
                "composite": round(0.7 * tuned_test_top1 + 0.3 * tuned_test_cont, 4),
                "macro_f1": round(tuned_test_f1, 4),
                "binary_accuracy": round(tuned_test_acc, 4),
                "log_loss": round(tuned_test_logloss, 4)
            },
            "delta": {
                "top1_acc": round(tuned_test_top1 - baseline_top1, 4),
                "continuity": round(tuned_test_cont - baseline_cont, 4),
                "composite": round((0.7 * tuned_test_top1 + 0.3 * tuned_test_cont) -
                                   (0.7 * baseline_top1 + 0.3 * baseline_cont), 4),
                "macro_f1": round(tuned_test_f1 - baseline_f1, 4),
                "log_loss": round(tuned_test_logloss - baseline_logloss, 4)
            },
            "optimization_info": {
                "n_trials": n_trials,
                "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "best_cv_score": round(study.best_value, 4),
                "objective_type": objective_type,
                "optimization_time_s": round(opt_duration, 1)
            }
        }

        comparison_path = os.path.join(RESULTS_DIR, "tuned_vs_baseline.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved baseline vs tuned comparison to: {comparison_path}")

        print(f"\n--- Baseline vs Tuned ---")
        print(f"  Top-1 Acc:   {baseline_top1:.4f} → {tuned_test_top1:.4f} (Δ {tuned_test_top1 - baseline_top1:+.4f})")
        print(f"  Continuity:  {baseline_cont:.4f} → {tuned_test_cont:.4f} (Δ {tuned_test_cont - baseline_cont:+.4f})")
        print(f"  Composite:   {0.7*baseline_top1 + 0.3*baseline_cont:.4f} → {0.7*tuned_test_top1 + 0.3*tuned_test_cont:.4f}")
        print(f"  Macro F1:    {baseline_f1:.4f} → {tuned_test_f1:.4f} (Δ {tuned_test_f1 - baseline_f1:+.4f})")
        print(f"  Log Loss:    {baseline_logloss:.4f} → {tuned_test_logloss:.4f} (Δ {tuned_test_logloss - baseline_logloss:+.4f})")
    else:
        print(f"[WARNING] Baseline model not found at {baseline_path}, skipping comparison.")
        comparison = None

    # =========================================================================
    # Feature Importance Export
    # =========================================================================
    if hasattr(best_clf, 'feature_importances_'):
        print("\nExtracting feature importances...")
        importances = best_clf.feature_importances_
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': importances
        }).sort_values('importance', ascending=False)

        fi_path = os.path.join(RESULTS_DIR, "feature_importances.csv")
        importance_df.to_csv(fi_path, index=False)
        print(f"Saved feature importances to: {fi_path}")
        print("\nFeature Importance Ranking:")
        for _, row in importance_df.iterrows():
            bar = "█" * int(row['importance'] * 80)
            print(f"  {row['feature']:<22s} {row['importance']:.4f}  {bar}")

    # =========================================================================
    # Save final tuned model artifact
    # =========================================================================
    final_model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump({
            "model": best_clf,
            "model_type": model_type,
            "best_params": best_params,
            "best_cv_composite": round(study.best_value, 4),
            "test_top1": round(tuned_test_top1, 4),
            "test_continuity": round(tuned_test_cont, 4),
            "test_composite": round(0.7 * tuned_test_top1 + 0.3 * tuned_test_cont, 4),
            "test_macro_f1": round(tuned_test_f1, 4),
            "test_logloss": round(tuned_test_logloss, 4),
            "optimization_time_s": round(opt_duration, 1),
            "n_trials": n_trials,
            "objective_type": objective_type
        }, f)
    print(f"\nSaved tuned final model artifact to: {final_model_path}")

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION COMPLETE")
    print("=" * 70)

    return study, best_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PathSnap - Bayesian Hyperparameter Optimization (Optuna TPE)"
    )
    parser.add_argument(
        "--trials", type=int, default=15,
        help="Number of Optuna optimization trials (default: 15)"
    )
    parser.add_argument(
        "--model", type=str, default="RandomForest",
        choices=["RandomForest", "HistGradientBoosting", "XGBoost"],
        help="Model type to optimize (default: RandomForest)"
    )
    parser.add_argument(
        "--objective", type=str, default="composite",
        choices=["composite", "top1_only"],
        help="Objective function: 'composite' (0.7×Top1 + 0.3×Continuity) or 'top1_only' (default: composite)"
    )
    args = parser.parse_args()

    run_bayesian_optimization(
        model_type=args.model,
        n_trials=args.trials,
        objective_type=args.objective
    )
