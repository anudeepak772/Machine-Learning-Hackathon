"""
PathSnap - Finalize Bayesian Optimization after 5 Iterations
Picks the best configuration from the completed trials, refits on all 80 training trips,
evaluates on student-test, benchmarks against baseline, exports feature importances,
and saves the final model artifact.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

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
    eval_df = df[['trip_id', 'obs_idx', 'arc_id', 'label']].copy()
    eval_df['pred_prob'] = probs

    top1 = eval_df.sort_values(
        ['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False]
    )
    top1_chosen = top1.groupby(['trip_id', 'obs_idx']).first().reset_index()
    top1_acc = (top1_chosen['label'] == 1).mean()

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


def finalize_optimization():
    print("=" * 70)
    print("FINALIZING BAYESIAN OPTIMIZATION (5 TRIALS)")
    print("=" * 70)

    trials_csv = os.path.join(RESULTS_DIR, "optuna_trials_RandomForest.csv")
    if not os.path.exists(trials_csv):
        raise FileNotFoundError(f"Trials CSV not found at: {trials_csv}")

    trials_df = pd.read_csv(trials_csv)
    print(f"Loaded {len(trials_df)} trials from {trials_csv}:")
    print(trials_df.to_string(index=False))

    # Filter complete trials
    complete_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    complete_trials['value'] = pd.to_numeric(complete_trials['value'])
    best_row = complete_trials.sort_values('value', ascending=False).iloc[0]

    print(f"\n---> Winning Trial: #{int(best_row['trial_number'])}")
    print(f"     Best CV Composite Score: {best_row['value']:.5f}")

    # Parse best parameters
    best_params = {
        'n_estimators': int(best_row['n_estimators']),
        'max_depth': int(best_row['max_depth']),
        'min_samples_split': int(best_row['min_samples_split']),
        'min_samples_leaf': int(best_row['min_samples_leaf']),
        'max_features': None if pd.isna(best_row['max_features']) or str(best_row['max_features']).strip() == '' else str(best_row['max_features']),
        'random_state': 42,
        'n_jobs': -1
    }
    print(f"     Hyperparameters: {json.dumps(best_params, indent=2)}")

    # Load preprocessor and datasets
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)
    scaler = prep_data["scaler"]

    train_df = pd.read_pickle(os.path.join(DATA_DIR, "train_features.pkl"))
    test_df  = pd.read_pickle(os.path.join(DATA_DIR, "student_test_features.pkl"))

    X_train = scaler.transform(train_df[FEATURE_COLUMNS])
    y_train = train_df['label'].values

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df['label'].values

    test_graphs = load_graph_arcs_dict("student_test")

    # Refit final tuned model on entire 80-trip training set
    print(f"\nRefitting tuned Random Forest on all 80 training trips ({len(train_df):,} candidate pairs)...")
    t_refit_start = time.time()
    best_clf = RandomForestClassifier(**best_params)
    best_clf.fit(X_train, y_train)
    refit_duration = time.time() - t_refit_start
    print(f"Refit completed in {refit_duration:.1f}s.")

    # Evaluate on student-test
    print("\nEvaluating tuned model on 10 student-test trips...")
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
    print(f"  Log Loss:                   {tuned_test_logloss:.4f}")

    # Baseline vs Tuned Comparison
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
                "model_type": "RandomForest",
                "top1_acc": round(baseline_top1, 4),
                "continuity": round(baseline_cont, 4),
                "composite": round(0.7 * baseline_top1 + 0.3 * baseline_cont, 4),
                "macro_f1": round(baseline_f1, 4),
                "binary_accuracy": round(baseline_acc, 4),
                "log_loss": round(baseline_logloss, 4)
            },
            "tuned": {
                "model_type": "RandomForest",
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
                "n_trials": len(trials_df),
                "completed_trials": len(complete_trials),
                "pruned_trials": len(trials_df[trials_df['state'] == 'PRUNED']),
                "best_cv_score": round(float(best_row['value']), 5),
                "objective_type": "composite",
                "winning_trial": int(best_row['trial_number'])
            }
        }

        comparison_path = os.path.join(RESULTS_DIR, "tuned_vs_baseline.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved baseline vs tuned comparison to: {comparison_path}")

        print(f"\n--- Baseline vs Tuned Comparison ---")
        print(f"  Top-1 Acc:   {baseline_top1:.4f} -> {tuned_test_top1:.4f} (Delta {tuned_test_top1 - baseline_top1:+.4f})")
        print(f"  Continuity:  {baseline_cont:.4f} -> {tuned_test_cont:.4f} (Delta {tuned_test_cont - baseline_cont:+.4f})")
        print(f"  Composite:   {0.7*baseline_top1 + 0.3*baseline_cont:.4f} -> {0.7*tuned_test_top1 + 0.3*tuned_test_cont:.4f} (Delta {(0.7*tuned_test_top1 + 0.3*tuned_test_cont) - (0.7*baseline_top1 + 0.3*baseline_cont):+.4f})")
        print(f"  Macro F1:    {baseline_f1:.4f} -> {tuned_test_f1:.4f} (Delta {tuned_test_f1 - baseline_f1:+.4f})")
        print(f"  Log Loss:    {baseline_logloss:.4f} -> {tuned_test_logloss:.4f} (Delta {tuned_test_logloss - baseline_logloss:+.4f})")

    # Feature Importance Export
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
        bar = "#" * int(row['importance'] * 60)
        print(f"  {row['feature']:<22s} {row['importance']:.4f}  {bar}")

    # Save final tuned model artifact
    final_model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump({
            "model": best_clf,
            "model_type": "RandomForest",
            "best_params": best_params,
            "best_cv_composite": round(float(best_row['value']), 5),
            "test_top1": round(tuned_test_top1, 4),
            "test_continuity": round(tuned_test_cont, 4),
            "test_composite": round(0.7 * tuned_test_top1 + 0.3 * tuned_test_cont, 4),
            "test_macro_f1": round(tuned_test_f1, 4),
            "test_logloss": round(tuned_test_logloss, 4),
            "n_trials": len(trials_df),
            "objective_type": "composite"
        }, f)
    print(f"\nSaved tuned final model artifact to: {final_model_path}")

    # Copy trials to full CSV
    full_csv = os.path.join(RESULTS_DIR, "optuna_trials_RandomForest_full.csv")
    trials_df.to_csv(full_csv, index=False)
    print(f"Saved full trials table to: {full_csv}")

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION FINALIZED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    finalize_optimization()
