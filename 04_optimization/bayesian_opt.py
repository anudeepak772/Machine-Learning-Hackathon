"""
PathSnap - Bayesian Hyperparameter Optimization
Optimizes the top-performing ensemble/gradient boosting model using Optuna (TPE Bayesian Optimization).
Enforces Group-aware Cross-Validation (GroupKFold by trip_id) to strictly prevent data leakage.
Logs all optimization trials, refits the tuned model on all 80 training trips, and saves final artifacts.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

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
    folder = os.path.join(DATA_DIR, partition)
    trips = sorted(os.listdir(folder))
    graph_dict = {}
    for t in trips:
        arcs_p = os.path.join(folder, t, f"{t}.arcs")
        with open(arcs_p) as f:
            arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
        graph_dict[t] = arcs
    return graph_dict

def compute_top1_acc(df, probs):
    eval_df = df[['trip_id', 'obs_idx', 'label']].copy()
    eval_df['pred_prob'] = probs
    top1 = eval_df.sort_values(['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False])
    top1_chosen = top1.groupby(['trip_id', 'obs_idx']).first().reset_index()
    return (top1_chosen['label'] == 1).mean()

def run_bayesian_optimization(model_type="RandomForest", n_trials=30):
    print("=" * 70)
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE) - {model_type}")
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

    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        if model_type == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 250, step=20),
                'max_depth': trial.suggest_int('max_depth', 8, 22),
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

        fold_top1_scores = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=trip_ids_train)):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_v, y_v = X_train[val_idx], y_train[val_idx]
            val_df_fold = train_df.iloc[val_idx]

            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_v)[:, 1]
            top1 = compute_top1_acc(val_df_fold, probs)
            fold_top1_scores.append(top1)

        return np.mean(fold_top1_scores)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    t_opt_start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_duration = time.time() - t_opt_start

    print(f"\nOptimization Finished in {opt_duration:.1f}s across {len(study.trials)} trials.")
    print(f"Best Trial Score (CV Top-1 Acc): {study.best_value:.4f}")
    print(f"Best Hyperparameters: {json.dumps(study.best_params, indent=2)}")

    # Save full trials log
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(RESULTS_DIR, f"optuna_trials_{model_type}.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved optimization trials log to: {trials_csv}")

    # Refit tuned model on entire 80-trip training set
    print("\nRefitting final tuned model on all 80 training trips...")
    best_params = study.best_params
    if model_type == "RandomForest":
        best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif model_type == "HistGradientBoosting":
        best_clf = HistGradientBoostingClassifier(**best_params, random_state=42)
    elif model_type == "XGBoost" and HAS_XGB:
        best_clf = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, n_jobs=-1)

    best_clf.fit(X_train, y_train)

    # Evaluate tuned model on student-test set
    test_probs = best_clf.predict_proba(X_test)[:, 1]
    test_preds = best_clf.predict(X_test)
    tuned_test_top1 = compute_top1_acc(test_df, test_probs)
    tuned_test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    tuned_test_acc = accuracy_score(y_test, test_preds)
    tuned_test_logloss = log_loss(y_test, best_clf.predict_proba(X_test))

    print(f"\nTuned Model Student-Test Performance:")
    print(f"  Top-1 Observation Accuracy: {tuned_test_top1:.4f}")
    print(f"  Macro F1-score:             {tuned_test_f1:.4f}")
    print(f"  Binary Accuracy:            {tuned_test_acc:.4f}")
    print(f"  Log Loss:                   {tuned_test_logloss:.4f}")

    # Save final tuned model artifact
    final_model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump({
            "model": best_clf,
            "model_type": model_type,
            "best_params": best_params,
            "best_cv_top1": study.best_value,
            "test_top1": tuned_test_top1,
            "test_macro_f1": tuned_test_f1,
            "test_logloss": tuned_test_logloss,
            "optimization_time_s": opt_duration
        }, f)
    print(f"Saved tuned final model artifact to: {final_model_path}")

    return study, best_clf

if __name__ == "__main__":
    # Optimize RandomForest or HistGradientBoosting based on benchmark
    run_bayesian_optimization("RandomForest", n_trials=25)
