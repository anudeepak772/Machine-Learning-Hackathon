"""
PathSnap - 5-Model Benchmark Runner
Trains and evaluates exactly 5 models from diverse algorithm families.
Uses Group-aware Cross-Validation (by trip_id), Top-1 observation accuracy, and route continuity.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'perp_dist', 'dist_start', 'dist_end', 't_ratio', 'arc_len',
    'gps_speed', 'time_delta', 'heading_diff', 'cos_heading_diff',
    'spatial_prior', 'dist_rank', 'dist_ratio_to_best', 'dist_diff_to_best'
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


def evaluate_top1_and_continuity(df, probs, graph_dict):
    """
    Computes Observation-level Top-1 Accuracy and Route Continuity.
    """
    eval_df = df[['trip_id', 'obs_idx', 'arc_id', 'label']].copy()
    eval_df['pred_prob'] = probs

    top1_candidates = eval_df.sort_values(
        ['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False]
    )
    top1_chosen = top1_candidates.groupby(['trip_id', 'obs_idx']).first().reset_index()
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


def run_benchmark():
    print("=" * 70)
    print("PATHSNAP: 5-MODEL BENCHMARK PIPELINE")
    print("=" * 70)

    # --- Load preprocessor and datasets ---
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)
    scaler = prep_data["scaler"]

    train_df = pd.read_pickle(os.path.join(DATA_DIR, "train_features.pkl"))
    test_df  = pd.read_pickle(os.path.join(DATA_DIR, "student_test_features.pkl"))
    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    X_train = scaler.transform(train_df[FEATURE_COLUMNS])
    y_train = train_df['label'].values
    groups  = train_df['trip_id'].values

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test  = test_df['label'].values

    train_graphs = load_graph_arcs_dict("train")
    test_graphs  = load_graph_arcs_dict("student_test")

    # --- 5 Models across diverse algorithm families ---
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "config": "C=1.0, max_iter=1000, L2 penalty",
            "subsample_train": None,
        },
        "Decision Tree": {
            "family": "Tree-based",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=42),
            "config": "max_depth=12, min_samples_leaf=10",
            "subsample_train": None,
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            ),
            "config": "n_estimators=100, max_depth=15, min_samples_leaf=5",
            "subsample_train": None,
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, learning_rate=0.1, random_state=42
            ),
            "config": "max_iter=100, max_depth=8, lr=0.1",
            "subsample_train": None,
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "model": KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
            "config": "k=7, weights=distance, metric=euclidean",
            "subsample_train": 35000,  # subsample for speed
        },
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, item in models.items():
        print(f"\n{'='*60}")
        print(f"--> [{name}]  ({item['family']})")
        clf = item["model"]
        sub = item["subsample_train"]

        cv_f1, cv_top1, cv_cont = [], [], []
        t0 = time.time()

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_v,  y_v  = X_train[val_idx], y_train[val_idx]
            val_df_fold = train_df.iloc[val_idx]

            if sub:
                pick = np.random.choice(len(X_tr), min(sub, len(X_tr)), replace=False)
                clf.fit(X_tr[pick], y_tr[pick])
                pick_v = np.random.choice(len(X_v), min(15000, len(X_v)), replace=False)
                preds  = clf.predict(X_v[pick_v])
                probs  = clf.predict_proba(X_v[pick_v])[:, 1]
                f1     = f1_score(y_v[pick_v], preds, average="macro", zero_division=0)
                top1, cont = evaluate_top1_and_continuity(val_df_fold.iloc[pick_v], probs, train_graphs)
            else:
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_v)
                probs = clf.predict_proba(X_v)[:, 1]
                f1    = f1_score(y_v, preds, average="macro", zero_division=0)
                top1, cont = evaluate_top1_and_continuity(val_df_fold, probs, train_graphs)

            cv_f1.append(f1); cv_top1.append(top1); cv_cont.append(cont)
            print(f"  Fold {fold+1}: F1={f1:.4f}  Top-1={top1:.4f}  Continuity={cont:.4f}")

        cv_time = time.time() - t0

        # --- Full fit on all training data ---
        t_fit = time.time()
        if sub:
            pick_all = np.random.choice(len(X_train), min(50000, len(X_train)), replace=False)
            clf.fit(X_train[pick_all], y_train[pick_all])
        else:
            clf.fit(X_train, y_train)
        fit_time = time.time() - t_fit

        # --- Evaluate on student-test set ---
        t_inf = time.time()
        if sub:
            pick_te = np.random.choice(len(X_test), min(30000, len(X_test)), replace=False)
            test_preds = clf.predict(X_test[pick_te])
            test_probs = clf.predict_proba(X_test[pick_te])[:, 1]
            test_logloss = log_loss(y_test[pick_te], clf.predict_proba(X_test[pick_te]))
            inf_ms = ((time.time() - t_inf) / len(pick_te)) * 1e6
            test_acc  = accuracy_score(y_test[pick_te], test_preds)
            test_prec = precision_score(y_test[pick_te], test_preds, average="macro", zero_division=0)
            test_rec  = recall_score(y_test[pick_te], test_preds, average="macro", zero_division=0)
            test_f1   = f1_score(y_test[pick_te], test_preds, average="macro", zero_division=0)
            test_top1, test_cont = evaluate_top1_and_continuity(test_df.iloc[pick_te], test_probs, test_graphs)
        else:
            test_preds = clf.predict(X_test)
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_logloss = log_loss(y_test, clf.predict_proba(X_test))
            inf_ms = ((time.time() - t_inf) / len(X_test)) * 1e6
            test_acc  = accuracy_score(y_test, test_preds)
            test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
            test_rec  = recall_score(y_test, test_preds, average="macro", zero_division=0)
            test_f1   = f1_score(y_test, test_preds, average="macro", zero_division=0)
            test_top1, test_cont = evaluate_top1_and_continuity(test_df, test_probs, test_graphs)

        row = {
            "Model Name":               name,
            "Algorithm Family":         item["family"],
            "Key Configuration":        item["config"],
            "CV Macro F1":              round(np.mean(cv_f1), 4),
            "CV Top-1 Acc":             round(np.mean(cv_top1), 4),
            "CV Route Continuity":      round(np.mean(cv_cont), 4),
            "Student-Test Accuracy":    round(test_acc, 4),
            "Student-Test Macro F1":    round(test_f1, 4),
            "Student-Test Top-1 Acc":   round(test_top1, 4),
            "Student-Test Continuity":  round(test_cont, 4),
            "Student-Test Precision":   round(test_prec, 4),
            "Student-Test Recall":      round(test_rec, 4),
            "Log Loss":                 round(test_logloss, 4),
            "CV Training Time (s)":     round(cv_time, 1),
            "Inference Time (ms/1k)":   round(inf_ms, 2),
        }
        results.append(row)
        print(f"  >> CV  Macro F1={row['CV Macro F1']:.4f} | Top-1={row['CV Top-1 Acc']:.4f} | Continuity={row['CV Route Continuity']:.4f}")
        print(f"  >> TEST Macro F1={row['Student-Test Macro F1']:.4f} | Top-1={row['Student-Test Top-1 Acc']:.4f} | Continuity={row['Student-Test Continuity']:.4f}")

    # --- Save results ---
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table -> {csv_path}")

    # Pretty print
    print("\n" + "=" * 100)
    print("CONSOLIDATED 5-MODEL BENCHMARK TABLE")
    print("=" * 100)
    print(results_df[[
        "Model Name", "Algorithm Family",
        "CV Macro F1", "CV Top-1 Acc",
        "Student-Test Macro F1", "Student-Test Top-1 Acc",
        "Student-Test Continuity", "Inference Time (ms/1k)"
    ]].to_string(index=False))

    # Identify best model
    best_row = results_df.sort_values(
        by=["Student-Test Top-1 Acc", "Student-Test Macro F1"], ascending=False
    ).iloc[0]
    best_name = best_row["Model Name"]
    print(f"\n*** BEST MODEL: {best_name} ***")
    print(f"    Student-Test Top-1 Acc: {best_row['Student-Test Top-1 Acc']:.4f}")
    print(f"    Student-Test Macro F1:  {best_row['Student-Test Macro F1']:.4f}")
    print(f"    Student-Test Continuity:{best_row['Student-Test Continuity']:.4f}")

    # Save best model artifact
    best_clf = models[best_name]["model"]
    # Refit on full training set if needed (already fitted above)
    best_model_path = os.path.join(ARTIFACTS_DIR, "baseline_best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump({"model": best_clf, "model_name": best_name, "feature_columns": FEATURE_COLUMNS}, f)
    print(f"    Saved baseline best model -> {best_model_path}")

    # Save experiment log
    log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved experiment log     -> {log_path}")

    return results_df, best_name


if __name__ == "__main__":
    run_benchmark()
