"""
PathSnap - Multi-Model Experimentation & Benchmark Runner
Trains and systematically evaluates >=5 distinct machine learning models across multiple algorithm families.
Includes Group-aware Cross-Validation (by trip_id), Top-1 observation accuracy, and route continuity.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier

# Try importing XGBoost & LightGBM if installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
    """Builds lookup of (from_node, to_node) for each trip."""
    folder = os.path.join(DATA_DIR, partition)
    trips = sorted(os.listdir(folder))
    graph_dict = {}
    for t in trips:
        arcs_p = os.path.join(folder, t, f"{t}.arcs")
        with open(arcs_p) as f:
            arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
        graph_dict[t] = arcs
    return graph_dict

def evaluate_observation_top1_and_continuity(df, probs, graph_dict):
    """
    Computes Observation-level Top-1 Accuracy and Route Continuity.
    Top-1: For each GPS observation (trip_id, obs_idx), candidate with highest predicted prob
           matches ground truth (label == 1).
    Continuity: Proportion of consecutive top-1 predicted arcs connected in road graph.
    """
    eval_df = df[['trip_id', 'obs_idx', 'arc_id', 'label']].copy()
    eval_df['pred_prob'] = probs

    # Group by (trip_id, obs_idx) and find candidate with max prob
    top1_candidates = eval_df.sort_values(['trip_id', 'obs_idx', 'pred_prob'], ascending=[True, True, False])
    top1_chosen = top1_candidates.groupby(['trip_id', 'obs_idx']).first().reset_index()

    top1_acc = (top1_chosen['label'] == 1).mean()

    # Route Continuity
    total_transitions = 0
    connected_transitions = 0

    for trip_id, trip_group in top1_chosen.groupby('trip_id'):
        if trip_id not in graph_dict:
            continue
        arcs = graph_dict[trip_id]
        sorted_arcs = trip_group.sort_values('obs_idx')['arc_id'].tolist()
        
        for i in range(len(sorted_arcs) - 1):
            a1, a2 = sorted_arcs[i], sorted_arcs[i+1]
            if a1 == a2:
                # Same road segment -> continuous
                connected_transitions += 1
            else:
                u1, v1 = arcs[a1]
                u2, v2 = arcs[a2]
                if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                    connected_transitions += 1
            total_transitions += 1

    continuity = connected_transitions / total_transitions if total_transitions > 0 else 0.0
    return top1_acc, continuity, top1_chosen

def run_experiments():
    print("=" * 70)
    print("PATHSNAP: MULTI-MODEL EXPERIMENTATION & COMPARISON PIPELINE")
    print("=" * 70)

    # 1. Load Preprocessor and Datasets
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor not found at {prep_path}. Run data_pipeline.py first!")

    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)
    scaler = prep_data["scaler"]

    train_df = pd.read_pickle(os.path.join(DATA_DIR, "train_features.pkl"))
    test_df  = pd.read_pickle(os.path.join(DATA_DIR, "student_test_features.pkl"))

    print(f"Loaded Train: {len(train_df)} rows, Test: {len(test_df)} rows.")

    X_train = scaler.transform(train_df[FEATURE_COLUMNS])
    y_train = train_df['label'].values
    trip_ids_train = train_df['trip_id'].values

    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df['label'].values
    trip_ids_test = test_df['trip_id'].values

    train_graphs = load_graph_arcs_dict("train")
    test_graphs  = load_graph_arcs_dict("student_test")

    # 2. Define Model Zoo across diverse algorithm families
    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "config": "C=1.0, max_iter=1000, l2 penalty"
        },
        "Decision Tree": {
            "family": "Tree-based",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=42),
            "config": "max_depth=12, min_samples_leaf=10"
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "config": "n_estimators=100, max_depth=15, min_samples_leaf=5"
        },
        "Extra Trees": {
            "family": "Ensemble (Extremely Randomized)",
            "model": ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "config": "n_estimators=100, max_depth=15, min_samples_leaf=5"
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42),
            "config": "max_iter=100, max_depth=8, lr=0.1"
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "model": KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
            "config": "k=7, weights=distance, metric=euclidean"
        },
        "MLP Neural Network": {
            "family": "Neural Networks",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, alpha=0.001, random_state=42),
            "config": "layers=(64, 32), relu, alpha=0.001, max_iter=150"
        }
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "family": "Boosting (Extreme Gradient Boosting)",
            "model": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1),
            "config": "n_estimators=150, max_depth=6, lr=0.1"
        }

    if HAS_LGBM:
        models["LightGBM"] = {
            "family": "Boosting (Light Gradient Boosting)",
            "model": LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
            "config": "n_estimators=150, max_depth=6, lr=0.1"
        }

    results = []
    gkf = GroupKFold(n_splits=5)

    print(f"\nEvaluating {len(models)} models using 5-Fold GroupKFold (group=trip_id)...")

    for name, item in models.items():
        print(f"\n---> Training: {name} ({item['family']})")
        clf = item["model"]

        # 5-Fold CV on Training Set
        cv_f1_scores = []
        cv_top1_scores = []
        cv_continuity_scores = []
        
        t_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=trip_ids_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_v, y_v = X_train[val_idx], y_train[val_idx]
            val_df_fold = train_df.iloc[val_idx]

            # Fast subsampling for computationally intensive O(N^2) instance-based algorithms
            if "K-Nearest" in name:
                sub_tr = np.random.choice(len(X_tr), min(35000, len(X_tr)), replace=False)
                clf.fit(X_tr[sub_tr], y_tr[sub_tr])
                sub_v = np.random.choice(len(X_v), min(15000, len(X_v)), replace=False)
                preds = clf.predict(X_v[sub_v])
                probs = clf.predict_proba(X_v[sub_v])[:, 1]
                f1 = f1_score(y_v[sub_v], preds, average="macro", zero_division=0)
                top1, cont, _ = evaluate_observation_top1_and_continuity(val_df_fold.iloc[sub_v], probs, train_graphs)
            elif "MLP" in name:
                sub_tr = np.random.choice(len(X_tr), min(60000, len(X_tr)), replace=False)
                clf.fit(X_tr[sub_tr], y_tr[sub_tr])
                preds = clf.predict(X_v)
                probs = clf.predict_proba(X_v)[:, 1]
                f1 = f1_score(y_v, preds, average="macro", zero_division=0)
                top1, cont, _ = evaluate_observation_top1_and_continuity(val_df_fold, probs, train_graphs)
            else:
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_v)
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_v)[:, 1]
                else:
                    probs = clf.decision_function(X_v)
                f1 = f1_score(y_v, preds, average="macro", zero_division=0)
                top1, cont, _ = evaluate_observation_top1_and_continuity(val_df_fold, probs, train_graphs)
            
            cv_f1_scores.append(f1)
            cv_top1_scores.append(top1)
            cv_continuity_scores.append(cont)

        training_time = time.time() - t_start

        # Full fit on entire 80-trip training set
        t_fit = time.time()
        if "K-Nearest" in name:
            sub_tr = np.random.choice(len(X_train), min(50000, len(X_train)), replace=False)
            clf.fit(X_train[sub_tr], y_train[sub_tr])
        elif "MLP" in name:
            sub_tr = np.random.choice(len(X_train), min(100000, len(X_train)), replace=False)
            clf.fit(X_train[sub_tr], y_train[sub_tr])
        else:
            clf.fit(X_train, y_train)
        full_fit_time = time.time() - t_fit

        # Evaluate on Student-Test set
        t_inf_start = time.time()
        if "K-Nearest" in name:
            sub_te = np.random.choice(len(X_test), min(30000, len(X_test)), replace=False)
            test_preds = clf.predict(X_test[sub_te])
            test_probs = clf.predict_proba(X_test[sub_te])[:, 1]
            test_logloss = log_loss(y_test[sub_te], clf.predict_proba(X_test[sub_te]))
            inf_time_ms = ((time.time() - t_inf_start) / len(sub_te)) * 1000.0 * 1000.0
            test_acc = accuracy_score(y_test[sub_te], test_preds)
            test_prec = precision_score(y_test[sub_te], test_preds, average="macro", zero_division=0)
            test_rec = recall_score(y_test[sub_te], test_preds, average="macro", zero_division=0)
            test_f1 = f1_score(y_test[sub_te], test_preds, average="macro", zero_division=0)
            test_top1, test_continuity, _ = evaluate_observation_top1_and_continuity(
                test_df.iloc[sub_te], test_probs, test_graphs
            )
        else:
            test_preds = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                test_probs = clf.predict_proba(X_test)[:, 1]
                test_logloss = log_loss(y_test, clf.predict_proba(X_test))
            else:
                test_probs = clf.decision_function(X_test)
                test_logloss = "N/A"
            inf_time_ms = ((time.time() - t_inf_start) / len(X_test)) * 1000.0 * 1000.0
            test_acc = accuracy_score(y_test, test_preds)
            test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
            test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
            test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
            test_top1, test_continuity, _ = evaluate_observation_top1_and_continuity(
                test_df, test_probs, test_graphs
            )

        res_row = {
            "Model Name": name,
            "Algorithm Family": item["family"],
            "Key Configuration": item["config"],
            "CV Macro F1": np.mean(cv_f1_scores),
            "CV Top-1 Acc": np.mean(cv_top1_scores),
            "CV Route Continuity": np.mean(cv_continuity_scores),
            "Student-Test Macro F1": test_f1,
            "Student-Test Top-1 Acc": test_top1,
            "Student-Test Continuity": test_continuity,
            "Student-Test Precision": test_prec,
            "Student-Test Recall": test_rec,
            "Log Loss": test_logloss if test_logloss == "N/A" else f"{test_logloss:.4f}",
            "Training Time (s)": round(training_time, 2),
            "Inference Time (ms/1k)": round(inf_time_ms, 2)
        }
        results.append(res_row)

        print(f"  CV Macro F1: {res_row['CV Macro F1']:.4f} | CV Top-1: {res_row['CV Top-1 Acc']:.4f}")
        print(f"  Test Macro F1: {res_row['Student-Test Macro F1']:.4f} | Test Top-1: {res_row['Student-Test Top-1 Acc']:.4f} | Continuity: {res_row['Student-Test Continuity']:.4f}")

    # 3. Save Consolidated Comparison Table
    results_df = pd.DataFrame(results)
    comparison_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved consolidated comparison table to: {comparison_csv}")

    # Print clean Markdown table
    print("\n" + "=" * 90)
    print("CONSOLIDATED MODEL BENCHMARK TABLE")
    print("=" * 90)
    print(results_df[[
        "Model Name", "Algorithm Family", "CV Macro F1", "CV Top-1 Acc",
        "Student-Test Macro F1", "Student-Test Top-1 Acc", "Student-Test Continuity", "Inference Time (ms/1k)"
    ]].to_string(index=False))

    # Identify Best Model by Student-Test Top-1 & Macro F1
    best_model_row = results_df.sort_values(by=["Student-Test Top-1 Acc", "Student-Test Macro F1"], ascending=False).iloc[0]
    best_model_name = best_model_row["Model Name"]
    print(f"\nWINNING MODEL FOR BAYESIAN OPTIMIZATION: {best_model_name}")

    # Save baseline best model artifact
    best_clf = models[best_model_name]["model"]
    best_clf.fit(X_train, y_train)
    best_model_path = os.path.join(ARTIFACTS_DIR, "baseline_best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_clf, f)
    print(f"Saved baseline best model to: {best_model_path}")

    return results_df, best_model_name

if __name__ == "__main__":
    run_experiments()
