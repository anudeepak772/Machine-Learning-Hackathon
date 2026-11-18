"""
PathSnap - HMM Route Reconstruction Evaluation
Compares Classifier-Only predictions with Classifier + HMM Viterbi decoding.
Quantifies Pointwise Accuracy, Route Continuity, and Error Reduction.
"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from viterbi_decoder import ViterbiMapMatchingDecoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
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

def load_trip_graph_arcs(trip_id, partition="student_test"):
    folder = os.path.join(DATA_DIR, partition, trip_id)
    arcs_p = os.path.join(folder, f"{trip_id}.arcs")
    with open(arcs_p) as f:
        return [tuple(map(int, line.strip().split())) for line in f if line.strip()]

def calculate_route_continuity(predicted_arcs, graph_arcs):
    """Proportion of consecutive transitions connected in graph or remaining on same arc."""
    if len(predicted_arcs) <= 1:
        return 1.0
    connected = 0
    total = len(predicted_arcs) - 1
    for i in range(total):
        a1, a2 = predicted_arcs[i], predicted_arcs[i+1]
        if a1 == a2:
            connected += 1
        else:
            u1, v1 = graph_arcs[a1]
            u2, v2 = graph_arcs[a2]
            if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                connected += 1
    return connected / total if total > 0 else 0.0

def evaluate_hmm_reconstruction(partition="student_test"):
    print("=" * 70)
    print(f"HMM / VITERBI EVALUATION & COMPARISON ({partition.upper()})")
    print("=" * 70)

    # Load Model & Preprocessor
    model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, "baseline_best_model.pkl")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    else:
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)
            clf = saved_data["model"]

    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "rb") as f:
        prep_data = pickle.load(f)
    scaler = prep_data["scaler"]

    # Load partition features
    feat_file = f"{partition}_features.pkl"
    df = pd.read_pickle(os.path.join(DATA_DIR, feat_file))

    # Predict candidate probabilities
    X = scaler.transform(df[FEATURE_COLUMNS])
    df['clf_prob'] = clf.predict_proba(X)[:, 1]

    decoder = ViterbiMapMatchingDecoder(sigma_z=8.0, same_arc_prob=0.88, direct_conn_prob=1.0)

    comparison_results = []
    total_obs = 0
    clf_only_correct = 0
    hmm_correct = 0

    all_clf_continuity = []
    all_hmm_continuity = []

    trip_ids = df['trip_id'].unique()

    for trip_id in trip_ids:
        trip_df = df[df['trip_id'] == trip_id]
        graph_arcs = load_trip_graph_arcs(trip_id, partition)

        # Structure observations
        obs_groups = trip_df.groupby('obs_idx')
        obs_candidates_list = []
        ground_truth_arcs = []
        clf_only_preds = []

        for obs_idx, obs_cands in obs_groups:
            cands_records = obs_cands.to_dict('records')
            obs_candidates_list.append(cands_records)

            # Ground truth for this observation
            gt_row = obs_cands[obs_cands['label'] == 1]
            if len(gt_row) > 0:
                ground_truth_arcs.append(gt_row.iloc[0]['arc_id'])
            else:
                ground_truth_arcs.append(None)

            # Classifier-only prediction: argmax of clf_prob
            best_cand = max(cands_records, key=lambda c: c['clf_prob'])
            clf_only_preds.append(best_cand['arc_id'])

        # Viterbi decoding
        t_decode_start = time.time()
        hmm_preds = decoder.decode(obs_candidates_list, graph_arcs)
        decode_time_ms = (time.time() - t_decode_start) * 1000.0

        # Pointwise accuracy
        trip_obs_cnt = len(ground_truth_arcs)
        trip_clf_correct = sum(1 for p, gt in zip(clf_only_preds, ground_truth_arcs) if gt is not None and p == gt)
        trip_hmm_correct = sum(1 for p, gt in zip(hmm_preds, ground_truth_arcs) if gt is not None and p == gt)

        valid_gt_cnt = sum(1 for gt in ground_truth_arcs if gt is not None)
        clf_acc = trip_clf_correct / valid_gt_cnt if valid_gt_cnt > 0 else 0.0
        hmm_acc = trip_hmm_correct / valid_gt_cnt if valid_gt_cnt > 0 else 0.0

        # Continuity
        clf_cont = calculate_route_continuity(clf_only_preds, graph_arcs)
        hmm_cont = calculate_route_continuity(hmm_preds, graph_arcs)

        total_obs += valid_gt_cnt
        clf_only_correct += trip_clf_correct
        hmm_correct += trip_hmm_correct
        all_clf_continuity.append(clf_cont)
        all_hmm_continuity.append(hmm_cont)

        comparison_results.append({
            "trip_id": trip_id,
            "observations": trip_obs_cnt,
            "clf_only_accuracy": clf_acc,
            "hmm_accuracy": hmm_acc,
            "acc_improvement": hmm_acc - clf_acc,
            "clf_continuity": clf_cont,
            "hmm_continuity": hmm_cont,
            "continuity_improvement": hmm_cont - clf_cont,
            "decode_time_ms": decode_time_ms
        })

    overall_clf_acc = clf_only_correct / total_obs if total_obs > 0 else 0.0
    overall_hmm_acc = hmm_correct / total_obs if total_obs > 0 else 0.0
    overall_clf_cont = np.mean(all_clf_continuity)
    overall_hmm_cont = np.mean(all_hmm_continuity)

    res_df = pd.DataFrame(comparison_results)
    out_csv = os.path.join(RESULTS_DIR, f"hmm_vs_classifier_{partition}.csv")
    res_df.to_csv(out_csv, index=False)

    print("\n--- Summary Performance Comparison ---")
    print(f"Overall Observations Evaluated: {total_obs}")
    print(f"Classifier-Only Top-1 Accuracy:  {overall_clf_acc * 100.0:.2f}%")
    print(f"Classifier + HMM Top-1 Accuracy: {overall_hmm_acc * 100.0:.2f}%")
    print(f"Accuracy Difference:             {(overall_hmm_acc - overall_clf_acc) * 100.0:+.2f}%")
    print(f"\nClassifier-Only Route Continuity: {overall_clf_cont * 100.0:.2f}%")
    print(f"Classifier + HMM Route Continuity:{overall_hmm_cont * 100.0:.2f}%")
    print(f"Continuity Improvement:           {(overall_hmm_cont - overall_clf_cont) * 100.0:+.2f}%")

    # Save HMM configuration artifact
    hmm_config = {
        "sigma_z": decoder.sigma_z,
        "beta": decoder.beta,
        "same_arc_prob": decoder.same_arc_prob,
        "direct_conn_prob": decoder.direct_conn_prob,
        "gap_threshold_s": decoder.gap_threshold_s,
        "overall_clf_acc": overall_clf_acc,
        "overall_hmm_acc": overall_hmm_acc,
        "overall_clf_continuity": overall_clf_cont,
        "overall_hmm_continuity": overall_hmm_cont
    }
    with open(os.path.join(ARTIFACTS_DIR, "hmm_config.json"), "w") as f:
        json.dump(hmm_config, f, indent=2)

    return res_df, hmm_config

if __name__ == "__main__":
    evaluate_hmm_reconstruction("student_test")
