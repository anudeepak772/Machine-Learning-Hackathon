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
