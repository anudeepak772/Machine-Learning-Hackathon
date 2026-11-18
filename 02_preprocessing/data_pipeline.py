"""
PathSnap - End-to-End Data Preprocessing & Feature Pipeline
Builds training and evaluation feature datasets with candidate generation and ground-truth alignment.
"""

import os
import glob
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    from .feature_extractor import FeatureExtractor
except (ImportError, ValueError):
    from feature_extractor import FeatureExtractor

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

def load_raw_trip_files(trip_folder, trip_id):
    """Loads raw .track, .nodes, .arcs, and .route if available."""
    track_p = os.path.join(trip_folder, f"{trip_id}.track")
    nodes_p = os.path.join(trip_folder, f"{trip_id}.nodes")
    arcs_p  = os.path.join(trip_folder, f"{trip_id}.arcs")
    route_p = os.path.join(trip_folder, f"{trip_id}.route")

    with open(track_p) as f:
        track = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(nodes_p) as f:
        nodes = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(arcs_p) as f:
        arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
    
    route = None
    if os.path.exists(route_p):
        with open(route_p) as f:
            route = [int(line.strip()) for line in f if line.strip()]

    return track, nodes, arcs, route

def build_dataset(partition="train", max_trips=None):
    """
    Extracts features for all trips in partition.
    Returns:
        full_df: combined DataFrame with all features and labels
        miss_rate: overall Candidate Miss Rate
    """
    folder = os.path.join(DATA_DIR, partition)
    trip_ids = sorted(os.listdir(folder))
    if max_trips:
        trip_ids = trip_ids[:max_trips]

    print(f"Building dataset for partition '{partition}' ({len(trip_ids)} trips)...")
    extractor = FeatureExtractor(search_radius_m=50.0, max_candidates=12)

    dfs = []
    total_misses = 0
    total_gps_points = 0
    t0 = time.time()

    for idx, trip_id in enumerate(trip_ids):
        trip_folder = os.path.join(folder, trip_id)
        track, nodes, arcs, route = load_raw_trip_files(trip_folder, trip_id)
        
        df, miss_cnt, n_pts = extractor.extract_features_for_trip(
            trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route
        )
        dfs.append(df)
        total_misses += miss_cnt
        total_gps_points += n_pts

        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(trip_ids)} trips ({elapsed:.1f}s)...")

    full_df = pd.concat(dfs, ignore_index=True)
    miss_rate = total_misses / total_gps_points if total_gps_points > 0 else 0.0

    print(f"Finished {partition}: {len(full_df)} candidate pairs generated.")
    print(f"Total GPS points: {total_gps_points}, Missed Candidates: {total_misses}")
    print(f"Candidate Miss Rate: {miss_rate * 100.0:.2f}% (Recall: {(1.0 - miss_rate) * 100.0:.2f}%)")

    return full_df, miss_rate

def run_preprocessing_pipeline():
    """Main pipeline execution."""
    print("=" * 60)
    print("STARTING FEATURE EXTRACTION & PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Train set
    train_df, train_miss_rate = build_dataset("train")
    train_feat_path = os.path.join(DATA_DIR, "train_features.pkl")
    train_df.to_pickle(train_feat_path)
    print(f"Saved training features to: {train_feat_path}")

    # 2. Student test set
    test_df, test_miss_rate = build_dataset("student_test")
    test_feat_path = os.path.join(DATA_DIR, "student_test_features.pkl")
    test_df.to_pickle(test_feat_path)
    print(f"Saved student test features to: {test_feat_path}")

    # 3. Fit Scaler strictly on train set (Leakage Prevention)
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    scaler_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "train_miss_rate": train_miss_rate,
            "test_miss_rate": test_miss_rate
        }, f)
    print(f"Saved fitted preprocessor artifact to: {scaler_path}")

    # Record metrics
    metrics = {
        "train_candidate_pairs": len(train_df),
        "test_candidate_pairs": len(test_df),
        "train_miss_rate": train_miss_rate,
        "test_miss_rate": test_miss_rate,
        "train_candidate_recall": 1.0 - train_miss_rate,
        "test_candidate_recall": 1.0 - test_miss_rate
    }
    with open(os.path.join(RESULTS_DIR, "candidate_generation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPreprocessing pipeline completed successfully!")

if __name__ == "__main__":
    import json
    run_preprocessing_pipeline()
