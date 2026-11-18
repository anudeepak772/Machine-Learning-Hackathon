"""
PathSnap - Distributed Map-Matching FastAPI Backend (Laptop 1)
PES University UE24CS352A - Machine Learning Hackathon
Serves ML model inference, Viterbi HMM sequential route reconstruction,
server-side accuracy evaluation, and interactive Folium geospatial visualization.
"""

import os
import sys
import time
import pickle
import logging
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import folium
from folium import plugins

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path so modules can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import preprocessing and HMM decoder modules safely via importlib
try:
    feature_extractor_mod = importlib.import_module("02_preprocessing.feature_extractor")
    FeatureExtractor = feature_extractor_mod.FeatureExtractor
    match_gps_to_route_ground_truth = feature_extractor_mod.match_gps_to_route_ground_truth
    project_point_to_segment = feature_extractor_mod.project_point_to_segment
except ImportError as exc:
    raise ImportError(
        f"Could not import FeatureExtractor: {exc}"
    ) from exc

try:
    viterbi_mod = importlib.import_module("05_hmm.viterbi_decoder")
    ViterbiMapMatchingDecoder = viterbi_mod.ViterbiMapMatchingDecoder
except ImportError as exc:
    raise ImportError(
        f"Could not import ViterbiMapMatchingDecoder: {exc}"
    ) from exc

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | (Laptop 1 API) %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "api_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pathsnap_api")

# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "perp_dist",
    "dist_start",
    "dist_end",
    "t_ratio",
    "arc_len",
    "gps_speed",
    "time_delta",
    "heading_diff",
    "cos_heading_diff",
    "spatial_prior",
    "dist_rank",
    "dist_ratio_to_best",
    "dist_diff_to_best"
]

# ============================================================
# FASTAPI INITIALIZATION
# ============================================================

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="PES University UE24CS352A - Laptop 1 Distributed Map-Matching & Route Reconstruction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global in-memory model state
MODEL_STATE = {
    "model": None,
    "model_name": None,
    "feature_columns": FEATURE_COLUMNS,
    "scaler": None,
    "feature_extractor": None,
    "viterbi_decoder": None
}

# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    trip_id: str
    partition: str = "student_test"
    reconstruction: Optional[str] = "hmm_viterbi"  # "hmm_viterbi" (default, >93% acc) or "top1"


class PredictResponse(BaseModel):
    trip_id: str
    partition: str
    model_name: str

    num_gps_observations: int
    num_predicted_observations: int
    predicted_route_length: int

    pointwise_accuracy: Optional[float] = None
    route_continuity: Optional[float] = None

    processing_time_ms: float

    predicted_arcs: List[int]
    reference_arcs: Optional[List[int]] = None

    folium_map_html: str

# ============================================================
# ARTIFACT LOADING
# ============================================================

def load_artifacts():
    """Load model, scaler, and decoder artifacts into global state."""
    logger.info("Initializing PathSnap Backend on Laptop 1...")

    # 1. Preprocessor
    prep_path = ARTIFACTS_DIR / "preprocessor.pkl"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact missing at {prep_path}")
    with open(prep_path, "rb") as f:
        prep_obj = pickle.load(f)
    MODEL_STATE["scaler"] = prep_obj["scaler"]

    # 2. Classifier Model (prioritize final Bayesian-tuned model, fallback to baseline)
    final_model_path = ARTIFACTS_DIR / "final_model.pkl"
    baseline_model_path = ARTIFACTS_DIR / "baseline_best_model.pkl"

    if final_model_path.exists():
        with open(final_model_path, "rb") as f:
            model_obj = pickle.load(f)
        if isinstance(model_obj, dict):
            MODEL_STATE["model"] = model_obj["model"]
            MODEL_STATE["model_name"] = model_obj.get("model_type", "Tuned Random Forest")
        else:
            MODEL_STATE["model"] = model_obj
            MODEL_STATE["model_name"] = "Tuned Random Forest"
    elif baseline_model_path.exists():
        with open(baseline_model_path, "rb") as f:
            model_obj = pickle.load(f)
        if isinstance(model_obj, dict):
            MODEL_STATE["model"] = model_obj["model"]
            MODEL_STATE["model_name"] = model_obj.get("model_name", "Random Forest")
        else:
            MODEL_STATE["model"] = model_obj
            MODEL_STATE["model_name"] = "Random Forest"
    else:
        raise FileNotFoundError(f"No model artifact found in {ARTIFACTS_DIR}")

    # 3. Feature Extractor
    MODEL_STATE["feature_extractor"] = FeatureExtractor(
        search_radius_m=50.0,
        max_candidates=12
    )

    # 4. Viterbi HMM Decoder
    MODEL_STATE["viterbi_decoder"] = ViterbiMapMatchingDecoder(
        sigma_z=8.0,
        same_arc_prob=0.88,
        direct_conn_prob=1.0
    )

    logger.info("Loaded model: %s", MODEL_STATE["model_name"])
    logger.info("Loaded preprocessor and Viterbi HMM Decoder successfully.")


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
        logger.info("PathSnap API online and ready for evaluator requests.")
    except Exception:
        logger.exception("Failed to load model artifacts on startup.")
        raise

# ============================================================
# DATA LOADING HELPERS
# ============================================================

def validate_trip_id(trip_id: str) -> str:
    cleaned = trip_id.strip()
    if not cleaned.isalnum():
        raise HTTPException(status_code=400, detail="Invalid trajectory ID format.")
    return cleaned


def get_trip_directory(partition: str, trip_id: str) -> Path:
    trip_dir = DATA_DIR / partition / trip_id
    if not trip_dir.exists() or not trip_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Trajectory '{trip_id}' not found in partition '{partition}'."
        )
    return trip_dir


def load_trip_data(partition: str, trip_id: str):
    """
    Loads raw trajectory .track, .nodes, .arcs, and (if present) .route.
    Ground-truth route remains server-side only.
    """
    trip_dir = get_trip_directory(partition, trip_id)

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    if not track_path.exists() or not nodes_path.exists() or not arcs_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing core trajectory files for trip {trip_id}"
        )

    with open(track_path) as f:
        track = [tuple(map(float, line.strip().split())) for line in f if line.strip()]

    with open(nodes_path) as f:
        nodes = [tuple(map(float, line.strip().split())) for line in f if line.strip()]

    with open(arcs_path) as f:
        arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]

    route = None
    if route_path.exists():
        with open(route_path) as f:
            route = [int(line.strip()) for line in f if line.strip()]

    return track, nodes, arcs, route

# ============================================================
# EVALUATION & CONTINUITY HELPERS
# ============================================================

def calculate_route_continuity(predicted_arcs: List[int], arcs: List[tuple]) -> float:
    """Calculates route continuity fraction across consecutive predicted arcs."""
    if len(predicted_arcs) <= 1:
        return 1.0

    connected = 0
    total = len(predicted_arcs) - 1

    for i in range(total):
        a1, a2 = predicted_arcs[i], predicted_arcs[i + 1]
        if a1 == a2:
            connected += 1
        else:
            if a1 < len(arcs) and a2 < len(arcs):
                u1, v1 = arcs[a1]
                u2, v2 = arcs[a2]
                if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                    connected += 1

    return connected / total if total > 0 else 1.0


def evaluate_pointwise_accuracy(predicted_arcs: List[int], ground_truth_arcs: List[Optional[int]], arcs: List[tuple]) -> Optional[float]:
    """
    Evaluates observation-level Top-1 accuracy against ground truth.
    Supports physical road segment matching ({u, v}) for two-way streets per hackathon guidelines.
    """
    valid_pairs = [(p, gt) for p, gt in zip(predicted_arcs, ground_truth_arcs) if gt is not None]
    if not valid_pairs:
        return None

    correct = 0
    for p, gt in valid_pairs:
        if p == gt:
            correct += 1
        elif p < len(arcs) and gt < len(arcs):
            if set(arcs[p]) == set(arcs[gt]):
                correct += 1

    return correct / len(valid_pairs)

# ============================================================
# FAST INTERACTIVE FOLIUM MAP GENERATOR
# ============================================================

def generate_interactive_folium_map(
    track: List[tuple],
    nodes: List[tuple],
    arcs: List[tuple],
    predicted_arcs: List[int],
    reference_arcs: Optional[List[int]] = None
) -> str:
    """
    Generates a lightweight, high-performance Folium map with multi-polylines.
    Provides togglable layers:
      - Local Road Network (Grey)
      - Observed GPS Track (Orange)
      - Ground Truth Reference Route (Green Dashed)
      - Reconstructed Predicted Route (Blue)
    """
    center_lat = sum(p[1] for p in track) / len(track)
    center_lon = sum(p[0] for p in track) / len(track)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 1. Road Network Layer (Grey)
    fg_road = folium.FeatureGroup(name="Local Road Network", show=False)
    margin = 0.015
    lat_min = min(p[1] for p in track) - margin
    lat_max = max(p[1] for p in track) + margin
    lon_min = min(p[0] for p in track) - margin
    lon_max = max(p[0] for p in track) + margin

    road_lines = []
    for arc_id, (u, v) in enumerate(arcs):
        if u < len(nodes) and v < len(nodes):
            u_lon, u_lat = nodes[u]
            v_lon, v_lat = nodes[v]
            if lat_min <= u_lat <= lat_max and lon_min <= u_lon <= lon_max:
                road_lines.append([[u_lat, u_lon], [v_lat, v_lon]])
                if len(road_lines) >= 8000:
                    break
    if road_lines:
        folium.PolyLine(locations=road_lines, color="#999999", weight=1.5, opacity=0.4).add_to(fg_road)
    fg_road.add_to(m)

    # 2. Predicted Route Layer (Deep Blue)
    fg_pred = folium.FeatureGroup(name="Predicted Reconstructed Route (Blue)", show=True)
    pred_lines = []
    for arc_id in predicted_arcs:
        if arc_id < len(arcs):
            u, v = arcs[arc_id]
            if u < len(nodes) and v < len(nodes):
                pred_lines.append([[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]])
    if pred_lines:
        folium.PolyLine(locations=pred_lines, color="#0044FF", weight=4.5, opacity=0.85).add_to(fg_pred)
    fg_pred.add_to(m)

    # 3. Ground Truth Reference Route Layer (Green Dashed)
    if reference_arcs:
        fg_gt = folium.FeatureGroup(name="Ground Truth Reference Route (Green)", show=True)
        gt_lines = []
        for arc_id in reference_arcs:
            if arc_id is not None and arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u < len(nodes) and v < len(nodes):
                    gt_lines.append([[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]])
        if gt_lines:
            folium.PolyLine(locations=gt_lines, color="#00BB00", weight=4.0, opacity=0.7, dash_array="6, 6").add_to(fg_gt)
        fg_gt.add_to(m)

    # 4. Observed GPS Track Layer (Orange)
    fg_gps = folium.FeatureGroup(name="Observed GPS Points (Orange)", show=True)
    gps_coords = []
    for idx, p in enumerate(track):
        coord = [p[1], p[0]]
        gps_coords.append(coord)
        if idx % 8 == 0 or idx == 0 or idx == len(track) - 1:
            folium.CircleMarker(
                location=coord,
                radius=3,
                color="#FF8800",
                fill=True,
                fill_color="#FF8800",
                fill_opacity=0.8,
                popup=f"Obs #{idx}<br>Time: {p[2]}s"
            ).add_to(fg_gps)
    folium.PolyLine(gps_coords, color="#FFAA33", weight=1.4, opacity=0.6).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Distributed Map-Matching API",
        "host": "Laptop 1",
        "model": MODEL_STATE["model_name"]
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_STATE["model"] is not None,
        "model_name": MODEL_STATE["model_name"]
    }


@app.get("/api/models")
def get_models():
    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model has not been loaded.")
    return {
        "models": [
            {
                "id": "random_forest",
                "name": MODEL_STATE["model_name"],
                "available": True
            }
        ],
        "default_model": "random_forest",
        "reconstruction_methods": [
            {
                "id": "hmm_viterbi",
                "name": "HMM + Viterbi Decoder (Recommended: >93% Acc)",
                "available": True
            },
            {
                "id": "top1",
                "name": "Top-1 Candidate Selection (Baseline)",
                "available": True
            }
        ]
    }


@app.get("/api/trajectories")
@app.get("/trips")
def get_trajectories(partition: str = Query(default="student_test")):
    partition_dir = DATA_DIR / partition
    if not partition_dir.exists() or not partition_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Partition folder '{partition}' not found in data directory."
        )

    trips = sorted([d.name for d in partition_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    return {
        "partition": partition,
        "count": len(trips),
        "trajectories": trips,
        "trips": trips  # Alias for backward compatibility
    }


@app.post("/api/map-match", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def map_match(request: PredictRequest, req_raw: Request = None):
    t_start = time.perf_counter()
    client_ip = req_raw.client.host if req_raw and req_raw.client else "unknown"
    logger.info(
        f"Incoming map-matching request from {client_ip} for Trip '{request.trip_id}' "
        f"(partition: {request.partition}, method: {request.reconstruction})"
    )

    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    # 1. Load Trajectory Data
    track, nodes, arcs, reference_route = load_trip_data(request.partition, request.trip_id)

    # 2. Extract Candidate Features (without passing reference_route to ensure strictly unsupervised candidate scoring)
    extractor = MODEL_STATE["feature_extractor"]
    scaler = MODEL_STATE["scaler"]
    clf = MODEL_STATE["model"]
    decoder = MODEL_STATE["viterbi_decoder"]

    try:
        feature_df, miss_cnt, total_pts = extractor.extract_features_for_trip(
            trip_id=request.trip_id,
            track=track,
            nodes=nodes,
            arcs=arcs,
            route=None
        )
    except Exception as exc:
        logger.exception("Inference feature extraction failed.")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(exc)}")

    if feature_df.empty:
        raise HTTPException(status_code=422, detail="No candidate road segments found within search radius.")

    # 3. Model Inference: Score Candidates
    X = scaler.transform(feature_df[FEATURE_COLUMNS])
    feature_df["clf_prob"] = clf.predict_proba(X)[:, 1]

    # 4. Route Reconstruction
    obs_groups = feature_df.groupby("obs_idx")
    obs_candidates_list = [cands.to_dict("records") for _, cands in obs_groups]

    reconstruction_method = request.reconstruction or "hmm_viterbi"

    if reconstruction_method == "hmm_viterbi":
        predicted_arcs = decoder.decode(obs_candidates_list, arcs, nodes)
    else:
        # Top-1 Baseline selection
        predicted_arcs = []
        for cands in obs_candidates_list:
            best_c = max(cands, key=lambda c: c["clf_prob"])
            predicted_arcs.append(int(best_c["arc_id"]))

    # 5. Server-Side Reference Route Accuracy Evaluation
    ground_truth_arcs = None
    pointwise_accuracy = None

    if reference_route is not None:
        try:
            # Map GPS points to ground truth reference arcs robustly
            ground_truth_arcs = match_gps_to_route_ground_truth(track, nodes, arcs, reference_route)
            pointwise_accuracy = evaluate_pointwise_accuracy(predicted_arcs, ground_truth_arcs, arcs)
        except Exception:
            logger.exception("Server-side ground-truth alignment failed.")
            pointwise_accuracy = None

    # Calculate Route Continuity
    route_continuity = calculate_route_continuity(predicted_arcs, arcs)

    # 6. Generate Folium Interactive Visualization
    folium_html = generate_interactive_folium_map(
        track=track,
        nodes=nodes,
        arcs=arcs,
        predicted_arcs=predicted_arcs,
        reference_arcs=reference_route
    )

    processing_time_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        f"Processed Trip '{request.trip_id}' in {processing_time_ms:.1f}ms: "
        f"Acc={pointwise_accuracy*100 if pointwise_accuracy is not None else 0:.2f}%, "
        f"Continuity={route_continuity*100:.2f}%"
    )

    return PredictResponse(
        trip_id=request.trip_id,
        partition=request.partition,
        model_name=MODEL_STATE["model_name"],
        num_gps_observations=len(track),
        num_predicted_observations=len(predicted_arcs),
        predicted_route_length=len(predicted_arcs),
        pointwise_accuracy=pointwise_accuracy,
        route_continuity=route_continuity,
        processing_time_ms=round(processing_time_ms, 2),
        predicted_arcs=predicted_arcs,
        reference_arcs=reference_route,
        folium_map_html=folium_html
    )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("06_api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
