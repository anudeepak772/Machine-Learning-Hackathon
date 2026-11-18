"""
PathSnap - Distributed Map-Matching FastAPI Backend (Laptop 1)
Serves model inference, Viterbi route reconstruction, server-side accuracy evaluation,
and Folium interactive visualization.
"""

import os
import sys
import time
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import folium

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from 02_preprocessing.feature_extractor import FeatureExtractor
from 05_hmm.viterbi_decoder import ViterbiMapMatchingDecoder

# Configure logging for evaluator inspection
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Laptop 1 API) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "results", "api_server.log"))
    ]
)
logger = logging.getLogger("PathSnapAPI")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="PES University ML Hackathon UE24CS352A - Laptop 1 Inference & Evaluation Server",
    version="1.0.0"
)

# Enable CORS for cross-laptop network communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_STATE = {}

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

class PredictRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = "student_test"

class PredictResponse(BaseModel):
    trip_id: str
    partition: str
    num_gps_observations: int
    predicted_route_length: int
    pointwise_accuracy: float
    route_continuity: float
    processing_time_ms: float
    predicted_arcs: List[int]
    folium_map_html: str

@app.on_event("startup")
def load_artifacts():
    logger.info("Initializing PathSnap Backend on Laptop 1...")
    
    # 1. Load Preprocessor
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    if not os.path.exists(prep_path):
        logger.error(f"Preprocessor not found at {prep_path}")
        raise RuntimeError("Preprocessor artifact missing!")
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)
    MODEL_STATE["scaler"] = prep_data["scaler"]
    logger.info("Fitted preprocessor successfully loaded.")

    # 2. Load Final Model
    model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, "baseline_best_model.pkl")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        model_name = "Baseline Best Model"
    else:
        with open(model_path, "rb") as f:
            saved_obj = pickle.load(f)
            clf = saved_obj["model"]
            model_name = saved_obj.get("model_type", "Tuned Model")

    MODEL_STATE["model"] = clf
    MODEL_STATE["model_name"] = model_name
    logger.info(f"Model successfully loaded: {model_name}")

    # 3. Load Feature Extractor & Viterbi Decoder
    MODEL_STATE["extractor"] = FeatureExtractor(search_radius_m=50.0, max_candidates=12)
    MODEL_STATE["decoder"] = ViterbiMapMatchingDecoder(sigma_z=8.0, same_arc_prob=0.88, direct_conn_prob=1.0)
    logger.info("Viterbi HMM Decoder initialized and ready for requests.")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Distributed Map-Matching API",
        "host": "Laptop 1",
        "model": MODEL_STATE.get("model_name", "Not Loaded")
    }

@app.get("/trips")
def list_trips(partition: str = "student_test"):
    """Lists available evaluation trips for the UI dropdown."""
    folder = os.path.join(DATA_DIR, partition)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Partition folder {partition} not found.")
    trips = sorted(os.listdir(folder))
    return {
        "partition": partition,
        "count": len(trips),
        "trips": trips
    }

def generate_interactive_folium_map(track, nodes, arcs, predicted_arcs, ground_truth_arcs=None):
    """Generates an interactive Folium map visualization comparing GPS, predicted route, and GT."""
    center_lat = sum(p[1] for p in track) / len(track)
    center_lon = sum(p[0] for p in track) / len(track)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 1. Road Network Arcs (Grey)
    fg_road = folium.FeatureGroup(name="Local Road Network", show=False)
    margin = 0.015
    lat_min, lat_max = min(p[1] for p in track) - margin, max(p[1] for p in track) + margin
    lon_min, lon_max = min(p[0] for p in track) - margin, max(p[0] for p in track) + margin

    drawn_count = 0
    for arc_id, (u, v) in enumerate(arcs):
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        if lat_min <= u_lat <= lat_max and lon_min <= u_lon <= lon_max:
            folium.PolyLine([[u_lat, u_lon], [v_lat, v_lon]], color="#999999", weight=1.5, opacity=0.4).add_to(fg_road)
            drawn_count += 1
            if drawn_count > 10000:
                break
    fg_road.add_to(m)

    # 2. Predicted Route (Deep Blue / Indigo)
    fg_pred = folium.FeatureGroup(name="Predicted Route (Blue)", show=True)
    pred_lines = []
    for arc_id in predicted_arcs:
        if arc_id < len(arcs):
            u, v = arcs[arc_id]
            u_lon, u_lat = nodes[u]
            v_lon, v_lat = nodes[v]
            pred_lines.append([[u_lat, u_lon], [v_lat, v_lon]])
    for line in pred_lines:
        folium.PolyLine(line, color="#0044FF", weight=4.0, opacity=0.85).add_to(fg_pred)
    fg_pred.add_to(m)

    # 3. Ground Truth Reference Route (Green, server-side overlay)
    if ground_truth_arcs:
        fg_gt = folium.FeatureGroup(name="Reference Route (Green)", show=True)
        gt_lines = []
        for arc_id in ground_truth_arcs:
            if arc_id is not None and arc_id < len(arcs):
                u, v = arcs[arc_id]
                u_lon, u_lat = nodes[u]
                v_lon, v_lat = nodes[v]
                gt_lines.append([[u_lat, u_lon], [v_lat, v_lon]])
        for line in gt_lines:
            folium.PolyLine(line, color="#00BB00", weight=4.0, opacity=0.6, dash_array="6, 6").add_to(fg_gt)
        fg_gt.add_to(m)

    # 4. Raw GPS Track (Orange points)
    fg_gps = folium.FeatureGroup(name="Observed GPS Points (Orange)", show=True)
    gps_coords = []
    for idx, p in enumerate(track):
        coord = [p[1], p[0]]
        gps_coords.append(coord)
        if idx % 5 == 0 or idx == 0 or idx == len(track) - 1:
            folium.CircleMarker(
                location=coord,
                radius=3,
                color="#FF8800",
                fill=True,
                fill_color="#FF8800",
                fill_opacity=0.8,
                popup=f"Obs #{idx}<br>Time: {p[2]}s"
            ).add_to(fg_gps)
    folium.PolyLine(gps_coords, color="#FFAA33", weight=1.2, opacity=0.6).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

@app.post("/predict", response_model=PredictResponse)
def predict_trip(req: PredictRequest, request: Request):
    t_start = time.time()
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming prediction request from {client_ip} for Trip '{req.trip_id}' (partition: {req.partition})")

    trip_folder = os.path.join(DATA_DIR, req.partition, req.trip_id)
    if not os.path.exists(trip_folder):
        logger.error(f"Trip folder not found: {trip_folder}")
        raise HTTPException(status_code=404, detail=f"Trip ID '{req.trip_id}' not found in {req.partition}")

    # Load raw files
    track_p = os.path.join(trip_folder, f"{req.trip_id}.track")
    nodes_p = os.path.join(trip_folder, f"{req.trip_id}.nodes")
    arcs_p  = os.path.join(trip_folder, f"{req.trip_id}.arcs")
    route_p = os.path.join(trip_folder, f"{req.trip_id}.route")

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

    # 1. Feature Extraction
    extractor = MODEL_STATE["extractor"]
    scaler = MODEL_STATE["scaler"]
    clf = MODEL_STATE["model"]
    decoder = MODEL_STATE["decoder"]

    df, miss_cnt, total_pts = extractor.extract_features_for_trip(
        trip_id=req.trip_id, track=track, nodes=nodes, arcs=arcs, route=route
    )

    # 2. Classifier Prediction
    X = scaler.transform(df[FEATURE_COLUMNS])
    df['clf_prob'] = clf.predict_proba(X)[:, 1]

    # 3. Structure Observations for Viterbi
    obs_groups = df.groupby('obs_idx')
    obs_candidates_list = []
    ground_truth_arcs = []

    for obs_idx, obs_cands in obs_groups:
        obs_candidates_list.append(obs_cands.to_dict('records'))
        gt_row = obs_cands[obs_cands['label'] == 1]
        ground_truth_arcs.append(gt_row.iloc[0]['arc_id'] if len(gt_row) > 0 else None)

    # 4. HMM Viterbi Route Reconstruction
    predicted_arcs = decoder.decode(obs_candidates_list, arcs, nodes)

    # 5. Server-Side Accuracy Evaluation (Ground truth never sent directly)
    valid_gt = [gt for gt in ground_truth_arcs if gt is not None]
    if valid_gt:
        correct = sum(1 for p, gt in zip(predicted_arcs, ground_truth_arcs) if gt is not None and p == gt)
        accuracy = correct / len(valid_gt)
    else:
        accuracy = 1.0

    # Route Continuity
    conn_count = 0
    total_trans = len(predicted_arcs) - 1
    for i in range(total_trans):
        a1, a2 = predicted_arcs[i], predicted_arcs[i+1]
        if a1 == a2:
            conn_count += 1
        else:
            u1, v1 = arcs[a1]
            u2, v2 = arcs[a2]
            if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                conn_count += 1
    continuity = conn_count / total_trans if total_trans > 0 else 1.0

    # 6. Generate Folium HTML
    folium_html = generate_interactive_folium_map(track, nodes, arcs, predicted_arcs, ground_truth_arcs)

    processing_time_ms = (time.time() - t_start) * 1000.0
    logger.info(f"Processed Trip '{req.trip_id}': Acc={accuracy*100:.2f}%, Continuity={continuity*100:.2f}%, Time={processing_time_ms:.1f}ms")

    return PredictResponse(
        trip_id=req.trip_id,
        partition=req.partition,
        num_gps_observations=len(track),
        predicted_route_length=len(predicted_arcs),
        pointwise_accuracy=accuracy,
        route_continuity=continuity,
        processing_time_ms=processing_time_ms,
        predicted_arcs=predicted_arcs,
        folium_map_html=folium_html
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
