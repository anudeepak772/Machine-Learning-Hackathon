
import os
import sys
import time
import pickle
import logging
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import folium

from fastapi import FastAPI, HTTPException, Query
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

# The preprocessing directory starts with a number, so add it
# directly to sys.path instead of importing it as a package.
PREPROCESSING_DIR = PROJECT_ROOT / "02_preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

try:
    from feature_extractor import FeatureExtractor
except ImportError as exc:
    raise ImportError(
        "Could not import FeatureExtractor. Check that "
        "02_preprocessing/feature_extractor.py exists and that "
        "its dependencies are installed."
    ) from exc


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "api_server.log"),
        logging.StreamHandler()
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
    description="API for GPS trajectory map matching and evaluation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

MODEL_STATE = {
    "model": None,
    "model_name": None,
    "feature_columns": None,
    "scaler": None,
    "feature_extractor": None
}


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    trip_id: str
    partition: str = "student_test"


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

    predicted_arcs: list[int]
    reference_arcs: Optional[list[int]] = None

    folium_map_html: str


# ============================================================
# ARTIFACT LOADING
# ============================================================

def load_artifacts():
    """
    Load the trained Random Forest artifact and preprocessing scaler.

    baseline_best_model.pkl is a dictionary containing:
        model
        model_name
        feature_columns
    """

    model_path = ARTIFACTS_DIR / "baseline_best_model.pkl"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: {preprocessor_path}"
        )

    # Load model artifact
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    if not isinstance(model_artifact, dict):
        raise ValueError(
            "Expected baseline_best_model.pkl to contain a dictionary."
        )

    required_keys = {"model", "model_name", "feature_columns"}

    if not required_keys.issubset(model_artifact.keys()):
        raise ValueError(
            f"Model artifact is missing keys: "
            f"{required_keys - model_artifact.keys()}"
        )

    MODEL_STATE["model"] = model_artifact["model"]
    MODEL_STATE["model_name"] = model_artifact["model_name"]
    MODEL_STATE["feature_columns"] = model_artifact["feature_columns"]

    # Load preprocessing artifact
    with open(preprocessor_path, "rb") as f:
        preprocessor_artifact = pickle.load(f)

    if not isinstance(preprocessor_artifact, dict):
        raise ValueError(
            "Expected preprocessor.pkl to contain a dictionary."
        )

    MODEL_STATE["scaler"] = preprocessor_artifact["scaler"]

    # Verify feature consistency
    if MODEL_STATE["feature_columns"] != FEATURE_COLUMNS:
        logger.warning(
            "Feature columns in model artifact differ from "
            "the API's expected feature columns."
        )

    MODEL_STATE["feature_extractor"] = FeatureExtractor(
        search_radius_m=50.0,
        max_candidates=12
    )

    logger.info(
        "Loaded model: %s",
        MODEL_STATE["model_name"]
    )

    logger.info(
        "Loaded %d model features",
        len(MODEL_STATE["feature_columns"])
    )


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
        logger.info("PathSnap API started successfully.")
    except Exception:
        logger.exception("Failed to load model artifacts.")
        raise


# ============================================================
# DATASET HELPERS
# ============================================================

VALID_PARTITIONS = {"train", "student_test"}


def validate_trip_id(trip_id: str):
    """
    Prevent path traversal and accept only numeric trajectory IDs.
    """

    if not trip_id.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid trajectory ID."
        )


def get_trip_directory(partition: str, trip_id: str) -> Path:

    if partition not in VALID_PARTITIONS:
        raise HTTPException(
            status_code=400,
            detail="Partition must be 'train' or 'student_test'."
        )

    validate_trip_id(trip_id)

    trip_dir = DATA_DIR / partition / trip_id

    if not trip_dir.exists() or not trip_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Trajectory {trip_id} not found in {partition}."
        )

    return trip_dir


def load_trip_data(partition: str, trip_id: str):
    """
    Load trajectory GPS points, road nodes, road arcs and reference route.

    Important:
    The reference route is loaded here, but it must not be passed
    into the inference feature extraction call.
    """

    trip_dir = get_trip_directory(partition, trip_id)

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    required_files = [
        track_path,
        nodes_path,
        arcs_path,
        route_path
    ]

    for path in required_files:
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Missing dataset file: {path.name}"
            )

    try:
        track = pd.read_csv(
            track_path,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        ).to_numpy(dtype=float)

        nodes = pd.read_csv(
            nodes_path,
            sep=r"\s+",
            header=None
        ).to_numpy(dtype=float)

        arcs = pd.read_csv(
            arcs_path,
            sep=r"\s+",
            header=None
        ).to_numpy(dtype=int)

        route = pd.read_csv(
            route_path,
            sep=r"\s+",
            header=None
        ).iloc[:, 0].to_numpy(dtype=int)

    except Exception as exc:
        logger.exception("Failed to read trajectory files.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read trajectory data: {str(exc)}"
        )

    if len(track) == 0:
        raise HTTPException(
            status_code=400,
            detail="Trajectory contains no GPS observations."
        )

    if len(nodes) == 0 or len(arcs) == 0:
        raise HTTPException(
            status_code=400,
            detail="Trajectory has an empty road network."
        )

    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise HTTPException(
            status_code=400,
            detail="Trajectory contains invalid coordinates."
        )

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise HTTPException(
            status_code=400,
            detail="Road network contains invalid node references."
        )

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise HTTPException(
            status_code=400,
            detail="Reference route contains invalid arc IDs."
        )

    return track, nodes, arcs, route


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route=None
):
    """
    Call the existing FeatureExtractor.

    The method signature is inspected so that common parameter names
    can be mapped to the corresponding data.

    Verify this integration against the actual feature_extractor.py.
    """

    extractor = MODEL_STATE["feature_extractor"]

    if not hasattr(extractor, "extract_features_for_trip"):
        raise RuntimeError(
            "FeatureExtractor does not have an "
            "extract_features_for_trip() method."
        )

    method = extractor.extract_features_for_trip
    signature = inspect.signature(method)

    available_values = {
        "trip_id": trip_id,
        "track": track,
        "gps": track,
        "trajectory": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "ground_truth_route": route
    }

    kwargs = {}

    for parameter_name, parameter in signature.parameters.items():

        if parameter_name in available_values:
            kwargs[parameter_name] = available_values[parameter_name]

        elif parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                "Unrecognized required FeatureExtractor parameter: "
                f"{parameter_name}. Check its method signature."
            )

    result = method(**kwargs)

    # Expected existing output:
    # (feature_dataframe, miss_count, num_points)
    if isinstance(result, tuple):
        feature_df = result[0]
        miss_count = result[1] if len(result) > 1 else None
        num_points = result[2] if len(result) > 2 else len(track)

    else:
        feature_df = result
        miss_count = None
        num_points = len(track)

    if not isinstance(feature_df, pd.DataFrame):
        raise RuntimeError(
            "FeatureExtractor must return a pandas DataFrame "
            "or a tuple whose first item is a DataFrame."
        )

    return feature_df, miss_count, num_points


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_top1(feature_df: pd.DataFrame):
    """
    Predict the most likely candidate arc for each GPS observation.

    The classifier is applied to every candidate. The candidate with
    the highest positive-class probability is selected per observation.
    """

    if feature_df.empty:
        raise ValueError(
            "Feature extraction returned no candidate road segments."
        )

    model = MODEL_STATE["model"]
    scaler = MODEL_STATE["scaler"]
    feature_columns = MODEL_STATE["feature_columns"]

    missing_columns = [
        col for col in feature_columns
        if col not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    X = feature_df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isnull().any().any():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)

    # Do not assume that the positive class is always column 1.
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Classifier does not contain positive class 1: {classes}"
        )

    positive_class_index = classes.index(1)

    feature_df = feature_df.copy()

    feature_df["prediction_probability"] = (
        probabilities[:, positive_class_index]
    )

    # Select the highest-probability candidate for each GPS observation.
    selected_indices = (
        feature_df
        .groupby("obs_idx")["prediction_probability"]
        .idxmax()
    )

    predictions = (
        feature_df.loc[selected_indices]
        .sort_values("obs_idx")
        .reset_index(drop=True)
    )

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def calculate_pointwise_accuracy(
    predictions: pd.DataFrame,
    ground_truth_df: pd.DataFrame
):
    """
    Compare predicted arcs against ground-truth matched arcs.

    Ground-truth labels are generated only after inference.
    """

    if "label" not in ground_truth_df.columns:
        return None

    if "obs_idx" not in ground_truth_df.columns:
        return None

    if "arc_id" not in ground_truth_df.columns:
        return None

    ground_truth = (
        ground_truth_df[ground_truth_df["label"] == 1]
        .drop_duplicates(subset=["obs_idx"])
        [["obs_idx", "arc_id"]]
        .rename(columns={"arc_id": "true_arc"})
    )

    if ground_truth.empty:
        return None

    comparison = predictions[["obs_idx", "arc_id"]].merge(
        ground_truth,
        on="obs_idx",
        how="inner"
    )

    if comparison.empty:
        return None

    return float(
        (comparison["arc_id"] == comparison["true_arc"]).mean()
    )


def calculate_route_continuity(
    predicted_arcs: list[int],
    arcs: np.ndarray
):
    """
    Calculate the fraction of consecutive predicted arc pairs
    that are connected in the directed road network.

    A transition is connected when:
        current arc == next arc
    or
        current arc's destination == next arc's origin
    """

    if len(predicted_arcs) < 2:
        return None

    connected = 0
    total = len(predicted_arcs) - 1

    for i in range(total):

        current_arc = predicted_arcs[i]
        next_arc = predicted_arcs[i + 1]

        if current_arc == next_arc:
            connected += 1
            continue

        current_destination = arcs[current_arc][1]
        next_origin = arcs[next_arc][0]

        if current_destination == next_origin:
            connected += 1

    return connected / total


# ============================================================
# MAP GENERATION
# ============================================================

def arc_coordinates(arc_id: int, nodes: np.ndarray, arcs: np.ndarray):
    """
    Convert an arc ID into its two endpoint coordinates.
    """

    start_node, end_node = arcs[arc_id]

    start = nodes[start_node]
    end = nodes[end_node]

    return [
        [float(start[1]), float(start[0])],
        [float(end[1]), float(end[0])]
    ]


def generate_interactive_folium_map(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    predicted_arcs: list[int],
    reference_arcs: Optional[list[int]] = None
):
    """
    Generate a Folium map with independently togglable layers:
      - GPS trajectory
      - Road network
      - Predicted route
      - Reference route
    """

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        control_scale=True
    )

    # --------------------------------------------------------
    # GPS trajectory
    # --------------------------------------------------------

    gps_layer = folium.FeatureGroup(
        name="GPS Trajectory",
        show=True
    )

    gps_points = [
        [float(row[1]), float(row[0])]
        for row in track
    ]

    folium.PolyLine(
        gps_points,
        color="orange",
        weight=3,
        opacity=0.8,
        tooltip="GPS trajectory"
    ).add_to(gps_layer)

    gps_layer.add_to(map_object)

    # --------------------------------------------------------
    # Road network
    # --------------------------------------------------------

    road_layer = folium.FeatureGroup(
        name="Road Network",
        show=True
    )

    min_lon = np.min(track[:, 0]) - 0.003
    max_lon = np.max(track[:, 0]) + 0.003
    min_lat = np.min(track[:, 1]) - 0.003
    max_lat = np.max(track[:, 1]) + 0.003

    # Draw only road segments whose bounding boxes overlap
    # the GPS trajectory's local bounding box.
    for arc_id, (u, v) in enumerate(arcs):

        start = nodes[u]
        end = nodes[v]

        arc_min_lon = min(start[0], end[0])
        arc_max_lon = max(start[0], end[0])
        arc_min_lat = min(start[1], end[1])
        arc_max_lat = max(start[1], end[1])

        if (
            arc_max_lon < min_lon
            or arc_min_lon > max_lon
            or arc_max_lat < min_lat
            or arc_min_lat > max_lat
        ):
            continue

        folium.PolyLine(
            [
                [float(start[1]), float(start[0])],
                [float(end[1]), float(end[0])]
            ],
            color="gray",
            weight=1,
            opacity=0.45
        ).add_to(road_layer)

    road_layer.add_to(map_object)

    # --------------------------------------------------------
    # Predicted route
    # --------------------------------------------------------

    predicted_layer = folium.FeatureGroup(
        name="Predicted Route",
        show=True
    )

    for arc_id in predicted_arcs:

        coordinates = arc_coordinates(
            arc_id,
            nodes,
            arcs
        )

        folium.PolyLine(
            coordinates,
            color="blue",
            weight=5,
            opacity=0.9,
            tooltip=f"Predicted arc {arc_id}"
        ).add_to(predicted_layer)

    predicted_layer.add_to(map_object)

    # --------------------------------------------------------
    # Reference route
    # --------------------------------------------------------

    if reference_arcs is not None:

        reference_layer = folium.FeatureGroup(
            name="Reference Route",
            show=False
        )

        for arc_id in reference_arcs:

            coordinates = arc_coordinates(
                arc_id,
                nodes,
                arcs
            )

            folium.PolyLine(
                coordinates,
                color="green",
                weight=4,
                opacity=0.8,
                tooltip=f"Reference arc {arc_id}"
            ).add_to(reference_layer)

        reference_layer.add_to(map_object)

    folium.LayerControl(
        collapsed=False
    ).add_to(map_object)

    return map_object.get_root().render()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "PathSnap Map-Matching API",
        "status": "running",
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
        raise HTTPException(
            status_code=503,
            detail="Model has not been loaded."
        )

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
                "id": "top1",
                "name": "Top-1 Candidate Selection",
                "available": True
            },
            {
                "id": "hmm_viterbi",
                "name": "HMM + Viterbi",
                "available": False
            }
        ]
    }


@app.get("/api/trajectories")
@app.get("/trips")
def get_trajectories(
    partition: str = Query(default="student_test")
):

    if partition not in VALID_PARTITIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid partition."
        )

    partition_dir = DATA_DIR / partition

    if not partition_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Partition directory not found: {partition}"
        )

    trips = sorted(
        [
            item.name
            for item in partition_dir.iterdir()
            if item.is_dir() and item.name.isdigit()
        ]
    )

    return {
        "partition": partition,
        "count": len(trips),
        "trajectories": trips
    }


@app.post("/api/map-match", response_model=PredictResponse)
@app.post("/predict", response_model=PredictResponse)
def map_match(request: PredictRequest):

    if MODEL_STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 1. Load trajectory data
    # --------------------------------------------------------

    track, nodes, arcs, reference_route = load_trip_data(
        request.partition,
        request.trip_id
    )

    # --------------------------------------------------------
    # 2. Extract inference features
    # --------------------------------------------------------

    # CRITICAL:
    # Do not pass reference_route into feature extraction.
    # The model must generate predictions without ground-truth labels.

    try:
        feature_df, miss_count, num_points = extract_features(
            request.trip_id,
            track,
            nodes,
            arcs,
            route=None
        )

    except Exception as exc:
        logger.exception("Inference feature extraction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(exc)}"
        )

    if feature_df.empty:
        raise HTTPException(
            status_code=422,
            detail="No candidate road segments were generated."
        )

    # --------------------------------------------------------
    # 3. Predict using Random Forest
    # --------------------------------------------------------

    try:
        predictions = predict_top1(feature_df)

    except Exception as exc:
        logger.exception("Model prediction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )

    predicted_arcs = (
        predictions["arc_id"]
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # 4. Evaluate after prediction
    # --------------------------------------------------------

    # Ground-truth matching is performed separately, after
    # the predictions have already been generated.

    pointwise_accuracy = None

    try:
        ground_truth_df, _, _ = extract_features(
            request.trip_id,
            track,
            nodes,
            arcs,
            route=reference_route
        )

        pointwise_accuracy = calculate_pointwise_accuracy(
            predictions,
            ground_truth_df
        )

    except Exception:
        logger.exception(
            "Ground-truth alignment failed. "
            "Returning predictions without pointwise accuracy."
        )

    route_continuity = calculate_route_continuity(
        predicted_arcs,
        arcs
    )

    # --------------------------------------------------------
    # 5. Generate map
    # --------------------------------------------------------

    try:
        folium_map_html = generate_interactive_folium_map(
            track,
            nodes,
            arcs,
            predicted_arcs,
            reference_route.tolist()
        )

    except Exception as exc:
        logger.exception("Map generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Map generation failed: {str(exc)}"
        )

    # --------------------------------------------------------
    # 6. Build response
    # --------------------------------------------------------

    processing_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "Trajectory %s processed in %.2f ms",
        request.trip_id,
        processing_time_ms
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
        reference_arcs=reference_route.tolist(),

        folium_map_html=folium_map_html
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
