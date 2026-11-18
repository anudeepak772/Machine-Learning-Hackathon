
import json
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

API_BASE_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://localhost:8000"
).rstrip("/")

TIMEOUT = 180


# ============================================================
# API HELPERS
# ============================================================

def api_get(path: str):
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=20
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def unwrap_list(data, key):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        value = data.get(key, [])
        return value if isinstance(value, list) else []

    return []


def normalise_id(item):
    if isinstance(item, dict):
        return str(
            item.get(
                "trajectory_id",
                item.get("id", item.get("name", ""))
            )
        )

    return str(item)


def normalise_model(item):
    if isinstance(item, dict):
        slug = str(item.get("id", item.get("name", "")))

        label = str(
            item.get(
                "label",
                item.get("display_name", item.get("name", slug))
            )
        )

        available = bool(item.get("available", True))

        return {
            "id": slug,
            "label": label,
            "available": available
        }

    value = str(item)

    return {
        "id": value,
        "label": value.replace("_", " ").title(),
        "available": True
    }


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

st.title("🗺️ PathSnap")

st.caption(
    "GPS trajectory map matching and route reconstruction evaluation"
)

with st.sidebar:

    st.subheader("Backend connection")

    api_url_input = st.text_input(
        "Laptop 1 API Base URL",
        value=API_BASE_URL,
        help="Enter the IP address of Laptop 1 (e.g. http://192.168.1.5:8000 or http://localhost:8000)"
    )
    API_BASE_URL = api_url_input.rstrip("/")

    if st.button("Check API connection", use_container_width=True):

        try:
            health = api_get("/api/health")

            if health.get("status") == "healthy":
                st.success("API connected successfully.")
            else:
                st.warning(f"API response: {health}")

        except requests.RequestException as exc:
            st.error("Could not connect to FastAPI.")
            st.exception(exc)

    st.divider()

    st.subheader("Data Partition")
    partition_choice = st.selectbox(
        "Select Partition",
        ["evaluator_test", "student_test", "train"],
        index=0,
        help="Evaluator reserved test set or student testing partition"
    )

    st.divider()

    st.subheader("Map layers")

    show_gps = st.checkbox("GPS trajectory", value=True)
    show_predicted = st.checkbox("Predicted route", value=True)
    show_reference = st.checkbox("Reference route", value=True)
    show_road = st.checkbox("Road network", value=True)


# ============================================================
# LOAD TRAJECTORIES AND MODELS
# ============================================================

try:

    trajectory_payload = api_get(f"/api/trajectories?partition={partition_choice}")

    trajectories = unwrap_list(
        trajectory_payload,
        "trajectories"
    )

    model_payload = api_get("/api/models")

    models = [
        normalise_model(item)
        for item in unwrap_list(model_payload, "models")
    ]

except requests.RequestException as exc:

    st.error(
        "Could not load trajectories/models from the API. "
        "Start FastAPI and check PATHSNAP_API_URL."
    )

    st.exception(exc)
    st.stop()


if not trajectories:

    st.warning(
        "The API returned no trajectories. "
        "Check the backend dataset directory."
    )

    st.stop()


trajectory_ids = [
    normalise_id(item)
    for item in trajectories
]

trajectory_ids = [
    item for item in trajectory_ids if item
]


if not trajectory_ids:

    st.error("Invalid trajectory response from API.")
    st.stop()


model_options = [
    model for model in models
    if model["available"]
]


if not model_options:

    st.warning("No trained models are currently available.")
    st.stop()


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [1, 2],
    gap="large"
)


# ============================================================
# LEFT: CONFIGURATION
# ============================================================

with left:

    st.subheader("Run map matching")

    selected_trip = st.selectbox(
        f"Dataset trajectory ({len(trajectory_ids)} available in {partition_choice})",
        trajectory_ids,
        index=0 if trajectory_ids else None,
        help="Select a trajectory for map matching."
    )

    model_labels = {
        model["label"]: model["id"]
        for model in model_options
    }

    default_model_index = next(
        (
            i for i, model in enumerate(model_options)
            if model["id"] == "random_forest"
        ),
        0
    )

    selected_model_label = st.selectbox(
        "Model",
        list(model_labels.keys()),
        index=default_model_index
    )

    selected_model = model_labels[selected_model_label]

    reconstruction = st.selectbox(
        "Route reconstruction",
        options=["hmm_viterbi", "top1"],
        format_func=lambda value: (
            "HMM + Viterbi Decoder (Recommended: >93% Acc)"
            if value == "hmm_viterbi"
            else "Top-1 Candidate Selection (Baseline)"
        ),
        index=0
    )

    run_clicked = st.button(
        "Run map matching",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.subheader("Model comparison")

    st.caption(
        "Model comparison will be enabled once the backend "
        "supports the comparison endpoint and saves multiple models."
    )

    st.button(
        "Compare available models",
        use_container_width=True,
        disabled=True
    )


# ============================================================
# MAP MATCHING REQUEST
# ============================================================

if run_clicked:

    request_body = {
        "trip_id": selected_trip,
        "partition": partition_choice,
        "reconstruction": reconstruction
    }

    with st.spinner(
        f"Running map matching for trajectory {selected_trip}..."
    ):

        try:

            result = api_post(
                "/api/map-match",
                request_body
            )

            st.session_state.last_result = result

            st.success("Map matching completed successfully.")

        except requests.RequestException as exc:

            st.error(
                f"Map-matching request failed: {exc}"
            )

            if exc.response is not None:
                try:
                    st.error(
                        f"Backend response: {exc.response.json()}"
                    )
                except ValueError:
                    st.error(exc.response.text)

        except Exception as exc:

            st.error(f"Unexpected error: {exc}")


# ============================================================
# RIGHT: MAP
# ============================================================

with right:

    st.subheader("Trajectory map")

    result = st.session_state.last_result

    if result:

        map_html = result.get("folium_map_html")

        if map_html:

            components.html(
                map_html,
                height=650,
                scrolling=True
            )

        else:

            st.warning(
                "The API did not return a Folium map."
            )

    else:

        st.info(
            "Select a trajectory and run map matching. "
            "The resulting map will appear here."
        )


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.last_result

if result:

    st.divider()

    st.subheader("Prediction results")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = result.get("pointwise_accuracy")
    continuity = result.get("route_continuity")
    processing_time = result.get("processing_time_ms")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "GPS observations",
        result.get("num_gps_observations", 0)
    )

    metric_columns[1].metric(
        "Predicted observations",
        result.get("num_predicted_observations", 0)
    )

    metric_columns[2].metric(
        "Pointwise accuracy",
        f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
    )

    metric_columns[3].metric(
        "Route continuity",
        f"{continuity * 100:.2f}%" if continuity is not None else "N/A"
    )

    st.markdown("**Processing metrics**")

    timing_df = pd.DataFrame([
        {
            "Processing time (ms)": processing_time,
            "Predicted route length": result.get(
                "predicted_route_length", 0
            )
        }
    ])

    st.dataframe(
        timing_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    st.markdown("**Result details**")

    details = {
        "trajectory_id": result.get("trip_id"),
        "partition": result.get("partition"),
        "model": result.get("model_name"),
        "reconstruction": reconstruction
    }

    st.json(details, expanded=False)

    # --------------------------------------------------------
    # ROUTE INFORMATION
    # --------------------------------------------------------

    with st.expander("Predicted and reference arc IDs"):

        predicted_arcs = result.get("predicted_arcs", [])
        reference_arcs = result.get("reference_arcs", [])

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Predicted route**")

            st.write(predicted_arcs)

        with col2:

            st.markdown("**Reference route**")

            st.write(reference_arcs)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "Download result JSON",
        data=json.dumps(
            result,
            indent=2,
            default=str
        ),
        file_name=(
            f"pathsnap_{result.get('trip_id', 'result')}_result.json"
        ),
        mime="application/json"
    )


# ============================================================
# FOOTER / API INFORMATION
# ============================================================

with st.expander("Backend API information"):

    st.markdown("""
    **Currently supported endpoints**

    | Endpoint | Purpose |
    |---|---|
    | `GET /api/health` | Backend health |
    | `GET /api/trajectories` | Available trajectories |
    | `GET /api/models` | Available trained models |
    | `POST /api/map-match` | Run map matching |

    **Current configuration**

    - Model: Random Forest
    - Reconstruction: Top-1 candidate selection
    - Processing: Synchronous
    - Evaluation: Reference route used after prediction
    - Map: Folium interactive map
    """)
