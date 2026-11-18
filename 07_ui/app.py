"""
PathSnap - Distributed Map-Matching Evaluator Interface (Laptop 2)
PES University UE24CS352A - Machine Learning Hackathon
Connects over LAN to Laptop 1 FastAPI backend, displays trip options,
submits evaluation requests, and renders interactive maps and performance metrics.
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import json
import time

# Page Configuration
st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/87/PES_University_Bangalore_logo.svg/1200px-PES_University_Bangalore_logo.svg.png", width=160)
st.sidebar.title("PathSnap Console")
st.sidebar.markdown("**PES University — UE24CS352A**")
st.sidebar.markdown("---")

api_host = st.sidebar.text_input("Laptop 1 API Base URL", value="http://localhost:8000")
partition_choice = st.sidebar.selectbox("Data Partition", ["student_test", "train"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Evaluator Controls")
st.sidebar.info("""
1. Select Laptop 1 API IP & Port.
2. Choose evaluation trip from dropdown.
3. Click **Execute Map-Matching Pipeline**.
4. Review reconstructed route and live accuracy.
""")

# Main Content
st.markdown('<div class="main-title">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Laptop Distributed Machine Learning & Viterbi HMM Inference Architecture</div>', unsafe_allow_html=True)

# Fetch Trips from Backend
trips_list = []
backend_online = False

try:
    health_resp = requests.get(f"{api_host}/", timeout=3)
    if health_resp.status_code == 200:
        backend_online = True
        trips_resp = requests.get(f"{api_host}/trips?partition={partition_choice}", timeout=3)
        if trips_resp.status_code == 200:
            trips_list = trips_resp.json().get("trips", [])
except Exception as e:
    backend_online = False

if not backend_online:
    st.error(f"Cannot connect to Laptop 1 API at `{api_host}`. Please verify that the FastAPI backend is running on Laptop 1 (`uvicorn 06_api.main:app --host 0.0.0.0 --port 8000`).")
    st.stop()

col_sel, col_btn = st.columns([3, 1])

with col_sel:
    selected_trip = st.selectbox(
        f"Select Evaluation Trip ({len(trips_list)} available in {partition_choice}):",
        trips_list,
        index=0 if trips_list else None
    )

with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# Processing when button is clicked
if run_btn and selected_trip:
    with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 for ML + Viterbi reconstruction..."):
        payload = {"trip_id": selected_trip, "partition": partition_choice}
        try:
            t0 = time.time()
            resp = requests.post(f"{api_host}/predict", json=payload, timeout=45)
            client_roundtrip_ms = (time.time() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()

                # Metric Cards
                st.markdown("### Evaluation Results (Server-side Reference Evaluation)")
                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Pointwise Accuracy", f"{data['pointwise_accuracy'] * 100.0:.2f}%")
                m2.metric("Route Continuity", f"{data['route_continuity'] * 100.0:.2f}%")
                m3.metric("Server Pipeline Time", f"{data['processing_time_ms']:.1f} ms")
                m4.metric("GPS Points Matched", f"{data['num_gps_observations']}")

                st.markdown("---")
                st.markdown("### Interactive Geographical Route Visualization")
                st.caption("Blue: Predicted Reconstructed Route | Green (Dashed): Reference Route | Orange: Raw GPS Points")

                # Render HTML Map returned by API
                folium_html = data.get("folium_map_html", "")
                if folium_html:
                    st.components.v1.html(folium_html, height=580, scrolling=True)

                # Route Details
                with st.expander("Detailed Reconstructed Route Arc Sequence"):
                    predicted_arcs = data.get("predicted_arcs", [])
                    st.write(f"Total Traversed Arcs: {len(predicted_arcs)}")
                    st.write(predicted_arcs)

            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")

        except Exception as err:
            st.error(f"Failed to communicate with Laptop 1: {err}")
