"""
PathSnap - Exploratory Data Analysis (EDA)
UE24CS352A - Machine Learning Hackathon
Comprehensive dataset inspection, road-network analysis, trajectory statistics, and Folium visualizations.
"""

import os
import glob
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "student_test")
EDA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "01_eda")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def haversine_m(lon1, lat1, lon2, lat2):
    """Haversine distance in meters between two lon/lat points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def point_to_segment_distance_m(px, py, ax, ay, bx, by, cos_lat):
    """Euclidean perpendicular distance in meters using flat earth projection around lat."""
    px_m, py_m = px * 111320.0 * cos_lat, py * 111320.0
    ax_m, ay_m = ax * 111320.0 * cos_lat, ay * 111320.0
    bx_m, by_m = bx * 111320.0 * cos_lat, by * 111320.0
    dx, dy = bx_m - ax_m, by_m - ay_m
    l2 = dx*dx + dy*dy
    if l2 == 0.0:
        return math.hypot(px_m - ax_m, py_m - ay_m)
    t = max(0.0, min(1.0, ((px_m - ax_m)*dx + (py_m - ay_m)*dy) / l2))
    proj_x = ax_m + t * dx
    proj_y = ay_m + t * dy
    return math.hypot(px_m - proj_x, py_m - proj_y)

def inspect_trip(trip_path, trip_id, has_route=True):
    """Loads and summarizes a single trip."""
    track_p = os.path.join(trip_path, f"{trip_id}.track")
    nodes_p = os.path.join(trip_path, f"{trip_id}.nodes")
    arcs_p  = os.path.join(trip_path, f"{trip_id}.arcs")
    route_p = os.path.join(trip_path, f"{trip_id}.route")

    # Read .track: lon, lat, timestamp (seconds)
    track_df = pd.read_csv(track_p, sep=r'\s+', header=None, names=['lon', 'lat', 'ts'])
    
    # Read .nodes: lon, lat (0-indexed row index is node_id)
    nodes_df = pd.read_csv(nodes_p, sep=r'\s+', header=None, names=['lon', 'lat'])
    
    # Read .arcs: from_node_id, to_node_id (0-indexed row index is arc_id)
    arcs_df = pd.read_csv(arcs_p, sep=r'\s+', header=None, names=['from_node', 'to_node'])
    
    route_arcs = []
    if has_route and os.path.exists(route_p):
        with open(route_p) as f:
            route_arcs = [int(line.strip()) for line in f if line.strip()]

    # Time differences
    dt = track_df['ts'].diff().dropna()
    dt_1s_pct = (dt == 1.0).mean() * 100.0 if len(dt) > 0 else 0.0
    max_gap = dt.max() if len(dt) > 0 else 0.0

    # Speeds between consecutive GPS points (m/s)
    speeds = []
    for i in range(len(track_df) - 1):
        p1 = (track_df.iloc[i]['lon'], track_df.iloc[i]['lat'])
        p2 = (track_df.iloc[i+1]['lon'], track_df.iloc[i+1]['lat'])
        d = haversine_m(p1[0], p1[1], p2[0], p2[1])
        t_diff = track_df.iloc[i+1]['ts'] - track_df.iloc[i]['ts']
        if t_diff > 0:
            speeds.append(d / t_diff)
        else:
            speeds.append(0.0)

    # Road network node degrees
    out_deg = arcs_df['from_node'].value_counts()
    in_deg = arcs_df['to_node'].value_counts()
    isolated_nodes = len(nodes_df) - len(set(arcs_df['from_node']).union(set(arcs_df['to_node'])))

    summary = {
        'trip_id': trip_id,
        'gps_points': len(track_df),
        'duration_s': track_df['ts'].iloc[-1] - track_df['ts'].iloc[0] if len(track_df) > 0 else 0,
        'dt_1s_pct': dt_1s_pct,
        'max_time_gap_s': max_gap,
        'median_speed_mps': np.median(speeds) if speeds else 0.0,
        'max_speed_mps': np.max(speeds) if speeds else 0.0,
        'num_nodes': len(nodes_df),
        'num_arcs': len(arcs_df),
        'isolated_nodes': isolated_nodes,
        'num_route_arcs': len(route_arcs),
        'min_lon': track_df['lon'].min(),
        'max_lon': track_df['lon'].max(),
        'min_lat': track_df['lat'].min(),
        'max_lat': track_df['lat'].max(),
    }
    return summary, track_df, nodes_df, arcs_df, route_arcs

def run_dataset_inventory():
    """Runs inventory on all train and student_test trips."""
    print("=" * 60)
    print("STEP 1: DATASET INVENTORY")
    print("=" * 60)
    
    train_trips = sorted(os.listdir(TRAIN_DIR))
    test_trips = sorted(os.listdir(TEST_DIR))
    
    print(f"Total Training Trips: {len(train_trips)}")
    print(f"Total Student Testing Trips: {len(test_trips)}")
    
    train_summaries = []
    for t in train_trips:
        s, _, _, _, _ = inspect_trip(os.path.join(TRAIN_DIR, t), t, has_route=True)
        train_summaries.append(s)
        
    test_summaries = []
    for t in test_trips:
        s, _, _, _, _ = inspect_trip(os.path.join(TEST_DIR, t), t, has_route=True)
        test_summaries.append(s)

    train_df = pd.DataFrame(train_summaries)
    test_df = pd.DataFrame(test_summaries)

    train_df.to_csv(os.path.join(EDA_OUTPUT_DIR, "train_trips_inventory.csv"), index=False)
    test_df.to_csv(os.path.join(EDA_OUTPUT_DIR, "student_test_inventory.csv"), index=False)

    print("\n--- Training Set Statistics Summary ---")
    print(train_df[['gps_points', 'duration_s', 'dt_1s_pct', 'max_time_gap_s', 'num_nodes', 'num_arcs', 'num_route_arcs']].describe().to_string())

    print("\n--- Student Test Statistics Summary ---")
    print(test_df[['gps_points', 'duration_s', 'dt_1s_pct', 'max_time_gap_s', 'num_nodes', 'num_arcs', 'num_route_arcs']].describe().to_string())

    return train_df, test_df

def plot_eda_distributions(train_df):
    """Generates comprehensive exploratory figures."""
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING EDA STATISTICAL PLOTS")
    print("=" * 60)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("PathSnap Dataset Exploratory Analysis - Training Partition (80 Trips)", fontsize=16, fontweight='bold')

    # 1. GPS points distribution
    sns.histplot(train_df['gps_points'], bins=20, kde=True, ax=axes[0, 0], color='royalblue')
    axes[0, 0].set_title("Distribution of GPS Observations per Trip", fontweight='bold')
    axes[0, 0].set_xlabel("Number of GPS Observations")
    axes[0, 0].set_ylabel("Count of Trips")

    # 2. Trajectory duration
    sns.histplot(train_df['duration_s'] / 60.0, bins=20, kde=True, ax=axes[0, 1], color='darkorange')
    axes[0, 1].set_title("Distribution of Trip Duration (Minutes)", fontweight='bold')
    axes[0, 1].set_xlabel("Duration (minutes)")
    axes[0, 1].set_ylabel("Count of Trips")

    # 3. 1Hz Sampling percentage
    sns.histplot(train_df['dt_1s_pct'], bins=15, kde=True, ax=axes[0, 2], color='forestgreen')
    axes[0, 2].set_title("Percentage of 1 Hz Consecutive Samples (dt = 1s)", fontweight='bold')
    axes[0, 2].set_xlabel("Percentage (%)")
    axes[0, 2].set_ylabel("Count of Trips")

    # 4. Road network nodes
    sns.histplot(train_df['num_nodes'], bins=20, kde=True, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title("Distribution of Road Network Nodes", fontweight='bold')
    axes[1, 0].set_xlabel("Number of Nodes")
    axes[1, 0].set_ylabel("Count of Trips")

    # 5. Road network arcs
    sns.histplot(train_df['num_arcs'], bins=20, kde=True, ax=axes[1, 1], color='crimson')
    axes[1, 1].set_title("Distribution of Road Network Arcs", fontweight='bold')
    axes[1, 1].set_xlabel("Number of Arcs")
    axes[1, 1].set_ylabel("Count of Trips")

    # 6. Max time gap
    sns.histplot(train_df['max_time_gap_s'], bins=20, kde=True, ax=axes[1, 2], color='teal')
    axes[1, 2].set_title("Distribution of Maximum GPS Gap (seconds)", fontweight='bold')
    axes[1, 2].set_xlabel("Max Gap (seconds)")
    axes[1, 2].set_ylabel("Count of Trips")

    plt.tight_layout()
    plot_path = os.path.join(EDA_OUTPUT_DIR, "eda_dataset_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved dataset distribution plots to: {plot_path}")

def generate_folium_map(trip_id, partition="train", max_road_arcs=15000):
    """Creates a high quality interactive Folium map for a given trip."""
    print(f"Generating interactive Folium map for trip {trip_id} ({partition})...")
    trip_path = os.path.join(DATA_DIR, partition, trip_id)
    _, track_df, nodes_df, arcs_df, route_arcs = inspect_trip(trip_path, trip_id, has_route=True)

    center_lat = track_df['lat'].mean()
    center_lon = track_df['lon'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 1. FeatureGroup for Road Network Arcs (sample if huge to keep map light and snappy)
    fg_network = folium.FeatureGroup(name="Road Network Arcs (Grey)", show=True)
    
    # Filter arcs near the trajectory bounds + margin
    margin = 0.015
    lat_min, lat_max = track_df['lat'].min() - margin, track_df['lat'].max() + margin
    lon_min, lon_max = track_df['lon'].min() - margin, track_df['lon'].max() + margin
    
    arc_lines = []
    nodes_arr = nodes_df.values
    arcs_arr = arcs_df.values
    
    # Fast filtering of local arcs
    for arc_id in range(len(arcs_arr)):
        u, v = arcs_arr[arc_id]
        u_lon, u_lat = nodes_arr[u]
        v_lon, v_lat = nodes_arr[v]
        if lat_min <= u_lat <= lat_max and lon_min <= u_lon <= lon_max:
            arc_lines.append([[u_lat, u_lon], [v_lat, v_lon]])
            if len(arc_lines) >= max_road_arcs:
                break

    for line in arc_lines:
        folium.PolyLine(line, color="#888888", weight=1.5, opacity=0.4).add_to(fg_network)
    fg_network.add_to(m)

    # 2. FeatureGroup for Ground Truth Route (Green)
    if route_arcs:
        fg_route = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        route_lines = []
        for arc_id in route_arcs:
            if arc_id < len(arcs_arr):
                u, v = arcs_arr[arc_id]
                u_lon, u_lat = nodes_arr[u]
                v_lon, v_lat = nodes_arr[v]
                route_lines.append([[u_lat, u_lon], [v_lat, v_lon]])
        
        for line in route_lines:
            folium.PolyLine(line, color="#00AA00", weight=4.5, opacity=0.9).add_to(fg_route)
        fg_route.add_to(m)

    # 3. FeatureGroup for Raw GPS Points (Blue)
    fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory (Blue Dots)", show=True)
    gps_coords = []
    for idx, row in track_df.iterrows():
        coord = [row['lat'], row['lon']]
        gps_coords.append(coord)
        # Add circle marker every 5 points to keep map responsive, plus start and end
        if idx % 5 == 0 or idx == 0 or idx == len(track_df) - 1:
            popup_html = f"<b>Pt #{idx}</b><br>Time: {row['ts']}s<br>Lat: {row['lat']:.5f}<br>Lon: {row['lon']:.5f}"
            folium.CircleMarker(
                location=coord,
                radius=3.5,
                color="#0066FF",
                fill=True,
                fill_color="#0066FF",
                fill_opacity=0.8,
                popup=popup_html
            ).add_to(fg_gps)

    # Add thin line connecting GPS points
    folium.PolyLine(gps_coords, color="#3388FF", weight=1.5, opacity=0.6, dash_array="4, 4").add_to(fg_gps)
    
    # Start and End Markers
    folium.Marker(
        location=gps_coords[0],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="Trip START"
    ).add_to(fg_gps)
    folium.Marker(
        location=gps_coords[-1],
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        popup="Trip END"
    ).add_to(fg_gps)
    
    fg_gps.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    map_path = os.path.join(EDA_OUTPUT_DIR, f"folium_trip_{trip_id}.html")
    m.save(map_path)
    print(f"Saved interactive Folium map to: {map_path}")
    return map_path

if __name__ == "__main__":
    train_df, test_df = run_dataset_inventory()
    plot_eda_distributions(train_df)
    
    # Generate Folium maps for representative trips
    generate_folium_map("00000001", partition="train")
    generate_folium_map("00000002", partition="train")
    if len(test_df) > 0:
        first_test = test_df.iloc[0]['trip_id']
        generate_folium_map(first_test, partition="student_test")
    print("\nEDA Completed Successfully!")
