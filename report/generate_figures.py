"""
PathSnap - Report Figure Generation Script
Generates publication-quality charts and diagrams for the final Technical Report PDF.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# -------------------------------------------------------------
# 1. Model Benchmark Comparison Bar Chart
# -------------------------------------------------------------
def plot_model_comparison():
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df = pd.read_csv(csv_path)
    
    models = df["Model Name"].tolist()
    cv_f1 = df["CV Macro F1"].tolist()
    test_f1 = df["Student-Test Macro F1"].tolist()
    top1_acc = df["Student-Test Top-1 Acc"].tolist()
    
    x = np.arange(len(models))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width, cv_f1, width, label='CV Macro F1 (80 Trips)', color='#3B82F6', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, test_f1, width, label='Student-Test Macro F1', color='#10B981', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, top1_acc, width, label='Student-Test Top-1 Accuracy', color='#8B5CF6', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Score / Ratio', fontweight='bold')
    ax.set_title('Cross-Family Model Benchmark Performance Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Value annotations on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=90)
            
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_model_benchmark_bars.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 2. Inference Latency & Log Loss Comparison
# -------------------------------------------------------------
def plot_latency_and_logloss():
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df = pd.read_csv(csv_path)
    
    models = df["Model Name"].tolist()
    latency = df["Inference Time (ms/1k)"].tolist()
    logloss = df["Log Loss"].tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Latency (log scale)
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6366F1']
    bars1 = ax1.bar(models, latency, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('Latency (ms / 1,000 obs) [Log Scale]', fontweight='bold')
    ax1.set_title('Inference Latency Comparison (Lower is Faster)', fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=20, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f'{yval:.1f}ms', ha='center', va='bottom', fontsize=8)
        
    # Log Loss
    bars2 = ax2.bar(models, logloss, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Log Loss / Cross-Entropy (Lower is Better)', fontweight='bold')
    ax2.set_title('Probability Calibration (Log Loss)', fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=20, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
        
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_latency_and_logloss.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 3. Feature Importances
# -------------------------------------------------------------
def plot_feature_importance():
    csv_path = os.path.join(RESULTS_DIR, "feature_importances.csv")
    df = pd.read_csv(csv_path).sort_values(by="importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["importance"], color='#2563EB', edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontweight='bold')
    ax.set_xlabel('Relative Gini Feature Importance', fontweight='bold')
    ax.set_title('Random Forest Feature Importance Hierarchy (13 Extracted Features)', fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.0, f'{width*100:.1f}%',
                ha='left', va='center', fontsize=8, fontweight='bold', color='#1E3A8A')
        
    ax.set_xlim(0, max(df["importance"]) * 1.15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 4. Bayesian Optimization Trials Plot
# -------------------------------------------------------------
def plot_bayesian_optimization():
    csv_path = os.path.join(RESULTS_DIR, "optuna_trials_RandomForest_full.csv")
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Trial vs Objective Value
    colors = ['#10B981' if s == 'COMPLETE' else '#EF4444' for s in df['state']]
    ax1.scatter(df['trial_number'], df['value'], color=colors, s=120, edgecolor='black', zorder=5)
    ax1.plot(df['trial_number'], df['value'], color='#94A3B8', linestyle='--', alpha=0.7, zorder=3)
    ax1.set_xlabel('Optuna Trial Number', fontweight='bold')
    ax1.set_ylabel('Composite Objective Score', fontweight='bold')
    ax1.set_title('Bayesian Optimization Progression (TPE Sampler)', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate winner
    winner = df.loc[df['state'] == 'COMPLETE'].sort_values(by='value', ascending=False).iloc[0]
    ax1.annotate(f"Optimal Trial {int(winner['trial_number'])}\nScore: {winner['value']:.4f}\n(max_depth=12, max_feat=None)",
                 xy=(winner['trial_number'], winner['value']),
                 xytext=(winner['trial_number'] - 1.2, winner['value'] - 0.04),
                 arrowprops=dict(facecolor='#1E40AF', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', fontsize=8.5, color='#1E40AF',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#3B82F6", lw=1))
    
    # Parameter Search space & impact (Trial duration vs n_estimators)
    sc = ax2.scatter(df['n_estimators'], df['max_depth'], c=df['value'], cmap='viridis', s=160, edgecolor='black')
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Composite Objective Value', fontweight='bold')
    ax2.set_xlabel('n_estimators', fontweight='bold')
    ax2.set_ylabel('max_depth', fontweight='bold')
    ax2.set_title('Hyperparameter Space Exploration', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_bayesian_optimization.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 5. HMM vs Classifier Per-Trip Comparison (Evaluator Test)
# -------------------------------------------------------------
def plot_hmm_comparison():
    csv_path = os.path.join(RESULTS_DIR, "hmm_vs_classifier_evaluator_test.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(RESULTS_DIR, "hmm_vs_classifier_student_test.csv")
    df = pd.read_csv(csv_path)
    
    trips = [f"T{str(t).zfill(8)[-2:]}" for t in df["trip_id"]]
    clf_acc = df["clf_only_accuracy"] * 100
    hmm_acc = df["hmm_accuracy"] * 100
    clf_cont = df["clf_continuity"] * 100
    hmm_cont = df["hmm_continuity"] * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=300, sharex=True)
    
    x = np.arange(len(trips))
    width = 0.35
    
    # Subplot 1: Accuracy
    ax1.bar(x - width/2, clf_acc, width, label='Classifier Only Top-1', color='#93C5FD', edgecolor='black', linewidth=0.5)
    bars_h1 = ax1.bar(x + width/2, hmm_acc, width, label='Classifier + Viterbi HMM', color='#1D4ED8', edgecolor='black', linewidth=0.5)
    ax1.axhline(90, color='#DC2626', linestyle='--', linewidth=1.2, label='Target Threshold (90%)')
    ax1.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax1.set_title('Evaluator-Testing Set: Pointwise Top-1 Road Matching Accuracy (Classifier vs Viterbi HMM)', fontweight='bold')
    ax1.set_ylim(80, 102)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars_h1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f'{yval:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Subplot 2: Route Continuity
    ax2.bar(x - width/2, clf_cont, width, label='Classifier Only Continuity', color='#A7F3D0', edgecolor='black', linewidth=0.5)
    bars_h2 = ax2.bar(x + width/2, hmm_cont, width, label='Classifier + Viterbi HMM Continuity', color='#047857', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Route Continuity (%)', fontweight='bold')
    ax2.set_xlabel('Evaluator-Testing Reserved Trip ID', fontweight='bold')
    ax2.set_title('Evaluator-Testing Set: Topological Route Continuity with Viterbi Decoding', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(trips, fontweight='bold')
    ax2.set_ylim(80, 103)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars_h2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_hmm_vs_classifier.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 6. Map Visualisation Export (Matplotlib Render of Evaluator Trajectory)
# -------------------------------------------------------------
def plot_folium_map_rendering():
    # Load Trip 00000015 from evaluator_test
    trip_id = "00000015"
    trip_dir = os.path.join(PROJECT_ROOT, "data", "evaluator_test", trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(PROJECT_ROOT, "data", "student_test", "00000000")
        trip_id = "00000000"
    
    track_p = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_p = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_p  = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_p = os.path.join(trip_dir, f"{trip_id}.route")
    
    with open(track_p) as f:
        track = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(nodes_p) as f:
        nodes = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(arcs_p) as f:
        arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
    with open(route_p) as f:
        route = [int(line.strip()) for line in f if line.strip()]
        
    lons = [p[0] for p in track]
    lats = [p[1] for p in track]
    margin = 0.008
    min_lon, max_lon = min(lons) - margin, max(lons) + margin
    min_lat, max_lat = min(lats) - margin, max(lats) + margin
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 1. Background Road Network
    road_drawn = 0
    for u, v in arcs:
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        if min_lon <= u_lon <= max_lon and min_lat <= u_lat <= max_lat:
            ax.plot([u_lon, v_lon], [u_lat, v_lat], color='#CBD5E1', linewidth=1.0, alpha=0.7, zorder=1)
            road_drawn += 1
            if road_drawn > 5000:
                break
                
    # 2. Ground Truth Reference Route
    for idx, arc_id in enumerate(route):
        u, v = arcs[arc_id]
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]],
                color='#10B981', linewidth=3.5, alpha=0.7, linestyle='--',
                label='Ground Truth Reference Route' if idx == 0 else "", zorder=2)
                
    # 3. Raw GPS Trajectory (with noise)
    ax.plot(lons, lats, color='#F59E0B', linewidth=1.2, alpha=0.6, label='Observed GPS Track (Noisy)', zorder=3)
    ax.scatter(lons[::15], lats[::15], color='#D97706', s=16, alpha=0.8, edgecolors='none', label='Sampled GPS Points', zorder=4)
    
    # 4. Highlight Start and End
    ax.scatter([lons[0]], [lats[0]], color='#2563EB', s=90, marker='o', edgecolors='black', linewidth=1.5, label='Trip Start', zorder=6)
    ax.scatter([lons[-1]], [lats[-1]], color='#DC2626', s=90, marker='X', edgecolors='black', linewidth=1.5, label='Trip End', zorder=6)
    
    ax.set_title(f"PathSnap Map-Matching Reconstruction (Evaluator Trip {trip_id})\nRoad Network Topology, Noisy GPS Observations & Reference Path", fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    with open(track_p) as f:
        track = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(nodes_p) as f:
        nodes = [tuple(map(float, line.strip().split())) for line in f if line.strip()]
    with open(arcs_p) as f:
        arcs = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
    with open(route_p) as f:
        route = [int(line.strip()) for line in f if line.strip()]
        
    lons = [p[0] for p in track]
    lats = [p[1] for p in track]
    margin = 0.008
    min_lon, max_lon = min(lons) - margin, max(lons) + margin
    min_lat, max_lat = min(lats) - margin, max(lats) + margin
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 1. Background Road Network
    road_drawn = 0
    for u, v in arcs:
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        if min_lon <= u_lon <= max_lon and min_lat <= u_lat <= max_lat:
            ax.plot([u_lon, v_lon], [u_lat, v_lat], color='#CBD5E1', linewidth=1.0, alpha=0.7, zorder=1)
            road_drawn += 1
            if road_drawn > 4000:
                break
                
    # 2. Ground Truth Reference Route
    for idx, arc_id in enumerate(route):
        u, v = arcs[arc_id]
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]],
                color='#10B981', linewidth=3.5, alpha=0.7, linestyle='--',
                label='Ground Truth Reference Route' if idx == 0 else "", zorder=2)
                
    # 3. Raw GPS Trajectory (with noise)
    ax.plot(lons, lats, color='#F59E0B', linewidth=1.2, alpha=0.6, label='Observed GPS Track (Noisy)', zorder=3)
    ax.scatter(lons[::8], lats[::8], color='#D97706', s=16, alpha=0.8, edgecolors='none', label='Sampled GPS Points', zorder=4)
    
    # 4. Highlight Start and End
    ax.scatter([lons[0]], [lats[0]], color='#2563EB', s=90, marker='o', edgecolors='black', linewidth=1.5, label='Trip Start', zorder=6)
    ax.scatter([lons[-1]], [lats[-1]], color='#DC2626', s=90, marker='X', edgecolors='black', linewidth=1.5, label='Trip End', zorder=6)
    
    ax.set_title(f"PathSnap Map-Matching Reconstruction (Trip {trip_id})\nRoad Network Topology, Noisy GPS Observations & Reference Path", fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig7_folium_map_rendering.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------
# 7. Two-Laptop Architecture Diagram
# -------------------------------------------------------------
def plot_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    
    # Laptop 1 Box (Backend Server)
    l1_box = patches.FancyBboxPatch((0.5, 0.8), 4.2, 4.0, boxstyle="round,pad=0.2",
                                    facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.0)
    ax.add_patch(l1_box)
    ax.text(2.6, 4.5, "LAPTOP 1: MODEL & API BACKEND", ha="center", va="center",
            fontweight="bold", fontsize=11, color="#1E3A8A")
    ax.text(2.6, 4.15, "FastAPI Service (Host: 0.0.0.0:8000)", ha="center", va="center",
            fontsize=8.5, color="#3B82F6", style="italic")
    
    l1_modules = [
        "1. Spatial Grid Index (Candidate Gen)",
        "2. 13-D Feature Engineering Pipeline",
        "3. Tuned Random Forest Classifier (Top-1)",
        "4. Viterbi HMM Sequential Decoder",
        "5. Server-Side Reference Evaluation",
        "6. Folium Map Generator Engine"
    ]
    for idx, text in enumerate(l1_modules):
        m_box = patches.FancyBboxPatch((0.8, 3.4 - idx*0.48), 3.6, 0.38, boxstyle="round,pad=0.08",
                                       facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.0)
        ax.add_patch(m_box)
        ax.text(2.6, 3.59 - idx*0.48, text, ha="center", va="center", fontsize=8.2, color="#1F2937")

    # Laptop 2 Box (Evaluator UI)
    l2_box = patches.FancyBboxPatch((6.3, 0.8), 4.2, 4.0, boxstyle="round,pad=0.2",
                                    facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=2.0)
    ax.add_patch(l2_box)
    ax.text(8.4, 4.5, "LAPTOP 2: EVALUATOR USER INTERFACE", ha="center", va="center",
            fontweight="bold", fontsize=11, color="#14532D")
    ax.text(8.4, 4.15, "Streamlit Dashboard (Port: 8501)", ha="center", va="center",
            fontsize=8.5, color="#16A34A", style="italic")
    
    l2_modules = [
        "1. Evaluation Trip Selection Dropdown (10 Trips)",
        "2. Configurable Backend IP Address Input",
        "3. Pipeline Execution Trigger Button",
        "4. Real-Time Top-1 Accuracy Metric Card",
        "5. Route Continuity (%) Metric Display",
        "6. Embedded Interactive Folium Map Viewer"
    ]
    for idx, text in enumerate(l2_modules):
        m_box = patches.FancyBboxPatch((6.6, 3.4 - idx*0.48), 3.6, 0.38, boxstyle="round,pad=0.08",
                                       facecolor="#FFFFFF", edgecolor="#86EFAC", linewidth=1.0)
        ax.add_patch(m_box)
        ax.text(8.4, 3.59 - idx*0.48, text, ha="center", va="center", fontsize=8.2, color="#1F2937")

    # Bidirectional Network Arrows
    # Forward: Request
    ax.annotate("", xy=(4.9, 2.9), xytext=(6.1, 2.9),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.2))
    ax.text(5.5, 3.15, "HTTP POST /predict\n{trip_id, partition}", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#DC2626")

    # Backward: Response
    ax.annotate("", xy=(6.1, 1.9), xytext=(4.9, 1.9),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2.2))
    ax.text(5.5, 1.6, "JSON Response\n{accuracy, continuity,\nfolium_map_html}", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color="#059669")
    
    # Network Boundary Line
    ax.plot([5.5, 5.5], [0.4, 4.8], color="#94A3B8", linestyle=":", linewidth=1.5)
    ax.text(5.5, 4.9, "LAN / Wi-Fi Network", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#475569")
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig8_system_architecture.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    plot_model_comparison()
    plot_latency_and_logloss()
    plot_feature_importance()
    plot_bayesian_optimization()
    plot_hmm_comparison()
    plot_folium_map_rendering()
    plot_system_architecture()
    print("All report figures successfully generated!")
