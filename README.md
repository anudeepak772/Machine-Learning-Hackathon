# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University — Department of Computer Science and Engineering**  
**Course:** UE24CS352A: Machine Learning  
**Project:** PathSnap Distributed Map-Matching Hackathon  

---

## 1. Executive Summary

GPS trajectory measurements suffer from multipath distortions, signal attenuation, urban canyon dropouts, and low sampling frequencies. Consequently, raw vehicle positions frequently deviate into adjacent streets or buildings. 

**PathSnap** is an end-to-end distributed machine learning system that accurately reconstructs realistic vehicle trajectories by matching noisy GPS observations to underlying road networks. It pairs:
1. **Spatial Candidate Generation & Feature Extraction**: sub-millisecond road network indexing and rich kinematic/directional feature engineering.
2. **Machine Learning Classifier**: comparative benchmarking of 7+ distinct algorithms (Ensemble Bagging, Boosting, Linear/Probabilistic, Trees, Instance-Based, Neural Networks) evaluated with **5-Fold GroupKFold (group=trip_id)** to prevent data leakage.
3. **Bayesian Hyperparameter Optimization**: Optuna Tree-structured Parzen Estimator (TPE) tuning over search spaces.
4. **HMM Viterbi Decoder**: global sequence decoding that enforces road-network graph connectivity, resolving transitions and bridging GPS gaps.
5. **Two-Laptop Distributed Deployment**: high-performance FastAPI server (Laptop 1) and an interactive Streamlit visualization console (Laptop 2).

---

## 2. Distributed Architecture

```
+-------------------------------------------------------------+
|                          LAPTOP 2                           |
|            Streamlit Evaluator Console (Frontend)           |
|                                                             |
|   1. Displays 10 reserved evaluation trips                  |
|   2. Evaluator selects any trip                             |
|   3. Sends request: POST http://<LAPTOP_1_IP>:8000/predict   |
|   4. Renders live Accuracy, Continuity, & Folium Map        |
+------------------------------+------------------------------+
                               |
                        LAN HTTP / REST
                               |
+------------------------------v------------------------------+
|                          LAPTOP 1                           |
|               FastAPI Server & Model Backend                |
|                                                             |
|   - Loads: Scaler, Tuned Model, HMM Viterbi Decoder         |
|   - Executes full inference pipeline on selected trip        |
|   - Server-side Ground Truth accuracy evaluation            |
|   - Generates interactive Folium map layers                 |
|   - Terminal displays real-time request logs                |
+-------------------------------------------------------------+
```

---

## 3. Repository Structure

```
E:\ML-HACKATHON\
├── 01_eda/
│   ├── eda_analysis.py              # Statistical EDA & Folium map generator
│   ├── train_trips_inventory.csv    # Per-trip summary statistics (80 train trips)
│   ├── student_test_inventory.csv   # Per-trip summary statistics (10 test trips)
│   ├── eda_dataset_distributions.png # Distribution plots (duration, dt, nodes, arcs)
│   └── folium_trip_*.html           # Standalone interactive Folium maps
├── 02_preprocessing/
│   ├── spatial_indexer.py           # High-speed spatial grid index for road networks
│   ├── feature_extractor.py         # Kinematic, spatial, & relative rank feature extraction
│   └── data_pipeline.py             # Feature pipeline & StandardScaler fitting
├── 03_models/
│   └── experiment_runner.py         # Multi-model benchmarking (>=5 models with GroupKFold)
├── 04_optimization/
│   └── bayesian_opt.py              # Bayesian hyperparameter optimization (Optuna TPE)
├── 05_hmm/
│   ├── viterbi_decoder.py           # Viterbi log-space dynamic programming decoder
│   └── evaluate_hmm.py              # Pointwise & sequence continuity comparison
├── 06_api/
│   └── main.py                      # FastAPI inference & evaluation server (Laptop 1)
├── 07_ui/
│   └── app.py                       # Streamlit evaluator interface (Laptop 2)
├── artifacts/
│   ├── preprocessor.pkl             # Fitted scaler & feature configuration
│   ├── baseline_best_model.pkl      # Pre-tuned best classifier
│   ├── final_model.pkl              # Final Bayesian-tuned model artifact
│   └── hmm_config.json              # Optimized HMM transition & emission parameters
├── data/
│   ├── train/                       # 80 training trips (.track, .nodes, .arcs, .route)
│   └── student_test/                # 10 student testing trips
├── results/
│   ├── model_comparison.csv         # Consolidated benchmark across all models
│   ├── optuna_trials_*.csv          # Bayesian search trials history
│   └── hmm_vs_classifier_*.csv      # Quantitative HMM vs classifier comparison
├── requirements.txt                 # Project dependencies
├── run_pipeline.py                  # Master automated end-to-end execution runner
└── README.md                        # Documentation & setup guide
```

---

## 4. Quick Start & Execution

### Step 1: Environment Setup
Ensure Python 3.11 or 3.12 is installed:
```bash
cd E:\ML-HACKATHON
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Run Complete End-to-End ML Pipeline
To execute EDA, candidate feature extraction, multi-model benchmarking, Bayesian optimization, and HMM evaluation in one command:
```bash
python run_pipeline.py
```

Or execute individual stages modularly:
```bash
# 1. Exploratory Data Analysis
python 01_eda\eda_analysis.py

# 2. Data Preprocessing & Candidate Generation
python 02_preprocessing\data_pipeline.py

# 3. Benchmark >=5 Models with Group-aware CV
python 03_models\experiment_runner.py

# 4. Bayesian Hyperparameter Optimization
python 04_optimization\bayesian_opt.py

# 5. HMM Viterbi Evaluation
python 05_hmm\evaluate_hmm.py
```

---

## 5. Deployment Instructions (Two-Laptop Setup)

### On Laptop 1 (Model Host & API Backend)
1. Find Laptop 1's local IP address (e.g., `192.168.1.5`):
   ```powershell
   ipconfig
   ```
2. Start the FastAPI server bound to all network interfaces (`0.0.0.0`):
   ```bash
   python -m uvicorn 06_api.main:app --host 0.0.0.0 --port 8000
   ```
3. Keep this terminal open during evaluation to display live request logs to the evaluators.

### On Laptop 2 (Evaluator User Interface)
1. Launch the Streamlit evaluator interface:
   ```bash
   streamlit run 07_ui\app.py
   ```
2. In the UI sidebar, enter Laptop 1's URL:
   `http://<LAPTOP_1_IP>:8000` (or `http://localhost:8000` if testing locally)
3. The UI will retrieve the available evaluation trips.
4. Select any trip and click **Reconstruct Route**.
5. Live pointwise accuracy, route continuity, and interactive map will display immediately.
