"""
PathSnap - Master Pipeline Runner
Executes all project stages end-to-end:
  1. Exploratory Data Analysis & Folium Maps
  2. Candidate Generation & Feature Extraction
  3. Multi-Model Benchmarking (>=5 models with GroupKFold)
  4. Bayesian Hyperparameter Optimization (Optuna)
  5. HMM & Viterbi Sequence Decoding & Comparison
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

def run_step(title, script_rel_path):
    print("\n" + "#" * 80)
    print(f"# {title.upper()}")
    print("#" * 80)
    script_full_path = os.path.join(PROJECT_ROOT, script_rel_path)
    cwd = os.path.dirname(script_full_path)
    
    t0 = time.time()
    res = subprocess.run([VENV_PYTHON, script_full_path], cwd=cwd)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{title}' failed with return code {res.returncode}!")
        sys.exit(res.returncode)
    else:
        print(f"\n[SUCCESS] Step '{title}' completed in {elapsed:.1f}s.")

if __name__ == "__main__":
    t_total = time.time()
    print("=" * 80)
    print("STARTING FULL PATHSNAP END-TO-END EXECUTION")
    print(f"Python: {VENV_PYTHON}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    # Step 1: EDA
    run_step("Exploratory Data Analysis", os.path.join("01_eda", "eda_analysis.py"))

    # Step 2: Data Preprocessing & Feature Extraction
    run_step("Data Preprocessing & Feature Extraction", os.path.join("02_preprocessing", "data_pipeline.py"))

    # Step 3: Multi-Model Benchmark
    run_step("Multi-Model Benchmark (>=5 Models)", os.path.join("03_models", "experiment_runner.py"))

    # Step 4: Bayesian Hyperparameter Optimization
    run_step("Bayesian Hyperparameter Optimization", os.path.join("04_optimization", "bayesian_opt.py"))

    # Step 5: HMM Viterbi Route Reconstruction
    run_step("HMM Viterbi Route Reconstruction", os.path.join("05_hmm", "evaluate_hmm.py"))

    print("\n" + "=" * 80)
    print(f"ALL PATHSNAP PIPELINE STAGES COMPLETED SUCCESSFULLY IN {time.time() - t_total:.1f}s!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Laptop 1 (Backend API): python -m uvicorn 06_api.main:app --host 0.0.0.0 --port 8000")
    print("  2. Laptop 2 (Frontend UI): streamlit run 07_ui/app.py")
