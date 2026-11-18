"""
PathSnap - Quick Benchmark Results Viewer
Loads and prints the consolidated 5-model benchmark results instantly.
"""
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "model_comparison.csv")

def show_results():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Results file not found at {CSV_PATH}")
        print("Run `python 03_models\\run_5model_benchmark.py` first to generate results.")
        return

    df = pd.read_csv(CSV_PATH)
    
    print("\n" + "=" * 115)
    print("                      PATHSNAP: 5-MODEL BENCHMARK RESULTS TABLE")
    print("=" * 115)
    
    display_cols = [
        "Model Name", "Algorithm Family", 
        "CV Macro F1", "CV Top-1 Acc", 
        "Student-Test Macro F1", "Student-Test Top-1 Acc", 
        "Student-Test Continuity", "Inference Time (ms/1k)"
    ]
    
    print(df[display_cols].to_string(index=False))
    print("=" * 115)
    
    # Identify winner
    best = df.sort_values(by=["Student-Test Top-1 Acc", "Student-Test Macro F1"], ascending=False).iloc[0]
    print(f"\n>>> WINNING MODEL: {best['Model Name']} ({best['Algorithm Family']})")
    print(f"    - Student-Test Top-1 Accuracy : {best['Student-Test Top-1 Acc']:.4f}")
    print(f"    - Student-Test Route Continuity: {best['Student-Test Continuity']:.4f}")
    print(f"    - Student-Test Macro F1       : {best['Student-Test Macro F1']:.4f}")
    print(f"    - Inference Latency           : {best['Inference Time (ms/1k)']:.2f} ms / 1,000 observations")
    print("\nFull CSV location: " + CSV_PATH + "\n")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    show_results()
