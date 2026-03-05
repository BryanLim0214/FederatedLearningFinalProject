"""Debug FedClust with exact run_final.py params."""
import sys, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from src.data_prep import load_client_splits
from src.baselines.fedclust import run_fedclust

splits = load_client_splits('data/processed/client_splits_v2', 32)

print("=== FedClust: 200 trees, 30 rounds (exact run_final.py config) ===")
try:
    r = run_fedclust(splits, n_clients=32, n_rounds=30, n_estimators=200,
                     results_dir='results_test/fedclust_debug')
    print(f"OK: F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
