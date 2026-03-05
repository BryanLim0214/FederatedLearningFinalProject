"""Quick test: check FedClust and RandomCluster for errors."""
import sys, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from src.data_prep import load_client_splits

splits = load_client_splits('data/processed/client_splits_v2', 32)
TRUE_LABELS = [0]*8 + [1]*6 + [2]*6 + [3]*8 + [4]*4

print("=== Testing FedClust (2 rounds) ===")
try:
    from src.baselines.fedclust import run_fedclust
    r = run_fedclust(splits, n_clients=32, n_rounds=2, n_estimators=50, results_dir='results_test')
    print(f"FedClust OK: F1={r['final_f1']:.4f}")
except Exception as e:
    print(f"FedClust FAILED:")
    traceback.print_exc()

print("\n=== Testing RandomCluster (2 rounds) ===")
try:
    from src.baselines.random_cluster import run_random_cluster
    np.random.seed(42)
    r = run_random_cluster(splits, n_clients=32, n_rounds=2, n_estimators=50,
                            n_clusters=5, true_labels=TRUE_LABELS, results_dir='results_test')
    print(f"RandomCluster OK: F1={r['final_f1']:.4f}")
except Exception as e:
    print(f"RandomCluster FAILED:")
    traceback.print_exc()
