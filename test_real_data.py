"""Quick smoke test: FedRankX on REAL data (2 rounds)."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from src.data_prep import load_client_splits
from src.server import run_fedrankx
from src.baselines.local_only import run_local_only

# Load REAL data splits
splits = load_client_splits('data/processed/client_splits_real', 32)

print("=" * 60)
print("  SMOKE TEST: FedRankX on REAL phishing data (2 rounds)")
print("=" * 60)

r = run_fedrankx(
    n_clients=32, n_rounds=2, top_k=30, n_estimators=50,
    split_dir='data/processed/client_splits_real',
    results_dir='results_test_real', seed=42,
    distance_threshold=0.5, rank_metric='kendall',
)
print(f"\nFedRankX:  F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

r2 = run_local_only(splits, n_clients=32, n_estimators=50, results_dir='results_test_real')
print(f"LocalOnly: F1={r2['final_f1']:.4f}  AUC={r2['final_auc']:.4f}")

gap = r['final_f1'] - r2['final_f1']
print(f"\nGap: F1={gap:+.4f}")
print("REAL DATA TEST COMPLETE!")
