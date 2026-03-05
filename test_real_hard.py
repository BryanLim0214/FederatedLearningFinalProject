"""Quick smoke test: FedRankX vs LocalOnly on HARD REAL data."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from src.data_prep import load_client_splits
from src.server import run_fedrankx
from src.baselines.local_only import run_local_only

SPLIT_DIR = 'data/processed/client_splits_real_hard'
splits = load_client_splits(SPLIT_DIR, 32)

print("=" * 60)
print("  SMOKE TEST: Real Data, Hard Non-IID (3 rounds)")
print("=" * 60)

r = run_fedrankx(
    n_clients=32, n_rounds=3, top_k=30, n_estimators=100,
    split_dir=SPLIT_DIR,
    results_dir='results_test_real_hard', seed=42,
    distance_threshold=0.5, rank_metric='kendall',
)
print(f"\nFedRankX:  F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

r2 = run_local_only(splits, n_clients=32, n_estimators=100, results_dir='results_test_real_hard')
print(f"LocalOnly: F1={r2['final_f1']:.4f}  AUC={r2['final_auc']:.4f}")
print(f"\nGap: F1={r['final_f1']-r2['final_f1']:+.4f}  AUC={r['final_auc']-r2['final_auc']:+.4f}")
