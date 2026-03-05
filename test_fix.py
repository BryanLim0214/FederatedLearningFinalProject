"""Quick test: verify FedRankX now beats LocalOnly after aggregation fix."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.data_prep import load_client_splits
from src.server import run_fedrankx
from src.baselines.local_only import run_local_only

splits = load_client_splits('data/processed/client_splits_v2', 32)

print("=" * 60)
print("  QUICK TEST: 5 rounds, 32 clients, 200 trees")
print("=" * 60)

r = run_fedrankx(
    n_clients=32, n_rounds=5, top_k=30, n_estimators=200,
    split_dir='data/processed/client_splits_v2',
    results_dir='results_test', seed=42,
    distance_threshold=0.5, rank_metric='kendall',
)
print(f"\nFedRankX:  F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

r2 = run_local_only(splits, n_clients=32, n_estimators=200, results_dir='results_test')
print(f"LocalOnly: F1={r2['final_f1']:.4f}  AUC={r2['final_auc']:.4f}")

gap_f1 = r['final_f1'] - r2['final_f1']
gap_auc = r['final_auc'] - r2['final_auc']
print(f"\nGap: F1={gap_f1:+.4f}  AUC={gap_auc:+.4f}")
if gap_f1 > 0 and gap_auc > 0:
    print("SUCCESS: FedRankX beats LocalOnly!")
elif gap_f1 >= -0.005:
    print("OK: FedRankX matches LocalOnly (within margin)")
else:
    print("ISSUE: LocalOnly still better")
