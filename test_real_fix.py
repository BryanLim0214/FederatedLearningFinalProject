"""Quick 3-round test with real data to verify the fix."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import shutil
from pathlib import Path
from src.data_prep import create_client_splits, load_client_splits
from src.server import run_fedrankx
from src.baselines.fedavg_standard import run_fedavg

# 1) Regenerate splits from real data
p = Path('data/processed/client_splits')
if p.exists():
    shutil.rmtree(p)

create_client_splits(
    'data/processed/client_splits', n_clients=16, samples_per_client=1500,
    use_real_data=True,
    iscx_path='data/raw/ISCX-URL2016/All.csv',
    email_path='data/raw/phishing_email/Phishing_Email.csv',
)

# 2) Quick FedRankX test
print("\n--- FedRankX 3 rounds ---")
r1 = run_fedrankx(
    n_clients=16, n_rounds=3, top_k=15, n_estimators=20,
    split_dir='data/processed/client_splits',
    results_dir='results/quick_real_test', seed=42,
)
print(f"FedRankX: F1={r1['final_f1']:.4f}  AUC={r1['final_auc']:.4f}  NMI={r1['final_nmi']:.3f}")

# 3) Quick FedAvg test
print("\n--- FedAvg 3 rounds ---")
splits = load_client_splits('data/processed/client_splits', 16)
r2 = run_fedavg(splits, n_clients=16, n_rounds=3, n_estimators=20,
                results_dir='results/quick_real_test')
print(f"FedAvg:   F1={r2['final_f1']:.4f}  AUC={r2['final_auc']:.4f}")

if r2['final_f1'] > 0.0 and r2['final_auc'] > 0.5:
    print("\n=== REAL DATA FIX VERIFIED ===")
else:
    print("\n!!! FedAvg still broken !!!")
