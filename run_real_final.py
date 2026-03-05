"""
run_real_final.py
=================
Full experiment on REAL data (ISCX-URL-2016 + Kaggle Phishing Emails)
with hard Non-IID splits.

Runs: FedRankX, FedAvg, LocalOnly, IFCA, FedClust, RandomCluster
All with 3 seeds, 30 rounds, 200 trees.
"""
import sys, json, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path
from src.data_prep import load_client_splits
from src.server import run_fedrankx
from src.baselines.local_only import run_local_only
from src.baselines.fedavg_standard import run_fedavg
from src.baselines.ifca import run_ifca
from src.baselines.fedclust import run_fedclust
from src.baselines.random_cluster import run_random_cluster

SPLIT_DIR = 'data/processed/client_splits_real_hard'
RESULTS = 'results_real'
SEEDS = [42, 123, 99]
N_CLIENTS = 32
N_ROUNDS = 30
N_TREES = 200

Path(RESULTS).mkdir(parents=True, exist_ok=True)
splits = load_client_splits(SPLIT_DIR, N_CLIENTS)


def safe_run(name, fn):
    try:
        return fn()
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        traceback.print_exc()
        return None


# ================================================================
# 1) FedRankX (3 seeds)
# ================================================================
print("=" * 60)
print("  FedRankX on REAL DATA")
print("=" * 60)
fr_f1, fr_auc, fr_nmi, fr_ari = [], [], [], []
for seed in SEEDS:
    np.random.seed(seed)
    rdir = f"{RESULTS}/fedrankx_seed_{seed}"
    r = run_fedrankx(
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=30, n_estimators=N_TREES,
        split_dir=SPLIT_DIR, results_dir=rdir, seed=seed,
        distance_threshold=0.5, rank_metric='kendall',
    )
    fr_f1.append(r['final_f1']); fr_auc.append(r['final_auc'])
    fr_nmi.append(r['final_nmi']); fr_ari.append(r['final_ari'])
    print(f"  Seed {seed}: F1={r['final_f1']:.4f} AUC={r['final_auc']:.4f} NMI={r['final_nmi']:.3f}")

# ================================================================
# 2) Baselines (3 seeds each)
# ================================================================
baselines = {
    'FedAvg': lambda s, rd: run_fedavg(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                         n_estimators=N_TREES, results_dir=rd),
    'LocalOnly': lambda s, rd: run_local_only(s, n_clients=N_CLIENTS,
                                                n_estimators=N_TREES, results_dir=rd),
    'IFCA': lambda s, rd: run_ifca(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                     n_estimators=N_TREES, results_dir=rd),
    'FedClust': lambda s, rd: run_fedclust(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                             n_estimators=N_TREES, results_dir=rd),
    'RandomCluster': lambda s, rd: run_random_cluster(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                                        n_estimators=N_TREES, results_dir=rd),
}

all_results = {
    'FedRankX': {
        'f1': f"{np.mean(fr_f1):.4f} +/- {np.std(fr_f1):.4f}",
        'auc': f"{np.mean(fr_auc):.4f} +/- {np.std(fr_auc):.4f}",
        'nmi': f"{np.mean(fr_nmi):.3f} +/- {np.std(fr_nmi):.3f}",
        'ari': f"{np.mean(fr_ari):.3f} +/- {np.std(fr_ari):.3f}",
        'f1_mean': float(np.mean(fr_f1)), 'f1_std': float(np.std(fr_f1)),
        'auc_mean': float(np.mean(fr_auc)), 'auc_std': float(np.std(fr_auc)),
        'nmi_mean': float(np.mean(fr_nmi)), 'nmi_std': float(np.std(fr_nmi)),
    }
}

for name, bfn in baselines.items():
    print(f"\n{'='*60}")
    print(f"  {name} on REAL DATA")
    print(f"{'='*60}")
    b_f1, b_auc = [], []
    for seed in SEEDS:
        np.random.seed(seed)
        rdir = f"{RESULTS}/{name.lower()}_seed_{seed}"
        Path(rdir).mkdir(parents=True, exist_ok=True)
        r = safe_run(f"{name} seed={seed}", lambda _bfn=bfn, _rdir=rdir: _bfn(splits, _rdir))
        if r:
            b_f1.append(r['final_f1'])
            b_auc.append(r['final_auc'])
            print(f"  Seed {seed}: F1={r['final_f1']:.4f} AUC={r['final_auc']:.4f}")

    if b_f1:
        all_results[name] = {
            'f1': f"{np.mean(b_f1):.4f} +/- {np.std(b_f1):.4f}",
            'auc': f"{np.mean(b_auc):.4f} +/- {np.std(b_auc):.4f}",
            'f1_mean': float(np.mean(b_f1)), 'f1_std': float(np.std(b_f1)),
            'auc_mean': float(np.mean(b_auc)), 'auc_std': float(np.std(b_auc)),
        }

# ================================================================
# Save all results
# ================================================================
with open(f"{RESULTS}/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("  FINAL RESULTS (REAL DATA)")
print(f"{'='*60}")
for name, res in all_results.items():
    extra = f"  NMI={res.get('nmi_mean', 'N/A')}" if 'nmi_mean' in res else ""
    print(f"  {name:15s}: F1={res['f1_mean']:.4f}  AUC={res['auc_mean']:.4f}{extra}")
print(f"\nResults saved to {RESULTS}/all_results.json")
