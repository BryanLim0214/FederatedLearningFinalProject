"""
run_all.py - Master experiment runner
=====================================
Runs everything in a single Python process. No PowerShell, no redirection.
"""
import sys, os, json, traceback
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.server import run_fedrankx
from src.data_prep import load_client_splits
from src.baselines.fedavg_standard import run_fedavg
from src.baselines.local_only import run_local_only
from src.baselines.fedclust import run_fedclust
from src.baselines.ifca import run_ifca

import numpy as np

SPLIT_DIR = "data/processed/client_splits"
N_CLIENTS = 16
N_ROUNDS = 30
N_EST = 50
RESULTS = "results"
SEEDS = [42, 123, 99]


def safe_run(name, fn):
    """Run a function, catch and print any error."""
    print(f"\n{'='*60}")
    print(f"  STARTING: {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        print(f"  DONE: {name}")
        return result
    except Exception:
        print(f"  FAILED: {name}")
        traceback.print_exc()
        return None


def run_fedrankx_seeds():
    all_f1, all_auc, all_nmi, all_ari = [], [], [], []
    for seed in SEEDS:
        r = run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=15,
            n_estimators=N_EST, split_dir=SPLIT_DIR,
            results_dir=f"{RESULTS}/seed_{seed}", seed=seed,
        )
        if r:
            all_f1.append(r["final_f1"])
            all_auc.append(r["final_auc"])
            all_nmi.append(r["final_nmi"])
            all_ari.append(r["final_ari"])

    if all_f1:
        summary = {
            "method": "FedRankX",
            "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
            "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
            "nmi_mean": float(np.mean(all_nmi)), "nmi_std": float(np.std(all_nmi)),
            "ari_mean": float(np.mean(all_ari)), "ari_std": float(np.std(all_ari)),
        }
        Path(RESULTS).mkdir(parents=True, exist_ok=True)
        with open(f"{RESULTS}/FedRankX.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  FedRankX FINAL: F1={summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
        print(f"  AUC={summary['auc_mean']:.4f}  NMI={summary['nmi_mean']:.3f}  ARI={summary['ari_mean']:.3f}")


def run_baseline_seeds(name, run_fn):
    all_f1, all_auc = [], []
    splits = load_client_splits(SPLIT_DIR, N_CLIENTS)
    for seed in SEEDS:
        r = run_fn(splits, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                   n_estimators=N_EST, results_dir=f"{RESULTS}/seed_{seed}")
        if r:
            all_f1.append(r["final_f1"])
            all_auc.append(r["final_auc"])

    if all_f1:
        summary = {
            "method": name,
            "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
            "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
        }
        Path(RESULTS).mkdir(parents=True, exist_ok=True)
        with open(f"{RESULTS}/{name}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  {name} FINAL: F1={summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")


def run_evaluate():
    """Generate tables and figures."""
    Path(f"{RESULTS}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{RESULTS}/figures").mkdir(parents=True, exist_ok=True)
    from src.evaluate import load_all_results, build_comparison_table
    results = load_all_results(RESULTS)
    if results:
        df = build_comparison_table(results)
        print("\n" + "="*60)
        print("FINAL COMPARISON TABLE")
        print("="*60)
        print(df.to_string(index=False))
        df.to_csv(f"{RESULTS}/tables/table2_comparison.csv", index=False)
        print(f"\nSaved -> {RESULTS}/tables/table2_comparison.csv")
    else:
        print("No results found to evaluate.")


if __name__ == "__main__":
    print("="*60)
    print("  FedRankX Full Experiment Suite")
    print("  Real datasets: ISCX-URL2016 + Kaggle Phishing Emails")
    print("="*60)

    # 1) FedRankX (3 seeds x 30 rounds)
    safe_run("FedRankX (3 seeds)", run_fedrankx_seeds)

    # 2) Baselines (3 seeds x 30 rounds each)
    safe_run("FedAvg", lambda: run_baseline_seeds("FedAvg", run_fedavg))
    safe_run("LocalOnly", lambda: run_baseline_seeds("LocalOnly", run_local_only))
    safe_run("FedClust", lambda: run_baseline_seeds("FedClust", run_fedclust))
    safe_run("IFCA", lambda: run_baseline_seeds("IFCA", run_ifca))

    # 3) Generate tables
    safe_run("Evaluation Tables", run_evaluate)

    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE!")
    print("="*60)
