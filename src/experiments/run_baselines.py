"""
experiments/run_baselines.py
============================
Runs all baseline methods (LocalOnly, FedAvg, FedClust, IFCA) and
saves results to the same results/ directory as FedRankX for comparison.

Usage
-----
    python -m src.experiments.run_baselines
    python -m src.experiments.run_baselines --methods local fedavg  # subset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import create_client_splits, load_client_splits
from src.baselines.local_only import run_local_only
from src.baselines.fedavg_standard import run_fedavg
from src.baselines.fedclust import run_fedclust
from src.baselines.ifca import run_ifca

ALL_METHODS = ["local", "fedavg", "fedclust", "ifca"]


def run_all_baselines(
    n_clients: int = 16,
    n_rounds: int = 30,
    n_estimators: int = 50,
    seeds: list = None,
    split_dir: str = "data/processed/client_splits",
    results_dir: str = "results",
    methods: list = None,
    samples_per_client: int = 1500,
):
    if seeds is None:
        seeds = [42, 123, 99]
    if methods is None:
        methods = ALL_METHODS

    # Ensure splits exist
    split_path = Path(split_dir)
    if not (split_path / "client_metadata.csv").exists():
        print("[baselines] Creating client splits...")
        create_client_splits(split_dir, n_clients=n_clients,
                             samples_per_client=samples_per_client, seed=seeds[0])

    client_splits = load_client_splits(split_dir, n_clients)

    # Read ground-truth labels for NMI/ARI computation
    meta_path = split_path / "client_metadata.csv"
    true_labels = None
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        true_labels = meta_df.sort_values("client_id")["type_label"].values

    agg: dict = {}

    for method in methods:
        all_f1, all_auc, all_nmi, all_ari = [], [], [], []

        for seed in seeds:
            rdir = f"{results_dir}/seed_{seed}"
            Path(rdir).mkdir(parents=True, exist_ok=True)

            if method == "local":
                res = run_local_only(client_splits, n_clients,
                                     n_estimators=n_estimators, results_dir=rdir)
            elif method == "fedavg":
                res = run_fedavg(client_splits, n_clients, n_rounds=n_rounds,
                                 n_estimators=n_estimators, results_dir=rdir)
            elif method == "fedclust":
                res = run_fedclust(client_splits, n_clients, n_rounds=n_rounds,
                                   true_client_labels=true_labels,
                                   n_estimators=n_estimators, results_dir=rdir)
            elif method == "ifca":
                res = run_ifca(client_splits, n_clients, n_rounds=n_rounds,
                               true_client_labels=true_labels,
                               n_estimators=n_estimators, results_dir=rdir)
            else:
                continue

            all_f1.append(res.get("final_f1", 0.0))
            all_auc.append(res.get("final_auc", 0.5))
            all_nmi.append(res.get("final_nmi", 0.0))
            all_ari.append(res.get("final_ari", 0.0))

        method_name = {
            "local": "LocalOnly",
            "fedavg": "FedAvg",
            "fedclust": "FedClust",
            "ifca": "IFCA",
        }[method]

        summary = {
            "method": method_name,
            "n_seeds": len(seeds),
            "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
            "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
            "nmi_mean": float(np.mean(all_nmi)), "nmi_std": float(np.std(all_nmi)),
            "ari_mean": float(np.mean(all_ari)), "ari_std": float(np.std(all_ari)),
            "per_seed": {"f1": all_f1, "auc": all_auc},
        }
        agg[method_name] = summary

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{results_dir}/{method_name}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[{method_name:10s}] F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}  "
              f"AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}")

    return agg


def main():
    parser = argparse.ArgumentParser(description="Run FedRankX Baselines")
    parser.add_argument("--n_clients", type=int, default=16)
    parser.add_argument("--n_rounds", type=int, default=30)
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--split_dir", default="data/processed/client_splits")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--samples_per_client", type=int, default=1500)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    args = parser.parse_args()

    seeds = [42, 123, 99][:args.n_seeds]
    run_all_baselines(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        n_estimators=args.n_estimators,
        seeds=seeds,
        split_dir=args.split_dir,
        results_dir=args.results_dir,
        methods=args.methods,
        samples_per_client=args.samples_per_client,
    )


if __name__ == "__main__":
    main()
