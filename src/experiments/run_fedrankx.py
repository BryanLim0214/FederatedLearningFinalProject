"""
experiments/run_fedrankx.py
===========================
One-click runner for the main FedRankX experiment (Table 2, Figures 2–3).
Runs FedRankX across 3 random seeds and averages results.

Usage
-----
    python -m src.experiments.run_fedrankx
    python -m src.experiments.run_fedrankx --n_rounds 50 --top_k 10 --n_clusters 5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.server import run_fedrankx


def main():
    parser = argparse.ArgumentParser(description="Run FedRankX Main Experiment")
    parser.add_argument("--n_clients", type=int, default=16)
    parser.add_argument("--n_rounds",  type=int, default=30)
    parser.add_argument("--top_k",     type=int, default=15)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--distance_threshold", type=float, default=0.5)
    parser.add_argument("--linkage_method", default="ward")
    parser.add_argument("--rank_metric", default="spearman",
                        choices=["spearman", "kendall", "hamming"])
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random seeds for significance testing")
    parser.add_argument("--split_dir", default="data/processed/client_splits")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--samples_per_client", type=int, default=1500)
    parser.add_argument("--use_real_data", action="store_true")
    args = parser.parse_args()

    seeds = [42, 123, 99][:args.n_seeds]
    all_f1, all_auc, all_nmi, all_ari = [], [], [], []

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  FedRankX — seed={seed}")
        print(f"{'='*55}")
        result = run_fedrankx(
            n_clients=args.n_clients,
            n_rounds=args.n_rounds,
            top_k=args.top_k,
            n_clusters=args.n_clusters,
            distance_threshold=args.distance_threshold,
            linkage_method=args.linkage_method,
            rank_metric=args.rank_metric,
            n_estimators=args.n_estimators,
            split_dir=args.split_dir,
            results_dir=f"{args.results_dir}/seed_{seed}",
            seed=seed,
            samples_per_client=args.samples_per_client,
            use_real_data=args.use_real_data,
        )
        all_f1.append(result["final_f1"])
        all_auc.append(result["final_auc"])
        all_nmi.append(result["final_nmi"])
        all_ari.append(result["final_ari"])

    # Aggregate across seeds
    summary = {
        "method": "FedRankX",
        "n_seeds": len(seeds),
        "top_k": args.top_k,
        "rank_metric": args.rank_metric,
        "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
        "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
        "nmi_mean": float(np.mean(all_nmi)), "nmi_std": float(np.std(all_nmi)),
        "ari_mean": float(np.mean(all_ari)), "ari_std": float(np.std(all_ari)),
        "per_seed": {"f1": all_f1, "auc": all_auc, "nmi": all_nmi, "ari": all_ari},
    }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{args.results_dir}/FedRankX.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  FedRankX FINAL ({len(seeds)} seeds)")
    print(f"  F1  = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  NMI = {summary['nmi_mean']:.3f} ± {summary['nmi_std']:.3f}")
    print(f"  ARI = {summary['ari_mean']:.3f} ± {summary['ari_std']:.3f}")
    print(f"  Saved -> {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
