"""
experiments/ablation.py
=======================
Ablation study sweeps for the paper (Table 4).

Experiments
-----------
  A) Top-K sensitivity: K ∈ {5, 10, 15, 20, 30}
  B) Distance metric:   Spearman vs Kendall vs Hamming
  C) Clustering method: Ward vs k-medoids vs complete linkage
  D) Number of rounds:  10, 20, 30, 50

Usage
-----
    python -m src.experiments.ablation
    python -m src.experiments.ablation --ablation topk  # single ablation
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_prep import create_client_splits, load_client_splits
from src.server import run_fedrankx

ABLATION_CONFIGS = {
    "topk": {
        "description": "Top-K sensitivity",
        "param": "top_k",
        "values": [5, 10, 15, 20, 30],
        "fixed": {"n_rounds": 30, "rank_metric": "spearman", "linkage_method": "ward"},
    },
    "metric": {
        "description": "Distance metric comparison",
        "param": "rank_metric",
        "values": ["spearman", "kendall", "hamming"],
        "fixed": {"n_rounds": 30, "top_k": 15, "linkage_method": "ward"},
    },
    "linkage": {
        "description": "Clustering linkage method",
        "param": "linkage_method",
        "values": ["ward", "complete", "average", "single"],
        "fixed": {"n_rounds": 30, "top_k": 15, "rank_metric": "spearman"},
    },
    "rounds": {
        "description": "Number of FL rounds",
        "param": "n_rounds",
        "values": [10, 20, 30, 50],
        "fixed": {"top_k": 15, "rank_metric": "spearman", "linkage_method": "ward"},
    },
}


def run_ablation(
    ablation_name: str,
    n_clients: int = 16,
    seeds: list = None,
    split_dir: str = "data/processed/client_splits",
    results_dir: str = "results/ablation",
    samples_per_client: int = 1500,
    n_estimators: int = 50,
):
    if seeds is None:
        seeds = [42, 123]  # 2 seeds for ablation (faster)

    config = ABLATION_CONFIGS[ablation_name]
    param_name = config["param"]
    values = config["values"]
    fixed = config["fixed"]

    print(f"\n{'='*55}")
    print(f"  Ablation: {config['description']}")
    print(f"  Sweeping {param_name} over {values}")
    print(f"{'='*55}")

    # Ensure splits exist
    split_path = Path(split_dir)
    if not (split_path / "client_metadata.csv").exists():
        create_client_splits(split_dir, n_clients=n_clients,
                             samples_per_client=samples_per_client, seed=42)

    ablation_results = []

    for value in values:
        params = {**fixed, param_name: value}
        all_f1, all_nmi = [], []

        for seed in seeds:
            res = run_fedrankx(
                n_clients=n_clients,
                n_estimators=n_estimators,
                split_dir=split_dir,
                results_dir=f"{results_dir}/{ablation_name}/{param_name}_{value}/seed_{seed}",
                seed=seed,
                samples_per_client=samples_per_client,
                distance_threshold=0.5,
                n_clusters=None,
                **params,
            )
            all_f1.append(res.get("final_f1", 0.0))
            all_nmi.append(res.get("final_nmi", 0.0))

        row = {
            param_name: value,
            "f1_mean": float(np.mean(all_f1)),
            "f1_std": float(np.std(all_f1)),
            "nmi_mean": float(np.mean(all_nmi)),
            "nmi_std": float(np.std(all_nmi)),
        }
        ablation_results.append(row)
        print(f"  {param_name}={value:12} | "
              f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
              f"NMI={row['nmi_mean']:.3f}±{row['nmi_std']:.3f}")

    # Save
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{results_dir}/{ablation_name}_results.json"
    with open(out_path, "w") as f:
        json.dump({"ablation": ablation_name, "results": ablation_results}, f, indent=2)
    print(f"\n  Ablation results saved -> {out_path}")
    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="FedRankX Ablation Study")
    parser.add_argument("--ablation", default="all",
                        choices=list(ABLATION_CONFIGS.keys()) + ["all"])
    parser.add_argument("--n_clients", type=int, default=16)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--split_dir", default="data/processed/client_splits")
    parser.add_argument("--results_dir", default="results/ablation")
    parser.add_argument("--samples_per_client", type=int, default=1500)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    seeds = [42, 123][:args.n_seeds]
    ablations_to_run = list(ABLATION_CONFIGS.keys()) if args.ablation == "all" else [args.ablation]

    for abl in ablations_to_run:
        run_ablation(
            abl,
            n_clients=args.n_clients,
            seeds=seeds,
            split_dir=args.split_dir,
            results_dir=args.results_dir,
            samples_per_client=args.samples_per_client,
            n_estimators=args.n_estimators,
        )


if __name__ == "__main__":
    main()
