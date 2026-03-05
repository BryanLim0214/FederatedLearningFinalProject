"""
server.py
=========
FedRankX simulation server — uses pure-Python fl_engine (no Ray required).

Usage
-----
    python -m src.server --n_clients 16 --n_rounds 30 --top_k 15
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.client import make_client_fn
from src.data_prep import create_client_splits, load_client_splits
from src.fedrank_strategy import FedRankXStrategy
from src.fl_engine import Parameters, run_simulation

warnings.filterwarnings("ignore")

DEFAULT_N_CLIENTS = 16
DEFAULT_N_ROUNDS = 30
DEFAULT_TOP_K = 15
DEFAULT_SPLIT_DIR = "data/processed/client_splits"
DEFAULT_RESULTS_DIR = "results"


def run_fedrankx(
    n_clients: int = DEFAULT_N_CLIENTS,
    n_rounds: int = DEFAULT_N_ROUNDS,
    top_k: int = DEFAULT_TOP_K,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.5,
    linkage_method: str = "ward",
    rank_metric: str = "spearman",
    n_estimators: int = 50,
    split_dir: str = DEFAULT_SPLIT_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    seed: int = 42,
    samples_per_client: int = 1500,
    use_real_data: bool = False,
    iscx_path: str = "data/raw/ISCX-URL2016/All.csv",
    email_path: str = "data/raw/phishing_email/Phishing_Email.csv",
    **kwargs,  # absorb any extra args
) -> dict:
    """Run the FedRankX simulation and return result dict."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # ---- Load or create splits ----
    split_path = Path(split_dir)
    if not (split_path / "client_metadata.csv").exists():
        print("[server] Creating client splits...")
        client_splits = create_client_splits(
            output_dir=split_dir, n_clients=n_clients,
            samples_per_client=samples_per_client,
            use_real_data=use_real_data,
            iscx_path=iscx_path, email_path=email_path, seed=seed,
        )
    else:
        client_splits = load_client_splits(split_dir, n_clients)

    # Ground-truth type labels for NMI/ARI
    meta_path = split_path / "client_metadata.csv"
    true_labels = None
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        true_labels = meta_df.sort_values("client_id")["type_label"].values

    # ---- Build components ----
    client_fn = make_client_fn(client_splits, top_k=top_k, n_estimators=n_estimators)
    strategy = FedRankXStrategy(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage_method=linkage_method,
        rank_metric=rank_metric,
        true_client_labels=true_labels,
        log_clusters=True,
    )

    print(f"\n{'='*58}")
    print(f"  FedRankX Simulation")
    print(f"  n_clients={n_clients}  n_rounds={n_rounds}  top_k={top_k}")
    print(f"  metric={rank_metric}  linkage={linkage_method}  threshold={distance_threshold}")
    print(f"{'='*58}")

    # ---- Run ----
    history = run_simulation(
        client_fn=client_fn,
        strategy=strategy,
        n_clients=n_clients,
        n_rounds=n_rounds,
        initial_parameters=Parameters([np.array([], dtype=np.uint8)]),
        verbose=False,  # strategy already prints per-round info
    )

    # ---- Collect results ----
    round_history = strategy.get_round_history()
    f1_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("f1", [])]
    auc_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("auc", [])]

    result = {
        "method": "FedRankX",
        "n_clients": n_clients, "n_rounds": n_rounds, "top_k": top_k,
        "rank_metric": rank_metric, "linkage_method": linkage_method,
        "distance_threshold": distance_threshold, "n_clusters_target": n_clusters,
        "losses": [(r, float(l)) for r, l in history.losses_distributed],
        "f1_history": f1_hist, "auc_history": auc_hist,
        "round_cluster_history": round_history,
        "final_f1":  f1_hist[-1][1]  if f1_hist  else 0.0,
        "final_auc": auc_hist[-1][1] if auc_hist else 0.5,
        "final_nmi": float(round_history[-1].get("nmi", 0.0)) if round_history else 0.0,
        "final_ari": float(round_history[-1].get("ari", 0.0)) if round_history else 0.0,
    }

    tag = f"fedrankx_{rank_metric}_{linkage_method}_k{top_k}_r{n_rounds}"
    out_file = results_path / f"{tag}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[server] F1={result['final_f1']:.4f}  AUC={result['final_auc']:.4f}  "
          f"NMI={result['final_nmi']:.3f}  ARI={result['final_ari']:.3f}")
    print(f"[server] Results -> {out_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedRankX FL Simulation")
    parser.add_argument("--n_clients", type=int, default=DEFAULT_N_CLIENTS)
    parser.add_argument("--n_rounds",  type=int, default=DEFAULT_N_ROUNDS)
    parser.add_argument("--top_k",     type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--distance_threshold", type=float, default=0.5)
    parser.add_argument("--linkage_method", default="ward",
                        choices=["ward", "average", "complete", "single"])
    parser.add_argument("--rank_metric", default="spearman",
                        choices=["spearman", "kendall", "hamming"])
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--split_dir",    default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--results_dir",  default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--samples_per_client", type=int, default=1500)
    parser.add_argument("--use_real_data", action="store_true")
    parser.add_argument("--iscx_path",  default="data/raw/ISCX-URL2016/All.csv")
    parser.add_argument("--email_path", default="data/raw/phishing_email/Phishing_Email.csv")
    args = parser.parse_args()
    run_fedrankx(**vars(args))
