"""
evaluate.py
===========
Result aggregation, metric computation, and paper-quality table/figure generation.

Functions
---------
  load_all_results(results_dir)  -> dict of method -> result_dict
  build_comparison_table(results) -> pd.DataFrame
  plot_f1_convergence(results)
  plot_cluster_nmi(results)
  plot_communication_overhead(results)
  run_privacy_analysis(client_splits, top_k_values)
  statistical_significance(a, b) -> p_value
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

PALETTE = {
    "FedRankX": "#2563EB",    # blue
    "FedAvg": "#DC2626",      # red
    "FedClust": "#16A34A",    # green
    "IFCA": "#D97706",        # amber
    "LocalOnly": "#7C3AED",   # purple
    "OracleCluster": "#0891B2",  # cyan
}


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir: str = "results") -> Dict[str, dict]:
    """Load all JSON result files from results_dir, keyed by method name."""
    rp = Path(results_dir)
    results = {}
    for p in sorted(rp.glob("*.json")):
        try:
            data = load_result(str(p))
            method = data.get("method", p.stem)
            results[method] = data
        except Exception as e:
            print(f"[evaluate] Could not load {p}: {e}")
    return results


# ---------------------------------------------------------------------------
# Comparison table (for paper)
# ---------------------------------------------------------------------------

def build_comparison_table(results: Dict[str, dict]) -> pd.DataFrame:
    """
    Build the main results table (Table 2 in the paper).
    Columns: Method | F1 | AUC | NMI | ARI | #Clusters | CommOverhead(bytes)
    """
    rows = []
    for method, res in results.items():
        n_rounds = res.get("n_rounds", 30)
        top_k = res.get("top_k", 15)
        n_clients = res.get("n_clients", 16)

        # Communication overhead: rank list bytes per round per client
        rank_bytes = top_k * 2  # uint16
        total_rank_bytes = rank_bytes * n_rounds * n_clients if method == "FedRankX" else 0

        rows.append({
            "Method": method,
            "F1": round(res.get("final_f1", 0.0), 4),
            "AUC": round(res.get("final_auc", 0.5), 4),
            "NMI": round(res.get("final_nmi", 0.0), 3),
            "ARI": round(res.get("final_ari", 0.0), 3),
            "#Clusters": res.get("n_clusters_discovered", res.get("n_clusters_target", "—")),
            "Rank Overhead (B)": total_rank_bytes if method == "FedRankX" else "N/A",
        })

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 2: F1 convergence across rounds
# ---------------------------------------------------------------------------

def plot_f1_convergence(
    results: Dict[str, dict],
    save_path: str = "results/figures/f1_convergence.png",
    figsize: tuple = (8, 5),
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    for method, res in results.items():
        f1_hist = res.get("f1_history", [])
        if not f1_hist:
            continue
        rounds = [r for r, _ in f1_hist]
        f1s = [v for _, v in f1_hist]
        color = PALETTE.get(method, "gray")
        lw = 2.5 if method == "FedRankX" else 1.5
        ls = "-" if method == "FedRankX" else "--"
        ax.plot(rounds, f1s, label=method, color=color, linewidth=lw, linestyle=ls)

    ax.set_xlabel("Federated Learning Round", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("F1 Convergence by Method", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved F1 convergence plot -> {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Cluster quality (NMI) per round for FedRankX
# ---------------------------------------------------------------------------

def plot_cluster_nmi_history(
    fedrankx_result: dict,
    save_path: str = "results/figures/cluster_nmi.png",
    figsize: tuple = (7, 4),
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    history = fedrankx_result.get("round_cluster_history", [])
    rounds = [h["round"] for h in history if "nmi" in h]
    nmis = [h["nmi"] for h in history if "nmi" in h]
    aris = [h["ari"] for h in history if "ari" in h]

    if not rounds:
        print("[evaluate] No NMI history to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rounds, nmis, label="NMI", color="#2563EB", linewidth=2.0, marker="o", markersize=4)
    ax.plot(rounds, aris, label="ARI", color="#16A34A", linewidth=2.0, marker="s", markersize=4, linestyle="--")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cluster Quality", fontsize=12)
    ax.set_title("FedRankX Cluster Recovery Quality (NMI/ARI)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved NMI history plot -> {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: Privacy leakage analysis (rank vs raw SHAP)
# ---------------------------------------------------------------------------

def run_privacy_analysis(
    client_splits: dict,
    top_k_values: List[int] = None,
    save_path: str = "results/figures/privacy_leakage.png",
) -> pd.DataFrame:
    """
    Train a LightGBM model, compute SHAP values, and measure reconstruction
    error as a function of K. Produces Figure 4 in the paper.
    """
    import lightgbm as lgb
    from src.rank_utils import compute_shap_for_lgbm, estimate_reconstruction_error, shap_to_rank

    if top_k_values is None:
        top_k_values = [5, 10, 15, 20, 30]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Use client 0 for the analysis
    train_df = client_splits[0]["train"]
    feature_cols = [c for c in train_df.columns if c != "label"]
    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["label"].values

    dtrain = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "n_jobs": 1},
        dtrain, num_boost_round=50,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    shap_importance = compute_shap_for_lgbm(booster, pd.DataFrame(X, columns=feature_cols))

    rows = []
    for k in top_k_values:
        rank_list = shap_to_rank(shap_importance, top_k=k)
        leak = estimate_reconstruction_error(shap_importance, rank_list)
        rows.append({
            "top_k": k,
            "rmse": leak["rmse"],
            "info_ratio": leak["information_ratio"],
            "rho": leak["rank_correlation_rho"],
        })

    df = pd.DataFrame(rows)
    print("\n[evaluate] Privacy Leakage Analysis:")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["top_k"], df["rmse"], marker="o", color="#2563EB", linewidth=2)
    axes[0].set_xlabel("Top-K", fontsize=11)
    axes[0].set_ylabel("Reconstruction RMSE", fontsize=11)
    axes[0].set_title("SHAP Reconstruction Error", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["top_k"], df["info_ratio"], marker="s", color="#DC2626", linewidth=2)
    axes[1].set_xlabel("Top-K", fontsize=11)
    axes[1].set_ylabel("Information Retained (fraction)", fontsize=11)
    axes[1].set_title("Info Ratio (lower = better privacy)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.suptitle("Privacy Analysis: Rank List vs. Raw SHAP Values", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved privacy analysis -> {save_path}")

    return df


# ---------------------------------------------------------------------------
# Statistical significance test
# ---------------------------------------------------------------------------

def statistical_significance(
    scores_a: List[float],
    scores_b: List[float],
    method_a: str = "FedRankX",
    method_b: str = "FedAvg",
) -> dict:
    """Welch's t-test between two score distributions (across seeds)."""
    stat, p = ttest_ind(scores_a, scores_b, equal_var=False)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "t_stat": float(stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Communication overhead comparison
# ---------------------------------------------------------------------------

def compute_communication_overhead(
    results: Dict[str, dict],
    model_size_bytes: int = 100_000,  # typical LightGBM model ~100KB
) -> pd.DataFrame:
    """
    Estimate total bytes uploaded per client per round for each method.
    FedRankX sends: model weights + rank list (K × 2 bytes)
    Others send: model weights only
    """
    rows = []
    for method, res in results.items():
        top_k = res.get("top_k", 15)
        rank_overhead = top_k * 2 if method == "FedRankX" else 0
        total_per_round = model_size_bytes + rank_overhead
        rows.append({
            "Method": method,
            "Model (bytes)": model_size_bytes,
            "Rank overhead (bytes)": rank_overhead,
            "Total per round per client (bytes)": total_per_round,
            "Rank overhead %": round(rank_overhead / total_per_round * 100, 4),
        })
    return pd.DataFrame(rows).sort_values("Total per round per client (bytes)")


# ---------------------------------------------------------------------------
# Generate full paper tables to stdout / CSV
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    # Ensure output directories exist before writing
    os.makedirs(os.path.join(args.results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)

    results = load_all_results(args.results_dir)
    if not results:
        print("[evaluate] No results found. Run experiments first.")
        return

    print("\n" + "="*60)
    print("TABLE 2: Main Comparison Results")
    print("="*60)
    df = build_comparison_table(results)
    print(df.to_string(index=False))
    df.to_csv(f"{args.results_dir}/tables/table2_comparison.csv", index=False)

    print("\n" + "="*60)
    print("TABLE 3: Communication Overhead")
    print("="*60)
    comm_df = compute_communication_overhead(results)
    print(comm_df.to_string(index=False))
    comm_df.to_csv(f"{results_dir}/tables/table3_communication.csv", index=False)

    # Plots
    plot_f1_convergence(results)
    if "FedRankX" in results:
        plot_cluster_nmi_history(results["FedRankX"])


if __name__ == "__main__":
    main()
