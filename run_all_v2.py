"""
run_all_v2.py - Enhanced Experiment Suite
==========================================
Tuned hyperparameters, 32 clients, all baselines including RandomCluster,
ablation sweeps, and communication cost analysis.
"""
import sys, os, json, traceback, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_prep import create_client_splits, load_client_splits
from src.server import run_fedrankx
from src.baselines.fedavg_standard import run_fedavg
from src.baselines.local_only import run_local_only
from src.baselines.fedclust import run_fedclust
from src.baselines.ifca import run_ifca
from src.baselines.random_cluster import run_random_cluster
from src.rank_utils import encode_rank_for_transmission

# ============================================================
# CONFIGURATION - TUNED FOR BEST RESULTS
# ============================================================
N_CLIENTS = 32
N_ROUNDS = 30
N_EST = 200        # was 50, more trees = better SHAP signals
TOP_K = 15
DIST_THRESH = 0.3  # was 0.5, finer clusters
RESULTS = "results_v2"
SEEDS = [42, 123, 99]
SAMPLES_PER_CLIENT = 1500

# True labels for NMI computation (based on CLIENT_TYPES in data_prep.py)
# Banking(ISCX)=8, Healthcare(email)=6, SME(email)=8, ISP(ISCX)=6, Mixed=4
# Total = 32 clients
TRUE_LABELS_32 = (
    [0]*8 + [1]*6 + [2]*8 + [3]*6 + [4]*4  # 32 clients
)


def safe_run(name, fn):
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


def step1_prep_data():
    """Regenerate client splits with 32 clients from real data."""
    p = Path('data/processed/client_splits_v2')
    if p.exists():
        shutil.rmtree(p)
    create_client_splits(
        str(p), n_clients=N_CLIENTS, samples_per_client=SAMPLES_PER_CLIENT,
        use_real_data=True,
        iscx_path='data/raw/ISCX-URL2016/All.csv',
        email_path='data/raw/phishing_email/Phishing_Email.csv',
    )


def step2_fedrankx():
    """Run FedRankX across 3 seeds with tuned hyperparameters."""
    all_f1, all_auc, all_nmi, all_ari = [], [], [], []
    for seed in SEEDS:
        r = run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=TOP_K,
            n_estimators=N_EST, split_dir='data/processed/client_splits_v2',
            results_dir=f"{RESULTS}/seed_{seed}", seed=seed,
            distance_threshold=DIST_THRESH, linkage_method='ward',
            rank_metric='spearman',
        )
        if r:
            all_f1.append(r["final_f1"])
            all_auc.append(r["final_auc"])
            all_nmi.append(r["final_nmi"])
            all_ari.append(r["final_ari"])

    if all_f1:
        summary = {
            "method": "FedRankX", "n_clients": N_CLIENTS, "n_rounds": N_ROUNDS,
            "n_estimators": N_EST, "top_k": TOP_K, "distance_threshold": DIST_THRESH,
            "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
            "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
            "nmi_mean": float(np.mean(all_nmi)), "nmi_std": float(np.std(all_nmi)),
            "ari_mean": float(np.mean(all_ari)), "ari_std": float(np.std(all_ari)),
        }
        Path(RESULTS).mkdir(parents=True, exist_ok=True)
        with open(f"{RESULTS}/FedRankX.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  FedRankX FINAL: F1={summary['f1_mean']:.4f}+/-{summary['f1_std']:.4f}  "
              f"AUC={summary['auc_mean']:.4f}  NMI={summary['nmi_mean']:.3f}")


def step3_baselines():
    """Run all baselines including RandomCluster."""
    splits = load_client_splits('data/processed/client_splits_v2', N_CLIENTS)

    for seed in SEEDS:
        rdir = f"{RESULTS}/seed_{seed}"

        # FedAvg
        print(f"\n--- FedAvg seed={seed} ---")
        run_fedavg(splits, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                   n_estimators=N_EST, results_dir=rdir)

        # FedClust
        print(f"\n--- FedClust seed={seed} ---")
        run_fedclust(splits, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                     n_estimators=N_EST, results_dir=rdir)

        # IFCA
        print(f"\n--- IFCA seed={seed} ---")
        run_ifca(splits, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                 n_estimators=N_EST, results_dir=rdir)

        # LocalOnly (no n_rounds param)
        print(f"\n--- LocalOnly seed={seed} ---")
        run_local_only(splits, n_clients=N_CLIENTS,
                       n_estimators=N_EST, results_dir=rdir)

        # RandomCluster
        print(f"\n--- RandomCluster seed={seed} ---")
        run_random_cluster(splits, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                           n_estimators=N_EST, n_clusters=4,
                           true_labels=TRUE_LABELS_32[:N_CLIENTS],
                           results_dir=rdir)

    # Aggregate baseline results across seeds
    for method, filename in [("FedAvg", "fedavg_standard"),
                              ("FedClust", "fedclust"),
                              ("IFCA", "ifca"),
                              ("LocalOnly", "local_only"),
                              ("RandomCluster", "random_cluster")]:
        all_f1, all_auc = [], []
        for seed in SEEDS:
            try:
                with open(f"{RESULTS}/seed_{seed}/{filename}.json") as f:
                    d = json.load(f)
                    all_f1.append(d["final_f1"])
                    all_auc.append(d["final_auc"])
            except:
                pass
        if all_f1:
            summary = {
                "method": method,
                "f1_mean": float(np.mean(all_f1)), "f1_std": float(np.std(all_f1)),
                "auc_mean": float(np.mean(all_auc)), "auc_std": float(np.std(all_auc)),
            }
            with open(f"{RESULTS}/{method}.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  {method} FINAL: F1={summary['f1_mean']:.4f}+/-{summary['f1_std']:.4f}  "
                  f"AUC={summary['auc_mean']:.4f}")


def step4_ablations():
    """Sweep top_k and distance metrics."""
    splits = load_client_splits('data/processed/client_splits_v2', N_CLIENTS)
    ablation_results = []

    # Top-K sweep
    print("\n=== ABLATION: Top-K Sweep ===")
    for k in [5, 10, 15, 20, 30, 50]:
        r = run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=k,
            n_estimators=N_EST, split_dir='data/processed/client_splits_v2',
            results_dir=f"{RESULTS}/ablation_topk", seed=42,
            distance_threshold=DIST_THRESH,
        )
        if r:
            ablation_results.append({
                "ablation": "top_k", "value": k,
                "f1": r["final_f1"], "auc": r["final_auc"],
                "nmi": r["final_nmi"], "ari": r["final_ari"],
            })
            print(f"  top_k={k:3d}: F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

    # Distance metric sweep
    print("\n=== ABLATION: Distance Metric Sweep ===")
    for metric in ["spearman", "kendall", "hamming"]:
        r = run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=TOP_K,
            n_estimators=N_EST, split_dir='data/processed/client_splits_v2',
            results_dir=f"{RESULTS}/ablation_metric", seed=42,
            distance_threshold=DIST_THRESH, rank_metric=metric,
        )
        if r:
            ablation_results.append({
                "ablation": "metric", "value": metric,
                "f1": r["final_f1"], "auc": r["final_auc"],
                "nmi": r["final_nmi"], "ari": r["final_ari"],
            })
            print(f"  metric={metric:10s}: F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

    # Threshold sweep
    print("\n=== ABLATION: Distance Threshold Sweep ===")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        r = run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=TOP_K,
            n_estimators=N_EST, split_dir='data/processed/client_splits_v2',
            results_dir=f"{RESULTS}/ablation_thresh", seed=42,
            distance_threshold=thresh,
        )
        if r:
            ablation_results.append({
                "ablation": "threshold", "value": thresh,
                "f1": r["final_f1"], "auc": r["final_auc"],
                "nmi": r["final_nmi"], "ari": r["final_ari"],
            })
            print(f"  thresh={thresh:.1f}: F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}  NMI={r['final_nmi']:.3f}")

    Path(f"{RESULTS}/ablations").mkdir(parents=True, exist_ok=True)
    with open(f"{RESULTS}/ablations/all_ablations.json", "w") as f:
        json.dump(ablation_results, f, indent=2)


def step5_comm_cost():
    """Calculate communication overhead for rank lists vs model weights."""
    print("\n=== COMMUNICATION COST ANALYSIS ===")

    # Load a sample client split to measure model size
    splits = load_client_splits('data/processed/client_splits_v2', N_CLIENTS)
    import lightgbm as lgb
    train_df = splits[0]["train"]
    feature_cols = [c for c in train_df.columns if c != "label"]
    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["label"].values.astype(int)

    dtrain = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "n_jobs": 1, "num_leaves": 31},
        dtrain, num_boost_round=N_EST,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    model_bytes = len(booster.model_to_string().encode("utf-8"))
    rank_bytes = TOP_K * 2  # uint16 per index

    n_features = len(feature_cols)
    comm_data = {
        "n_features": n_features,
        "model_size_bytes": model_bytes,
        "rank_list_size_bytes": rank_bytes,
        "compression_ratio": model_bytes / rank_bytes,
        "per_round_model_total": model_bytes * N_CLIENTS * 2,  # up + down
        "per_round_rank_total": rank_bytes * N_CLIENTS,         # up only (rank lists)
    }

    print(f"  Model size:         {model_bytes:>10,} bytes")
    print(f"  Rank list size:     {rank_bytes:>10,} bytes")
    print(f"  Compression ratio:  {comm_data['compression_ratio']:.0f}x")
    print(f"  Per round (model):  {comm_data['per_round_model_total']:>10,} bytes")
    print(f"  Per round (rank):   {comm_data['per_round_rank_total']:>10,} bytes")

    Path(f"{RESULTS}/tables").mkdir(parents=True, exist_ok=True)
    with open(f"{RESULTS}/tables/communication_cost.json", "w") as f:
        json.dump(comm_data, f, indent=2)


def step6_final_table():
    """Generate the final comparison table."""
    Path(f"{RESULTS}/tables").mkdir(parents=True, exist_ok=True)
    methods = ["FedRankX", "FedAvg", "FedClust", "IFCA", "LocalOnly", "RandomCluster"]
    rows = []
    for m in methods:
        try:
            with open(f"{RESULTS}/{m}.json") as f:
                d = json.load(f)
            rows.append({
                "Method": m,
                "F1": f"{d.get('f1_mean', d.get('final_f1', 0)):.4f}",
                "AUC": f"{d.get('auc_mean', d.get('final_auc', 0)):.4f}",
                "NMI": f"{d.get('nmi_mean', d.get('final_nmi', 0)):.3f}",
                "ARI": f"{d.get('ari_mean', d.get('final_ari', 0)):.3f}",
            })
        except:
            pass

    df = pd.DataFrame(rows)
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE (Table 2)")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv(f"{RESULTS}/tables/table2_comparison.csv", index=False)
    print(f"\nSaved -> {RESULTS}/tables/table2_comparison.csv")


if __name__ == "__main__":
    print("="*60)
    print("  FedRankX Enhanced Experiment Suite v2")
    print(f"  {N_CLIENTS} clients, {N_ROUNDS} rounds, {N_EST} estimators")
    print(f"  top_k={TOP_K}, threshold={DIST_THRESH}")
    print("="*60)

    safe_run("Data Preparation (32 clients, real data)", step1_prep_data)
    safe_run("FedRankX (3 seeds, tuned)", step2_fedrankx)
    safe_run("All Baselines (3 seeds)", step3_baselines)
    safe_run("Ablation Studies", step4_ablations)
    safe_run("Communication Cost Analysis", step5_comm_cost)
    safe_run("Final Comparison Table", step6_final_table)

    print("\n" + "="*60)
    print("  ALL V2 EXPERIMENTS COMPLETE!")
    print("="*60)
