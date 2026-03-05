"""
run_final.py - Clean Final Experiment Run
==========================================
Optimal config from ablation sweeps:
  - 32 clients, 30 rounds, 200 estimators
  - top_k=30, kendall metric, threshold=0.5
  - All baselines properly saved per-seed
"""
import sys, os, json, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np

from src.data_prep import load_client_splits
from src.server import run_fedrankx
from src.baselines.fedavg_standard import run_fedavg
from src.baselines.local_only import run_local_only
from src.baselines.fedclust import run_fedclust
from src.baselines.ifca import run_ifca
from src.baselines.random_cluster import run_random_cluster

# ============================================================
# OPTIMAL CONFIG (from ablation sweeps)
# ============================================================
N_CLIENTS = 32
N_ROUNDS = 30
N_EST = 200
TOP_K = 30          # ablation showed k=30 > k=15
METRIC = "kendall"  # ablation showed kendall > spearman
THRESH = 0.5        # ablation showed 0.5 gives best NMI
RESULTS = "results_final"
SEEDS = [42, 123, 99]
SPLIT_DIR = "data/processed/client_splits_v2"  # reuse existing splits

# True labels: banking(8)=0, healthcare(6)=1, government(6)=2, sme(8)=3, mixed(4)=4
TRUE_LABELS = [0]*8 + [1]*6 + [2]*6 + [3]*8 + [4]*4


def safe_run(name, fn):
    print(f"\n{'='*60}\n  STARTING: {name}\n{'='*60}")
    try:
        result = fn()
        print(f"  DONE: {name}")
        return result
    except Exception:
        print(f"  FAILED: {name}")
        traceback.print_exc()
        return None


def run_all():
    splits = load_client_splits(SPLIT_DIR, N_CLIENTS)
    Path(RESULTS).mkdir(parents=True, exist_ok=True)

    all_results = {}

    # --- FedRankX (3 seeds) ---
    fr_f1, fr_auc, fr_nmi, fr_ari = [], [], [], []
    for seed in SEEDS:
        rdir = f"{RESULTS}/fedrankx_seed_{seed}"
        r = safe_run(f"FedRankX seed={seed}", lambda s=seed: run_fedrankx(
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, top_k=TOP_K,
            n_estimators=N_EST, split_dir=SPLIT_DIR,
            results_dir=rdir, seed=s,
            distance_threshold=THRESH, linkage_method='ward',
            rank_metric=METRIC,
        ))
        if r:
            fr_f1.append(r["final_f1"]); fr_auc.append(r["final_auc"])
            fr_nmi.append(r["final_nmi"]); fr_ari.append(r["final_ari"])

    if fr_f1:
        all_results["FedRankX"] = {
            "f1": f"{np.mean(fr_f1):.4f} +/- {np.std(fr_f1):.4f}",
            "auc": f"{np.mean(fr_auc):.4f} +/- {np.std(fr_auc):.4f}",
            "nmi": f"{np.mean(fr_nmi):.3f} +/- {np.std(fr_nmi):.3f}",
            "ari": f"{np.mean(fr_ari):.3f} +/- {np.std(fr_ari):.3f}",
            "f1_mean": float(np.mean(fr_f1)), "f1_std": float(np.std(fr_f1)),
            "auc_mean": float(np.mean(fr_auc)), "auc_std": float(np.std(fr_auc)),
            "nmi_mean": float(np.mean(fr_nmi)), "nmi_std": float(np.std(fr_nmi)),
            "ari_mean": float(np.mean(fr_ari)), "ari_std": float(np.std(fr_ari)),
        }
        print(f"\n>>> FedRankX: F1={np.mean(fr_f1):.4f}+/-{np.std(fr_f1):.4f}  "
              f"AUC={np.mean(fr_auc):.4f}  NMI={np.mean(fr_nmi):.3f}")

    # --- Baselines (3 seeds each, properly saved) ---
    baselines = {
        "FedAvg": lambda s, rd: run_fedavg(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                             n_estimators=N_EST, results_dir=rd),
        "FedClust": lambda s, rd: run_fedclust(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                                 n_estimators=N_EST, results_dir=rd),
        "IFCA": lambda s, rd: run_ifca(s, n_clients=N_CLIENTS, n_rounds=N_ROUNDS,
                                         n_estimators=N_EST, results_dir=rd),
        "LocalOnly": lambda s, rd: run_local_only(s, n_clients=N_CLIENTS,
                                                     n_estimators=N_EST, results_dir=rd),
        "RandomCluster": lambda s, rd: run_random_cluster(s, n_clients=N_CLIENTS,
                                                            n_rounds=N_ROUNDS,
                                                            n_estimators=N_EST, n_clusters=5,
                                                            true_labels=TRUE_LABELS,
                                                            results_dir=rd),
    }

    for bname, bfn in baselines.items():
        b_f1, b_auc = [], []
        for seed in SEEDS:
            np.random.seed(seed)
            rdir = f"{RESULTS}/{bname.lower()}_seed_{seed}"
            # Capture bfn and rdir by value to avoid closure scoping bug
            r = safe_run(f"{bname} seed={seed}",
                         lambda _bfn=bfn, _rdir=rdir: _bfn(splits, _rdir))
            if r:
                b_f1.append(r["final_f1"])
                b_auc.append(r["final_auc"])

        if b_f1:
            all_results[bname] = {
                "f1": f"{np.mean(b_f1):.4f} +/- {np.std(b_f1):.4f}",
                "auc": f"{np.mean(b_auc):.4f} +/- {np.std(b_auc):.4f}",
                "f1_mean": float(np.mean(b_f1)), "f1_std": float(np.std(b_f1)),
                "auc_mean": float(np.mean(b_auc)), "auc_std": float(np.std(b_auc)),
            }
            print(f"\n>>> {bname}: F1={np.mean(b_f1):.4f}+/-{np.std(b_f1):.4f}  "
                  f"AUC={np.mean(b_auc):.4f}")

    # --- Save everything ---
    with open(f"{RESULTS}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Communication Cost ---
    import lightgbm as lgb
    train_df = splits[0]["train"]
    fcols = [c for c in train_df.columns if c != "label"]
    dtrain = lgb.Dataset(train_df[fcols].values.astype(np.float32),
                         label=train_df["label"].values.astype(int))
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 31},
                        dtrain, num_boost_round=N_EST,
                        callbacks=[lgb.log_evaluation(period=-1)])
    model_bytes = len(booster.model_to_string().encode("utf-8"))
    rank_bytes = TOP_K * 2
    comm = {"model_bytes": model_bytes, "rank_bytes": rank_bytes,
            "compression": model_bytes / rank_bytes,
            "per_round_model": model_bytes * N_CLIENTS * 2,
            "per_round_rank": rank_bytes * N_CLIENTS}
    with open(f"{RESULTS}/communication_cost.json", "w") as f:
        json.dump(comm, f, indent=2)

    # --- Final Table ---
    print("\n" + "="*70)
    print("  FINAL COMPARISON TABLE (Optimal Config)")
    print(f"  {N_CLIENTS} clients, {N_ROUNDS} rounds, {N_EST} trees")
    print(f"  top_k={TOP_K}, metric={METRIC}, threshold={THRESH}")
    print("="*70)
    for method, data in all_results.items():
        nmi = data.get('nmi', 'N/A')
        print(f"  {method:15s}  F1={data['f1']}  AUC={data['auc']}  NMI={nmi}")
    print(f"\n  Communication: {model_bytes:,} bytes (model) vs {rank_bytes} bytes (rank) = {model_bytes//rank_bytes}x savings")
    print("="*70)
    print("  ALL FINAL EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    run_all()
