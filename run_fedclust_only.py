"""Run FedClust with 3 seeds and merge results into all_results.json."""
import sys, json, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from src.data_prep import load_client_splits
from src.baselines.fedclust import run_fedclust

RESULTS = "results_final"
SEEDS = [42, 123, 99]
splits = load_client_splits('data/processed/client_splits_v2', 32)

print("=" * 60)
print("  FedClust: 3 seeds x 30 rounds x 200 trees")
print("=" * 60)

b_f1, b_auc = [], []
for seed in SEEDS:
    np.random.seed(seed)
    rdir = f"{RESULTS}/fedclust_seed_{seed}"
    print(f"\n--- Seed {seed} ---")
    try:
        r = run_fedclust(splits, n_clients=32, n_rounds=30, n_estimators=200,
                         results_dir=rdir)
        b_f1.append(r["final_f1"])
        b_auc.append(r["final_auc"])
        print(f"  Seed {seed}: F1={r['final_f1']:.4f}  AUC={r['final_auc']:.4f}")
    except Exception:
        print(f"  Seed {seed} FAILED:")
        traceback.print_exc()

if b_f1:
    fc_result = {
        "f1": f"{np.mean(b_f1):.4f} +/- {np.std(b_f1):.4f}",
        "auc": f"{np.mean(b_auc):.4f} +/- {np.std(b_auc):.4f}",
        "f1_mean": float(np.mean(b_f1)), "f1_std": float(np.std(b_f1)),
        "auc_mean": float(np.mean(b_auc)), "auc_std": float(np.std(b_auc)),
    }

    # Merge into all_results.json
    with open(f"{RESULTS}/all_results.json") as f:
        all_results = json.load(f)
    all_results["FedClust"] = fc_result
    with open(f"{RESULTS}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n>>> FedClust: F1={np.mean(b_f1):.4f}+/-{np.std(b_f1):.4f}  AUC={np.mean(b_auc):.4f}")
    print("Merged into all_results.json!")
else:
    print("All seeds failed!")
