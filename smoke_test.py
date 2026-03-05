"""smoke_test.py — verifies FedRankX + FedAvg baseline end-to-end."""
from src.data_prep import load_client_splits
from src.baselines.fedavg_standard import run_fedavg
from src.server import run_fedrankx

splits = load_client_splits("data/processed/client_splits", 16)

# Verify unified feature schema
for cid, s in splits.items():
    ncols = len(s["train"].columns)
    assert ncols == 80, f"Client {cid}: expected 80 cols (79 feat + label), got {ncols}"
print("Feature alignment: PASSED — all 16 clients have 79 features + label\n")

# FedAvg baseline (3 rounds, fast)
r_fa = run_fedavg(splits, n_clients=16, n_rounds=3, n_estimators=20,
                  results_dir="results/smoke_test")
print(f"FedAvg  smoke: F1={r_fa['final_f1']:.4f}  AUC={r_fa['final_auc']:.4f}\n")

# FedRankX (3 rounds, fast)
r_fx = run_fedrankx(n_clients=16, n_rounds=3, top_k=15, n_estimators=20,
                    results_dir="results/smoke_test",
                    split_dir="data/processed/client_splits")
print(f"FedRankX smoke: F1={r_fx['final_f1']:.4f}  AUC={r_fx['final_auc']:.4f}  "
      f"NMI={r_fx['final_nmi']:.3f}  ARI={r_fx['final_ari']:.3f}")

print("\n=== SMOKE TEST PASSED ===")
