"""run_fast_real_smoke.py"""
from src.server import run_fedrankx

print('Running 2-round FedRankX on real ISCX/Kaggle datasets...')
result = run_fedrankx(
    n_clients=16,
    n_rounds=2,
    top_k=15,
    n_clusters=None,
    distance_threshold=0.5,
    linkage_method='ward',
    rank_metric='spearman',
    n_estimators=20,
    split_dir='data/processed/client_splits',
    results_dir='results/smoke_test_real',
    seed=42,
    samples_per_client=1500,
    use_real_data=True
)
print('\n=== 2-Round Real Data Test Results ===')
print(f'F1:  {result["final_f1"]:.4f}')
print(f'AUC: {result["final_auc"]:.4f}')
print(f'NMI: {result["final_nmi"]:.4f}')
print(f'ARI: {result["final_ari"]:.4f}')
print('======================================')
