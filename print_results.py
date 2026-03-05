import json, glob, os

print('='*70)
print('FINAL COMPARISON TABLE')
print('='*70)

for name in ['FedRankX', 'FedAvg', 'FedClust', 'IFCA', 'LocalOnly', 'RandomCluster']:
    try:
        with open(f'results_v2/{name}.json') as f:
            d = json.load(f)
        f1 = d.get('f1_mean', d.get('final_f1', 0))
        f1s = d.get('f1_std', 0)
        auc = d.get('auc_mean', d.get('final_auc', 0))
        aucs = d.get('auc_std', 0)
        nmi = d.get('nmi_mean', d.get('final_nmi', '-'))
        ari = d.get('ari_mean', d.get('final_ari', '-'))
        print(f'  {name:15s} F1={f1:.4f}+/-{f1s:.4f}  AUC={auc:.4f}+/-{aucs:.4f}  NMI={nmi}  ARI={ari}')
    except Exception as e:
        print(f'  {name:15s} NOT FOUND: {e}')

print()
print('='*70)
print('ABLATION RESULTS')
print('='*70)
try:
    with open('results_v2/ablations/all_ablations.json') as f:
        ablations = json.load(f)
    for a in ablations:
        print(f"  {a['ablation']:10s} = {str(a['value']):10s}  F1={a['f1']:.4f}  AUC={a['auc']:.4f}  NMI={a['nmi']:.3f}  ARI={a['ari']:.3f}")
except Exception as e:
    print(f'  Error: {e}')

print()
print('='*70)
print('COMMUNICATION COST')
print('='*70)
try:
    with open('results_v2/tables/communication_cost.json') as f:
        cc = json.load(f)
    for k, v in cc.items():
        print(f'  {k:30s} = {v}')
except Exception as e:
    print(f'  Error: {e}')
