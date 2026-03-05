"""Create client_metadata.csv for real data splits."""
import pandas as pd

SECTOR = {
    **{i: "banking" for i in range(8)},
    **{i: "healthcare" for i in range(8, 14)},
    **{i: "government" for i in range(14, 20)},
    **{i: "smb" for i in range(20, 28)},
    **{i: "mixed" for i in range(28, 32)},
}
TYPE_LABEL = {
    **{i: 0 for i in range(8)},       # banking
    **{i: 1 for i in range(8, 14)},    # healthcare
    **{i: 2 for i in range(14, 20)},   # government
    **{i: 3 for i in range(20, 28)},   # smb
    **{i: 4 for i in range(28, 32)},   # mixed
}

rows = []
for cid in range(32):
    train = pd.read_csv(f'data/processed/client_splits_real/client_{cid:02d}_train.csv')
    test = pd.read_csv(f'data/processed/client_splits_real/client_{cid:02d}_test.csv')
    rows.append({
        'client_id': cid,
        'domain': SECTOR[cid],
        'type_label': TYPE_LABEL[cid],
        'n_train': len(train),
        'n_test': len(test),
        'phish_rate': train['label'].mean(),
    })

meta = pd.DataFrame(rows)
meta.to_csv('data/processed/client_splits_real/client_metadata.csv', index=False)
print(meta.to_string())
print("\nSaved client_metadata.csv!")
