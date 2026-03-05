"""
prep_real_hard.py
=================
Create HARD Non-IID client splits from REAL ISCX-URL-2016 + Kaggle data.

To make the task realistically challenging (so methods actually differ):
1. Feature subsets: Each client uses only 15-20 of 79 features (simulates
   different organizations collecting different data about URLs/emails)
2. Dirichlet label skew (α=0.3): Each client gets skewed class distribution
3. Small training sets (~300 samples): Simulates limited per-org data
4. Label noise (5-10%): Simulates real-world annotation errors

This creates a scenario where:
- Individual models are weak (limited features + small data)
- FedRankX's clustering discovers which orgs have similar feature sets
- Ensemble predictions from similar orgs dramatically improve performance
"""
import sys, os, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
N_CLIENTS = 32
OUTPUT_DIR = "data/processed/client_splits_real_hard"

# Feature groups (simulating what different org types collect)
# Banking: URL structure features (length, tokens, TLD)
# Healthcare: Content-based email features
# Government: Network/host features
# SMB: Basic email features
# Each group covers ~20 features from the full 79

def load_and_prepare():
    """Load real data, merge ISCX and email, create unified feature matrix."""
    np.random.seed(SEED)

    # -- Load ISCX-URL-2016 --
    print("  Loading ISCX-URL-2016 (All.csv)...")
    iscx = pd.read_csv('All.csv')
    # Binary: phishing=1, everything else=0 (benign, malware, spam, defacement)
    # Actually keep phishing vs benign only for cleaner classification
    iscx = iscx[iscx['URL_Type_obf_Type'].isin(['phishing', 'benign'])].copy()
    iscx['label'] = (iscx['URL_Type_obf_Type'] == 'phishing').astype(int)
    iscx = iscx.drop(columns=['URL_Type_obf_Type'])

    feature_cols = [c for c in iscx.columns if c != 'label']
    iscx[feature_cols] = iscx[feature_cols].fillna(0).astype(np.float32)
    print(f"    ISCX: {len(iscx)} samples, {len(feature_cols)} features "
          f"({iscx['label'].sum()} phishing, {len(iscx)-iscx['label'].sum()} benign)")

    # -- Load Kaggle Phishing Emails --
    kaggle_path = os.path.expanduser(
        '~/.cache/kagglehub/datasets/naserabdullahalam/'
        'phishing-email-dataset/versions/1/phishing_email.csv'
    )
    print(f"  Loading Kaggle Phishing Emails...")
    import re
    emails = pd.read_csv(kaggle_path, on_bad_lines='skip', encoding='utf-8')
    emails = emails.sample(n=min(15000, len(emails)), random_state=SEED).reset_index(drop=True)

    # Extract NLP features into the same 79-column schema
    print(f"    Extracting NLP features from {len(emails)} emails...")
    email_features = []
    for i, text in enumerate(emails['text_combined']):
        if not isinstance(text, str):
            text = ""
        f = []
        # Match ISCX feature count: extract 79 features from text
        words = text.split()
        urls = re.findall(r'https?://\S+', text)

        # Lexical (20 features matching ISCX query/domain/path lengths)
        f.append(len(text))
        f.append(len(words))
        f.append(np.mean([len(w) for w in words]) if words else 0)
        f.append(max([len(w) for w in words], default=0))
        f.append(len(text.split('\n')))
        f.append(text.count('.'))
        f.append(text.count(','))
        f.append(text.count('!'))
        f.append(text.count('?'))
        f.append(text.count('@'))
        f.append(text.count(':'))
        f.append(text.count(';'))
        f.append(text.count('-'))
        f.append(text.count('(') + text.count(')'))
        f.append(text.count('"'))
        f.append(text.count('/'))
        f.append(text.count('='))
        f.append(text.count('&'))
        f.append(text.count('%'))
        f.append(text.count('#'))

        # URL features (15)
        f.append(len(urls))
        f.append(sum(len(u) for u in urls))
        f.append(max([len(u) for u in urls], default=0))
        f.append(1 if any('http://' in u for u in urls) else 0)
        f.append(1 if any('https://' in u for u in urls) else 0)
        f.append(text.lower().count('click'))
        f.append(text.lower().count('link'))
        f.append(text.lower().count('www'))
        f.append(text.lower().count('.com'))
        f.append(text.lower().count('.org'))
        f.append(text.lower().count('.net'))
        f.append(text.lower().count('.exe'))
        f.append(1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text) else 0)
        f.append(text.lower().count('download'))
        f.append(text.lower().count('install'))

        # Urgency/phishing keywords (20)
        for w in ['urgent', 'immediately', 'action', 'required', 'verify',
                   'confirm', 'suspend', 'account', 'password', 'update',
                   'expire', 'warning', 'alert', 'security', 'unauthorized',
                   'bank', 'paypal', 'login', 'credit', 'ssn']:
            f.append(text.lower().count(w))

        # Structural (15)
        total_c = max(len(text), 1)
        f.append(sum(1 for c in text if c.isupper()) / total_c)
        f.append(sum(1 for c in text if c.isdigit()) / total_c)
        f.append(sum(1 for c in text if c.isalpha()) / total_c)
        f.append(sum(1 for c in text if c == ' ') / total_c)
        f.append(sum(1 for c in text if not c.isalnum() and c != ' ') / total_c)
        f.append(len(set(words)) / max(len(words), 1))
        f.append(text.count('\n\n'))
        f.append(1 if 'dear' in text.lower() else 0)
        f.append(1 if 'sir' in text.lower() or 'madam' in text.lower() else 0)
        f.append(text.lower().count('free'))
        f.append(text.lower().count('winner'))
        f.append(text.lower().count('prize'))
        f.append(text.lower().count('offer'))
        f.append(text.lower().count('congratulation'))
        f.append(1 if 'unsubscribe' in text.lower() else 0)

        # Entropy (4)
        from collections import Counter
        cf = Counter(text.lower())
        tc = sum(cf.values())
        ent = -sum((c/tc)*math.log2(c/tc) for c in cf.values() if c > 0) if tc > 0 else 0
        f.append(ent)
        wf = Counter(text.lower().split())
        tw = sum(wf.values())
        went = -sum((c/tw)*math.log2(c/tw) for c in wf.values() if c > 0) if tw > 0 else 0
        f.append(went)
        f.append(len(set(text.lower().split())))
        f.append(len(text) / max(len(words), 1))

        # Pad/truncate to 79
        f = f[:79]
        while len(f) < 79:
            f.append(0.0)
        email_features.append(f)

        if (i+1) % 3000 == 0:
            print(f"      {i+1}/{len(emails)} emails...")

    email_df = pd.DataFrame(email_features, columns=feature_cols)
    email_df['label'] = emails['label'].values.astype(int)
    print(f"    Emails: {len(email_df)} samples "
          f"({email_df['label'].sum()} phishing, {len(email_df)-email_df['label'].sum()} legit)")

    return iscx, email_df, feature_cols


def create_hard_splits(iscx, email_df, feature_cols):
    """Create 32 hard Non-IID client splits."""
    np.random.seed(SEED)
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    n_features = len(feature_cols)

    # Define 5 feature groups (overlapping subsets of the 79 features)
    # Each group has ~20 features, with some overlap
    indices = np.arange(n_features)
    np.random.shuffle(indices)

    feature_groups = {
        'banking':    sorted(indices[:25]),       # First 25 shuffled features
        'healthcare': sorted(indices[18:43]),      # features 18-43 (7 overlap with banking)
        'government': sorted(indices[35:60]),      # features 35-60 (8 overlap with healthcare)
        'smb':        sorted(indices[50:75]),       # features 50-75 (10 overlap with government)
        'mixed':      sorted(indices[10:35]),       # features 10-35 (overlaps banking+healthcare)
    }

    # Client assignments
    SECTORS = (
        ['banking']*8 + ['healthcare']*6 + ['government']*6 +
        ['smb']*8 + ['mixed']*4
    )

    # Data source: banking/government use ISCX, healthcare/smb/mixed use email
    DATA_SOURCE = {
        'banking': iscx, 'government': iscx,
        'healthcare': email_df, 'smb': email_df, 'mixed': email_df,
    }

    # Dirichlet allocation for label skew
    alpha = 0.5  # Moderate skew (not too extreme)
    samples_per_client = 500  # Moderate training sets

    all_splits = {}
    meta_rows = []

    for cid in range(N_CLIENTS):
        sector = SECTORS[cid]
        source_df = DATA_SOURCE[sector]
        feat_indices = feature_groups[sector]

        # Sample with Dirichlet-based label imbalance
        label_prob = np.random.dirichlet([alpha, alpha])
        # Skew: some clients get more phishing, others more benign
        phishing_ratio = np.clip(label_prob[0], 0.20, 0.80)  # Clamp to 20-80%

        n_phishing = max(10, int(samples_per_client * phishing_ratio))
        n_benign = samples_per_client - n_phishing

        phish_pool = source_df[source_df['label'] == 1]
        benign_pool = source_df[source_df['label'] == 0]

        n_phishing = min(n_phishing, len(phish_pool))
        n_benign = min(n_benign, len(benign_pool))

        client_phish = phish_pool.sample(n=n_phishing, random_state=SEED+cid, replace=True)
        client_benign = benign_pool.sample(n=n_benign, random_state=SEED+cid+100, replace=True)
        client_data = pd.concat([client_phish, client_benign], ignore_index=True)

        # Apply feature masking: zero out features not in this client's group
        for i in range(n_features):
            if i not in feat_indices:
                client_data.iloc[:, i] = 0.0

        # Add label noise (3-8% per client)
        noise_rate = np.random.uniform(0.03, 0.08)
        n_noisy = int(len(client_data) * noise_rate)
        noisy_idx = np.random.choice(len(client_data), n_noisy, replace=False)
        client_data.iloc[noisy_idx, -1] = 1 - client_data.iloc[noisy_idx, -1]

        # Shuffle
        client_data = client_data.sample(frac=1, random_state=SEED+cid).reset_index(drop=True)

        # Rename columns to feat_0..feat_78, label
        client_data.columns = [f"feat_{i}" for i in range(n_features)] + ['label']

        # Train/test split
        train_df, test_df = train_test_split(
            client_data, test_size=0.25, random_state=SEED,
            stratify=client_data['label']
        )

        train_df.to_csv(output / f"client_{cid:02d}_train.csv", index=False)
        test_df.to_csv(output / f"client_{cid:02d}_test.csv", index=False)
        all_splits[cid] = {"train": train_df, "test": test_df}

        phish_rate = client_data['label'].mean()
        print(f"  Client {cid:02d} ({sector:12s}): {len(train_df)} train, "
              f"{len(test_df)} test, phish={phish_rate:.1%}, "
              f"feats={len(feat_indices)}, noise={noise_rate:.1%}")

        meta_rows.append({
            'client_id': cid, 'domain': sector,
            'type_label': ['banking','healthcare','government','smb','mixed'].index(sector),
            'n_train': len(train_df), 'n_test': len(test_df),
            'phish_rate': phish_rate, 'n_features': len(feat_indices),
            'noise_rate': noise_rate,
        })

    # Save metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(output / "client_metadata.csv", index=False)

    print(f"\n  Saved {N_CLIENTS} hard Non-IID client splits to {OUTPUT_DIR}/")
    return all_splits


def main():
    print("=" * 70)
    print("  CREATING HARD NON-IID SPLITS FROM REAL DATA")
    print("  ISCX-URL-2016 + Kaggle Phishing Emails")
    print("=" * 70)
    print()

    iscx, email_df, feature_cols = load_and_prepare()
    print()
    splits = create_hard_splits(iscx, email_df, feature_cols)

    print("\n  Ready for FedRankX experiment!")


if __name__ == "__main__":
    main()
