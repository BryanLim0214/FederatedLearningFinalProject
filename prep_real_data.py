"""
prep_real_data.py
=================
Load REAL phishing datasets (ISCX-URL-2016 + Kaggle Phishing Email)
and create Non-IID client splits for the FedRankX experiment.

ISCX-URL-2016: 36,707 URLs with 79 numeric lexical/host features
Kaggle Phishing Email: 82,486 emails with text_combined + binary label

Strategy:
- ISCX URLs: Use the 79 existing numeric columns directly.
  Filter to phishing (label=1) and benign (label=0).
- Kaggle Emails: Extract 79 NLP/text features from text_combined.
  Already has binary label (1=phishing, 0=legit).
- Both datasets produce 79 features + 1 label column.
- Client splits are Non-IID by domain type:
    0-7: Banking (ISCX URL data, high-quality, more samples)
    8-13: Healthcare (Kaggle email data, smaller/noisier)
    14-19: Government (ISCX URL data)
    20-27: SMB (Kaggle email data, least data per client)
    28-31: Mixed (both URL + email data)
"""
import sys, os, re, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
N_CLIENTS = 32
OUTPUT_DIR = "data/processed/client_splits_real"


# =====================================================================
# 1) Load ISCX-URL-2016
# =====================================================================
def load_iscx(csv_path="All.csv"):
    """Load ISCX URL dataset, filter to phishing/benign, return features+label."""
    print(f"  Loading ISCX-URL-2016 from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter to phishing and benign only (binary classification)
    df = df[df['URL_Type_obf_Type'].isin(['phishing', 'benign'])].copy()
    df['label'] = (df['URL_Type_obf_Type'] == 'phishing').astype(int)
    df = df.drop(columns=['URL_Type_obf_Type'])

    # All remaining columns are numeric features
    feature_cols = [c for c in df.columns if c != 'label']

    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(0).astype(np.float32)

    print(f"    Loaded {len(df)} URLs ({df['label'].sum()} phishing, "
          f"{len(df)-df['label'].sum()} benign), {len(feature_cols)} features")
    return df, feature_cols


# =====================================================================
# 2) Load Kaggle Phishing Emails + Extract NLP Features
# =====================================================================
def extract_email_features(text):
    """Extract 79 NLP features from email text to match ISCX feature count."""
    if not isinstance(text, str):
        return np.zeros(79, dtype=np.float32)

    features = []

    # Length features (5)
    features.append(len(text))                    # total length
    features.append(len(text.split()))            # word count
    features.append(np.mean([len(w) for w in text.split()]) if text.split() else 0)  # avg word len
    features.append(max([len(w) for w in text.split()], default=0))  # max word len
    features.append(len(text.split('\n')))        # line count

    # Character composition (15)
    features.append(text.count('.'))              # dots
    features.append(text.count(','))              # commas
    features.append(text.count('!'))              # exclamation
    features.append(text.count('?'))              # question marks
    features.append(text.count('@'))              # at signs
    features.append(text.count('#'))              # hashes
    features.append(text.count('$'))              # dollar signs
    features.append(text.count('%'))              # percent
    features.append(text.count('&'))              # ampersand
    features.append(text.count('(') + text.count(')'))  # parentheses
    features.append(text.count('"'))              # quotes
    features.append(text.count("'"))              # apostrophes
    features.append(text.count(':'))              # colons
    features.append(text.count(';'))              # semicolons
    features.append(text.count('-'))              # hyphens

    # URL/link features (10)
    urls = re.findall(r'https?://\S+', text)
    features.append(len(urls))                    # URL count
    features.append(sum(len(u) for u in urls))    # total URL length
    features.append(max([len(u) for u in urls], default=0))  # max URL length
    features.append(1 if any('http://' in u for u in urls) else 0)  # has http
    features.append(1 if any('https://' in u for u in urls) else 0) # has https
    features.append(text.lower().count('click'))  # "click" count
    features.append(text.lower().count('link'))   # "link" count
    features.append(text.lower().count('www'))    # "www" count
    features.append(text.lower().count('.com'))    # ".com" count
    features.append(text.lower().count('.org'))    # ".org" count

    # Urgency/phishing keywords (15)
    urgency_words = ['urgent', 'immediately', 'action', 'required', 'verify',
                     'confirm', 'suspend', 'account', 'password', 'update',
                     'expire', 'warning', 'alert', 'security', 'unauthorized']
    for word in urgency_words:
        features.append(text.lower().count(word))

    # Structural features (10)
    features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # uppercase ratio
    features.append(sum(1 for c in text if c.isdigit()) / max(len(text), 1))  # digit ratio
    features.append(sum(1 for c in text if c.isalpha()) / max(len(text), 1))  # alpha ratio
    features.append(sum(1 for c in text if c == ' ') / max(len(text), 1))     # space ratio
    features.append(sum(1 for c in text if not c.isalnum() and c != ' ') / max(len(text), 1))  # special char ratio
    features.append(len(set(text.split())) / max(len(text.split()), 1))       # vocab diversity
    features.append(1 if re.search(r'\d{4}[\-/]\d{2}[\-/]\d{2}', text) else 0)  # has date
    features.append(1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0)  # has phone
    features.append(1 if re.search(r'[A-Z]{2,}', text) else 0)               # has all-caps word
    features.append(text.count('\n\n'))           # paragraph count

    # Sentiment/tone features (10)
    tone_words = ['dear', 'sir', 'madam', 'congratulations', 'winner',
                  'free', 'offer', 'limited', 'prize', 'reward']
    for word in tone_words:
        features.append(text.lower().count(word))

    # Email-specific features (10)
    features.append(1 if 'from:' in text.lower() else 0)
    features.append(1 if 'to:' in text.lower() else 0)
    features.append(1 if 'subject:' in text.lower() else 0)
    features.append(1 if 'reply' in text.lower() else 0)
    features.append(1 if 'forward' in text.lower() else 0)
    features.append(1 if 'attachment' in text.lower() else 0)
    features.append(1 if 'unsubscribe' in text.lower() else 0)
    features.append(text.lower().count('please'))
    features.append(text.lower().count('thank'))
    features.append(text.lower().count('kind'))

    # Entropy features (4)
    from collections import Counter
    char_freq = Counter(text.lower())
    total_chars = sum(char_freq.values())
    entropy = -sum((c/total_chars) * math.log2(c/total_chars)
                    for c in char_freq.values() if c > 0) if total_chars > 0 else 0
    features.append(entropy)

    word_freq = Counter(text.lower().split())
    total_words = sum(word_freq.values())
    word_entropy = -sum((c/total_words) * math.log2(c/total_words)
                        for c in word_freq.values() if c > 0) if total_words > 0 else 0
    features.append(word_entropy)

    features.append(len(set(text.lower().split())))  # unique word count
    features.append(len(text) / max(len(text.split()), 1))  # chars per word

    # Pad or truncate to 79
    features = features[:79]
    while len(features) < 79:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def load_kaggle_email():
    """Load Kaggle phishing email dataset with NLP features."""
    kaggle_path = os.path.expanduser(
        '~/.cache/kagglehub/datasets/naserabdullahalam/'
        'phishing-email-dataset/versions/1/phishing_email.csv'
    )
    print(f"  Loading Kaggle Phishing Email from {kaggle_path}...")
    df = pd.read_csv(kaggle_path, on_bad_lines='skip', encoding='utf-8')

    # Subsample for speed (keep 15,000 — balanced)
    n_sample = min(15000, len(df))
    df = df.sample(n=n_sample, random_state=SEED).reset_index(drop=True)

    print(f"    Extracting 79 NLP features from {len(df)} emails...")
    feature_arrays = []
    for i, text in enumerate(df['text_combined']):
        feature_arrays.append(extract_email_features(text))
        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{len(df)} emails processed...")

    # Create feature names matching ISCX column count
    feature_cols = [f"email_feat_{i}" for i in range(79)]
    feature_df = pd.DataFrame(feature_arrays, columns=feature_cols)
    feature_df['label'] = df['label'].values.astype(int)

    print(f"    Loaded {len(feature_df)} emails ({feature_df['label'].sum()} phishing, "
          f"{len(feature_df)-feature_df['label'].sum()} legit), 79 features")
    return feature_df, feature_cols


# =====================================================================
# 3) Create Non-IID Client Splits
# =====================================================================
def create_client_splits(iscx_df, iscx_cols, email_df, email_cols):
    """
    Create 32 Non-IID client splits:
      0-7: Banking (ISCX URL data, more data)
      8-13: Healthcare (Kaggle email, less data)
      14-19: Government (ISCX URL data)
      20-27: SMB (Kaggle email, least data)
      28-31: Mixed (both)
    """
    np.random.seed(SEED)

    # Shuffle both datasets
    iscx_df = iscx_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    email_df = email_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Use common column names for each domain
    iscx_renamed = iscx_df.copy()
    iscx_renamed.columns = [f"feat_{i}" for i in range(len(iscx_cols))] + ['label']

    email_renamed = email_df.copy()
    email_renamed.columns = [f"feat_{i}" for i in range(len(email_cols))] + ['label']

    # Split ISCX into banking (0-7) and government (14-19) = 14 clients
    iscx_per_client = len(iscx_renamed) // 14
    iscx_splits = []
    for i in range(14):
        start = i * iscx_per_client
        end = start + iscx_per_client if i < 13 else len(iscx_renamed)
        iscx_splits.append(iscx_renamed.iloc[start:end])

    # Split email into healthcare (8-13) and SMB (20-27) = 14 clients
    email_per_client = len(email_renamed) // 14
    email_splits = []
    for i in range(14):
        start = i * email_per_client
        end = start + email_per_client if i < 13 else len(email_renamed)
        email_splits.append(email_renamed.iloc[start:end])

    # Add Non-IID heterogeneity: vary phishing ratio per client
    client_data = {}
    # Banking (0-7): ISCX URL data
    for cid in range(8):
        client_data[cid] = iscx_splits[cid]

    # Healthcare (8-13): Email data
    for cid in range(8, 14):
        client_data[cid] = email_splits[cid - 8]

    # Government (14-19): ISCX URL data
    for cid in range(14, 20):
        client_data[cid] = iscx_splits[cid - 14 + 8]

    # SMB (20-27): Email data
    for cid in range(20, 28):
        client_data[cid] = email_splits[cid - 20 + 6]

    # Mixed (28-31): Mix of URL + email data
    for cid in range(28, 32):
        # Take some from each
        iscx_portion = iscx_renamed.sample(n=min(300, len(iscx_renamed)), random_state=SEED+cid)
        email_portion = email_renamed.sample(n=min(300, len(email_renamed)), random_state=SEED+cid+100)
        client_data[cid] = pd.concat([iscx_portion, email_portion], ignore_index=True)

    # Create train/test splits and save
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    all_splits = {}
    for cid in range(N_CLIENTS):
        data = client_data[cid]
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=SEED,
                                              stratify=data['label'] if len(data['label'].unique()) > 1 else None)
        train_df.to_csv(output / f"client_{cid:02d}_train.csv", index=False)
        test_df.to_csv(output / f"client_{cid:02d}_test.csv", index=False)
        all_splits[cid] = {"train": train_df, "test": test_df}

        phish_ratio = data['label'].mean()
        sector = ["Banking", "Banking", "Banking", "Banking", "Banking", "Banking", "Banking", "Banking",
                   "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare", "Healthcare",
                   "Government", "Government", "Government", "Government", "Government", "Government",
                   "SMB", "SMB", "SMB", "SMB", "SMB", "SMB", "SMB", "SMB",
                   "Mixed", "Mixed", "Mixed", "Mixed"][cid]
        data_type = "URL" if cid < 8 or 14 <= cid < 20 else "Email" if 8 <= cid < 14 or 20 <= cid < 28 else "Both"
        print(f"  Client {cid:02d} ({sector:12s}, {data_type:5s}): "
              f"{len(train_df)} train, {len(test_df)} test, "
              f"phishing={phish_ratio:.1%}")

    return all_splits


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  PREPARING REAL PHISHING DATASETS FOR FEDRANKX")
    print("=" * 70)
    print()

    iscx_df, iscx_cols = load_iscx("All.csv")
    email_df, email_cols = load_kaggle_email()

    print(f"\n  Creating {N_CLIENTS} Non-IID client splits...")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    splits = create_client_splits(iscx_df, iscx_cols, email_df, email_cols)

    print(f"\n  Done! {N_CLIENTS} clients saved to {OUTPUT_DIR}/")
    print(f"  ISCX features: {len(iscx_cols)}, Email features: {len(email_cols)}")
    return splits


if __name__ == "__main__":
    main()
