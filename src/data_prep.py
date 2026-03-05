"""
data_prep.py
============
Downloads / loads the two phishing datasets and partitions them into
Non-IID splits simulating heterogeneous federated clients.

Datasets
--------
A) ISCX-URL2016  -- URL-structure features (banking/government clients)
B) Phishing Email (Kaggle composite) -- NLP+metadata features (healthcare/SMB clients)

If the raw files are not present, the script synthesizes a realistic
stand-in dataset following the same feature distributions so that the
full pipeline can be tested immediately. Replace with real data for
the final paper runs.

Usage
-----
    python -m src.data_prep --output_dir data/processed/client_splits --n_clients 16
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_CLIENTS = 16

# ISCX-URL2016 feature names (79 features from the URL lexical/host/content-based set)
ISCX_FEATURES = [
    "url_length", "hostname_length", "path_length", "fd_length",
    "count_dots", "count_hyphens", "count_at", "count_qm",
    "count_excl", "count_and", "count_eq", "count_underscore",
    "count_tilde", "count_pct", "count_slash", "count_star",
    "count_colon", "count_comma", "count_semicolon",
    "use_of_ip", "count_http_token", "https_in_url",
    "shortening_service", "count_www", "count_path_from_domain",
    "ratio_digits", "count_digits", "port",
    "tld_in_path", "tld_in_subdomain", "prefix_suffix",
    "random_domain", "punycode", "count_subdomain", "count_path_components",
    "double_slash_redirect", "root_domain_length",
    "url_entropy", "hostname_entropy", "path_entropy",
    "count_embedding_domain", "count_letter", "count_vowels",
    "count_consonants", "count_special_chars", "max_len_split",
    "mean_len_split", "longest_word", "shortest_word",
    "brand_in_subdomain", "brand_in_path", "brand_in_url",
    "suspicious_tld", "domain_in_brand", "path_in_brand",
    "ratio_phish_hints", "has_client", "has_admin",
    "has_server", "has_login", "has_secure", "has_account",
    "has_update", "has_signin", "has_webscr",
    "mx_domain", "spf_record", "time_domain_activation",
    "time_domain_expire", "time_domain_update",
    "dns_record", "domain_registration_length", "domain_age",
    "web_traffic", "page_rank", "google_index",
    "links_in_tags", "statistical_report", "iframe"
]

# Email feature names (NLP + metadata derived)
EMAIL_FEATURES = [
    "urgency_score", "fear_score", "reward_score", "auth_request_score",
    "subject_length", "subject_excl_count", "subject_qm_count",
    "subject_caps_ratio", "body_length", "body_link_count",
    "body_html_ratio", "body_excl_count", "body_suspicious_keywords",
    "attachment_flag", "attachment_count", "attachment_exe_flag",
    "sender_domain_mismatch", "sender_display_name_mismatch",
    "reply_to_mismatch", "spf_pass", "dkim_pass",
    "recipient_count", "cc_count", "bcc_count",
    "hour_of_day", "day_of_week", "is_weekend",
    "has_login_link", "has_verify_link", "has_update_link"
]

# Client type definitions — which feature set and data distribution bias
CLIENT_TYPES = {
    # (feature_set_name, phishing_rate_bias, n_clients, label)
    "banking":    ("iscx",  0.45, 8, 0),  # URL-heavy, moderate phishing rate
    "healthcare": ("email", 0.40, 6, 1),  # Email-heavy, urgency/auth attacks
    "government": ("iscx",  0.35, 6, 2),  # URL-heavy, defacement/redirect focused
    "sme":        ("email", 0.50, 8, 3),  # Email, high phishing rate
    "mixed":      ("mixed", 0.42, 4, 4),  # Both feature sets
}
N_CLIENTS = sum(v[2] for v in CLIENT_TYPES.values())  # 32


# ---------------------------------------------------------------------------
# Synthetic data generation (stand-in until real datasets are in place)
# ---------------------------------------------------------------------------

def _make_iscx_like(n_samples: int, phishing_rate: float,
                    rng: np.random.Generator,
                    feature_bias: np.ndarray = None) -> pd.DataFrame:
    """Generate realistic ISCX-URL2016-like feature vectors."""
    n_phish = int(n_samples * phishing_rate)
    n_benign = n_samples - n_phish
    n_feats = len(ISCX_FEATURES)

    # Benign URLs: shorter, fewer suspicious markers
    benign = rng.normal(loc=0.2, scale=0.15, size=(n_benign, n_feats))
    # Phishing URLs: more suspicious feature activations
    phish = rng.normal(loc=0.6, scale=0.2, size=(n_phish, n_feats))

    # Apply domain-specific bias to distinguish client types
    if feature_bias is not None:
        phish = phish + feature_bias * 0.25

    data = np.vstack([benign, phish]).clip(0, 1)
    labels = np.array([0] * n_benign + [1] * n_phish)

    df = pd.DataFrame(data, columns=ISCX_FEATURES)
    df["label"] = labels
    return df.sample(frac=1, random_state=rng.integers(0, 10000)).reset_index(drop=True)


def _make_email_like(n_samples: int, phishing_rate: float,
                     rng: np.random.Generator,
                     feature_bias: np.ndarray = None) -> pd.DataFrame:
    """Generate realistic phishing-email-like feature vectors."""
    n_phish = int(n_samples * phishing_rate)
    n_benign = n_samples - n_phish
    n_feats = len(EMAIL_FEATURES)

    benign = rng.normal(loc=0.15, scale=0.1, size=(n_benign, n_feats))
    phish = rng.normal(loc=0.65, scale=0.2, size=(n_phish, n_feats))

    if feature_bias is not None:
        phish = phish + feature_bias * 0.25

    data = np.vstack([benign, phish]).clip(0, 1)
    labels = np.array([0] * n_benign + [1] * n_phish)

    df = pd.DataFrame(data, columns=EMAIL_FEATURES)
    df["label"] = labels
    return df.sample(frac=1, random_state=rng.integers(0, 10000)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real data loaders (to replace synthetic when files exist)
# ---------------------------------------------------------------------------

def load_iscx_url2016(path: str) -> pd.DataFrame:
    """
    Load ISCX-URL2016 dataset.
    Download from: https://www.unb.ca/cic/datasets/url-2016.html
    Expected file: data/raw/ISCX-URL2016/All.csv
    """
    df = pd.read_csv(path)
    
    # Map the primary label column if present
    if "URL_Type_obf_Type" in df.columns:
        df = df.rename(columns={"URL_Type_obf_Type": "Type"})
        
    # Keep phishing vs benign only (drop spam/malware/defacement)
    df = df[df["Type"].isin(["benign", "phishing"])].copy()
    df["label"] = (df["Type"] == "phishing").astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in ["url", "Type", "URL_Type_obf_Type"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Use ALL remaining numeric columns as features
    feature_cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    
    # Drop columns that are all NaN
    feature_cols = [c for c in feature_cols if df[c].notna().any()]
    
    df = df[feature_cols + ["label"]].copy()
    df[feature_cols] = df[feature_cols].fillna(0).astype(np.float32)
    
    print(f"[data_prep] ISCX loaded: {len(df)} rows, {len(feature_cols)} features")
    return df.reset_index(drop=True)


def load_phishing_email(path: str) -> pd.DataFrame:
    """
    Load Kaggle phishing email dataset.
    Download: https://www.kaggle.com/datasets/subhajournal/phishingemails
    Expected file: data/raw/phishing_email/Phishing_Email.csv
    This dataset has 'Email Text' and 'Email Type' columns.
    We extract features manually.
    """
    import re

    df = pd.read_csv(path, usecols=["Email Text", "Email Type"])
    df = df.dropna()
    df["label"] = (df["Email Type"] == "Phishing Email").astype(int)

    urgency_words = ["urgent", "immediately", "action required", "verify",
                     "account suspended", "click here", "confirm", "expire",
                     "limited time", "act now", "warning", "alert"]
    fear_words = ["suspended", "blocked", "unauthorized", "fraud", "illegal",
                  "failure", "locked", "terminated", "breach"]
    reward_words = ["winner", "prize", "gift", "reward", "bonus",
                    "congratulations", "free", "offer"]
    auth_words = ["password", "username", "login", "sign in", "credentials",
                  "verify identity", "update account"]

    def extract_features(text: str) -> dict:
        text_lower = str(text).lower()
        return {
            "urgency_score": sum(w in text_lower for w in urgency_words) / len(urgency_words),
            "fear_score": sum(w in text_lower for w in fear_words) / len(fear_words),
            "reward_score": sum(w in text_lower for w in reward_words) / len(reward_words),
            "auth_request_score": sum(w in text_lower for w in auth_words) / len(auth_words),
            "body_length": min(len(text) / 5000, 1.0),
            "body_link_count": min(len(re.findall(r"http[s]?://", text)) / 20, 1.0),
            "body_html_ratio": min(len(re.findall(r"<[^>]+>", text)) / max(len(text), 1), 1.0),
            "body_excl_count": min(text.count("!") / 20, 1.0),
            "body_suspicious_keywords": min(
                sum(w in text_lower for w in urgency_words + fear_words) / 10, 1.0
            ),
            "has_login_link": float("login" in text_lower or "signin" in text_lower),
            "has_verify_link": float("verify" in text_lower),
            "has_update_link": float("update" in text_lower),
            "attachment_flag": 0.0,  # not inferable from text alone
            "attachment_count": 0.0,
            "attachment_exe_flag": 0.0,
            "sender_domain_mismatch": 0.0,
            "sender_display_name_mismatch": 0.0,
            "reply_to_mismatch": 0.0,
            "spf_pass": 0.5,
            "dkim_pass": 0.5,
            "recipient_count": 1.0,
            "cc_count": 0.0,
            "bcc_count": 0.0,
            "hour_of_day": 0.5,
            "day_of_week": 0.5,
            "is_weekend": 0.0,
            "subject_length": 0.5,
            "subject_excl_count": 0.0,
            "subject_qm_count": 0.0,
            "subject_caps_ratio": 0.0,
        }

    features = df["Email Text"].apply(extract_features)
    feat_df = pd.DataFrame(list(features))
    feat_df["label"] = df["label"].values
    return feat_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Non-IID partitioning
# ---------------------------------------------------------------------------

def create_client_splits(
    output_dir: str,
    n_clients: int = N_CLIENTS,
    samples_per_client: int = 1500,
    use_real_data: bool = False,
    iscx_path: str = None,
    email_path: str = None,
    seed: int = RANDOM_SEED,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Create Non-IID data splits for N clients.

    Returns dict: client_id -> {"train": DataFrame, "test": DataFrame}
    Also saves each split as CSV to output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    client_splits: Dict[int, Dict[str, pd.DataFrame]] = {}
    client_meta: List[dict] = []

    # --- Load or generate base datasets ---
    if use_real_data and iscx_path and Path(iscx_path).exists():
        print(f"[data_prep] Loading real ISCX-URL2016 from {iscx_path}")
        iscx_df = load_iscx_url2016(iscx_path)
    else:
        print("[data_prep] ISCX data not found — using synthetic stand-in")
        iscx_df = None  # will generate per-client

    if use_real_data and email_path and Path(email_path).exists():
        print(f"[data_prep] Loading real phishing email data from {email_path}")
        email_df = load_phishing_email(email_path)
    else:
        print("[data_prep] Email data not found — using synthetic stand-in")
        email_df = None

    # === Build unified feature schema from loaded data ===
    # Collect all feature column names across all data sources
    all_feature_names = set()
    if iscx_df is not None:
        all_feature_names |= set(c for c in iscx_df.columns if c != "label")
    else:
        all_feature_names |= set(ISCX_FEATURES)
    if email_df is not None:
        all_feature_names |= set(c for c in email_df.columns if c != "label")
    else:
        all_feature_names |= set(EMAIL_FEATURES)
    all_feature_names = sorted(all_feature_names)  # deterministic order
    print(f"[data_prep] Unified feature schema: {len(all_feature_names)} features")

    client_id = 0

    for domain, (feat_type, phishing_rate, n_clients_domain, type_label) in CLIENT_TYPES.items():
        num_clients_this_domain = n_clients_domain

        for i in range(num_clients_this_domain):
            # Each client has a slightly different feature bias (intra-type heterogeneity)
            n_feats = len(ISCX_FEATURES) if feat_type in ("iscx", "mixed") else len(EMAIL_FEATURES)
            feature_bias = rng.uniform(-0.2, 0.2, size=n_feats)

            client_phish_rate = np.clip(
                phishing_rate + rng.uniform(-0.05, 0.05), 0.2, 0.7
            )

            # Determine data source
            if feat_type == "iscx":
                if iscx_df is not None:
                    df = _sample_from_real(iscx_df, samples_per_client, client_phish_rate, rng)
                else:
                    df = _make_iscx_like(samples_per_client, client_phish_rate, rng, feature_bias)
            elif feat_type == "email":
                if email_df is not None:
                    df = _sample_from_real(email_df, samples_per_client, client_phish_rate, rng)
                else:
                    df = _make_email_like(samples_per_client, client_phish_rate, rng, feature_bias)
            else:  # mixed
                n_half = samples_per_client // 2
                if iscx_df is not None:
                    df_iscx = _sample_from_real(iscx_df, n_half, client_phish_rate, rng)
                else:
                    feat_bias_iscx = rng.uniform(-0.2, 0.2, size=len(ISCX_FEATURES))
                    df_iscx = _make_iscx_like(n_half, client_phish_rate, rng, feat_bias_iscx)
                if email_df is not None:
                    df_email = _sample_from_real(email_df, samples_per_client - n_half, client_phish_rate, rng)
                else:
                    feat_bias_email = rng.uniform(-0.2, 0.2, size=len(EMAIL_FEATURES))
                    df_email = _make_email_like(samples_per_client - n_half, client_phish_rate, rng, feat_bias_email)
                # Align columns: fill missing with 0
                all_cols = list(set(df_iscx.columns) | set(df_email.columns))
                df = pd.concat([
                    df_iscx.reindex(columns=all_cols, fill_value=0),
                    df_email.reindex(columns=all_cols, fill_value=0)
                ], ignore_index=True).sample(frac=1, random_state=rng.integers(0, 9999)).reset_index(drop=True)

            # Ensure label column is clean integer
            df["label"] = df["label"].astype(int)

            # Train / test split (80/20 stratified)
            train_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df["label"],
                random_state=int(rng.integers(0, 9999))
            )

            # === Pad to unified feature schema ===
            for split_name, split_df in [("train", train_df), ("test", test_df)]:
                for col in all_feature_names:
                    if col not in split_df.columns:
                        split_df[col] = 0.0
                split_df = split_df[all_feature_names + ["label"]].copy()
                if split_name == "train":
                    train_df = split_df
                else:
                    test_df = split_df

            client_splits[client_id] = {"train": train_df, "test": test_df}

            # Save to disk
            train_df.to_csv(out_path / f"client_{client_id:02d}_train.csv", index=False)
            test_df.to_csv(out_path / f"client_{client_id:02d}_test.csv", index=False)

            client_meta.append({
                "client_id": client_id,
                "domain": domain,
                "feature_type": feat_type,
                "type_label": type_label,
                "phishing_rate": round(float(client_phish_rate), 3),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })

            print(
                f"  [client {client_id:02d}] domain={domain:12s} "
                f"feat={feat_type:5s} phish_rate={client_phish_rate:.2f} "
                f"n_train={len(train_df)} n_test={len(test_df)}"
            )

            client_id += 1

    # Save metadata
    meta_df = pd.DataFrame(client_meta)
    meta_df.to_csv(out_path / "client_metadata.csv", index=False)
    print(f"\n[data_prep] Saved {client_id} client splits to {out_path}")
    print(meta_df.to_string(index=False))

    return client_splits


def _sample_from_real(df: pd.DataFrame, n: int, phish_rate: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from a real dataset honoring the target phishing rate."""
    n_phish = int(n * phish_rate)
    n_benign = n - n_phish
    phish_pool = df[df["label"] == 1]
    benign_pool = df[df["label"] == 0]
    phish_sample = phish_pool.sample(
        min(n_phish, len(phish_pool)),
        random_state=int(rng.integers(0, 9999))
    )
    benign_sample = benign_pool.sample(
        min(n_benign, len(benign_pool)),
        random_state=int(rng.integers(0, 9999))
    )
    return pd.concat([phish_sample, benign_sample], ignore_index=True) \
             .sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)


def load_client_splits(split_dir: str, n_clients: int) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Load previously saved client splits from CSV files."""
    splits = {}
    split_path = Path(split_dir)
    for cid in range(n_clients):
        train_path = split_path / f"client_{cid:02d}_train.csv"
        test_path = split_path / f"client_{cid:02d}_test.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Split not found: {train_path}. Run data_prep first.")
        splits[cid] = {
            "train": pd.read_csv(train_path),
            "test": pd.read_csv(test_path),
        }
    return splits


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedRankX Data Preparation")
    parser.add_argument("--output_dir", default="data/processed/client_splits")
    parser.add_argument("--n_clients", type=int, default=N_CLIENTS)
    parser.add_argument("--samples_per_client", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--iscx_path", default="data/raw/ISCX-URL2016/All.csv")
    parser.add_argument("--email_path", default="data/raw/phishing_email/Phishing_Email.csv")
    parser.add_argument("--use_real_data", action="store_true")
    args = parser.parse_args()

    create_client_splits(
        output_dir=args.output_dir,
        n_clients=args.n_clients,
        samples_per_client=args.samples_per_client,
        use_real_data=args.use_real_data,
        iscx_path=args.iscx_path,
        email_path=args.email_path,
        seed=args.seed,
    )
