"""
rank_utils.py
=============
Converts local SHAP importance vectors into privacy-preserving rank lists.

Core idea
---------
  Raw SHAP values carry cardinal magnitude information that could leak
  internal model behaviour. By converting to an *ordinal* rank list
  (just the sorted indices of the top-K features), we discard magnitude
  information while retaining the structural signal needed for clustering.

Key functions
-------------
  shap_to_rank(shap_values, top_k) -> np.ndarray
  rank_distance_spearman(rank_a, rank_b) -> float
  rank_distance_kendall(rank_a, rank_b) -> float
  rank_distance_hamming(rank_a, rank_b) -> float
  compute_shap_for_lgbm(model, X) -> np.ndarray
  privacy_leakage_mutual_information(shap_values, rank_list) -> float
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_for_lgbm(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    subsample: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute per-feature SHAP importances using LightGBM's TreeExplainer.

    Returns the *mean absolute* SHAP value per feature, aggregated over a
    random subsample of training rows (fast on CPU).

    Parameters
    ----------
    model : lightgbm.Booster or sklearn-compatible LightGBM estimator
    X : feature matrix (numpy or pandas)
    subsample : number of rows to use (200 is usually enough for stable ranks)
    seed : random seed for subsampling

    Returns
    -------
    shap_importance : np.ndarray of shape (n_features,)
        Mean absolute SHAP value per feature.
    """
    import shap

    rng = np.random.default_rng(seed)
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    n = X_arr.shape[0]
    if n > subsample:
        idx = rng.choice(n, subsample, replace=False)
        X_sub = X_arr[idx]
    else:
        X_sub = X_arr

    # TreeExplainer is optimised for gradient boosting — very fast on CPU
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # For binary classification LightGBM returns a list [neg_class, pos_class]
    # or a single 2D array depending on version. Handle both.
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive (phishing) class
    else:
        sv = shap_values

    # Mean absolute SHAP across rows -> shape (n_features,)
    importance = np.mean(np.abs(sv), axis=0)
    return importance


# ---------------------------------------------------------------------------
# Rank conversion
# ---------------------------------------------------------------------------

def shap_to_rank(shap_importance: np.ndarray, top_k: int = 15) -> np.ndarray:
    """
    Convert a SHAP importance vector to an ordinal top-K rank list.

    Privacy guarantee: Only feature *indices* (integers) are returned,
    not their magnitudes. The rank order encodes which features matter
    most but not by how much.

    Parameters
    ----------
    shap_importance : np.ndarray shape (n_features,)
        Mean absolute SHAP values.
    top_k : int
        Number of top features to include in the rank list.

    Returns
    -------
    rank_list : np.ndarray of ints, shape (top_k,)
        Feature indices sorted by descending SHAP importance.
        rank_list[0] = most important feature index.
    """
    top_k = min(top_k, len(shap_importance))
    sorted_idx = np.argsort(shap_importance)[::-1]
    return sorted_idx[:top_k].astype(np.int32)


# ---------------------------------------------------------------------------
# Rank-based distance metrics
# ---------------------------------------------------------------------------

def rank_distance_spearman(rank_a: np.ndarray, rank_b: np.ndarray,
                            n_features: int = None) -> float:
    """
    Spearman rank-correlation distance between two rank lists.

    Both lists are converted to position vectors over the union of features,
    with unranked features receiving a penalty position.

    Returns distance in [0, 2]: 0 = identical, 2 = perfectly inverse.
    """
    if n_features is None:
        n_features = max(int(rank_a.max()), int(rank_b.max())) + 1

    penalty = n_features  # position assigned to unranked features
    all_feats = list(set(rank_a.tolist()) | set(rank_b.tolist()))

    pos_a = {feat: i for i, feat in enumerate(rank_a.tolist())}
    pos_b = {feat: i for i, feat in enumerate(rank_b.tolist())}

    vec_a = [pos_a.get(f, penalty) for f in all_feats]
    vec_b = [pos_b.get(f, penalty) for f in all_feats]

    if len(set(vec_a)) == 1 or len(set(vec_b)) == 1:
        return 1.0  # degenerate case

    rho, _ = spearmanr(vec_a, vec_b)
    return float(1.0 - rho)


def rank_distance_kendall(rank_a: np.ndarray, rank_b: np.ndarray,
                           n_features: int = None) -> float:
    """
    Kendall τ distance between two rank lists.
    Returns distance in [0, 2].
    """
    if n_features is None:
        n_features = max(int(rank_a.max()), int(rank_b.max())) + 1

    penalty = n_features
    all_feats = list(set(rank_a.tolist()) | set(rank_b.tolist()))

    pos_a = {f: i for i, f in enumerate(rank_a.tolist())}
    pos_b = {f: i for i, f in enumerate(rank_b.tolist())}

    vec_a = [pos_a.get(f, penalty) for f in all_feats]
    vec_b = [pos_b.get(f, penalty) for f in all_feats]

    if len(set(vec_a)) == 1 or len(set(vec_b)) == 1:
        return 1.0

    tau, _ = kendalltau(vec_a, vec_b)
    return float(1.0 - tau)


def rank_distance_hamming(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """
    Jaccard-based set overlap distance (simplified Hamming on top-K sets).
    Measures how different the *sets* of top-K features are.
    Returns distance in [0, 1]: 0 = same set, 1 = completely disjoint.
    """
    set_a = set(rank_a.tolist())
    set_b = set(rank_b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


# ---------------------------------------------------------------------------
# Privacy leakage analysis
# ---------------------------------------------------------------------------

def estimate_reconstruction_error(
    shap_importance: np.ndarray,
    rank_list: np.ndarray,
) -> dict:
    """
    Estimate how well an adversary could reconstruct SHAP magnitudes
    from the rank list alone (empirical privacy analysis for paper).

    Strategy: The adversary sees only the rank list (indices + order).
    They guess SHAP values as 1/rank (harmonic weights).
    We measure the RMSE of this guess vs. true SHAP values.

    Returns
    -------
    dict with:
      'rmse': float — reconstruction RMSE (higher = more private)
      'rank_correlation': float — ρ between true and reconstructed
      'information_ratio': float — fraction of original info retained (≤1)
    """
    n_feats = len(shap_importance)
    top_k = len(rank_list)

    # Adversary's best guess: harmonic weights for ranked features, 0 for rest
    reconstructed = np.zeros(n_feats)
    for rank_pos, feat_idx in enumerate(rank_list):
        reconstructed[feat_idx] = 1.0 / (rank_pos + 1)  # harmonic guess

    # Normalise to same scale as true SHAP
    if reconstructed.max() > 0:
        reconstructed = reconstructed * (shap_importance.max() / reconstructed.max())

    rmse = float(np.sqrt(np.mean((shap_importance - reconstructed) ** 2)))
    rho, _ = spearmanr(shap_importance, reconstructed)

    # Information ratio: how much variance in SHAP is explained by the rank guess
    ss_res = np.sum((shap_importance - reconstructed) ** 2)
    ss_tot = np.sum((shap_importance - shap_importance.mean()) ** 2)
    info_ratio = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": rmse,
        "rank_correlation_rho": float(rho),
        "information_ratio": max(0.0, float(info_ratio)),
    }


def encode_rank_for_transmission(rank_list: np.ndarray) -> bytes:
    """
    Serialise a rank list to bytes for communication overhead calculation.
    Each index is a uint16 (2 bytes) -> max 65535 features supported.
    """
    return rank_list.astype(np.uint16).tobytes()


def decode_rank_from_transmission(data: bytes) -> np.ndarray:
    """Deserialise bytes back to rank list."""
    return np.frombuffer(data, dtype=np.uint16)


def rank_payload_size_bytes(top_k: int) -> int:
    """Returns byte size of a rank list payload (uint16 per index)."""
    return top_k * 2  # 2 bytes per uint16


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Draw all values in [0, 0.7] so hand-set values are guaranteed top-3
    fake_shap = rng.uniform(0, 0.7, size=79)
    fake_shap[3] = 0.99   # rank 1
    fake_shap[12] = 0.88  # rank 2
    fake_shap[27] = 0.72  # rank 3 (above max uniform draw)

    rank = shap_to_rank(fake_shap, top_k=15)
    print(f"Top-15 rank list: {rank}")
    assert rank[0] == 3, "Highest SHAP should be ranked first"
    assert rank[1] == 12, "Second highest should be ranked second"

    # Distance self-consistency
    d_aa = rank_distance_spearman(rank, rank)
    assert abs(d_aa) < 1e-6, f"Distance to self should be 0, got {d_aa}"

    rng2 = np.random.default_rng(99)
    rank_b = shap_to_rank(rng2.uniform(0, 1, size=79), top_k=15)
    d_ab = rank_distance_spearman(rank, rank_b)
    print(f"Spearman distance A↔B: {d_ab:.4f}")
    print(f"Kendall distance A↔B:  {rank_distance_kendall(rank, rank_b):.4f}")
    print(f"Hamming distance A↔B:  {rank_distance_hamming(rank, rank_b):.4f}")

    leak = estimate_reconstruction_error(fake_shap, rank)
    print(f"Privacy leak analysis: {leak}")
    print(f"Rank payload size (K=15): {rank_payload_size_bytes(15)} bytes")
    print("rank_utils.py self-test PASSED")
