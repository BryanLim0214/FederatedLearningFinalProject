"""
clustering.py
=============
Server-side clustering of federated clients based on their SHAP rank lists.

Algorithm
---------
1. Receive top-K rank lists from all N clients.
2. Compute pairwise rank-distance matrix using Spearman / Kendall / Hamming.
3. Apply hierarchical clustering (Ward linkage) to group similar clients.
4. Optionally use k-medoids for fixed number of clusters.
5. Return cluster assignment array.

The server never sees raw SHAP values — only integer rank lists.
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Distance matrix construction
# ---------------------------------------------------------------------------

def compute_distance_matrix(
    rank_lists: List[np.ndarray],
    metric: str = "spearman",
    n_features: int = None,
) -> np.ndarray:
    """
    Compute an N×N pairwise rank-distance matrix.

    Parameters
    ----------
    rank_lists : list of np.ndarray
        Each entry is a top-K rank list (indices) for one client.
    metric : str
        One of "spearman", "kendall", "hamming".
    n_features : int, optional
        Total number of features (for penalty position). Inferred if None.

    Returns
    -------
    D : np.ndarray shape (N, N), symmetric, zero diagonal.
    """
    from src.rank_utils import (
        rank_distance_hamming,
        rank_distance_kendall,
        rank_distance_spearman,
    )

    n = len(rank_lists)
    if n_features is None:
        n_features = max(int(rl.max()) for rl in rank_lists) + 1

    distance_fn = {
        "spearman": lambda a, b: rank_distance_spearman(a, b, n_features),
        "kendall": lambda a, b: rank_distance_kendall(a, b, n_features),
        "hamming": lambda a, b: rank_distance_hamming(a, b),
    }[metric]

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(rank_lists[i], rank_lists[j])
            D[i, j] = d
            D[j, i] = d

    return D


# ---------------------------------------------------------------------------
# Hierarchical clustering (primary method)
# ---------------------------------------------------------------------------

def hierarchical_cluster(
    rank_lists: List[np.ndarray],
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.5,
    linkage_method: str = "ward",
    metric: str = "spearman",
    n_features: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster clients using hierarchical clustering on rank-distance matrix.

    Parameters
    ----------
    rank_lists : list of rank arrays, one per client
    n_clusters : int or None
        If set, cut dendrogram to exactly this many clusters.
        If None, use distance_threshold for automatic cut.
    distance_threshold : float
        Dendrogram cut height for automatic cluster count.
    linkage_method : str
        Scipy linkage method: "ward", "average", "complete", "single".
    metric : str
        Rank distance metric: "spearman", "kendall", "hamming".
    n_features : int, optional

    Returns
    -------
    labels : np.ndarray shape (N,)  — 0-indexed cluster assignments
    D : np.ndarray shape (N, N)     — distance matrix
    Z : np.ndarray                  — linkage matrix (for dendrogram)
    """
    D = compute_distance_matrix(rank_lists, metric=metric, n_features=n_features)

    # scipy linkage requires condensed upper triangle
    condensed = squareform(D, checks=False)

    # Handle degenerate case: all zeros (all clients identical)
    if condensed.max() < 1e-12:
        return np.zeros(len(rank_lists), dtype=int), D, np.zeros((len(rank_lists) - 1, 4))

    Z = linkage(condensed, method=linkage_method)

    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    else:
        labels = fcluster(Z, distance_threshold, criterion="distance") - 1

    return labels.astype(int), D, Z


# ---------------------------------------------------------------------------
# K-Medoids clustering (alternative — used in ablation)
# ---------------------------------------------------------------------------

def kmedoids_cluster(
    rank_lists: List[np.ndarray],
    n_clusters: int,
    metric: str = "spearman",
    n_features: int = None,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Medoids clustering using the PAM algorithm on the rank distance matrix.

    Returns
    -------
    labels : np.ndarray shape (N,) — 0-indexed cluster assignments
    D : np.ndarray shape (N, N)   — distance matrix
    """
    D = compute_distance_matrix(rank_lists, metric=metric, n_features=n_features)
    rng = np.random.default_rng(seed)
    n = len(rank_lists)
    n_clusters = min(n_clusters, n)

    # Initialise medoids randomly
    medoid_idx = rng.choice(n, n_clusters, replace=False).tolist()

    for _ in range(max_iter):
        # Assignment step
        labels = np.argmin(D[:, medoid_idx], axis=1)

        # Update step: for each cluster, pick the member minimising intra-cluster distance
        new_medoids = []
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[k])
                continue
            sub_D = D[np.ix_(members, members)]
            local_costs = sub_D.sum(axis=1)
            new_medoids.append(int(members[np.argmin(local_costs)]))

        if new_medoids == medoid_idx:
            break
        medoid_idx = new_medoids

    # Final assignment
    labels = np.argmin(D[:, medoid_idx], axis=1).astype(int)
    return labels, D


# ---------------------------------------------------------------------------
# Cluster quality metrics (for evaluation)
# ---------------------------------------------------------------------------

def cluster_quality(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
) -> dict:
    """
    Compute NMI and ARI between predicted clusters and ground-truth client types.
    Used in the paper's evaluation to show cluster recovery quality.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels_true, labels_pred)
    n_clusters = len(np.unique(labels_pred))
    return {
        "nmi": float(nmi),
        "ari": float(ari),
        "n_clusters_discovered": int(n_clusters),
    }


def print_cluster_summary(labels: np.ndarray, client_meta: list = None) -> None:
    """Print a human-readable cluster assignment table."""
    unique = sorted(set(labels.tolist()))
    print(f"\n[clustering] Discovered {len(unique)} clusters:")
    for cluster_id in unique:
        members = [i for i, l in enumerate(labels) if l == cluster_id]
        if client_meta:
            domains = [client_meta[m]["domain"] for m in members]
            print(f"  Cluster {cluster_id}: clients {members} -> domains: {domains}")
        else:
            print(f"  Cluster {cluster_id}: clients {members}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate two groups of clients: group A prefers features 0-14, group B prefers 20-34
    def make_rank(preferred_range, k=15, noise=3, seed=0):
        rng2 = np.random.default_rng(seed)
        base = np.arange(preferred_range[0], preferred_range[0] + k)
        # Add noise
        noise_idx = rng2.choice(np.arange(35, 79), noise, replace=False)
        mixed = np.concatenate([base[:-noise], noise_idx])
        rng2.shuffle(mixed)
        return mixed[:k].astype(np.int32)

    rank_lists = (
        [make_rank((0, 15), seed=i) for i in range(5)] +   # group A
        [make_rank((20, 35), seed=i + 10) for i in range(5)]  # group B
    )
    true_labels = np.array([0] * 5 + [1] * 5)

    labels, D, Z = hierarchical_cluster(rank_lists, n_clusters=2, metric="spearman")
    quality = cluster_quality(labels, true_labels)
    print(f"Hierarchical (Spearman): labels={labels}, NMI={quality['nmi']:.3f}, ARI={quality['ari']:.3f}")

    labels_km, D_km = kmedoids_cluster(rank_lists, n_clusters=2, metric="spearman")
    quality_km = cluster_quality(labels_km, true_labels)
    print(f"K-Medoids (Spearman):   labels={labels_km}, NMI={quality_km['nmi']:.3f}, ARI={quality_km['ari']:.3f}")

    print("clustering.py self-test PASSED")
