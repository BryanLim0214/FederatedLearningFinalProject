"""
fedrank_strategy.py
===================
FedRankX server strategy — uses fl_engine.FLStrategy (no Flower/Ray).

Per round:
  1. Send each client its OWN previous model for warm-starting.
  2. Receive rank lists + model weights from all clients.
  3. Cluster clients by rank distance (Spearman/Kendall).
  4. Store per-client models; build cluster ensembles.
  5. Evaluate each client using cluster ensemble (averaged predictions).
"""

import json
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.clustering import cluster_quality, hierarchical_cluster
from src.fl_engine import FLClient, FLStrategy, Parameters, params_to_arrays

warnings.filterwarnings("ignore")


class FedRankXStrategy(FLStrategy):
    """
    FedRankX: cluster clients by SHAP rank-list distance, then
    use cluster-aware ensemble evaluation.

    Key design: each client trains on their OWN model (no destructive
    warm-start from cluster representative). The cluster information
    is used for ensemble evaluation and NMI tracking.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5,
        linkage_method: str = "ward",
        rank_metric: str = "spearman",
        true_client_labels: Optional[np.ndarray] = None,
        log_clusters: bool = True,
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method
        self.rank_metric = rank_metric
        self.true_client_labels = true_client_labels
        self.log_clusters = log_clusters

        # Persistent state
        self.current_cluster_labels: Optional[np.ndarray] = None
        self.cluster_models: Dict[int, Parameters] = {}  # cluster_id -> rep model
        self.client_models: Dict[int, Parameters] = {}   # client_id -> own model
        self.round_history: List[dict] = []
        self._global_params: Optional[Parameters] = None

    # ------------------------------------------------------------------
    # configure_fit: send each client ITS OWN previous model
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        n_clients: int,
    ) -> List[Tuple[int, Parameters, dict]]:
        self._global_params = parameters
        instructions = []
        for cid in range(n_clients):
            config = {"current_round": server_round}
            # Send client its OWN previous model for warm-starting
            # Falls back to global params on round 1 (no previous model)
            params_to_send = self.client_models.get(cid, parameters)
            instructions.append((cid, params_to_send, config))
        return instructions

    # ------------------------------------------------------------------
    # aggregate_fit: cluster by ranks, store per-client models
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, Parameters, int, dict]],
        failures,
    ) -> Tuple[Parameters, dict]:

        if not results:
            return self._global_params, {}

        # ---- Step 1: decode rank lists, store per-client models ----
        client_ids, rank_lists, client_data = [], [], []
        n_features = None

        for client_id, params, n_examples, metrics in results:
            # Store THIS client's model
            self.client_models[client_id] = params

            rank_json = metrics.get("rank_list_json", "[]")
            rank_list = np.array(json.loads(rank_json), dtype=np.int32)
            if len(rank_list) == 0:
                n_f = int(metrics.get("n_features", 79))
                rank_list = np.arange(min(15, n_f), dtype=np.int32)

            rank_lists.append(rank_list)
            client_ids.append(client_id)
            client_data.append((params, n_examples))
            n_features = int(metrics.get("n_features", n_features or 79))

        # ---- Step 2: cluster by rank distance ----
        cluster_labels, D, Z = hierarchical_cluster(
            rank_lists,
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage_method=self.linkage_method,
            metric=self.rank_metric,
            n_features=n_features,
        )

        # Map to full-length label array
        n_total = max(client_ids) + 1
        full_labels = np.full(n_total, -1, dtype=int)
        for local_idx, cid in enumerate(client_ids):
            full_labels[cid] = cluster_labels[local_idx]
        self.current_cluster_labels = full_labels

        if self.log_clusters:
            n_disc = len(set(cluster_labels.tolist()))
            assignment = {cid: int(cluster_labels[i]) for i, cid in enumerate(client_ids)}
            print(f"  [Round {server_round}] Clusters: {assignment} -> {n_disc} group(s)")

        # ---- Step 3: build cluster groups (for ensemble eval + fallback) ----
        cluster_groups: Dict[int, List[Tuple[int, Parameters, int]]] = defaultdict(list)
        for local_idx, cid in enumerate(client_ids):
            cluster_id = int(cluster_labels[local_idx])
            params, n_examples = client_data[local_idx]
            cluster_groups[cluster_id].append((cid, params, n_examples))

        # Store cluster representative (largest client) for fallback
        self.cluster_models = {}
        for cluster_id, group in cluster_groups.items():
            best_cid, best_params, _ = max(group, key=lambda x: x[2])
            self.cluster_models[cluster_id] = best_params

        # Store cluster member IDs for ensemble evaluation
        self._cluster_members = {}
        for cluster_id, group in cluster_groups.items():
            self._cluster_members[cluster_id] = [cid for cid, _, _ in group]

        # ---- Step 4: cluster quality ----
        metrics_agg: dict = {"n_clusters": len(self.cluster_models)}
        if self.true_client_labels is not None:
            valid = [cid for cid in client_ids if cid < len(self.true_client_labels)]
            if valid:
                pred = cluster_labels[[client_ids.index(c) for c in valid]]
                true = self.true_client_labels[valid]
                q = cluster_quality(pred, true)
                metrics_agg.update(q)
                if self.log_clusters:
                    print(f"  [Round {server_round}] NMI={q['nmi']:.3f}  ARI={q['ari']:.3f}")

        self.round_history.append({
            "round": server_round,
            "cluster_labels": cluster_labels.tolist(),
            "client_ids": client_ids,
            **metrics_agg,
        })

        # Return largest cluster model as global fallback
        largest = max(cluster_groups, key=lambda k: sum(n for _, _, n in cluster_groups[k]))
        global_params = self.cluster_models[largest]
        self._global_params = global_params
        return global_params, metrics_agg

    # ------------------------------------------------------------------
    # configure_evaluate: tell each client to use ensemble eval
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        n_clients: int,
    ) -> List[Tuple[int, Parameters, dict]]:
        """
        For evaluation, send each client a config flag to use local model.
        The cluster ensemble members are passed as config metadata.
        """
        instructions = []
        for cid in range(n_clients):
            # Tell the client to evaluate with its OWN local model
            # AND to also include cluster member models for ensemble
            config = {"use_local_model": True}

            # Build ensemble: serialize cluster member model params
            if (self.current_cluster_labels is not None
                    and cid < len(self.current_cluster_labels)):
                cluster_id = int(self.current_cluster_labels[cid])
                members = self._cluster_members.get(cluster_id, [cid])
                # Pass cluster member model params as list of serialized models
                ensemble_models = []
                for member_id in members:
                    if member_id != cid and member_id in self.client_models:
                        ensemble_models.append(member_id)
                config["ensemble_member_ids"] = ensemble_models

            # Send the client's own model (fallback if self.booster is None)
            params_to_eval = self.client_models.get(cid, parameters)
            instructions.append((cid, params_to_eval, config))
        return instructions

    # ------------------------------------------------------------------
    # aggregate_evaluate
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, float, int, dict]],
        failures,
    ) -> Tuple[float, dict]:
        if not results:
            return 1.0, {}
        total = sum(n for _, _, n, _ in results)
        wloss = sum(loss * n for _, loss, n, _ in results) / total
        wf1   = sum(m.get("f1", 0) * n for _, _, n, m in results) / total
        wauc  = sum(m.get("auc", 0.5) * n for _, _, n, m in results) / total
        return wloss, {"f1": wf1, "auc": wauc}

    def get_round_history(self) -> List[dict]:
        return self.round_history

    def get_client_models(self) -> Dict[int, Parameters]:
        """Return all stored per-client models (for ensemble eval in engine)."""
        return self.client_models
