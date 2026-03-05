"""
fl_engine.py
============
Lightweight pure-Python synchronous federated learning simulation engine.

Replaces Flower's Ray-backed Virtual Client Engine for portability on
Windows / Python 3.13 where Ray is unavailable.

Design
------
- Runs N client fit() calls sequentially in a for loop per round
- Zero external dependencies beyond the project itself
- Fully reproducible (no distributed scheduling randomness)
- Collects History identical in structure to Flower's History object
- All FedRankX logic is preserved exactly — only the scheduler changes

This is scientifically sound: sequential simulation is standard in FL
research papers (e.g., Li et al. 2020 FedProx, Ghosh et al. 2020 IFCA).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Lightweight parameter container (mirrors flwr.common.Parameters)
# ---------------------------------------------------------------------------

@dataclass
class Parameters:
    """Wraps a list of numpy arrays as model parameters."""
    tensors: List[np.ndarray]

    def copy(self) -> "Parameters":
        return Parameters([a.copy() for a in self.tensors])


def params_to_arrays(p: Parameters) -> List[np.ndarray]:
    return p.tensors


def arrays_to_params(arrays: List[np.ndarray]) -> Parameters:
    return Parameters(arrays)


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------

class FLClient:
    """
    Abstract base for a federated client.
    Subclasses implement fit() and evaluate().
    """
    def fit(
        self,
        parameters: Parameters,
        config: dict,
    ) -> Tuple[Parameters, int, dict]:
        """Train and return (updated_params, n_examples, metrics)."""
        raise NotImplementedError

    def evaluate(
        self,
        parameters: Parameters,
        config: dict,
    ) -> Tuple[float, int, dict]:
        """Evaluate and return (loss, n_examples, metrics)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Server strategy protocol
# ---------------------------------------------------------------------------

class FLStrategy:
    """
    Abstract server strategy.
    Subclasses implement aggregate_fit() and aggregate_evaluate().
    """
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        n_clients: int,
    ) -> List[Tuple[int, Parameters, dict]]:
        """
        Return list of (client_id, parameters_to_send, config) for this round.
        Default: broadcast same parameters to all clients.
        """
        return [(cid, parameters, {"current_round": server_round})
                for cid in range(n_clients)]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, Parameters, int, dict]],
        # (client_id, parameters, n_examples, metrics)
    ) -> Tuple[Parameters, dict]:
        raise NotImplementedError

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        n_clients: int,
    ) -> List[Tuple[int, Parameters, dict]]:
        return [(cid, parameters, {}) for cid in range(n_clients)]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, float, int, dict]],
        # (client_id, loss, n_examples, metrics)
    ) -> Tuple[float, dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------

@dataclass
class History:
    losses_distributed: List[Tuple[int, float]] = field(default_factory=list)
    losses_centralized: List[Tuple[int, float]] = field(default_factory=list)
    metrics_distributed: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    metrics_centralized: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    client_fn: Callable[[int], FLClient],
    strategy: FLStrategy,
    n_clients: int,
    n_rounds: int,
    initial_parameters: Optional[Parameters] = None,
    verbose: bool = True,
) -> History:
    """
    Run a synchronous federated learning simulation.

    Parameters
    ----------
    client_fn : callable(cid: int) -> FLClient
        Factory that returns a fresh client for the given ID.
    strategy : FLStrategy
        Server-side aggregation strategy.
    n_clients : int
        Total number of clients to simulate.
    n_rounds : int
        Number of FL rounds.
    initial_parameters : Parameters, optional
        Global starting parameters. If None, obtained from client 0.
    verbose : bool
        Print round-level progress.

    Returns
    -------
    History
    """
    history = History()

    # Instantiate all clients once (they persist across rounds, holding local data)
    clients: Dict[int, FLClient] = {cid: client_fn(cid) for cid in range(n_clients)}

    # Initialise global parameters from client 0 if not provided
    if initial_parameters is None:
        init_p, _, _ = clients[0].fit(Parameters([np.array([], dtype=np.uint8)]), {"current_round": 0})
        global_params = init_p
    else:
        global_params = initial_parameters

    t_start_total = time.time()

    for round_num in range(1, n_rounds + 1):
        t_start = time.time()

        # ---- FIT PHASE ----
        fit_instructions = strategy.configure_fit(round_num, global_params, n_clients)
        fit_results: List[Tuple[int, Parameters, int, dict]] = []

        for client_id, params_to_send, config in fit_instructions:
            params_out, n_examples, metrics = clients[client_id].fit(params_to_send, config)
            fit_results.append((client_id, params_out, n_examples, metrics))

        # Aggregate
        global_params, agg_fit_metrics = strategy.aggregate_fit(round_num, fit_results, [])

        # ---- EVALUATE PHASE ----
        # Support cluster ensemble: if strategy stores per-client models and
        # cluster labels, evaluate each client using ensemble of cluster peers.
        eval_instructions = strategy.configure_evaluate(round_num, global_params, n_clients)
        eval_results: List[Tuple[int, float, int, dict]] = []

        # Get cluster member models for ensemble evaluation
        client_models_map = {}
        cluster_members_map = {}
        if hasattr(strategy, 'get_client_models') and hasattr(strategy, '_cluster_members'):
            client_models_map = strategy.get_client_models()
            cluster_members_map = strategy._cluster_members
            cluster_labels = strategy.current_cluster_labels

        for client_id, params_to_eval, config in eval_instructions:
            # Build ensemble for this client's cluster
            if (cluster_members_map and
                    hasattr(strategy, 'current_cluster_labels') and
                    strategy.current_cluster_labels is not None and
                    client_id < len(strategy.current_cluster_labels)):
                cid_cluster = int(strategy.current_cluster_labels[client_id])
                member_ids = cluster_members_map.get(cid_cluster, [client_id])
                ensemble_params = [client_models_map[mid] for mid in member_ids
                                   if mid in client_models_map]
                if ensemble_params:
                    config = dict(config)  # copy
                    config['ensemble_params'] = ensemble_params

            loss, n_examples, metrics = clients[client_id].evaluate(params_to_eval, config)
            eval_results.append((client_id, loss, n_examples, metrics))

        agg_loss, agg_eval_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])

        # ---- Record history ----
        if agg_loss is not None:
            history.losses_distributed.append((round_num, agg_loss))
        for k, v in agg_eval_metrics.items():
            history.metrics_distributed.setdefault(k, []).append((round_num, v))

        elapsed = time.time() - t_start
        if verbose:
            print(f"  Round {round_num:3d}/{n_rounds} | "
                  f"Loss={agg_loss:.4f} | "
                  f"F1={agg_eval_metrics.get('f1', 0.0):.4f} | "
                  f"AUC={agg_eval_metrics.get('auc', 0.0):.4f} | "
                  f"{elapsed:.1f}s")

    total_time = time.time() - t_start_total
    if verbose:
        print(f"\n  Simulation complete in {total_time:.1f}s")

    return history
