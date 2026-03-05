"""
baselines/ifca.py  — pure fl_engine version (no Flower/Ray)
IFCA: Iterative Federated Clustering Algorithm (Ghosh et al. 2020)
Each client picks the lowest-loss model from K cluster models per round.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.fl_engine import FLClient, FLStrategy, Parameters, run_simulation
from src.client import lgbm_to_params, params_to_lgbm
from src.clustering import cluster_quality

warnings.filterwarnings("ignore")

K_CLUSTERS = 5


def _loss(booster, X, y):
    yp = np.clip(booster.predict(X), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(yp) + (1 - y) * np.log(1 - yp)))


class IFCAStrategy(FLStrategy):
    def __init__(self, k, n_clients, true_client_labels=None):
        self.k = k
        self.n_clients = n_clients
        self.true_client_labels = true_client_labels
        self.cluster_models: List[Optional[Parameters]] = [None] * k
        self.assignments: Dict[int, int] = {}
        self.round_history = []

    def _packed(self) -> Parameters:
        """Pack all K models as separate arrays in one Parameters object."""
        arrays = []
        for km in self.cluster_models:
            arrays.append(km.tensors[0] if km else np.array([], dtype=np.uint8))
        return Parameters(arrays)

    def configure_fit(self, server_round, parameters, n_clients):
        packed = self._packed()
        return [(cid, packed, {"round": server_round, "k": self.k})
                for cid in range(n_clients)]

    def aggregate_fit(self, server_round, results, failures):
        grps: Dict[int, list] = defaultdict(list)
        new_assignments = {}
        for cid, params, n_ex, metrics in results:
            kid = int(metrics.get("cluster_id", 0))
            grps[kid].append((params, n_ex))
            new_assignments[cid] = kid
            self.assignments[cid] = kid

        for kid, group in grps.items():
            self.cluster_models[kid] = max(group, key=lambda x: x[1])[0]

        m: dict = {}
        if self.true_client_labels is not None and new_assignments:
            valid = [c for c in new_assignments if c < len(self.true_client_labels)]
            if valid:
                pred = np.array([new_assignments[c] for c in valid])
                q = cluster_quality(pred, self.true_client_labels[valid])
                m = {"nmi": q["nmi"], "ari": q["ari"]}
        self.round_history.append({"round": server_round, "assignments": new_assignments, **m})
        print(f"  [IFCA Round {server_round}] Assignments: {new_assignments}")
        return self._packed(), m

    def configure_evaluate(self, server_round, parameters, n_clients):
        packed = self._packed()
        return [(cid, packed, {}) for cid in range(n_clients)]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 1.0, {}
        total = sum(n for _,_,n,_ in results)
        wf1   = sum(m.get("f1",0)*n for _,_,n,m in results)/total
        wauc  = sum(m.get("auc",0.5)*n for _,_,n,m in results)/total
        wloss = sum(loss*n for _,loss,n,_ in results)/total
        print(f"  [IFCA Round {server_round}] F1={wf1:.4f}  AUC={wauc:.4f}")
        return wloss, {"f1": wf1, "auc": wauc}


class IFCAClient(FLClient):
    def __init__(self, cid, train_df, test_df, k=K_CLUSTERS, n_estimators=50):
        self.cid = cid
        self.k = k
        self.n_estimators = n_estimators
        fc = [c for c in train_df.columns if c != "label"]
        self.X_train = train_df[fc].values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(int)
        self.X_test  = test_df[fc].values.astype(np.float32)
        self.y_test  = test_df["label"].values.astype(int)
        self.current_k = 0
        self.booster = None

    def fit(self, parameters, config):
        k = config.get("k", self.k)
        # Each tensor in parameters is one cluster model
        best_k, best_booster, best_loss = 0, None, float("inf")
        for kid in range(k):
            if kid < len(parameters.tensors) and len(parameters.tensors[kid]) > 0:
                try:
                    b = lgb.Booster(model_str=parameters.tensors[kid].tobytes().decode("utf-8"))
                    l = _loss(b, self.X_train, self.y_train)
                    if l < best_loss:
                        best_loss, best_k, best_booster = l, kid, b
                except Exception:
                    pass

        self.current_k = best_k
        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.booster = lgb.train(
            {"objective":"binary","verbose":-1,"n_jobs":1},
            dtrain, num_boost_round=self.n_estimators, init_model=best_booster,
            callbacks=[lgb.log_evaluation(period=-1)], keep_training_booster=True,
        )
        return lgbm_to_params(self.booster), len(self.X_train), {"cluster_id": self.current_k}

    def evaluate(self, parameters, config):
        b = None
        if (self.current_k < len(parameters.tensors) and
                len(parameters.tensors[self.current_k]) > 0):
            try:
                b = lgb.Booster(model_str=parameters.tensors[self.current_k].tobytes().decode("utf-8"))
            except Exception:
                b = self.booster
        b = b or self.booster
        if b is None:
            return 1.0, len(self.X_test), {"f1":0.0,"auc":0.5}
        yp = b.predict(self.X_test)
        f1 = float(f1_score(self.y_test,(yp>=0.5).astype(int),zero_division=0))
        try: auc = float(roc_auc_score(self.y_test, yp))
        except: auc = 0.5
        loss = _loss(b, self.X_test, self.y_test)
        return loss, len(self.X_test), {"f1":f1,"auc":auc}


def run_ifca(client_splits, n_clients, k_clusters=K_CLUSTERS, n_rounds=30,
             true_client_labels=None, n_estimators=50, results_dir="results", **kwargs) -> dict:
    def client_fn(cid):
        return IFCAClient(cid, client_splits[cid]["train"], client_splits[cid]["test"],
                          k=k_clusters, n_estimators=n_estimators)
    strategy = IFCAStrategy(k_clusters, n_clients, true_client_labels)
    history = run_simulation(client_fn, strategy, n_clients, n_rounds,
                             initial_parameters=Parameters([np.array([], dtype=np.uint8)] * k_clusters),
                             verbose=False)
    f1h  = [(r,float(v)) for r,v in history.metrics_distributed.get("f1",[])]
    auch = [(r,float(v)) for r,v in history.metrics_distributed.get("auc",[])]
    final_nmi = strategy.round_history[-1].get("nmi",0.0) if strategy.round_history else 0.0
    final_ari = strategy.round_history[-1].get("ari",0.0) if strategy.round_history else 0.0
    result = {
        "method": "IFCA", "n_clients": n_clients, "n_rounds": n_rounds,
        "losses": [(r,float(l)) for r,l in history.losses_distributed],
        "f1_history": f1h, "auc_history": auch,
        "final_f1":  f1h[-1][1]  if f1h  else 0.0,
        "final_auc": auch[-1][1] if auch else 0.5,
        "final_nmi": float(final_nmi), "final_ari": float(final_ari),
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/ifca.json","w") as f:
        json.dump(result, f, indent=2)
    print(f"[IFCA] F1={result['final_f1']:.4f}  AUC={result['final_auc']:.4f}")
    return result
