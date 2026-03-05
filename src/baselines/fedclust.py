"""
baselines/fedclust.py  — pure fl_engine version (no Flower/Ray)
FedClust: clusters clients by cosine distance of LightGBM feature-importance
weight vectors, then FedAvg within each cluster.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import f1_score, roc_auc_score

from src.fl_engine import FLClient, FLStrategy, Parameters, run_simulation
from src.client import lgbm_to_params, params_to_lgbm
from src.clustering import cluster_quality

warnings.filterwarnings("ignore")


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    # Clamp to [0, 2] to prevent floating-point negative distances
    return float(max(0.0, 1.0 - np.dot(a, b) / (na * nb)))


def _booster_weight_vec(booster: lgb.Booster) -> np.ndarray:
    imp = booster.feature_importance(importance_type="gain").astype(float)
    nrm = np.linalg.norm(imp)
    return imp / nrm if nrm > 0 else imp


class FedClustStrategy(FLStrategy):
    def __init__(self, n_clients, n_clusters=None, distance_threshold=0.5,
                 true_client_labels=None):
        self.n_clients = n_clients
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.true_client_labels = true_client_labels
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_models: Dict[int, Parameters] = {}
        self.round_history = []
        self._global: Optional[Parameters] = None

    def configure_fit(self, server_round, parameters, n_clients):
        self._global = parameters
        inst = []
        for cid in range(n_clients):
            if self.cluster_models and self.cluster_labels is not None and cid < len(self.cluster_labels):
                p = self.cluster_models.get(int(self.cluster_labels[cid]), parameters)
            else:
                p = parameters
            inst.append((cid, p, {"current_round": server_round}))
        return inst

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return self._global, {}

        client_ids, wvecs, cdata = [], [], []
        for cid, params, n_ex, _ in results:
            b = params_to_lgbm(params)
            wvecs.append(_booster_weight_vec(b) if b else np.zeros(79))
            client_ids.append(cid)
            cdata.append((params, n_ex))

        n = len(wvecs)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = D[j,i] = _cosine_dist(wvecs[i], wvecs[j])

        condensed = np.maximum(squareform(D, checks=False), 0.0)  # clamp negatives
        if condensed.max() < 1e-12:
            labels = np.zeros(n, dtype=int)
        else:
            Z = linkage(condensed, method="ward")
            labels = (fcluster(Z, self.n_clusters, criterion="maxclust") - 1
                      if self.n_clusters else
                      fcluster(Z, self.distance_threshold, criterion="distance") - 1)

        full = np.full(max(client_ids)+1, -1, dtype=int)
        for li, cid in enumerate(client_ids):
            full[cid] = labels[li]
        self.cluster_labels = full

        grps: Dict[int, list] = defaultdict(list)
        for li, cid in enumerate(client_ids):
            grps[int(labels[li])].append(cdata[li])
        self.cluster_models = {k: max(g, key=lambda x: x[1])[0] for k, g in grps.items()}

        m: dict = {}
        if self.true_client_labels is not None:
            valid = [c for c in client_ids if c < len(self.true_client_labels)]
            if valid:
                pred = labels[[client_ids.index(c) for c in valid]]
                q = cluster_quality(pred, self.true_client_labels[valid])
                m = {"nmi": q["nmi"], "ari": q["ari"]}

        self.round_history.append({"round": server_round,
                                   "cluster_labels": labels.tolist(), **m})
        largest = max(grps, key=lambda k: sum(n for _, n in grps[k]))
        self._global = self.cluster_models[largest]
        return self._global, m

    def configure_evaluate(self, server_round, parameters, n_clients):
        inst = []
        for cid in range(n_clients):
            if self.cluster_models and self.cluster_labels is not None and cid < len(self.cluster_labels):
                p = self.cluster_models.get(int(self.cluster_labels[cid]), parameters)
            else:
                p = parameters
            inst.append((cid, p, {}))
        return inst

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 1.0, {}
        total = sum(n for _, _, n, _ in results)
        wf1   = sum(m.get("f1",0)*n for _,_,n,m in results)/total
        wauc  = sum(m.get("auc",0.5)*n for _,_,n,m in results)/total
        wloss = sum(loss*n for _,loss,n,_ in results)/total
        print(f"  [FedClust Round {server_round}] F1={wf1:.4f}  AUC={wauc:.4f}")
        return wloss, {"f1": wf1, "auc": wauc}


class FedClustClient(FLClient):
    def __init__(self, cid, train_df, test_df, n_estimators=50):
        self.cid = cid
        self.fc = [c for c in train_df.columns if c != "label"]
        self.X_train = train_df[self.fc].values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(int)
        self.X_test  = test_df[self.fc].values.astype(np.float32)
        self.y_test  = test_df["label"].values.astype(int)
        self.n_estimators = n_estimators
        self.booster = None

    def fit(self, parameters, config):
        init = params_to_lgbm(parameters)
        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.booster = lgb.train(
            {"objective":"binary","verbose":-1,"n_jobs":1},
            dtrain, num_boost_round=self.n_estimators, init_model=init,
            callbacks=[lgb.log_evaluation(period=-1)], keep_training_booster=True,
        )
        return lgbm_to_params(self.booster), len(self.X_train), {}

    def evaluate(self, parameters, config):
        b = params_to_lgbm(parameters) or self.booster
        if b is None:
            return 1.0, len(self.X_test), {"f1":0.0,"auc":0.5}
        yp = b.predict(self.X_test)
        f1 = float(f1_score(self.y_test,(yp>=0.5).astype(int),zero_division=0))
        try: auc = float(roc_auc_score(self.y_test, yp))
        except: auc = 0.5
        loss = float(-np.mean(self.y_test*np.log(np.clip(yp,1e-9,1))+
                               (1-self.y_test)*np.log(np.clip(1-yp,1e-9,1))))
        return loss, len(self.X_test), {"f1":f1,"auc":auc}


def run_fedclust(client_splits, n_clients, n_rounds=30, n_clusters=None,
                 distance_threshold=0.5, true_client_labels=None,
                 n_estimators=50, results_dir="results", **kwargs) -> dict:
    def client_fn(cid):
        return FedClustClient(cid, client_splits[cid]["train"],
                              client_splits[cid]["test"], n_estimators)
    strategy = FedClustStrategy(n_clients, n_clusters, distance_threshold, true_client_labels)
    history = run_simulation(client_fn, strategy, n_clients, n_rounds,
                             initial_parameters=Parameters([np.array([], dtype=np.uint8)]),
                             verbose=False)
    f1h  = [(r,float(v)) for r,v in history.metrics_distributed.get("f1",[])]
    auch = [(r,float(v)) for r,v in history.metrics_distributed.get("auc",[])]
    final_nmi = strategy.round_history[-1].get("nmi",0.0) if strategy.round_history else 0.0
    final_ari = strategy.round_history[-1].get("ari",0.0) if strategy.round_history else 0.0
    result = {
        "method": "FedClust", "n_clients": n_clients, "n_rounds": n_rounds,
        "losses": [(r,float(l)) for r,l in history.losses_distributed],
        "f1_history": f1h, "auc_history": auch,
        "final_f1":  f1h[-1][1]  if f1h  else 0.0,
        "final_auc": auch[-1][1] if auch else 0.5,
        "final_nmi": float(final_nmi), "final_ari": float(final_ari),
        "round_cluster_history": strategy.round_history,
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/fedclust.json","w") as f:
        json.dump(result, f, indent=2)
    print(f"[FedClust] F1={result['final_f1']:.4f}  AUC={result['final_auc']:.4f}")
    return result
