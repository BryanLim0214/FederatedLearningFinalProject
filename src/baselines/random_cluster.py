"""
baselines/random_cluster.py
============================
Random Clustering baseline: randomly assign clients to clusters,
then aggregate within each cluster. This proves that SHAP rank-based
clustering carries real structural information vs random assignment.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score

from src.fl_engine import FLClient, FLStrategy, Parameters, run_simulation
from src.client import lgbm_to_params, params_to_lgbm

warnings.filterwarnings("ignore")


class RandomClusterStrategy(FLStrategy):
    """Randomly assign clients to K clusters each round, aggregate within."""

    def __init__(self, n_clients: int, n_clusters: int = 4, true_labels=None):
        self._params: Optional[Parameters] = None
        self.n_clients = n_clients
        self.n_clusters = n_clusters
        self.true_labels = true_labels
        self.cluster_models = {}
        self.labels = None

    def aggregate_fit(self, server_round, results, failures):
        # Random cluster assignment
        self.labels = np.random.randint(0, self.n_clusters, size=self.n_clients)

        # Aggregate within each cluster (pick largest client)
        cluster_groups = defaultdict(list)
        for client_id, params, n_examples, metrics in results:
            cluster_groups[int(self.labels[client_id])].append((params, n_examples))

        for cid, group in cluster_groups.items():
            best = max(group, key=lambda x: x[1])
            self.cluster_models[cid] = best[0]

        # Pick global model as largest overall
        best_global = max(results, key=lambda x: x[2])
        self._params = best_global[1]

        # Compute NMI/ARI if we have ground truth
        nmi, ari = 0.0, 0.0
        if self.true_labels is not None:
            nmi = float(normalized_mutual_info_score(self.true_labels, self.labels))
            ari = float(adjusted_rand_score(self.true_labels, self.labels))
            print(f"  [RandomClust Round {server_round}] NMI={nmi:.3f}  ARI={ari:.3f}")

        return self._params, {"nmi": nmi, "ari": ari}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 1.0, {}
        total = sum(n for _, _, n, _ in results)
        wf1 = sum(m.get("f1", 0) * n for _, _, n, m in results) / total
        wauc = sum(m.get("auc", 0.5) * n for _, _, n, m in results) / total
        wloss = sum(loss * n for _, loss, n, _ in results) / total
        print(f"  [RandomClust Round {server_round}] F1={wf1:.4f}  AUC={wauc:.4f}")
        return wloss, {"f1": wf1, "auc": wauc}

    def configure_fit(self, server_round, parameters, n_clients):
        configs = []
        for cid in range(n_clients):
            if self.labels is not None and int(self.labels[cid]) in self.cluster_models:
                p = self.cluster_models[int(self.labels[cid])]
            else:
                p = parameters
            configs.append((cid, p, {"current_round": server_round}))
        return configs


class RandomClustClient(FLClient):
    def __init__(self, client_id, train_df, test_df, n_estimators=50):
        self.client_id = client_id
        self.feature_cols = [c for c in train_df.columns if c != "label"]
        self.X_train = train_df[self.feature_cols].values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(int)
        self.X_test = test_df[self.feature_cols].values.astype(np.float32)
        self.y_test = test_df["label"].values.astype(int)
        self.n_estimators = n_estimators
        self.booster = None

    def fit(self, parameters, config):
        init_model = params_to_lgbm(parameters)
        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.booster = lgb.train(
            {"objective": "binary", "verbose": -1, "n_jobs": 1,
             "learning_rate": 0.05, "num_leaves": 31},
            dtrain, num_boost_round=self.n_estimators, init_model=init_model,
            callbacks=[lgb.log_evaluation(period=-1)], keep_training_booster=True,
        )
        return lgbm_to_params(self.booster), len(self.X_train), {}

    def evaluate(self, parameters, config):
        booster = params_to_lgbm(parameters) or self.booster
        if booster is None:
            return 1.0, len(self.X_test), {"f1": 0.0, "auc": 0.5}
        y_prob = booster.predict(self.X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(self.y_test, y_prob))
        except:
            auc = 0.5
        loss = float(-np.mean(self.y_test * np.log(np.clip(y_prob, 1e-9, 1)) +
                               (1 - self.y_test) * np.log(np.clip(1 - y_prob, 1e-9, 1))))
        return loss, len(self.X_test), {"f1": f1, "auc": auc}


def run_random_cluster(client_splits, n_clients, n_rounds=30, n_estimators=50,
                       n_clusters=4, true_labels=None, results_dir="results", **kwargs):
    def client_fn(cid):
        return RandomClustClient(cid, client_splits[cid]["train"],
                                 client_splits[cid]["test"], n_estimators)

    strategy = RandomClusterStrategy(n_clients, n_clusters, true_labels)
    history = run_simulation(client_fn, strategy, n_clients, n_rounds,
                             initial_parameters=Parameters([np.array([], dtype=np.uint8)]),
                             verbose=False)

    f1_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("f1", [])]
    auc_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("auc", [])]
    nmi_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("nmi", [])]
    result = {
        "method": "RandomCluster",
        "n_clients": n_clients, "n_rounds": n_rounds,
        "f1_history": f1_hist, "auc_history": auc_hist,
        "final_f1": f1_hist[-1][1] if f1_hist else 0.0,
        "final_auc": auc_hist[-1][1] if auc_hist else 0.5,
        "final_nmi": nmi_hist[-1][1] if nmi_hist else 0.0,
        "final_ari": 0.0,
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/random_cluster.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[RandomClust] F1={result['final_f1']:.4f}  AUC={result['final_auc']:.4f}")
    return result
