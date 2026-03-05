"""
baselines/fedavg_standard.py  — pure fl_engine version (no Flower/Ray)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.fl_engine import FLClient, FLStrategy, Parameters, params_to_arrays, run_simulation
from src.client import lgbm_to_params, params_to_lgbm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Simple FedAvg strategy (largest-sample representative)
# ---------------------------------------------------------------------------

class FedAvgStrategy(FLStrategy):
    def __init__(self, n_clients: int):
        self._params: Optional[Parameters] = None
        self.n_clients = n_clients
        self._round_f1: List[Tuple[int, float]] = []

    def aggregate_fit(self, server_round, results, failures):
        best = max(results, key=lambda x: x[2])  # most examples
        self._params = best[1]
        return self._params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return 1.0, {}
        total = sum(n for _, _, n, _ in results)
        wf1   = sum(m.get("f1", 0) * n for _, _, n, m in results) / total
        wauc  = sum(m.get("auc", 0.5) * n for _, _, n, m in results) / total
        wloss = sum(loss * n for _, loss, n, _ in results) / total
        print(f"  [FedAvg Round {server_round}] F1={wf1:.4f}  AUC={wauc:.4f}")
        return wloss, {"f1": wf1, "auc": wauc}


class FedAvgClient(FLClient):
    def __init__(self, client_id, train_df, test_df, n_estimators=50):
        self.client_id = client_id
        self.feature_cols = [c for c in train_df.columns if c != "label"]
        self.X_train = train_df[self.feature_cols].values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(int)
        self.X_test  = test_df[self.feature_cols].values.astype(np.float32)
        self.y_test  = test_df["label"].values.astype(int)
        self.n_estimators = n_estimators
        self.booster = None

    def fit(self, parameters, config):
        init_model = params_to_lgbm(parameters)
        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.booster = lgb.train(
            {"objective": "binary", "verbose": -1, "n_jobs": 1, "learning_rate": 0.05, "num_leaves": 31},
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
        try:   auc = float(roc_auc_score(self.y_test, y_prob))
        except: auc = 0.5
        loss = float(-np.mean(self.y_test * np.log(np.clip(y_prob,1e-9,1)) +
                               (1-self.y_test) * np.log(np.clip(1-y_prob,1e-9,1))))
        return loss, len(self.X_test), {"f1": f1, "auc": auc}


def run_fedavg(client_splits, n_clients, n_rounds=30, n_estimators=50,
               results_dir="results", **kwargs) -> dict:
    def client_fn(cid):
        return FedAvgClient(cid, client_splits[cid]["train"],
                            client_splits[cid]["test"], n_estimators)

    strategy = FedAvgStrategy(n_clients)
    history = run_simulation(client_fn, strategy, n_clients, n_rounds,
                             initial_parameters=Parameters([np.array([], dtype=np.uint8)]),
                             verbose=False)

    f1_hist  = [(r, float(v)) for r, v in history.metrics_distributed.get("f1", [])]
    auc_hist = [(r, float(v)) for r, v in history.metrics_distributed.get("auc", [])]
    result = {
        "method": "FedAvg", "n_clients": n_clients, "n_rounds": n_rounds,
        "losses": [(r, float(l)) for r, l in history.losses_distributed],
        "f1_history": f1_hist, "auc_history": auc_hist,
        "final_f1":  f1_hist[-1][1]  if f1_hist  else 0.0,
        "final_auc": auc_hist[-1][1] if auc_hist else 0.5,
        "final_nmi": 0.0, "final_ari": 0.0,
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/fedavg_standard.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[FedAvg] F1={result['final_f1']:.4f}  AUC={result['final_auc']:.4f}")
    return result
