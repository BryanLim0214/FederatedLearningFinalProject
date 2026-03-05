"""
client.py
=========
FedRankX FL client — uses fl_engine.FLClient (no Flower/Ray dependency).

Each client:
  1. Receives model parameters from the server.
  2. Trains LightGBM locally on private phishing data.
  3. Extracts SHAP importances -> top-K rank list.
  4. Returns updated model parameters + rank list in metrics.
"""

import json
import warnings
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.fl_engine import FLClient, Parameters, arrays_to_params, params_to_arrays
from src.rank_utils import compute_shap_for_lgbm, shap_to_rank

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# LightGBM serialisation helpers
# ---------------------------------------------------------------------------

def lgbm_to_params(booster: lgb.Booster) -> Parameters:
    model_bytes = np.frombuffer(booster.model_to_string().encode("utf-8"), dtype=np.uint8)
    return Parameters([model_bytes])


def params_to_lgbm(params: Parameters) -> Optional[lgb.Booster]:
    arrays = params_to_arrays(params)
    if not arrays or len(arrays[0]) == 0:
        return None
    try:
        model_str = arrays[0].tobytes().decode("utf-8")
        return lgb.Booster(model_str=model_str)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FedRankX Client
# ---------------------------------------------------------------------------

class FedRankXClient(FLClient):
    """
    Federated client that trains LightGBM locally, extracts SHAP
    rank lists, and communicates them back to the server via metrics.
    """

    def __init__(
        self,
        client_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        top_k: int = 15,
        n_estimators: int = 50,
        shap_subsample: int = 200,
        lgb_params: Optional[dict] = None,
    ):
        self.client_id = client_id
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.shap_subsample = shap_subsample

        self.feature_cols = [c for c in train_df.columns if c != "label"]
        self.X_train = train_df[self.feature_cols].values.astype(np.float32)
        self.y_train = train_df["label"].values.astype(int)
        self.X_test = test_df[self.feature_cols].values.astype(np.float32)
        self.y_test = test_df["label"].values.astype(int)

        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": 1,
        }
        self.booster: Optional[lgb.Booster] = None

    def fit(self, parameters: Parameters, config: dict) -> Tuple[Parameters, int, dict]:
        current_round = config.get("current_round", 1)
        init_model = params_to_lgbm(parameters)

        dtrain = lgb.Dataset(self.X_train, label=self.y_train)
        self.booster = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=self.n_estimators,
            init_model=init_model,
            callbacks=[lgb.log_evaluation(period=-1)],
            keep_training_booster=True,
        )

        # SHAP -> rank list (the key FedRankX operation)
        shap_importance = compute_shap_for_lgbm(
            self.booster,
            pd.DataFrame(self.X_train, columns=self.feature_cols),
            subsample=self.shap_subsample,
        )
        rank_list = shap_to_rank(shap_importance, top_k=self.top_k)

        metrics = {
            "client_id": self.client_id,
            "rank_list_json": json.dumps(rank_list.tolist()),
            "n_features": len(self.feature_cols),
            "top_k": self.top_k,
            "round": current_round,
        }

        return lgbm_to_params(self.booster), len(self.X_train), metrics

    def evaluate(self, parameters: Parameters, config: dict) -> Tuple[float, int, dict]:
        # ---- Cluster Ensemble Evaluation ----
        # If we have ensemble_params (list of Parameters from cluster members),
        # average predictions from ALL cluster models. This gives the federated
        # ensemble diversity benefit that makes FedRankX beat LocalOnly.
        ensemble_params = config.get("ensemble_params", None)
        if ensemble_params and len(ensemble_params) > 1:
            all_probs = []
            for ep in ensemble_params:
                b = params_to_lgbm(ep)
                if b is not None:
                    try:
                        probs = b.predict(self.X_test)
                        all_probs.append(probs)
                    except Exception:
                        pass
            if all_probs:
                # Average predictions across all cluster member models
                y_prob = np.mean(all_probs, axis=0)
                y_pred = (y_prob >= 0.5).astype(int)
                f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
                try:
                    auc = float(roc_auc_score(self.y_test, y_prob))
                except ValueError:
                    auc = 0.5
                loss = float(-np.mean(
                    self.y_test * np.log(np.clip(y_prob, 1e-9, 1)) +
                    (1 - self.y_test) * np.log(np.clip(1 - y_prob, 1e-9, 1))
                ))
                return loss, len(self.X_test), {"f1": f1, "auc": auc, "client_id": self.client_id}

        # ---- Single model fallback ----
        # Prefer local model (self.booster) over server parameters.
        if config.get("use_local_model", False) and self.booster is not None:
            booster = self.booster
        else:
            booster = params_to_lgbm(parameters) or self.booster
        if booster is None:
            return 1.0, len(self.X_test), {"f1": 0.0, "auc": 0.5}

        y_prob = booster.predict(self.X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(self.y_test, y_prob))
        except ValueError:
            auc = 0.5
        loss = float(-np.mean(
            self.y_test * np.log(np.clip(y_prob, 1e-9, 1)) +
            (1 - self.y_test) * np.log(np.clip(1 - y_prob, 1e-9, 1))
        ))
        return loss, len(self.X_test), {"f1": f1, "auc": auc, "client_id": self.client_id}


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client_fn(
    client_splits: Dict[int, Dict[str, pd.DataFrame]],
    top_k: int = 15,
    n_estimators: int = 50,
):
    def client_fn(cid: int) -> FedRankXClient:
        split = client_splits[cid]
        return FedRankXClient(
            client_id=cid,
            train_df=split["train"],
            test_df=split["test"],
            top_k=top_k,
            n_estimators=n_estimators,
        )
    return client_fn
