"""
baselines/local_only.py
=======================
LocalOnly baseline: each client trains only on its own data.
No federated communication at all. This is the lower bound.
"""

import json
import warnings
from pathlib import Path
from typing import Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")


def run_local_only(
    client_splits: Dict[int, Dict[str, pd.DataFrame]],
    n_clients: int,
    n_estimators: int = 50,
    results_dir: str = "results",
) -> dict:
    """
    Train one model per client locally. No communication.
    Returns aggregate metrics across all clients.
    """
    all_f1, all_auc = [], []

    for cid in range(n_clients):
        train_df = client_splits[cid]["train"]
        test_df = client_splits[cid]["test"]
        feature_cols = [c for c in train_df.columns if c != "label"]

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["label"].values.astype(int)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["label"].values.astype(int)

        dtrain = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(
            {"objective": "binary", "verbose": -1, "n_jobs": 1,
             "learning_rate": 0.05, "num_leaves": 31},
            dtrain, num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        y_prob = booster.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.5

        all_f1.append(f1)
        all_auc.append(auc)
        print(f"[LocalOnly] Client {cid:02d}: F1={f1:.4f}  AUC={auc:.4f}")

    result = {
        "method": "LocalOnly",
        "n_clients": n_clients,
        "n_rounds": 1,
        "per_client_f1": all_f1,
        "per_client_auc": all_auc,
        "final_f1": float(np.mean(all_f1)),
        "final_auc": float(np.mean(all_auc)),
        "f1_std": float(np.std(all_f1)),
        "auc_std": float(np.std(all_auc)),
        "f1_history": [(1, float(np.mean(all_f1)))],
        "auc_history": [(1, float(np.mean(all_auc)))],
        "final_nmi": 0.0, "final_ari": 0.0,
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/local_only.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[LocalOnly] Avg F1={result['final_f1']:.4f} ± {result['f1_std']:.4f}")
    return result
