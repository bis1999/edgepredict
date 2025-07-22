"""Model training & evaluation supporting multiple classifiers

Supported models
----------------
* RandomForestClassifier (sklearn)
* LogisticRegression (sklearn)
* Support‑Vector Machine (SVC, sklearn – RBF kernel)
* XGBClassifier (xgboost, optional)

The class runs a simple grid search with 5‑fold cross‑validation over a
cartesian product of hyper‑parameter grids defined per model.  Scores are
aggregated (mean) across folds for two metrics: AUC and precision@k.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGB = True
except ImportError:  # pragma: no cover
    _HAVE_XGB = False

from .logger import get_logger
from .metrics import auc_score, precision_at_k, recall_at_k, f1_at_k

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_model(name: str, params: Dict[str, Any]):
    if name == "random_forest":
        return RandomForestClassifier(random_state=params.pop("random_state"), **params)
    if name == "log_reg":
        return LogisticRegression(max_iter=1000, solver="liblinear", **params)
    if name == "svm":
        return SVC(probability=True, **params)
    if name == "xgb":
        if not _HAVE_XGB:
            raise RuntimeError("xgboost not installed – run `pip install xgboost`. ")
        return XGBClassifier(eval_metric="logloss", use_label_encoder=False, **params)
    raise ValueError(f"Unknown model '{name}'")

# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class LinkPredictionTrainer:
    """Grid‑search trainer with Top‑K evaluation for multiple models."""

    def __init__(
        self,
        X_train_cv: Dict[int, pd.DataFrame],
        y_train_cv: Dict[int, np.ndarray],
        X_test_cv: Dict[int, pd.DataFrame],
        y_test_cv: Dict[int, np.ndarray],
        df_tr_top: pd.DataFrame,
        df_ho_top: pd.DataFrame,
        feature_set: List[str],
        random_state: int = 42,
    ) -> None:
        self.X_train_cv, self.y_train_cv = X_train_cv, y_train_cv
        self.X_test_cv, self.y_test_cv = X_test_cv, y_test_cv
        self.df_tr_top, self.df_ho_top = df_tr_top, df_ho_top
        self.feature_set = feature_set
        self.random_state = random_state

        # ----------------------------------------------
        # Define per‑model hyper‑parameter grids
        # ----------------------------------------------
        self.param_grid = {
            "random_forest": {
                "max_depth": [3, 6],
                "n_estimators": [100, 200],
                "random_state": [random_state],
            },
            "log_reg": {
                "C": [0.1, 1, 10],
            },
            "svm": {
                "C": [0.1, 1, 10],
                "gamma": ["scale"],
                "kernel": ["rbf"],
            },
        }
        if _HAVE_XGB:
            self.param_grid["xgb"] = {
                "max_depth": [3, 6],
                "n_estimators": [100, 200],
                "learning_rate": [0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "random_state": [random_state],
            }

    # ---------------------------------------------------------
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df.fillna(df.mean())

    # ---------------------------------------------------------
    def cross_val_search(self, k_top: int = 5):
        """Exhaustive grid search over all model families & params."""
        results: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, float]] = {}

        for model_name, grid in self.param_grid.items():
            keys, values = zip(*grid.items())  # type: ignore[arg-type]
            for combo in product(*values):
                params = dict(zip(keys, combo))
                scores_auc, scores_p = [], []

                for cv in range(1, 6):
                    Xtr = self._clean(self.X_train_cv[cv])[self.feature_set]
                    ytr = self.y_train_cv[cv]
                    Xte = self._clean(self.X_test_cv[cv])[self.feature_set]
                    yte = self.y_test_cv[cv]

                    clf = _make_model(model_name, params.copy())
                    clf.fit(Xtr, ytr)
                    proba = (
                        clf.predict_proba(Xte)[:, 1]
                        if hasattr(clf, "predict_proba")
                        else clf.decision_function(Xte)  # SVM
                    )

                    scores_auc.append(auc_score(yte, proba))
                    scores_p.append(precision_at_k(yte, proba, k_top))

                key = (model_name, tuple(sorted(params.items())))
                results[key] = {
                    "auc": float(np.mean(scores_auc)),
                    "precision@k": float(np.mean(scores_p)),
                }
                log.info(
                    "%s %s | AUC=%.3f P@%d=%.3f",
                    model_name,
                    params,
                    results[key]["auc"],
                    k_top,
                    results[key]["precision@k"],
                )

        # pick best by AUC first, then precision@k
        best_key = max(results, key=lambda k: (results[k]["auc"], results[k]["precision@k"]))
        self.best_model_name, param_tuple = best_key
        self.best_params = dict(param_tuple)
        log.info("Selected model → %s %s", self.best_model_name, self.best_params)
        return results, best_key

    # ---------------------------------------------------------
    def predict_holdout(self, k_top: int = 10):
        if not hasattr(self, "best_model_name"):
            raise RuntimeError("Run cross_val_search() first.")

        clf = _make_model(self.best_model_name, self.best_params.copy())

        Xtr = self._clean(self.df_tr_top)[self.feature_set]
        ytr = self.df_tr_top["label"].values
        Xho = self._clean(self.df_ho_top)[self.feature_set]
        yho = self.df_ho_top["label"].values

        clf.fit(Xtr, ytr)
        proba = (
            clf.predict_proba(Xho)[:, 1]
            if hasattr(clf, "predict_proba")
            else clf.decision_function(Xho)
        )
        pred = (proba >= 0.5).astype(int)

        cm = confusion_matrix(yho, pred)
        result = {
            "model": self.best_model_name,
            "params": self.best_params,
            "auc": auc_score(yho, proba),
            "precision@k": precision_at_k(yho, proba, k_top),
            "recall@k": recall_at_k(yho, proba, k_top),
            "f1@k": f1_at_k(yho, proba, k_top),
            "tp_rate": cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0,
            "fn_rate": cm[0, 1] / cm[0].sum() if cm[0].sum() else 0.0,
        }
        for k, v in result.items():
            log.info("%s: %s" if not isinstance(v, float) else "%s: %.4f", k, v)
        return result
