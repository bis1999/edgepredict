"""
ML trainer for link prediction with CV model selection.
- Works with dict-style folds produced by LinkPredictionTrainer._prepare_cv_data_for_ml()
  (each fold has X_train, y_train, X_test, y_test)
  and also supports legacy list-of-(train_idx, val_idx) folds.
- Metrics from metrics.evaluate_multi_k (exposes 'auc' and 'roc_auc').
- LightGBM removed per request. XGBoost is optional.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


from ..utils.metrics import evaluate_multi_k

try:
    from ..utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


def _fmt(v: Optional[float]) -> str:
    try:
        return f"{v:.4f}"
    except Exception:
        return "N/A"


def _get_model(name: str, params: Dict[str, Any]):
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "svc":
        return SVC(**params)
    if name == "xgboost":
        if not HAVE_XGB:
            raise ValueError("xgboost not available")
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def _scores(model, X: np.ndarray) -> np.ndarray:
    """Return scores in [0,1] whenever possible (probabilities preferred)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        # logistic squashing for comparability
        return 1.0 / (1.0 + np.exp(-z))
    # last resort: predicted labels as floats (not ideal, but safe)
    return model.predict(X).astype(float)


class MLLinkPredictionTrainer:
    """Grid-search ML trainer using upstream CV folds (dict-splits or index-folds)."""

    def __init__(
        self,
        cv_data: Dict[str, Any],
        random_state: int = 42,
        models: Optional[List[str]] = None,
        custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        scale_features: bool = True,
        primary_metric: str = "auc",
    ):
        # Expect cv_data structure from LinkPredictionTrainer._prepare_cv_data_for_ml()
        self.cv_folds = cv_data["cv_folds"]["folds"]               # dict OR list
        self.feature_columns = cv_data["cv_folds"]["feature_columns"]
        self.cv_dataset = cv_data["cv_dataset"]                    # full DF (has 'label')

        self.random_state = random_state
        self.scale_features = scale_features
        self.primary_metric = primary_metric  # 'auc' recommended (alias of roc_auc)

        default_models = ["random_forest", "logistic_regression", "svc"]
        if HAVE_XGB:
            default_models.append("xgboost")
        self.models = models or default_models

        self.param_grids = self._setup_param_grids(custom_grids)

        # Fit artifacts
        self.results: Dict[Any, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf
        self.scaler: Optional[StandardScaler] = None
        self.best_fitted_model = None

        log.info(
            f"ML Trainer | folds={len(self.cv_folds) if hasattr(self.cv_folds, '__len__') else 'N/A'} "
            f"| features={len(self.feature_columns)} | models={self.models}"
        )

    # ----------------- internals -----------------

    def _setup_param_grids(self, custom: Optional[Dict[str, Dict[str, List[Any]]]]) -> Dict[str, Dict[str, List[Any]]]:
        grid = {
            "random_forest": {
                "n_estimators": [50,300],
                "max_depth": [5,30],
                "min_samples_split": [2, 5],
                "n_jobs": [-1],
                "random_state": [self.random_state],
            },
            "logistic_regression": {
                "C": [0.5, 1.0, 2.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "max_iter": [2000],
                "random_state": [self.random_state],
            },
            "svc": {
                "C": [0.5, 1.0, 2.0],
                "kernel": ["rbf"],
                "probability": [True],  # so predict_proba is available
                "gamma": ["scale"],
            },
        }
        if HAVE_XGB:
            grid["xgboost"] = {
                "n_estimators": [100,200],
                "max_depth": [4, 6],
                "learning_rate": [0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "eval_metric": ["logloss"],
                "random_state": [self.random_state],
                "n_jobs": [-1],
                "use_label_encoder": [False],
            }
        if custom:
            for k, v in custom.items():
                grid[k] = v
        return grid

    def _iter_params(self, grid: Dict[str, List[Any]]):
        keys = list(grid.keys())
        for values in product(*[grid[k] for k in keys]):
            yield dict(zip(keys, values))

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and (optionally) scale features from a DataFrame."""
        X = df[self.feature_columns].to_numpy(dtype=float)
        if self.scale_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        return X

    # ----------------- training -----------------

    def fit(self, ks: List[int] = [10, 20]) -> None:
        """
        Grid-search across models/params using CV folds.
        Supports:
          - dict-splits: {fold_id: {"X_train", "y_train", "X_test", "y_test"}}
          - index-folds: list of (train_idx, val_idx)
        """
        best = -np.inf
        best_summary = None

        # Detect dict-split style
        dict_style = isinstance(self.cv_folds, dict) and len(self.cv_folds) > 0
        if dict_style:
            first_val = next(iter(self.cv_folds.values()))
            dict_style = isinstance(first_val, dict) and all(
                k in first_val for k in ("X_train", "y_train", "X_test", "y_test")
            )

        for model_name in self.models:
            grid = self.param_grids.get(model_name, {})
            for params in self._iter_params(grid):
                fold_scores = []

                if dict_style:
                    # --- Use pre-split DataFrames/arrays from upstream trainer ---
                    for fid, fold in self.cv_folds.items():
                        df_tr = fold["X_train"]
                        ytr = np.asarray(fold["y_train"], dtype=int)
                        df_te = fold["X_test"]
                        yte = np.asarray(fold["y_test"], dtype=int)

                        self.scaler = None  # fit scaler per fold to avoid leakage
                        Xtr = self._prepare_X(df_tr)
                        model = _get_model(model_name, params)
                        model.fit(Xtr, ytr)

                        Xte = self._prepare_X(df_te)
                        s = _scores(model, Xte)
                        m = evaluate_multi_k(yte, s, threshold=0.5, ks=ks)
                        fold_scores.append(m)

                else:
                    # --- Fallback: folds are list-like of (train_idx, val_idx) ---
                    folds_iterable = (
                        self.cv_folds.items() if isinstance(self.cv_folds, dict) else enumerate(self.cv_folds)
                    )
                    for _, (tr_idx, te_idx) in folds_iterable:
                        df_tr = self.cv_dataset.iloc[tr_idx]
                        df_te = self.cv_dataset.iloc[te_idx]
                        ytr = df_tr["label"].astype(int).to_numpy()
                        yte = df_te["label"].astype(int).to_numpy()

                        self.scaler = None
                        Xtr = self._prepare_X(df_tr)
                        model = _get_model(model_name, params)
                        model.fit(Xtr, ytr)

                        Xte = self._prepare_X(df_te)
                        s = _scores(model, Xte)
                        m = evaluate_multi_k(yte, s, threshold=0.5, ks=ks)
                        fold_scores.append(m)

                primary = float(np.nanmean([m.get(self.primary_metric, np.nan) for m in fold_scores]))
                if primary > best:
                    best = primary
                    best_summary = {
                        "model": model_name,
                        "params": params,
                        "cv_mean_primary": primary,
                        "cv_all": fold_scores,
                    }
                log.info(f"{model_name} {params} | {self.primary_metric}={_fmt(primary)}")

        if best_summary is None:
            raise RuntimeError("No model evaluated successfully.")

        self.best_model_name = best_summary["model"]
        self.best_params = best_summary["params"]
        self.best_score = best_summary["cv_mean_primary"]

        # Fit best on full dataset (uses labels in cv_dataset)
        self.scaler = None
        Xall = self._prepare_X(self.cv_dataset)
        yall = self.cv_dataset["label"].astype(int).to_numpy()
        self.best_fitted_model = _get_model(self.best_model_name, self.best_params)
        self.best_fitted_model.fit(Xall, yall)
        log.info(f"Selected {self.best_model_name} with {self.primary_metric}={_fmt(self.best_score)}")

        self.results = {
            "best": {
                "model": self.best_model_name,
                "params": self.best_params,
                self.primary_metric: self.best_score
            }
        }

    # ----------------- API used by trainer.py -----------------

    def get_summary(self) -> Dict[str, Any]:
        return {
            "best_model": self.best_model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
        }

    def predict_holdout(self, df_holdout: pd.DataFrame, ks: List[int] = [10, 20]) -> Dict[str, float]:
        if self.best_fitted_model is None:
            raise RuntimeError("Call fit() first.")
        X = self._prepare_X(df_holdout)
        y = df_holdout["label"].astype(int).to_numpy()
        s = _scores(self.best_fitted_model, X)
        return evaluate_multi_k(y, s, threshold=0.5, ks=ks)

    def predict_candidates(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Score unlabeled candidate pairs; returns a copy with 'prediction_score'.
        """
        if self.best_fitted_model is None:
            raise RuntimeError("Call fit() first.")
        X = df_candidates[self.feature_columns].to_numpy(dtype=float)
        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)
        s = _scores(self.best_fitted_model, X)
        out = df_candidates.copy()
        out["prediction_score"] = s
        return out


# Optional convenience wrapper (used by LinkPredictionTrainer.train_ml_models)
def train_ml_models(
    cv_data: Dict[str, Any],
    models: Optional[List[str]] = None,
    custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ks: List[int] = [10, 20],
    primary_metric: str = "auc",
    scale_features: bool = True,
) -> MLLinkPredictionTrainer:
    trainer = MLLinkPredictionTrainer(
        cv_data=cv_data,
        models=models,
        custom_grids=custom_grids,
        scale_features=scale_features,
        primary_metric=primary_metric,
    )
    trainer.fit(ks=ks)
    return trainer
