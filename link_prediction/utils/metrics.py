"""
Minimal metrics for link prediction: AUC + confusion-at-threshold + ranking@K.
Consistency notes:
- Exposes both 'roc_auc' and 'auc' (alias).
- Ranking metrics computed on scores sorted desc; binary relevance in y_true.
"""

from __future__ import annotations
from typing import Dict, Any, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,  # AP kept in case you ever want it
    f1_score, precision_score, recall_score, accuracy_score
)

__all__ = [
    "evaluate", "evaluate_multi_k",
    "evaluate_df", "evaluate_df_multi_k",
    "aggregate_fold_metrics", "confusion_at_threshold"
]

# ---------- helpers ----------

def _safe(f, *a, **k) -> float:
    try:
        v = f(*a, **k)
        return float(v)
    except Exception:
        return float("nan")

def _validate(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length")
    if y_true.shape[0] == 0:
        raise ValueError("Empty input arrays")
    return y_true, y_score

def confusion_at_threshold(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    try:
        y_true, y_score = _validate(y_true, y_score)
    except ValueError:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "accuracy": float('nan'), "precision": float('nan'),
                "recall": float('nan'), "f1": float('nan'),
                "threshold": float(threshold)}
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
    }

def _ranking_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {f"precision@{k}": float('nan'), f"recall@{k}": float('nan'),
                f"f1@{k}": float('nan'), f"hits@{k}": float('nan')}
    k = max(1, min(int(k), len(y_true)))
    order = np.argsort(-y_score)
    topk_y = y_true[order[:k]]
    hits = int(np.sum(topk_y))
    total_pos = int(np.sum(y_true))
    precision_k = hits / k
    recall_k = hits / total_pos if total_pos > 0 else float('nan')
    f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) > 0 and np.isfinite(recall_k) else float('nan')
    return {
        f"precision@{k}": precision_k,
        f"recall@{k}": recall_k,
        f"f1@{k}": f1_k,
        f"hits@{k}": float(hits),
    }

# ---------- main API ----------

def evaluate(y_true, y_score, *, threshold: float = 0.5, k: int = 10) -> Dict[str, float]:
    """Classification + ranking@K (single K)."""
    try:
        y_true, y_score = _validate(y_true, y_score)
    except ValueError as e:
        return {"error": str(e)}
    out = {
        "roc_auc": _safe(roc_auc_score, y_true, y_score),
        "avg_precision": _safe(average_precision_score, y_true, y_score),
        "f1_at_thresh": _safe(f1_score, y_true, (y_score >= threshold).astype(int), zero_division=0),
    }
    out.update(confusion_at_threshold(y_true, y_score, threshold))
    out.update(_ranking_at_k(y_true, y_score, k))
    out["auc"] = out["roc_auc"]  # alias
    return out

def evaluate_multi_k(
    y_true, y_score, *, threshold: float = 0.5, ks: Union[int, Sequence[int]] = (5, 10, 20)
) -> Dict[str, float]:
    """Classification + ranking for multiple K."""
    try:
        y_true, y_score = _validate(y_true, y_score)
    except ValueError as e:
        return {"error": str(e)}
    if isinstance(ks, int):
        ks = [ks]
    out = {
        "roc_auc": _safe(roc_auc_score, y_true, y_score),
        "avg_precision": _safe(average_precision_score, y_true, y_score),
        "f1_at_thresh": _safe(f1_score, y_true, (y_score >= threshold).astype(int), zero_division=0),
    }
    out.update(confusion_at_threshold(y_true, y_score, threshold))
    for k in sorted(set(int(k) for k in ks)):
        out.update(_ranking_at_k(y_true, y_score, k))
    out["auc"] = out["roc_auc"]
    return out

def evaluate_df(df: pd.DataFrame, *, score_col="prediction_score", label_col="label",
                threshold: float = 0.5, k: int = 10) -> Dict[str, float]:
    if label_col not in df.columns or score_col not in df.columns:
        raise ValueError("DataFrame must contain label and score columns")
    return evaluate(df[label_col].values, df[score_col].values, threshold=threshold, k=k)

def evaluate_df_multi_k(df: pd.DataFrame, *, score_col="prediction_score", label_col="label",
                        threshold: float = 0.5, ks: Union[int, Sequence[int]] = (5, 10, 20)) -> Dict[str, float]:
    if label_col not in df.columns or score_col not in df.columns:
        raise ValueError("DataFrame must contain label and score columns")
    return evaluate_multi_k(df[label_col].values, df[score_col].values, threshold=threshold, ks=ks)

def aggregate_fold_metrics(fold_results: Dict[Any, Dict[str, float]]) -> pd.DataFrame:
    if not fold_results:
        return pd.DataFrame(columns=["metric", "mean", "std", "n_folds"])
    rows = [{"fold_id": fid, "metric": m, "value": v} for fid, d in fold_results.items() for m, v in d.items()]
    df = pd.DataFrame(rows)
    out = df.groupby("metric")["value"].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["metric", "mean", "std", "n_folds"]
    return out
