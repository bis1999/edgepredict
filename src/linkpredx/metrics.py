"""Evaluation metrics, including Top‑K variants."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

__all__ = [
    "auc_score",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
]

def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:  # type: ignore
    """Area‑under‑ROC using scikit‑learn."""
    return roc_auc_score(y_true, y_score)

def _topk_mask(y_score: np.ndarray, k: int) -> np.ndarray:
    """Return boolean mask selecting K highest‑score indices."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(y_score):
        k = len(y_score)
    top_idx = np.argpartition(-y_score, k - 1)[:k]
    mask = np.zeros_like(y_score, dtype=bool)
    mask[top_idx] = True
    return mask

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    mask = _topk_mask(y_score, k)
    return (y_true[mask].sum() / k) if k else 0.0

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    mask = _topk_mask(y_score, k)
    return y_true[mask].sum() / positives

def f1_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    p = precision_at_k(y_true, y_score, k)
    r = recall_at_k(y_true, y_score, k)
    return (2 * p * r / (p + r)) if (p + r) else 0.0