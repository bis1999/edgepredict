"""
linkpredx
=========

Unified ML + GNN link prediction toolkit.

This package provides:
- Dataset preparation (simulation, discovery, specific scenarios)
- Feature extraction (node, pairwise, SVD, configurable via YAML)
- Training with ML backends (logistic regression, random forest, etc.)
- Training with GNN backends (GCN, GraphSAGE)
- Unified metrics for evaluation
- Pretrained meta-models (GraphPredictor) for best-model selection
"""

__version__ = "0.1.0"

# Expose core API
from .config import DatasetConfig
from .dataset_preparer import LinkPredictionDatasetPreparer
from .trainer import LinkPredictionTrainer
from .metrics import (
    precision_at_k,
    recall_at_k,
    hits_at_k,
    auc_score,
)

# Optional imports (these can be heavy, so guard them if needed)
try:
    from .graph_predictor import GraphPredictor
except ImportError:
    GraphPredictor = None

__all__ = [
    "DatasetConfig",
    "LinkPredictionDatasetPreparer",
    "LinkPredictionTrainer",
    "GraphPredictor",
    "precision_at_k",
    "recall_at_k",
    "hits_at_k",
    "auc_score",
]
