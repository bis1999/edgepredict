"""
GNN trainer: BCEWithLogitsLoss, early stopping on validation AUC.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Literal, Sequence

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling

from gnn_models import _SAGE, _GCN, _MLPPredictor, generate_node_features
from metrics import evaluate_multi_k  # for final eval packaging
from sklearn.metrics import roc_auc_score

try:
    from logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger("gnn")
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

def _fmt(x):
    try:
        return f"{x:.4f}"
    except Exception:
        return "N/A"

class UnifiedGNNLinkPredictor:
    """Link predictor with early stopping on validation AUC."""

    def __init__(
        self,
        model_type: Literal["gcn", "sage"] = "sage",
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        node_feature_type: str = "random",
        node_feature_dim: int = 64,
        device: str = "cpu",
        random_state: int = 42
    ):
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.node_feature_type = node_feature_type
        self.node_feature_dim = node_feature_dim
        self.device = torch.device(device)
        self.random_state = random_state

        self.model: Optional[nn.Module] = None
        self.predictor: Optional[nn.Module] = None
        self.node_map: Optional[Dict[Any, int]] = None
        self.x: Optional[torch.Tensor] = None
        self.adj_t: Optional[SparseTensor] = None

        self.training_scenario: Optional[str] = None
        self.training_history: Dict[str, List[float]] = {"train_losses": [], "val_auc": []}

    def set_training_scenario(self, scenario: Optional[str]) -> None:
        self.training_scenario = scenario

    def _create_model(self, in_dim: int) -> None:
        dims = [in_dim] + self.hidden_dims
        if self.model_type == "gcn":
            self.model = _GCN(dims, dropout=self.dropout)
        elif self.model_type == "sage":
            self.model = _SAGE(dims, dropout=self.dropout)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.predictor = _MLPPredictor(
            dim=self.hidden_dims[-1],
            hidden=max(64, self.hidden_dims[-1] * 2),
            layers=3,
            dropout=self.dropout,
        )
        self.model = self.model.to(self.device)
        self.predictor = self.predictor.to(self.device)

    def _build_train_edges(self, G: nx.Graph) -> torch.Tensor:
        self.node_map = {n: i for i, n in enumerate(G.nodes())}
        edges = [(self.node_map[u], self.node_map[v]) for u, v in G.edges()]
        # undirected -> both directions
        edges = edges + [(v, u) for (u, v) in edges]
        return torch.tensor(edges, dtype=torch.long, device=self.device).T

    def _make_val_edges(self, train_edge_index: torch.Tensor, num_nodes: int, n_val_pos: int = 2048) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample validation pos/neg pairs and build labels for AUC."""
        m = train_edge_index.size(1)
        n_pos = min(n_val_pos, max(32, m // 20))
        # take first n_pos as positives (already doubled, but fine for scoring)
        pos_pairs = train_edge_index[:, :n_pos]

        # sample same number of negatives
        neg_pairs = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=n_pos,
            method="sparse",
        )
        # concatenate
        val_pairs = torch.cat([pos_pairs, neg_pairs], dim=1)
        y_true = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_pos, dtype=int)], axis=0)
        return val_pairs, y_true

    def _train_step(self, train_edge_index: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        assert self.model and self.predictor and self.x is not None and self.adj_t is not None
        self.model.train(); self.predictor.train()
        optimizer.zero_grad()
        z = self.model(self.x, self.adj_t)

        pos_logits = self.predictor(z[train_edge_index[0]], z[train_edge_index[1]])
        pos_loss = loss_fn(pos_logits, torch.ones_like(pos_logits))

        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=self.x.size(0),
            num_neg_samples=train_edge_index.size(1),
            method="sparse",
        )
        neg_logits = self.predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
        neg_loss = loss_fn(neg_logits, torch.zeros_like(neg_logits))

        loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.predictor.parameters()), 1.0)
        optimizer.step()
        return float(loss.item())

    def _eval_auc(self, val_pairs: torch.Tensor, y_true: np.ndarray) -> Optional[float]:
        self.model.eval(); self.predictor.eval()
        with torch.no_grad():
            z = self.model(self.x, self.adj_t)
            logits = self.predictor(z[val_pairs[0]], z[val_pairs[1]])
            probs = torch.sigmoid(logits).cpu().numpy()
        try:
            return float(roc_auc_score(y_true, probs))
        except Exception:
            return None

    def fit(self, training_graph: nx.Graph, epochs: int = 600, log_every: int = 10,
            early_stopping: bool = True, patience: int = 20) -> Dict[str, List[float]]:
        if training_graph.number_of_nodes() == 0 or training_graph.number_of_edges() == 0:
            raise ValueError("Training graph must have nodes and edges.")
        random.seed(self.random_state); np.random.seed(self.random_state); torch.manual_seed(self.random_state)

        train_edge_index = self._build_train_edges(training_graph)
        num_nodes = training_graph.number_of_nodes()
        self.x = generate_node_features(
            num_nodes=num_nodes,
            dim=(num_nodes if self.node_feature_type == "identity" else self.node_feature_dim),
            feature_type=("identity" if self.node_feature_type == "identity" else "random"),
            seed=self.random_state,
        ).to(self.device)
        self.adj_t = SparseTensor.from_edge_index(train_edge_index, sparse_sizes=(num_nodes, num_nodes)).to(self.device)
        self._create_model(self.x.size(1))

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # labeled val set (pos+neg) for AUC early stopping
        val_pairs, y_val = self._make_val_edges(train_edge_index, num_nodes)

        best_auc, best_state, bad = -1.0, None, 0
        self.training_history = {"train_losses": [], "val_auc": []}

        for epoch in range(1, epochs + 1):
            loss = self._train_step(train_edge_index, optimizer, loss_fn)
            self.training_history["train_losses"].append(loss)

            cur_auc = self._eval_auc(val_pairs, y_val)
            self.training_history["val_auc"].append(cur_auc if cur_auc is not None else float("nan"))

            if (epoch % log_every == 0) or epoch == 1 or epoch == epochs:
                log.info(f"Epoch {epoch:4d} | loss {_fmt(loss)} | valAUC {_fmt(cur_auc)}")

            if early_stopping and cur_auc is not None:
                if cur_auc > best_auc + 1e-6:
                    best_auc = cur_auc; bad = 0
                    best_state = {
                        "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                        "pred": {k: v.detach().cpu() for k, v in self.predictor.state_dict().items()},
                    }
                else:
                    bad += 1
                    if bad >= patience:
                        log.info(f"Early stopping at epoch {epoch} (best valAUC={_fmt(best_auc)})")
                        break

        if best_state:
            self.model.load_state_dict(best_state["model"])
            self.predictor.load_state_dict(best_state["pred"])

        return self.training_history

    def predict(
        self,
        edges_to_predict: List[Tuple[Any, Any]],
        labels: Optional[Sequence[int]] = None,
        threshold: float = 0.5,
        batch_size: int = 65536,
        return_dataframe: bool = True,
        return_metrics: bool = True,
        ks: List[int] = [5, 10, 20]
    ):
        if self.model is None or self.predictor is None or self.x is None or self.adj_t is None or self.node_map is None:
            raise RuntimeError("Model state incomplete. Train first.")
        if labels is not None and len(labels) != len(edges_to_predict):
            raise ValueError("labels length must equal edges_to_predict length.")

        valid_idx, valid_mask = [], []
        for (u, v) in edges_to_predict:
            if (u in self.node_map) and (v in self.node_map):
                valid_idx.append((self.node_map[u], self.node_map[v])); valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            rows = []
            for i, (u, v) in enumerate(edges_to_predict):
                tlab = None if labels is None else int(labels[i])
                rows.append({"u": u, "v": v, "prediction_score": 0.0, "predicted_label": 0,
                             "true_label": tlab, "model_type": f"GNN_{self.model_type.upper()}",
                             "valid_prediction": False, "training_scenario": self.training_scenario})
            out = pd.DataFrame(rows) if return_dataframe else rows
            return (out, None) if return_metrics else out

        self.model.eval(); self.predictor.eval()
        with torch.no_grad():
            z = self.model(self.x, self.adj_t)

        idx_t = torch.tensor(valid_idx, dtype=torch.long, device=self.device).T
        m = idx_t.size(1)
        probs = torch.empty(m, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            s = 0
            while s < m:
                e = min(s + batch_size, m)
                idx = idx_t[:, s:e]
                logits = self.predictor(z[idx[0]], z[idx[1]])
                p = torch.sigmoid(logits).cpu().view(-1)
                probs[s:e] = p
                s = e

        rows, ptr = [], 0
        for i, ((u, v), ok) in enumerate(zip(edges_to_predict, valid_mask)):
            if ok:
                score = float(probs[ptr]); ptr += 1; valid = True
            else:
                score = 0.0; valid = False
            tlab = None if labels is None else int(labels[i])
            rows.append({
                "u": u, "v": v,
                "prediction_score": score,
                "predicted_label": int(score >= threshold),
                "true_label": tlab,
                "model_type": f"GNN_{self.model_type.upper()}",
                "valid_prediction": valid,
                "training_scenario": self.training_scenario
            })
        out = pd.DataFrame(rows) if return_dataframe else rows

        metrics = None
        if return_metrics and labels is not None:
            dfv = out if return_dataframe else pd.DataFrame(out)
            mask = (dfv["valid_prediction"] == True) & dfv["true_label"].isin([0, 1])
            if mask.any():
                y_true = dfv.loc[mask, "true_label"].astype(int).to_numpy()
                y_score = dfv.loc[mask, "prediction_score"].astype(float).to_numpy()
                metrics = evaluate_multi_k(y_true, y_score, threshold=threshold, ks=ks)
                metrics.update({
                    "n_total": int(len(edges_to_predict)),
                    "n_valid_scored": int(mask.sum()),
                    "n_unseen_or_invalid": int(len(edges_to_predict) - mask.sum()),
                    "model_type": f"GNN_{self.model_type.upper()}",
                    "training_scenario": self.training_scenario,
                })
            else:
                metrics = {
                    "n_total": int(len(edges_to_predict)),
                    "n_valid_scored": 0,
                    "n_unseen_or_invalid": int(len(edges_to_predict)),
                    "auc": None, "accuracy": None, "precision": None, "recall": None, "f1": None,
                    "threshold": float(threshold),
                    "model_type": f"GNN_{self.model_type.upper()}",
                    "training_scenario": self.training_scenario,
                }

        return (out, metrics) if return_metrics else out
