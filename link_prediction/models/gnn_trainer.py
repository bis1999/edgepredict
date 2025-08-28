"""
GNN trainer with cross-validation support and hyperparameter optimization.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Literal, Sequence
import itertools
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from .gnn_models import _SAGE, _GCN, _MLPPredictor, generate_node_features
from ..utils.metrics import evaluate_multi_k
from sklearn.metrics import roc_auc_score

try:
    from ..utils.logger import get_logger
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
            early_stopping: bool = False, patience: int = 20) -> Dict[str, List[float]]:
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


class GNNHyperparameterOptimizer:
    """
    GNN hyperparameter optimizer with cross-validation support.
    Integrated with the workflow results from dataset_preparer.
    """
    
    def __init__(
        self,
        cv_data: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        random_state: int = 42,
        max_trials: int = 20,
        cv_epochs: int = 200
    ):
        """
        Args:
            cv_data: CV data in the format from trainer._prepare_cv_data_for_gnn()
            search_space: Hyperparameter search space
            device: Device for training
            random_state: Random seed
            max_trials: Maximum number of hyperparameter combinations to try
            cv_epochs: Number of epochs for each CV fold (reduced for speed)
        """
        self.cv_data = cv_data
        self.search_space = search_space or self._get_default_search_space()
        self.device = device
        self.random_state = random_state
        self.max_trials = max_trials
        self.cv_epochs = cv_epochs
        
        self.best_params = None
        self.best_score = 0.0
        self.cv_results = {}
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space"""
        return {
            'model_type': ['gcn', 'sage'],
            'hidden_dims': [[32], [64], [64, 32], [128, 64]],
            'node_feature_type': ['random', 'identity'],
            'learning_rate': [0.001, 0.01, 0.05],
            'dropout': [0.1, 0.2, 0.3],
            'node_feature_dim': [32, 64]
        }
    
    def _evaluate_config_on_fold(self, params: Dict[str, Any], fold_data: Dict[str, Any]) -> Optional[float]:
        """
        Evaluate a single configuration on one CV fold.
        
        Args:
            params: Parameter configuration to evaluate
            fold_data: Single fold data with train/val graphs and edges
            
        Returns:
            AUC score or None if evaluation failed
        """
        try:
            # Extract fold data
            train_graph = fold_data['train_graph']
            val_edges = fold_data['val_edges']
            val_labels = fold_data['val_labels']
            
            if not val_edges or val_labels is None:
                return None
            
            # Create model with given parameters
            model = UnifiedGNNLinkPredictor(
                model_type=params['model_type'],
                hidden_dims=params['hidden_dims'],
                node_feature_type=params['node_feature_type'],
                learning_rate=params['learning_rate'],
                dropout=params['dropout'],
                node_feature_dim=params.get('node_feature_dim', 64),
                device=self.device,
                random_state=self.random_state
            )
            
            # Train on fold
            model.fit(
                training_graph=train_graph, 
                epochs=self.cv_epochs, 
                log_every=self.cv_epochs,  # Only log at end
                early_stopping=True,
                patience=20
            )
            
            # Evaluate on validation set
            _, metrics = model.predict(
                edges_to_predict=val_edges,
                labels=val_labels,
                return_metrics=True
            )
            
            if metrics and metrics.get('auc') is not None:
                return metrics['auc']
            else:
                return None
                
        except Exception as e:
            log.warning(f"Error evaluating config on fold: {e}")
            return None
    
    def _evaluate_config_cv(self, params: Dict[str, Any]) -> Tuple[Optional[float], List[float]]:
        """
        Evaluate a configuration using cross-validation.
        
        Args:
            params: Parameter configuration to evaluate
            
        Returns:
            Tuple of (mean_auc, fold_aucs)
        """
        fold_aucs = []
        
        for fold_id, fold_data in self.cv_data['cv_folds'].items():
            auc = self._evaluate_config_on_fold(params, fold_data)
            if auc is not None:
                fold_aucs.append(auc)
        
        if len(fold_aucs) == 0:
            return None, []
        
        mean_auc = np.mean(fold_aucs)
        return mean_auc, fold_aucs
    
    def optimize(self) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """
        Run hyperparameter optimization with cross-validation.
        
        Returns:
            Tuple of (best_params, best_score, cv_results)
        """
        log.info(f"Starting GNN hyperparameter optimization with {self.max_trials} trials")
        
        # Set random seed
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Generate parameter combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > self.max_trials:
            log.info(f"Sampling {self.max_trials} from {len(all_combinations)} possible combinations")
            combinations = random.sample(all_combinations, self.max_trials)
        else:
            combinations = all_combinations
        
        best_auc = -1.0
        best_params = None
        cv_results = {}
        
        for i, param_combo in enumerate(tqdm(combinations, desc="GNN CV optimization")):
            params = dict(zip(param_names, param_combo))
            
            try:
                mean_auc, fold_aucs = self._evaluate_config_cv(params)
                
                if mean_auc is not None:
                    config_key = f"config_{i}"
                    cv_results[config_key] = {
                        'params': params,
                        'mean_auc': mean_auc,
                        'std_auc': np.std(fold_aucs),
                        'fold_aucs': fold_aucs
                    }
                    
                    log.info(f"Config {i+1}/{len(combinations)}: AUC = {mean_auc:.4f} Â± {np.std(fold_aucs):.4f}")
                    
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = params.copy()
                        log.info(f"New best: {best_auc:.4f}")
            
            except Exception as e:
                log.error(f"Error with GNN config {params}: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_auc
        self.cv_results = cv_results
        
        if best_params is None:
            log.warning("No valid configuration found, using defaults")
            best_params = {
                'model_type': 'sage',
                'hidden_dims': [64],
                'node_feature_type': 'random',
                'learning_rate': 0.01,
                'dropout': 0.2,
                'node_feature_dim': 64
            }
            best_auc = 0.0
        
        log.info(f"GNN CV optimization completed: Best AUC = {best_auc:.4f} Â± {cv_results.get(f'config_{len(combinations)-1}', {}).get('std_auc', 0):.4f}")
        log.info(f"Best GNN parameters: {best_params}")
        
        return best_params, best_auc, cv_results


def train_gnn_with_cv(
    cv_data: Dict[str, Any],
    search_space: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    random_state: int = 42,
    max_trials: int = 20,
    cv_epochs: int = 200,
    final_epochs: int = 600,
    early_stopping    = False
) -> Dict[str, Any]:
    """
    Train GNN with hyperparameter optimization using cross-validation.
    
    Args:
        cv_data: CV data prepared by trainer
        search_space: Optional hyperparameter search space
        device: Device for training
        random_state: Random seed
        max_trials: Max hyperparameter combinations
        cv_epochs: Epochs per CV fold
        final_epochs: Epochs for final model
        
    Returns:
        Results dictionary with best model, parameters, and CV results
    """
    
    # Stage 1: Hyperparameter optimization with CV
    optimizer = GNNHyperparameterOptimizer(
        cv_data=cv_data,
        search_space=search_space,
        device=device,
        random_state=random_state,
        max_trials=max_trials,
        cv_epochs=cv_epochs
    )
    
    best_params, best_score, cv_results = optimizer.optimize()
    
    # Stage 2: Train final model with best parameters on full training data
    log.info("Training final GNN model with best parameters...")
    
    final_model = UnifiedGNNLinkPredictor(
        model_type=best_params['model_type'],
        hidden_dims=best_params['hidden_dims'],
        node_feature_type=best_params['node_feature_type'],
        learning_rate=best_params['learning_rate'],
        dropout=best_params['dropout'],
        node_feature_dim=best_params.get('node_feature_dim', 64),
        device=device,
        random_state=random_state
    )
    
    # Train on full training graph
    training_graph = cv_data['full_training_graph']
    training_history = final_model.fit(
        training_graph=training_graph,
        epochs=final_epochs,
        log_every=max(1, final_epochs // 20),
        early_stopping=True,
        patience=30
    )
    
    return {
        'best_model': final_model,
        'best_params': best_params,
        'best_cv_score': best_score,
        'cv_results': cv_results,
        'training_history': training_history,
        'model_type': f"GNN_{best_params['model_type'].upper()}"
    }