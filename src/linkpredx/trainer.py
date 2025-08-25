"""
Unified training orchestrator for link prediction.
Fixed to work with the actual dataset_preparer output structure.
MINIMAL FIXES APPLIED - keeps your original structure with compatibility updates.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Sequence, Union
import numpy as np  # FIX 1: Added missing import
import pandas as pd
import networkx as nx

from ml_models import MLLinkPredictionTrainer, train_ml_models
from gnn_trainer import UnifiedGNNLinkPredictor

try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class LinkPredictionTrainer:
    """
    Unified trainer for link prediction supporting both ML and GNN backends.
    
    Expects workflow_result from dataset_preparer with structure:
    {
        "df_tr_top": pd.DataFrame,          # Training data
        "df_ho_top": pd.DataFrame,          # Holdout/test data  
        "cv_folds": [(train_idx, val_idx)], # CV fold indices
        "graphs": {
            "original": nx.Graph,
            "observed": nx.Graph,
            "training": nx.Graph
        },
        "feature_info": {
            "n_features": int,
            "feature_names": List[str]
        },
        "metadata": {
            "scenario": str,                # "simulation", "discovery", "specific"
            ...
        }
    }
    """
    
    def __init__(self, workflow_result: Dict[str, Any], device: str = "cpu"):
        self.workflow_result = workflow_result
        self.device = device
        
        # Validate required fields
        if "metadata" not in workflow_result:
            raise ValueError("workflow_result must contain 'metadata'")
        if "scenario" not in workflow_result["metadata"]:
            raise ValueError("workflow_result['metadata'] must contain 'scenario'")
            
        self.scenario = workflow_result["metadata"]["scenario"]
        
        # Extract main components
        self.df_tr_top = workflow_result.get("df_tr_top")
        self.df_ho_top = workflow_result.get("df_ho_top") 
        self.cv_folds = workflow_result.get("cv_folds", [])
        self.graphs = workflow_result.get("graphs", {})
        self.feature_info = workflow_result.get("feature_info", {})
        
        if self.df_tr_top is None:
            raise ValueError("workflow_result must contain 'df_tr_top'")

    def _prepare_cv_data_for_ml(self) -> Dict[str, Any]:
        """Convert dataset_preparer output to ML trainer expected format."""
        
        # Get feature columns from feature_info or infer them
        if "feature_names" in self.feature_info:
            feature_columns = [name for name in self.feature_info["feature_names"] if name != "..."]
        else:
            # Infer feature columns
            numeric_cols = self.df_tr_top.select_dtypes(include=[np.number]).columns
            reserved_cols = {"u", "v", "label", "split", "scenario", "source"}
            feature_columns = [c for c in numeric_cols if c not in reserved_cols]
        
        # Convert CV folds from index tuples to data splits
        folds = {}
        for fold_id, (train_idx, val_idx) in enumerate(self.cv_folds):
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)
            
            # Extract fold data
            X_train = self.df_tr_top.iloc[train_idx]
            y_train = self.df_tr_top.iloc[train_idx]["label"].astype(int).values
            X_val = self.df_tr_top.iloc[val_idx]
            y_val = self.df_tr_top.iloc[val_idx]["label"].astype(int).values
            
            folds[fold_id] = {
                "X_train": X_train,
                "y_train": y_train, 
                "X_test": X_val,    # ML trainer expects X_test/y_test
                "y_test": y_val
            }
        
        return {
            "cv_folds": {
                "folds": folds, 
                "feature_columns": feature_columns
            },
            "cv_dataset": self.df_tr_top
        }

    def train_ml_models(
        self,
        models: Optional[List[str]] = None,
        custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        ks: List[int] = [10, 20],
        primary_metric: str = "auc",  # FIX 2: Changed from "roc_auc" to "auc"
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Train ML models with hyperparameter search.
        """
        log.info(f"Training ML models for {self.scenario} scenario")
        
        # Prepare CV data in the format ML trainer expects
        cv_data = self._prepare_cv_data_for_ml()
        
        # Train models
        trainer = train_ml_models(
            cv_data=cv_data,
            models=models,
            custom_grids=custom_grids,
            ks=ks,
            primary_metric=primary_metric,
            scale_features=scale_features
        )
        
        results = {
            "backend": "ml",
            "scenario": self.scenario,
            "trainer": trainer,
            "cv_results": trainer.results,
            "best_summary": trainer.get_summary(),
        }
        
        # Add holdout evaluation for simulation scenario
        if self.scenario == "simulation" and self.df_ho_top is not None:
            if "label" in self.df_ho_top.columns:
                try:
                    holdout_results = trainer.predict_holdout(self.df_ho_top, ks=ks)
                    results["holdout_metrics"] = holdout_results
                    log.info(f"Holdout AUC: {holdout_results.get('roc_auc', 'N/A'):.4f}")
                except Exception as e:
                    log.warning(f"Holdout evaluation failed: {e}")
        
        # Add candidate scoring for discovery/specific scenarios
        if self.scenario in ["discovery", "specific"] and self.df_ho_top is not None:
            try:
                candidate_preds = trainer.predict_candidates(self.df_ho_top)
                results["candidate_predictions"] = candidate_preds
                log.info(f"Scored {len(candidate_preds)} candidates")
            except Exception as e:
                log.warning(f"Candidate scoring failed: {e}")
        
        return results

    def train_gnn_model(
        self,
        model_type: str = "sage",
        epochs: int = 600,
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 0.01,
        dropout: float = 0.2,
        threshold: float = 0.5,
        ks: List[int] = [10, 20]  # FIX 3: Added ks parameter
    ) -> Dict[str, Any]:
        """
        Train GNN model for link prediction.
        """
        log.info(f"Training {model_type.upper()} for {self.scenario} scenario")
        
        # Get training graph - use the appropriate graph for the scenario
        if self.scenario == "simulation":
            # For simulation, train on the training graph (G'')
            training_graph = self.graphs.get("training")
        else:
            # For discovery/specific, train on observed graph (G')
            training_graph = self.graphs.get("observed") or self.graphs.get("original")
            
        if training_graph is None:
            raise ValueError("No suitable training graph found in workflow_result")
        
        if not isinstance(training_graph, nx.Graph):
            raise ValueError("Training graph must be nx.Graph")
        
        # Initialize and train model
        model = UnifiedGNNLinkPredictor(
            model_type=model_type,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            dropout=dropout,
            device=self.device
        )
        
        model.set_training_scenario(self.scenario)
        training_history = model.fit(training_graph, epochs=epochs)
        
        results = {
            "backend": "gnn",
            "scenario": self.scenario,
            "model": model,
            "training_history": training_history,
            "model_type": model_type
        }
        
        # Handle predictions based on scenario
        if self.scenario == "simulation" and self.df_ho_top is not None:
            # Evaluate on holdout edges with labels
            test_edges = list(zip(self.df_ho_top["u"], self.df_ho_top["v"]))
            test_labels = self.df_ho_top["label"].values if "label" in self.df_ho_top.columns else None
            
            predictions_df, metrics = model.predict(
                edges_to_predict=test_edges,
                labels=test_labels,
                threshold=threshold,
                return_dataframe=True,
                return_metrics=True,
                ks=ks  # FIX 4: Added ks parameter
            )
            
            results["simulation_predictions"] = predictions_df
            results["simulation_metrics"] = metrics
            
            if metrics and metrics.get("auc") is not None:
                log.info(f"Test AUC: {metrics['auc']:.4f}")
        
        elif self.scenario in ["discovery", "specific"] and self.df_ho_top is not None:
            # Score candidate edges (no labels)
            candidate_edges = list(zip(self.df_ho_top["u"], self.df_ho_top["v"]))
            
            predictions_df, _ = model.predict(
                edges_to_predict=candidate_edges,
                labels=None,
                threshold=threshold,
                return_dataframe=True,
                return_metrics=True,
                ks=ks  # FIX 5: Added ks parameter
            )
            
            results["candidate_predictions"] = predictions_df
            log.info(f"Scored {len(predictions_df)} candidates")
        
        return results

    def run(
        self, 
        backend: str = "ml",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run training with specified backend.
        
        Args:
            backend: "ml" or "gnn"  
            **kwargs: Backend-specific arguments
            
        Returns:
            Training results dictionary
        """
        backend = backend.lower()
        
        if backend == "ml":
            return self.train_ml_models(**kwargs)
        elif backend == "gnn":
            return self.train_gnn_model(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def compare_backends(
        self,
        ml_kwargs: Optional[Dict[str, Any]] = None,
        gnn_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train and compare both ML and GNN backends.
        """
        results = {"scenario": self.scenario}
        
        # Train ML models
        try:
            log.info("Training ML models...")
            ml_results = self.train_ml_models(**(ml_kwargs or {}))
            results["ml"] = ml_results
        except Exception as e:
            log.error(f"ML training failed: {e}")
            results["ml"] = {"error": str(e)}
        
        # Train GNN model  
        try:
            log.info("Training GNN model...")
            gnn_results = self.train_gnn_model(**(gnn_kwargs or {}))
            results["gnn"] = gnn_results
        except Exception as e:
            log.error(f"GNN training failed: {e}")
            results["gnn"] = {"error": str(e)}
        
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the workflow result."""
        return {
            "scenario": self.scenario,
            "training_samples": len(self.df_tr_top) if self.df_tr_top is not None else 0,
            "holdout_samples": len(self.df_ho_top) if self.df_ho_top is not None else 0,
            "cv_folds": len(self.cv_folds),
            "n_features": self.feature_info.get("n_features", 0),
            "graphs_available": list(self.graphs.keys()),
            "has_labels_in_holdout": "label" in self.df_ho_top.columns if self.df_ho_top is not None else False
        }


# Convenience functions for backward compatibility
def train_scenario_models(
    workflow_result: Dict[str, Any],
    backend: str = "ml",
    device: str = "cpu",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train models on a workflow result from dataset_preparer.
    
    Args:
        workflow_result: Output from LinkPredictionDatasetPreparer.prepare_dataset()
        backend: "ml" or "gnn" 
        device: Device for GNN training
        **kwargs: Backend-specific arguments
        
    Returns:
        Training results
    """
    trainer = LinkPredictionTrainer(workflow_result, device=device)
    return trainer.run(backend=backend, **kwargs)


# Helper function to add graphs to workflow result for GNN training
def add_graphs_for_gnn(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure graphs are available in workflow_result for GNN training.
    The dataset_preparer should already include these, but this is a safety check.
    """
    if "graphs" not in workflow_result:
        log.warning("No graphs found in workflow_result. GNN training may fail.")
        return workflow_result
    
    graphs = workflow_result["graphs"]
    required_graphs = []
    
    scenario = workflow_result["metadata"]["scenario"]
    if scenario == "simulation":
        required_graphs = ["original", "observed", "training"]
    elif scenario in ["discovery", "specific"]:
        required_graphs = ["original", "observed"]
    
    missing_graphs = [g for g in required_graphs if g not in graphs or graphs[g] is None]
    if missing_graphs:
        log.warning(f"Missing required graphs for {scenario}: {missing_graphs}")
    
    return workflow_result