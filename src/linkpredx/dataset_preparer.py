"""
Dataset Preparer for Link Prediction Tasks (dict-return version)

Scenarios
---------
1) simulation : Hold out edges from G to make test; train on G' and G''.
2) discovery  : Score non-edges in G; train via connected subset.
3) specific   : Score user-provided pairs; train via connected subset.
"""

from __future__ import annotations

import time
import json
import os
import random
from math import ceil
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import networkx as nx
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, KFold

from config import DatasetConfig  # alias of GeneratorConfig
from edge_sampler import sample_negative_edges
from graph_processor import build_connected_observed_edges
from pair_dataset import build_pairwise_feature_dataset


@dataclass
class _SplitArtifacts:
    """Internal holder during scenario building."""
    df_tr_top: pd.DataFrame
    df_ho_top: pd.DataFrame
    G_obs: nx.Graph              # observed graph (for features of holdout)
    G_tr: nx.Graph               # training graph (for features of training)
    ho_missing: List[Tuple[Any, Any]]  # Y  (E \ E')
    tr_missing: List[Tuple[Any, Any]]  # Y' (E' \ E'')


class LinkPredictionDatasetPreparer:
    def __init__(self, G: nx.Graph, config: DatasetConfig, rng: Optional[random.Random] = None):
        self.G: nx.Graph = G.copy()
        self.cfg = config
        self.rng = rng or random.Random(self.cfg.random_state)

        # filled per scenario to expose in results
        self.G_obs: Optional[nx.Graph] = None
        self.G_tr: Optional[nx.Graph] = None
        self.ho_missing: List[Tuple[Any, Any]] = []
        self.tr_missing: List[Tuple[Any, Any]] = []

        # stage timings
        self._last_holdout_time: Optional[float] = None
        self._last_train_time: Optional[float] = None

    # ------------------------ Public API ------------------------ #
    def prepare_dataset(
        self,
        scenario: Literal["simulation", "discovery", "specific"],
        predict_edges: Optional[List[Tuple[Any, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build datasets and return a comprehensive results dict.
        """
        total_t0 = time.perf_counter()
        stage_times: Dict[str, float] = {}

        # ---- edge splitting (graphs + positive sets) ----
        t0 = time.perf_counter()
        if scenario == "simulation":
            split = self._build_simulation_split()
        elif scenario == "discovery":
            split = self._build_discovery_split()
        elif scenario == "specific":
            split = self._build_specific_split(predict_edges=predict_edges)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        stage_times["edge_splitting"] = time.perf_counter() - t0

        # expose for packaging
        self.G_obs = split.G_obs
        self.G_tr = split.G_tr
        self.ho_missing = split.ho_missing
        self.tr_missing = split.tr_missing

        # ---- CV folds ----
        t0 = time.perf_counter()
        folds = self._make_cv_folds(split.df_tr_top)
        stage_times["cv_folds"] = time.perf_counter() - t0

        # ---- package final results ----
        results = self._package_results(
            df_tr_top=split.df_tr_top,
            df_ho_top=split.df_ho_top,
            folds=folds,
            stage_times=stage_times,
            scenario=scenario,
        )
        results["metadata"]["execution_time"] = time.perf_counter() - total_t0
        return results

    # ----------------------- Split builders --------------------- #
    def _build_simulation_split(self) -> _SplitArtifacts:
        # Build observed G' (connectivity-preserving)
        n_obs = max(1, int(self.cfg.obs_frac * self.G.number_of_edges()))
        E_obs = build_connected_observed_edges(self.G, n_obs=n_obs, rng=self.rng)
        G_obs = nx.Graph()
        G_obs.add_nodes_from(self.G.nodes())
        G_obs.add_edges_from(E_obs)

        # Holdout positives Y = E \ E'
        allE = set(map(tuple, map(sorted, self.G.edges())))
        E_obs_set = set(map(tuple, map(sorted, E_obs)))
        ho_pos = list(allE - E_obs_set)

        # ---- Holdout negatives: **controlled** by sim_ho_* knobs (NEW) ----
        t0 = time.perf_counter()
        n_ho_neg = self._compute_sim_ho_negatives_count(
            n_pos=len(ho_pos),
            graph_for_theoretical=self.G
        )
        ho_neg = sample_negative_edges(
            G=self.G, num_samples=n_ho_neg, rng=self.rng, avoid_edges=self.G.edges()
        )
        # Holdout features computed on G'
        df_ho_top = build_pairwise_feature_dataset(
            G=G_obs,
            pos_edges=ho_pos,
            neg_edges=ho_neg,
            feature_config_path=self.cfg.feature_config_path,
            scenario="simulation",
            source="holdout_on_Gprime",
        )
        holdout_time = time.perf_counter() - t0

        # Training support graph G'' from E' (connectivity-preserving)
        n_train = max(1, int(self.cfg.train_frac * len(E_obs)))
        E_train = build_connected_observed_edges(G_obs, n_obs=n_train, rng=self.rng)
        G_tr = nx.Graph()
        G_tr.add_nodes_from(G_obs.nodes())
        G_tr.add_edges_from(E_train)

        # Training positives Y' = E' \ E''
        E_train_set = set(map(tuple, map(sorted, E_train)))
        tr_pos = list(E_obs_set - E_train_set)

        # Training negatives from G' (avoid E'); COUNT controlled by config
        t0 = time.perf_counter()
        n_tr_neg = self._compute_train_negatives_count(
            n_pos=len(tr_pos),
            graph_for_theoretical=G_obs
        )
        tr_neg = sample_negative_edges(
            G=G_obs, num_samples=n_tr_neg, rng=self.rng, avoid_edges=E_obs
        )
        # Training features computed on G''
        df_tr_top = build_pairwise_feature_dataset(
            G=G_tr,
            pos_edges=tr_pos,
            neg_edges=tr_neg,
            feature_config_path=self.cfg.feature_config_path,
            scenario="simulation",
            source="train_on_Gdprime",
        )
        # Optional class balancing (train only)
        df_tr_top = self._maybe_oversample(df_tr_top)
        train_time = time.perf_counter() - t0

        # record missing sets
        ho_missing = ho_pos
        tr_missing = tr_pos

        # cache stage timings for metadata (we’ll pass these up)
        self._last_holdout_time = holdout_time
        self._last_train_time = train_time

        return _SplitArtifacts(
            df_tr_top=df_tr_top,
            df_ho_top=df_ho_top,
            G_obs=G_obs,
            G_tr=G_tr,
            ho_missing=ho_missing,
            tr_missing=tr_missing,
        )

    def _build_discovery_split(self) -> _SplitArtifacts:
        # Test = cap non-edges from G
        cap = int(self.cfg.max_negative_samples)
        t0 = time.perf_counter()
        non_edges = sample_negative_edges(
            G=self.G, num_samples=cap, rng=self.rng, avoid_edges=self.G.edges()
        )
        df_ho_top = build_pairwise_feature_dataset(
            G=self.G,
            pos_edges=[],
            neg_edges=non_edges,
            feature_config_path=self.cfg.feature_config_path,
            scenario="discovery",
            source="predict_non_edges_on_G",
        )
        holdout_time = time.perf_counter() - t0

        # Training via observed G' (connectivity-preserving)
        n_obs = max(1, int(self.cfg.obs_frac * self.G.number_of_edges()))
        E_obs = build_connected_observed_edges(self.G, n_obs=n_obs, rng=self.rng)
        G_obs = nx.Graph()
        G_obs.add_nodes_from(self.G.nodes())
        G_obs.add_edges_from(E_obs)

        # Positives E \ E', negatives from G (count controlled)
        t0 = time.perf_counter()
        allE = set(map(tuple, map(sorted, self.G.edges())))
        E_obs_set = set(map(tuple, map(sorted, E_obs)))
        tr_pos = list(allE - E_obs_set)

        n_tr_neg = self._compute_train_negatives_count(
            n_pos=len(tr_pos),
            graph_for_theoretical=self.G
        )
        tr_neg = sample_negative_edges(
            G=self.G, num_samples=n_tr_neg, rng=self.rng, avoid_edges=self.G.edges()
        )

        df_tr_top = build_pairwise_feature_dataset(
            G=G_obs,
            pos_edges=tr_pos,
            neg_edges=tr_neg,
            feature_config_path=self.cfg.feature_config_path,
            scenario="discovery",
            source="train_on_Gprime",
        )
        df_tr_top = self._maybe_oversample(df_tr_top)
        train_time = time.perf_counter() - t0

        self._last_holdout_time = holdout_time
        self._last_train_time = train_time

        return _SplitArtifacts(
            df_tr_top=df_tr_top,
            df_ho_top=df_ho_top,
            G_obs=G_obs,      # observed used for training features
            G_tr=G_obs,       # same as training graph in this scenario
            ho_missing=[],    # no positives in test for discovery
            tr_missing=tr_pos,
        )

    def _build_specific_split(self, predict_edges: Optional[List[Tuple[Any, Any]]]) -> _SplitArtifacts:
        if not predict_edges:
            raise ValueError("predict_edges must be provided for 'specific' scenario")

        # Test = given pairs; features on G
        t0 = time.perf_counter()
        df_ho_top = build_pairwise_feature_dataset(
            G=self.G,
            pos_edges=[],
            neg_edges=predict_edges,   # unlabeled scoring set
            feature_config_path=self.cfg.feature_config_path,
            scenario="specific",
            source="user_pairs_on_G",
        )
        holdout_time = time.perf_counter() - t0

        # Training support graph = connectivity-preserving subset of G
        n_train = max(1, int(self.cfg.train_frac * self.G.number_of_edges()))
        E_train = build_connected_observed_edges(self.G, n_obs=n_train, rng=self.rng)
        G_tr = nx.Graph()
        G_tr.add_nodes_from(self.G.nodes())
        G_tr.add_edges_from(E_train)

        # Positives E \ E_train ; negatives from G (count controlled)
        t0 = time.perf_counter()
        allE = set(map(tuple, map(sorted, self.G.edges())))
        E_train_set = set(map(tuple, map(sorted, E_train)))
        tr_pos = list(allE - E_train_set)

        n_tr_neg = self._compute_train_negatives_count(
            n_pos=len(tr_pos),
            graph_for_theoretical=self.G
        )
        tr_neg = sample_negative_edges(
            G=self.G, num_samples=n_tr_neg, rng=self.rng, avoid_edges=self.G.edges()
        )
        df_tr_top = build_pairwise_feature_dataset(
            G=G_tr,
            pos_edges=tr_pos,
            neg_edges=tr_neg,
            feature_config_path=self.cfg.feature_config_path,
            scenario="specific",
            source="train_on_Gtrain",
        )
        df_tr_top = self._maybe_oversample(df_tr_top)
        train_time = time.perf_counter() - t0

        self._last_holdout_time = holdout_time
        self._last_train_time = train_time

        return _SplitArtifacts(
            df_tr_top=df_tr_top,
            df_ho_top=df_ho_top,
            G_obs=self.G,   # test features computed on original G
            G_tr=G_tr,
            ho_missing=[],  # test is unlabeled user pairs
            tr_missing=tr_pos,
        )

    # ------------------------ Helpers --------------------------- #
    def _theoretical_max_non_edges(self, G: nx.Graph) -> int:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        return max(0, (n * (n - 1)) // 2 - m)

    def _compute_train_negatives_count(self, n_pos: int, graph_for_theoretical: nx.Graph) -> int:
        """Count negatives for TRAIN (uses train_* knobs)."""
        if n_pos <= 0:
            return 0
        # absolute override
        if self.cfg.train_neg_samples is not None:
            n = int(self.cfg.train_neg_samples)
        else:
            n = int(ceil(float(self.cfg.train_neg_per_pos) * n_pos))
        if self.cfg.train_neg_max_cap is not None:
            n = min(n, int(self.cfg.train_neg_max_cap))
        n = min(n, self._theoretical_max_non_edges(graph_for_theoretical))
        return max(0, n)

    def _compute_sim_ho_negatives_count(self, n_pos: int, graph_for_theoretical: nx.Graph) -> int:
        """Count negatives for SIMULATION holdout (uses sim_ho_* knobs)."""
        if n_pos <= 0:
            return 0
        # absolute override
        if self.cfg.sim_ho_neg_samples is not None:
            n = int(self.cfg.sim_ho_neg_samples)
        else:
            n = int(ceil(float(self.cfg.sim_ho_neg_per_pos) * n_pos))
        if self.cfg.sim_ho_neg_max_cap is not None:
            n = min(n, int(self.cfg.sim_ho_neg_max_cap))
        n = min(n, self._theoretical_max_non_edges(graph_for_theoretical))
        return max(0, n)

    def _maybe_oversample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.balance_classes and "label" in df.columns:
            X = df.drop(columns=["label"])
            y = df["label"]
            if len(set(y)) > 1:
                ros = RandomOverSampler(random_state=self.cfg.random_state)
                X_res, y_res = ros.fit_resample(X, y)
                df = pd.concat([X_res, y_res.rename("label")], axis=1).reset_index(drop=True)
        return df

    def _make_cv_folds(self, df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
        """Return a list of (train_idx, val_idx) folds."""
        n_splits = int(self.cfg.n_folds)
        if n_splits < 2 or len(df) < 2 * n_splits:
            idx = list(range(len(df)))
            return [(idx, idx)]

        if "label" in df.columns and self.cfg.use_stratified_cv and len(set(df["label"])) > 1:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.cfg.random_state)
            y = df["label"].values
            folds = [(list(tr), list(te)) for tr, te in kf.split(df, y)]
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.cfg.random_state)
            folds = [(list(tr), list(te)) for tr, te in kf.split(df)]
        return folds

    def _package_results(
        self,
        df_tr_top: pd.DataFrame,
        df_ho_top: pd.DataFrame,
        folds: List[Tuple[List[int], List[int]]],
        stage_times: Dict[str, float],
        scenario: str,
    ) -> Dict[str, Any]:
        # Feature column list
        feature_cols = [c for c in df_tr_top.columns if c not in {"u", "v", "label", "scenario", "source"}]

        # Dataset stats
        def _stats(df: pd.DataFrame) -> Dict[str, Any]:
            pos = int(df["label"].sum()) if "label" in df.columns else None
            neg = int(len(df) - pos) if pos is not None else None
            return {"rows": len(df), "pos": pos, "neg": neg}

        # Graph connectivity
        connectivity_analysis = {
            "original_is_connected": nx.is_connected(self.G) if self.G.number_of_nodes() > 0 else True,
            "original_connected_components": nx.number_connected_components(self.G) if self.G.number_of_nodes() > 0 else 0,
        }

        # Config block (only known fields)
        cfg_block = {
            "obs_frac": self.cfg.obs_frac,
            "train_frac": self.cfg.train_frac,
            "n_folds": self.cfg.n_folds,
            "use_stratified_cv": self.cfg.use_stratified_cv,
            "balance_classes": self.cfg.balance_classes,
            "random_state": self.cfg.random_state,
            "max_negative_samples": getattr(self.cfg, "max_negative_samples", None),
            "feature_config_path": getattr(self.cfg, "feature_config_path", None),
            # TRAIN knobs
            "train_neg_samples": getattr(self.cfg, "train_neg_samples", None),
            "train_neg_per_pos": getattr(self.cfg, "train_neg_per_pos", None),
            "train_neg_max_cap": getattr(self.cfg, "train_neg_max_cap", None),
            # SIMULATION holdout knobs (NEW)
            "sim_ho_neg_samples": getattr(self.cfg, "sim_ho_neg_samples", None),
            "sim_ho_neg_per_pos": getattr(self.cfg, "sim_ho_neg_per_pos", None),
            "sim_ho_neg_max_cap": getattr(self.cfg, "sim_ho_neg_max_cap", None),
        }

        # Build results dict
        results: Dict[str, Any] = {
            # Primary datasets
            "df_tr_top": df_tr_top,
            "df_ho_top": df_ho_top,
            "cv_folds": folds,

            # Graph objects for further analysis
            "graphs": {
                "original": self.G,
                "observed": self.G_obs,
                "training": self.G_tr,
            },

            # Edge lists for reconstruction/analysis
            "edge_lists": {
                "training_edges": list(self.G_tr.edges()) if self.G_tr is not None else [],
                "observed_edges": list(self.G_obs.edges()) if self.G_obs is not None else [],
                "holdout_missing": self.ho_missing,
                "training_missing": self.tr_missing,
            },

            # Feature information
            "feature_info": {
                "n_features": len(feature_cols),
                "feature_names": feature_cols[:20] + ["..."] if len(feature_cols) > 20 else feature_cols,
                "computation_graph": "training" if self.G_tr is not None else "original",
            },

            # Comprehensive metadata
            "metadata": {
                "execution_time": None,  # filled by caller after packaging
                "stage_times": {
                    "edge_splitting": stage_times.get("edge_splitting", None),
                    "training_dataset": getattr(self, "_last_train_time", None),
                    "holdout_dataset": getattr(self, "_last_holdout_time", None),
                    "cv_folds": stage_times.get("cv_folds", None),
                },
                "configuration": cfg_block,
                "graph_properties": connectivity_analysis,
                "dataset_statistics": {
                    "training": _stats(df_tr_top),
                    "holdout": _stats(df_ho_top),
                },
                "scenario": scenario,
            },
        }
        return results


# ------------------------------ persistence ------------------------------ #

def save_results(results: Dict[str, Any], out_dir: str) -> None:
    """
    Save results to a directory:
      - train.csv, test.csv
      - graphs/*.gpickle
      - metadata.json (no DataFrame dumps inside)
    """
    os.makedirs(out_dir, exist_ok=True)

    # DataFrames
    results["df_tr_top"].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    results["df_ho_top"].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # Graphs
    graphs = results.get("graphs", {})
    graphs_dir = os.path.join(out_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    for name in ("original", "observed", "training"):
        G = graphs.get(name)
        if G is not None:
            nx.write_gpickle(G, os.path.join(graphs_dir, f"{name}.gpickle"))

    # Metadata (JSON)
    meta = results.get("metadata", {})
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
