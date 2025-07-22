"""
linkpredx.generator
===================
Creates balanced (+/–) training and hold-out edge datasets, attaches
topological features, and returns stratified 5-fold CV splits.

Public class
------------
LinkPredictionDataGenerator
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

from .logger import get_logger
from .features import compute_all_features

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    neg_sample_strategy: str | int | float = "equal"   # "equal", 0.5, 100, …
    obs_frac: float = 0.8                              # fraction of edges kept “observed”
    train_frac: float = 0.8                            # fraction of observed used for training
    random_state: int = 42
    k_svd: int = 50                                    # SVD rank


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class LinkPredictionDataGenerator:
    """
    Prepare train / hold-out datasets for link prediction.

    Parameters
    ----------
    edge_list : list[tuple[int, int]]
        Undirected edges of the *original* network (no self-loops).
    cfg : GeneratorConfig
        Controls negative sampling, split ratios, and randomness.

    Methods
    -------
    generate_dataset()
        Returns a dict with CV folds, full feature frames, and edge lists.
    """

    # ---------------------------------------------------------------------

    def __init__(self, edge_list: List[Tuple[int, int]], cfg: GeneratorConfig = GeneratorConfig()):
        self.cfg = cfg
        rng = random.Random(cfg.random_state)

        # Normalise & deduplicate
        edges = {tuple(sorted(e)) for e in edge_list if e[0] != e[1]}
        if not edges:
            raise ValueError("edge_list empty after sanitisation")

        self.G = nx.Graph(list(edges))
        log.info("Graph: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())

        # Bind RNG for later negative sampling
        self._rng = rng

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def generate_dataset(self) -> Dict[str, Any]:
        """
        Build features and cross-validation splits.

        Returns
        -------
        dict
            Keys:
            • X_train_cv, y_train_cv, X_test_cv, y_test_cv – 5-fold dicts
            • df_ho_top, df_tr_top                        – full feature frames
            • Gho_edges, Gtr_edges                        – edge lists for bookkeeping
        """
        self._split_edges()  # populates self.G_obs / self.G_tr and missing edge lists

        df_ho = self._make_samples_df(self.G_obs, self.ho_missing)
        df_tr = self._make_samples_df(self.G_tr,  self.tr_missing)

        df_ho_top = compute_all_features(self.G_obs, df_ho, k_svd=self.cfg.k_svd)
        df_tr_top = compute_all_features(self.G_tr,  df_tr, k_svd=self.cfg.k_svd)

        cv = self._stratified_cv(df_tr_top)

        return {
            **cv,
            "df_ho_top": df_ho_top,
            "df_tr_top": df_tr_top,
            "Gho_edges": list(self.G_obs.edges()),
            "Gtr_edges": list(self.G_tr.edges()),
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _split_edges(self):
        """Create observed/train subgraphs and record missing edges."""
        edges = list(self.G.edges())
        self._rng.shuffle(edges)

        # Observe 80 % (default) of edges
        n_obs = int(len(edges) * self.cfg.obs_frac)
        obs_edges = set(edges[:n_obs])
        ho_missing = set(edges[n_obs:])

        # Training graph uses 80 % of observed edges
        obs_edges_list = list(obs_edges)
        self._rng.shuffle(obs_edges_list)
        n_tr = int(len(obs_edges_list) * self.cfg.train_frac)
        tr_edges = set(obs_edges_list[:n_tr])
        tr_missing = set(obs_edges_list[n_tr:])

        # Build NetworkX sub-graphs
        self.G_obs = nx.Graph(); self.G_obs.add_nodes_from(self.G.nodes()); self.G_obs.add_edges_from(obs_edges)
        self.G_tr  = nx.Graph(); self.G_tr.add_nodes_from(self.G.nodes()); self.G_tr.add_edges_from(tr_edges)

        self.ho_missing, self.tr_missing = list(ho_missing), list(tr_missing)

        log.info(
            "Observed=%d, Train=%d (HO missing=%d, TR missing=%d)",
            len(obs_edges), len(tr_edges), len(ho_missing), len(tr_missing)
        )

    # ---------------------------------------------------------------------

    def _make_samples_df(self, G_sub: nx.Graph, positives: List[Tuple[int, int]]) -> pd.DataFrame:
        """Return a DataFrame with +/– samples for *one* graph (train or HO)."""
        nodes = list(G_sub.nodes())
        forbidden = {tuple(sorted(e)) for e in itertools.chain(G_sub.edges(), positives)}

        # Determine #negatives
        n_pos = len(positives)
        k = self.cfg.neg_sample_strategy
        n_neg = (
            n_pos if k == "equal"
            else int(n_pos * k) if isinstance(k, float)
            else int(k) if isinstance(k, int)
            else None
        )
        if n_neg is None:
            raise ValueError(f"Invalid neg_sample_strategy {k}")

        max_possible = len(nodes) * (len(nodes) - 1) // 2 - len(forbidden)
        n_neg = min(n_neg, max_possible)

        negatives = self._sample_negatives(nodes, forbidden, n_neg)

        df = pd.DataFrame({"edge": positives + negatives})
        df["label"] = (df["edge"].isin(positives)).astype(int)
        df[["u", "v"]] = pd.DataFrame(df["edge"].tolist(), index=df.index)
        return df

    # ---------------------------------------------------------------------

    def _sample_negatives(
        self,
        nodes: List[int],
        forbidden: set[Tuple[int, int]],
        n_neg: int,
    ) -> List[Tuple[int, int]]:
        """Uniform negative edge sampling avoiding `forbidden`."""
        negs: set[Tuple[int, int]] = set()
        while len(negs) < n_neg:
            u, v = self._rng.sample(nodes, 2)
            e = (u, v) if u < v else (v, u)
            if e not in forbidden and e not in negs:
                negs.add(e)
        return list(negs)

    # ---------------------------------------------------------------------

    def _stratified_cv(self, df_tr_top: pd.DataFrame):
        """Create 5-fold stratified CV splits with class balancing."""
        X = df_tr_top.drop(columns=["label"])
        y = df_tr_top["label"].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.random_state)
        Xtrain, ytrain, Xtest, ytest = {}, {}, {}, {}

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
            ros = RandomOverSampler(random_state=self.cfg.random_state)
            Xtr, ytr = ros.fit_resample(X.iloc[tr_idx], y[tr_idx])
            Xte, yte = X.iloc[te_idx], y[te_idx]

            Xtrain[fold], ytrain[fold] = Xtr, ytr
            Xtest[fold],  ytest[fold]  = Xte, yte

        return {
            "X_train_cv": Xtrain,
            "y_train_cv": ytrain,
            "X_test_cv": Xtest,
            "y_test_cv": ytest,
        }
