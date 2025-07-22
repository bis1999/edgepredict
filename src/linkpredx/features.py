"""
linkpredx.features
==================
Helpers to compute node-level, pairwise, and SVD-based link-prediction
features for a given NetworkX graph.

Functions
---------
compute_all_features(G, df, k_svd=50)
    Returns `df` augmented with a rich set of topological features.

Internal helpers (prefixed with “_”) are not exported.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from .logger import get_logger

log = get_logger(__name__)

__all__ = [
    "compute_all_features",
]

# ---------------------------------------------------------------------------
# Node-level features
# ---------------------------------------------------------------------------


def _node_level(G: nx.Graph) -> dict[str, dict[int, float]]:
    """Return a dict mapping feature name → {node: value}."""
    log.info("Computing node-level features …")
    return {
        "triangles": nx.triangles(G),
        "pagerank": nx.pagerank(G),
        "clustering": nx.clustering(G),
        "avg_neigh_deg": nx.average_neighbor_degree(G),
        "deg_cent": nx.degree_centrality(G),
        "betw_cent": nx.betweenness_centrality(G, k=100, seed=42),
        "eig_cent": nx.eigenvector_centrality(G, max_iter=1000),
        "katz_cent": nx.katz_centrality(G, alpha=0.005, beta=1.0),
    }


# ---------------------------------------------------------------------------
# Pairwise (edge) features
# ---------------------------------------------------------------------------


def _pairwise(G: nx.Graph, edges: list[tuple[int, int]]) -> dict[str, list[float]]:
    """Return dict mapping pairwise feature name → list aligned with `edges`."""
    log.info("Computing pairwise features …")

    cn = [len(list(nx.common_neighbors(G, u, v))) for u, v in edges]
    sp = [
        nx.shortest_path_length(G, u, v) if nx.has_path(G, u, v) else np.inf
        for u, v in edges
    ]
    pa = [G.degree[u] * G.degree[v] for u, v in edges]
    jc = [score for *_, score in nx.jaccard_coefficient(G, edges)]
    aa = [score for *_, score in nx.adamic_adar_index(G, edges)]
    ra = [score for *_, score in nx.resource_allocation_index(G, edges)]

    # Leicht–Holme–Newman index (LHN)
    lhn = [
        len(set(G[u]) & set(G[v])) / (G.degree[u] * G.degree[v])
        if G.degree[u] * G.degree[v]
        else 0
        for u, v in edges
    ]

    return {
        "cn": cn,
        "sp": sp,
        "pa": pa,
        "jc": jc,
        "aa": aa,
        "ra": ra,
        "lhn": lhn,
    }


# ---------------------------------------------------------------------------
# Low-rank / SVD features
# ---------------------------------------------------------------------------


def _svd_features(
    G: nx.Graph, edges: list[tuple[int, int]], k: int = 50
) -> dict[str, list[float]]:
    """Return truncated-SVD features (k components)."""
    log.info("Computing truncated SVD features (k=%d) …", k)

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    U, s, Vt = svds(A, k=k)

    S = np.diag(s)
    Â = U @ S @ Vt
    latent_U = U @ np.sqrt(S)
    latent_V = Vt.T @ np.sqrt(S)

    neigh = {n: set(G.neighbors(n)) for n in G.nodes()}
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731

    svd_dot = [float(np.dot(latent_U[u], latent_V[v])) for u, v in edges]
    svd_mean = [
        mean(np.dot(latent_U[u], latent_V[n]) for n in neigh.get(v, []))
        for u, v in edges
    ]
    lra = [float(Â[u, v]) for u, v in edges]
    dlra = [float(np.inner(Â[u, :], Â[:, v])) for u, v in edges]
    mlra = [mean(Â[u, list(G[v])]) if G[v] else 0.0 for u, v in edges]

    return {
        "svd_dot": svd_dot,
        "svd_mean": svd_mean,
        "lra": lra,
        "dlra": dlra,
        "mlra": mlra,
    }


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compute_all_features(
    G: nx.Graph, df: pd.DataFrame, k_svd: int = 50
) -> pd.DataFrame:
    """
    Enrich the edge DataFrame with node-level, pairwise, and SVD features.

    Parameters
    ----------
    G : nx.Graph
        Graph containing *observed* edges (for shortest-path calculations).
    df : pd.DataFrame
        Must contain columns ``edge`` (tuple), ``u``, ``v``, and ``label``.
    k_svd : int
        Number of singular vectors to keep (default 50).

    Returns
    -------
    pd.DataFrame
        Original columns plus ~30 engineered feature columns.
    """
    edges = df["edge"].tolist()

    node_f = _node_level(G)
    pair_f = _pairwise(G, edges)
    svd_f = _svd_features(G, edges, k=k_svd)

    log.info("Assembling feature frame …")
    # Flatten node-feature dict into u/v-specific columns
    node_cols = {
        f"{name}_{suffix}": [node_f[name][n] for n in df[suffix]]
        for name in node_f
        for suffix in ("u", "v")
    }

    return pd.concat(
        [
            df[["u", "v", "label"]].reset_index(drop=True),
            pd.DataFrame({**pair_f, **svd_f, **node_cols}),
        ],
        axis=1,
    )
