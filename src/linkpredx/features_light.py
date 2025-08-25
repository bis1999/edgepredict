"""
Lightweight graph feature extraction (no Torch/SciPy/igraph/Numba).

- YAML-driven feature toggles (optional)
- NetworkX-based node & pair features
- Same public API: compute_all_features(G, uv_df, feature_config_path)
- Safe fallbacks & clear errors for missing columns
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Optional YAML
try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ----------------------------- Toggles -----------------------------

DEFAULT_TOGGLES: Dict[str, bool] = {
    # node-level
    "triangles": True,
    "clustering": True,
    "degree_centrality": True,
    # pair-level
    "common_neighbors": True,
    "jaccard_coefficient": True,
    "preferential_attachment": True,
    "adamic_adar": False,          # enable if you want these (slower on large graphs)
    "resource_allocation": False,  # enable if you want these
    "lhn_index": False,            # enable if you want these
    # convenience copies
    "deg_cent_u": True,
    "deg_cent_v": True,
    # SVD family intentionally omitted in the light backend
    "svd_dot": False, "svd_mean": False,
    "lra": False, "dlra": False, "mlra": False,
    "lra_approx": False, "dlra_approx": False, "mlra_approx": False,
    # igraph-only things intentionally off
    "pagerank": False, "closeness": False, "betweenness": False, "eigenvector": False,
    "shortest_paths": False,
}

def _load_feature_toggles_from_yaml(path: Optional[str]) -> Dict[str, bool]:
    if not path or not YAML_AVAILABLE:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            if isinstance(raw, dict):
                return {str(k): bool(v) for k, v in raw.items()}
    except FileNotFoundError:
        logger.info(f"Feature YAML not found at {path!r}; using defaults.")
    except Exception as e:
        logger.warning(f"Failed to read {path!r}: {e}; using defaults.")
    return {}

# ------------------------ Feature Extractor ------------------------

class GraphFeatureExtractor:
    """
    Lightweight extractor: NetworkX + NumPy only.

    Notes
    -----
    - SVD features and igraph-based centralities are not computed here.
    - Pair features rely on NetworkX (generators or simple set ops).
    """

    def __init__(self, feature_toggles: Optional[Dict[str, bool]] = None):
        toggles = DEFAULT_TOGGLES.copy()
        if feature_toggles:
            toggles.update({str(k): bool(v) for k, v in feature_toggles.items()})
        self.toggles = toggles

    def _on(self, name: str) -> bool:
        return bool(self.toggles.get(name, False))

    # -------------------- node-level features --------------------

    def _triangles(self, G: nx.Graph) -> Dict[Any, float]:
        return {k: float(v) for k, v in nx.triangles(G).items()}

    def _clustering(self, G: nx.Graph) -> Dict[Any, float]:
        return {k: float(v) for k, v in nx.clustering(G).items()}

    def _degree_centrality(self, G: nx.Graph) -> Dict[Any, float]:
        # normalized degree centrality: deg / max_deg
        deg = dict(G.degree())
        maxd = max(deg.values()) if deg else 1
        scale = float(maxd) if maxd > 0 else 1.0
        return {n: float(d) / scale for n, d in deg.items()}

    # -------------------- pair-level features --------------------

    def _common_neighbors(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        adj = {n: set(G.neighbors(n)) for n in G.nodes()}
        return [float(len(adj.get(u, set()) & adj.get(v, set()))) for u, v in pairs]

    def _jaccard(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        out = [0.0] * len(pairs)
        valid = [(u, v) for (u, v) in pairs if u in G and v in G]
        if not valid:
            return out
        idx = {e: i for i, e in enumerate(pairs)}
        for u, v, s in nx.jaccard_coefficient(G, valid):
            out[idx[(u, v)]] = float(s)
        return out

    def _preferential_attachment(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        deg = dict(G.degree())
        return [float(deg.get(u, 0) * deg.get(v, 0)) for u, v in pairs]

    def _adamic_adar(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        out = [0.0] * len(pairs)
        valid = [(u, v) for (u, v) in pairs if u in G and v in G]
        if not valid:
            return out
        idx = {e: i for i, e in enumerate(pairs)}
        for u, v, s in nx.adamic_adar_index(G, valid):
            out[idx[(u, v)]] = float(s)
        return out

    def _resource_allocation(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        out = [0.0] * len(pairs)
        valid = [(u, v) for (u, v) in pairs if u in G and v in G]
        if not valid:
            return out
        idx = {e: i for i, e in enumerate(pairs)}
        for u, v, s in nx.resource_allocation_index(G, valid):
            out[idx[(u, v)]] = float(s)
        return out

    def _lhn_index(self, G: nx.Graph, pairs: List[Tuple[Any, Any]]) -> List[float]:
        # LHN(u,v) = |Γ(u) ∩ Γ(v)| / (deg(u) * deg(v))
        adj = {n: set(G.neighbors(n)) for n in G.nodes()}
        deg = dict(G.degree())
        out: List[float] = []
        for u, v in pairs:
            inter = len(adj.get(u, set()) & adj.get(v, set()))
            denom = deg.get(u, 0) * deg.get(v, 0)
            out.append(float(inter) / float(denom) if denom > 0 else 0.0)
        return out

    # -------------------- main entrypoint --------------------

    def compute_feature_df(self, uv_df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
        """
        Given a DataFrame with columns ['u','v'], return uv_df with added feature columns.
        """
        if not {"u", "v"}.issubset(uv_df.columns):
            raise ValueError("uv_df must contain columns ['u','v'].")

        df = uv_df.copy()
        pairs = list(zip(df["u"].tolist(), df["v"].tolist()))

        # node maps
        if self._on("triangles"):
            tri = self._triangles(G)
            df["triangles_u"] = df["u"].map(tri).astype(float)
            df["triangles_v"] = df["v"].map(tri).astype(float)

        if self._on("clustering"):
            clu = self._clustering(G)
            df["clustering_u"] = df["u"].map(clu).astype(float)
            df["clustering_v"] = df["v"].map(clu).astype(float)

        if self._on("degree_centrality"):
            degc = self._degree_centrality(G)
            df["degree_centrality_u"] = df["u"].map(degc).astype(float)
            df["degree_centrality_v"] = df["v"].map(degc).astype(float)

        # pair features
        if self._on("common_neighbors"):
            df["common_neighbors"] = self._common_neighbors(G, pairs)

        if self._on("jaccard_coefficient"):
            df["jaccard_coefficient"] = self._jaccard(G, pairs)

        if self._on("preferential_attachment"):
            df["preferential_attachment"] = self._preferential_attachment(G, pairs)

        if self._on("adamic_adar"):
            df["adamic_adar"] = self._adamic_adar(G, pairs)

        if self._on("resource_allocation"):
            df["resource_allocation"] = self._resource_allocation(G, pairs)

        if self._on("lhn_index"):
            df["lhn_index"] = self._lhn_index(G, pairs)

        # convenience duplicates for normalized degree centrality
        if self._on("deg_cent_u"):
            df["deg_cent_u"] = df.get("degree_centrality_u", 0.0)
        if self._on("deg_cent_v"):
            df["deg_cent_v"] = df.get("degree_centrality_v", 0.0)

        return df

# ---------------- convenience API (stable for the pipeline) ----------------

def compute_all_features(
    G: nx.Graph,
    uv_df: pd.DataFrame,
    feature_config_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Drop-in replacement for the heavy backend.
    - Reads toggles from YAML (if provided),
    - Computes a basic but useful set of features,
    - Returns a dataframe aligned with uv_df rows.
    """
    toggles = _load_feature_toggles_from_yaml(feature_config_path)
    extractor = GraphFeatureExtractor(feature_toggles=toggles)
    return extractor.compute_feature_df(uv_df, G)
