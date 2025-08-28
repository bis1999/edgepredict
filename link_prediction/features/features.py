"""
Graph feature extraction module (no caching, no dataclasses).

- No disk or in-memory caching of graph-scoped computations
- YAML-driven feature toggles (optional)
- igraph-backed centralities + NetworkX/Numba pairwise scores
- Optional SVD-based edge features (SciPy + PyTorch)
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ------------------------- optional deps (guarded) -------------------------
try:
    import igraph as ig  # type: ignore
    IG_AVAILABLE = True
except Exception:
    ig = None  # type: ignore
    IG_AVAILABLE = False

try:
    from scipy.sparse.linalg import svds  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    svds = None  # type: ignore
    SCIPY_AVAILABLE = False

try:
    import scipy.sparse as sp  # type: ignore
    SCIPY_SPARSE_AVAILABLE = True
except Exception:
    sp = None  # type: ignore
    SCIPY_SPARSE_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from numba import njit, prange  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco
    def prange(*args, **kwargs):
        return range
    NUMBA_AVAILABLE = False

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    yaml = None  # type: ignore
    YAML_AVAILABLE = False

# ------------------------------- logging -----------------------------------
logger = logging.getLogger(__name__)

# ----------------------------- feature toggles -----------------------------
def load_feature_toggles_from_yaml(path: str = "configs/feature.yaml") -> Dict[str, bool]:
    """
    Load a {feature_name: bool} mapping from YAML.

    - If file missing / malformed / PyYAML absent, return {}.
    - Non-bool values are coerced with bool(...).
    """
    if not YAML_AVAILABLE:
        logger.info("PyYAML not available; using empty feature toggles.")
        return {}

    loaded: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.info(f"Feature YAML not found at {path!r}; using empty toggles.")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read {path!r} ({e}); using empty toggles.")
        return {}

    if not isinstance(loaded, dict):
        logger.warning(f"Feature YAML at {path!r} is not a mapping; using empty toggles.")
        return {}

    out: Dict[str, bool] = {}
    for k, v in loaded.items():
        out[str(k)] = bool(v)
    return out

# ------------------------------- utilities ---------------------------------
def _timing(method):
    """Decorator to log wall-time of computations."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        out = method(self, *args, **kwargs)
        logger.info(f"✅ {method.__name__} completed in {time.time() - t0:.3f}s")
        return out
    return wrapper

# ---------------------------------------------------------------------
# Feature extractor (NO CACHING)
# ---------------------------------------------------------------------
class GraphFeatureExtractor:
    """
    Compute graph features with efficient backends and simple per-feature toggles.

    Parameters
    ----------
    svd_rank : int
        Target rank for SVD features; <=0 disables all SVD-based features.
    feature_toggles : dict[str,bool] | None
        Mapping of toggles. YAML toggles are loaded first, and then this
        dictionary is merged on top (overrides YAML).
    """

    def __init__(
        self,
        svd_rank: int = 50,
        feature_toggles: Optional[Dict[str, bool]] = None,
    ):
        self.svd_rank = int(svd_rank)

        base = load_feature_toggles_from_yaml()
        if feature_toggles:
            base.update({str(k): bool(v) for k, v in feature_toggles.items()})
        self.feature_toggles = base

    # --------------------------- helpers -----------------------------
    def _on(self, name: str) -> bool:
        """
        True if a feature should be computed under current configuration.
        SVD-based features also require svd_rank > 0 and libs available.
        """
        t = bool(self.feature_toggles.get(name, False))

        svd_keys = {
            "svd_dot", "svd_mean", "lra", "dlra", "mlra",
            "lra_approx", "dlra_approx", "mlra_approx",
        }
        if name in svd_keys:
            return t and (self.svd_rank > 0) and SCIPY_AVAILABLE and TORCH_AVAILABLE
        # igraph-backed centralities need igraph
        if name in {"pagerank", "closeness", "betweenness", "eigenvector"}:
            return t and IG_AVAILABLE
        return t

    def _to_igraph(self, G: nx.Graph) -> Tuple[Any, List[Any], Dict[Any, int]]:
        """
        Convert a NetworkX graph to an igraph, created fresh each call (no caching).

        Returns
        -------
        (ig_graph, nodes, node_to_idx)
        """
        if not IG_AVAILABLE:
            raise ImportError("python-igraph is required for this feature. Install: pip install igraph")
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges = [(node_to_idx[u], node_to_idx[v]) for (u, v) in G.edges()]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        return ig_graph, nodes, node_to_idx

    # ================================================================
    # Topological features (node-level) via igraph + NX
    # ================================================================
    @_timing
    def compute_triangles(self, G: nx.Graph) -> Dict[Any, int]:
        """Count triangles per node using adjacency intersection (pure Python)."""
        adj: Dict[Any, set] = {n: set(G.neighbors(n)) for n in G.nodes()}
        out: Dict[Any, int] = {}
        for u in G.nodes():
            nbrs = sorted(adj[u])
            c = 0
            for i in range(len(nbrs)):
                a = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    b = nbrs[j]
                    if b in adj[a]:
                        c += 1
            out[u] = c
        return out

    @_timing
    def compute_clustering(self, G: nx.Graph) -> Dict[Any, float]:
        """Local clustering coefficient per node using NetworkX."""
        vals = nx.clustering(G)
        return {k: float(v) for k, v in vals.items()}

    @_timing
    def compute_pagerank(self, G: nx.Graph) -> Dict[Any, float]:
        """PageRank per node using igraph (damping=0.85)."""
        ig_graph, nodes, _ = self._to_igraph(G)
        pr = ig_graph.pagerank(damping=0.85)
        return {nodes[i]: float(pr[i]) for i in range(len(nodes))}

    @_timing
    def compute_betweenness_centrality(self, G: nx.Graph) -> Dict[Any, float]:
        """Betweenness centrality per node using igraph."""
        ig_graph, nodes, _ = self._to_igraph(G)
        bc = ig_graph.betweenness()
        return {nodes[i]: float(bc[i]) for i in range(len(nodes))}

    @_timing
    def compute_closeness_centrality(self, G: nx.Graph) -> Dict[Any, float]:
        """Closeness centrality per node using igraph."""
        ig_graph, nodes, _ = self._to_igraph(G)
        cc = ig_graph.closeness()
        return {nodes[i]: float(cc[i]) for i in range(len(nodes))}

    @_timing
    def compute_eigenvector_centrality(self, G: nx.Graph) -> Dict[Any, float]:
        """Eigenvector centrality per node using igraph."""
        ig_graph, nodes, _ = self._to_igraph(G)
        eig = ig_graph.eigenvector_centrality()
        return {nodes[i]: float(eig[i]) for i in range(len(nodes))}

    @_timing
    def compute_degree_centrality(self, G: nx.Graph) -> Dict[Any, float]:
        """Normalized degree centrality per node (degree / max_degree)."""
        degrees = dict(G.degree())
        maxd = max(degrees.values()) if degrees else 1
        return {n: (float(d) / maxd if maxd > 0 else 0.0) for n, d in degrees.items()}

    # ================================================================
    # Pairwise features (edge-level)
    # ================================================================
    @_timing
    def compute_common_neighbors(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[int]:
        """
        Common neighbors per edge.

        - Fast path: Numba kernel on CSR rows (requires SciPy sparse).
        - Fallback: Python set intersection per pair.
        """
        edges_list = list(edges)
        if not edges_list:
            return []
        if SCIPY_SPARSE_AVAILABLE:
            A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int32)
            node_to_idx = {n: i for i, n in enumerate(G.nodes())}
            u_idx = np.array([node_to_idx.get(u, -1) for u, _ in edges_list], dtype=np.int32)
            v_idx = np.array([node_to_idx.get(v, -1) for _, v in edges_list], dtype=np.int32)
            valid = (u_idx >= 0) & (v_idx >= 0)
            out = np.zeros(len(edges_list), dtype=np.int32)
            if np.any(valid):
                out[valid] = self._numba_common_neighbors(
                    A.data, A.indices, A.indptr, u_idx[valid], v_idx[valid]
                )
            return out.tolist()
        else:
            adj = {n: set(G.neighbors(n)) for n in G.nodes()}
            return [len(adj.get(u, set()) & adj.get(v, set())) for u, v in edges_list]

    @_timing
    def compute_preferential_attachment(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """Preferential attachment score per edge: deg(u) * deg(v)."""
        edges_list = list(edges)
        if not edges_list:
            return []
        deg = dict(G.degree())
        du = np.array([deg.get(u, 0) for u, _ in edges_list], dtype=np.float32)
        dv = np.array([deg.get(v, 0) for _, v in edges_list], dtype=np.float32)
        return self._numba_preferential_attachment(du, dv).tolist()

    @_timing
    def compute_jaccard_coefficient(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """Jaccard coefficient per edge via igraph similarity on pairs (fallback to NX)."""
        edges_list = list(edges)
        if not edges_list:
            return []
        if IG_AVAILABLE:
            ig_graph, _, node_to_idx = self._to_igraph(G)
            pairs = [(node_to_idx.get(u, -1), node_to_idx.get(v, -1)) for (u, v) in edges_list]
            valid_pairs = [(a, b) for (a, b) in pairs if a >= 0 and b >= 0]
            if not valid_pairs:
                return [0.0] * len(edges_list)
            vals = ig_graph.similarity_jaccard(pairs=valid_pairs)
            out = [0.0] * len(edges_list)
            j = 0
            for i, (a, b) in enumerate(pairs):
                if a >= 0 and b >= 0:
                    out[i] = float(vals[j]); j += 1
            return out
        else:
            valid = [(u, v) for (u, v) in edges_list if u in G and v in G]
            idx = {e: i for i, e in enumerate(edges_list)}
            out = [0.0] * len(edges_list)
            if valid:
                for u, v, s in nx.jaccard_coefficient(G, valid):
                    out[idx[(u, v)]] = float(s)
            return out

    @_timing
    def compute_adamic_adar(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """Adamic–Adar index per edge using NetworkX generator."""
        edges_list = list(edges)
        if not edges_list:
            return []
        valid = [(u, v) for (u, v) in edges_list if u in G and v in G]
        idx = {e: i for i, e in enumerate(edges_list)}
        out = [0.0] * len(edges_list)
        if valid:
            for u, v, s in nx.adamic_adar_index(G, valid):
                out[idx[(u, v)]] = float(s)
        return out

    @_timing
    def compute_resource_allocation(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """Resource Allocation index per edge using NetworkX generator."""
        edges_list = list(edges)
        if not edges_list:
            return []
        valid = [(u, v) for (u, v) in edges_list if u in G and v in G]
        idx = {e: i for i, e in enumerate(edges_list)}
        out = [0.0] * len(edges_list)
        if valid:
            for u, v, s in nx.resource_allocation_index(G, valid):
                out[idx[(u, v)]] = float(s)
        return out

    @_timing
    def compute_lhn_index(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """
        Leicht–Holme–Newman index per edge:

            LHN(u,v) = |Γ(u) ∩ Γ(v)| / (deg(u) * deg(v))

        Fast path uses Numba on CSR rows; fallback is Python sets.
        """
        edges_list = list(edges)
        if not edges_list:
            return []
        degrees = dict(G.degree())
        if SCIPY_SPARSE_AVAILABLE:
            A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.int32)
            nodes = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            u_idx = np.array([node_to_idx.get(u, -1) for u, _ in edges_list], dtype=np.int32)
            v_idx = np.array([node_to_idx.get(v, -1) for _, v in edges_list], dtype=np.int32)
            du = np.array([degrees.get(u, 0) for u, _ in edges_list], dtype=np.float32)
            dv = np.array([degrees.get(v, 0) for _, v in edges_list], dtype=np.float32)
            valid = (u_idx >= 0) & (v_idx >= 0)
            out = np.zeros(len(edges_list), dtype=np.float32)
            if np.any(valid):
                out[valid] = self._numba_lhn_index(
                    A.data, A.indices, A.indptr, u_idx[valid], v_idx[valid],
                    du[valid], dv[valid]
                )
            return out.tolist()
        else:
            adj = {n: set(G.neighbors(n)) for n in G.nodes()}
            out: List[float] = []
            for u, v in edges_list:
                inter = len(adj.get(u, set()) & adj.get(v, set()))
                denom = degrees.get(u, 0) * degrees.get(v, 0)
                out.append(float(inter) / denom if denom > 0 else 0.0)
            return out

    @_timing
    def compute_shortest_paths(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> List[float]:
        """
        Shortest path lengths per edge using igraph. Returns inf when unreachable.
        Requires: igraph and toggle enabled.
        """
        if not self._on("shortest_paths"):
            return [float("inf")] * len(edges)
        ig_graph, _, node_to_idx = self._to_igraph(G)
        out: List[float] = []
        for u, v in edges:
            a = node_to_idx.get(u, -1)
            b = node_to_idx.get(v, -1)
            if a < 0 or b < 0:
                out.append(float("inf"))
                continue
            try:
                d = ig_graph.distances(source=[a], target=[b])[0][0]
                out.append(float("inf") if d == np.inf else float(d))
            except Exception:
                out.append(float("inf"))
        return out

    # ================================================================
    # Matrix factorization (edge-level) via SciPy (svds) + PyTorch
    # ================================================================
    def _svd_factors(self, G: nx.Graph):
        """
        Return (U, s, Vt, nodes, node_to_idx) for graph G at self.svd_rank.
        Recomputed fresh every call (no caching).
        """
        if not (self.svd_rank > 0 and SCIPY_AVAILABLE):
            return None

        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        A_sparse = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)

        max_rank = max(0, min(A_sparse.shape) - 1)
        r = min(self.svd_rank, max_rank)
        if r <= 0:
            return (None, None, None, nodes, node_to_idx)

        U, s, Vt = svds(A_sparse, k=r, which="LM")
        idx = np.argsort(s)[::-1]
        U = U[:, idx].astype(np.float32, copy=False)
        s = s[idx].astype(np.float32, copy=False)
        Vt = Vt[idx, :].astype(np.float32, copy=False)
        return (U, s, Vt, nodes, node_to_idx)

    @_timing
    def compute_svd_features(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> Dict[str, List[float]]:
        """
        Compute SVD-based features for the provided `edges`.
        """
        if not (self.svd_rank > 0 and SCIPY_AVAILABLE and TORCH_AVAILABLE):
            return self._get_zero_svd_features(edges)

        factors = self._svd_factors(G)
        if factors is None:
            return self._get_zero_svd_features(edges)

        U, s, Vt, nodes, node_to_idx = factors
        if U is None or s is None or Vt is None:
            return self._get_zero_svd_features(edges)

        # PyTorch tensors
        U_t = torch.from_numpy(U)
        s_t = torch.from_numpy(s)
        Vt_t = torch.from_numpy(Vt)

        # Embeddings (scaled by sqrt singular values)
        sqrt_s = torch.sqrt(s_t)
        Ue = U_t * sqrt_s.unsqueeze(0)
        Ve = Vt_t.T * sqrt_s.unsqueeze(0)

        neighbor_mapping = self._build_neighbor_mapping(G, node_to_idx, Ve.shape[0])

        features: Dict[str, List[float]] = {}
        features["svd_dot"]  = self._compute_svd_dot_pytorch(edges, Ue, Ve, node_to_idx)
        features["svd_mean"] = self._compute_svd_mean_pytorch(edges, Ue, Ve, neighbor_mapping, node_to_idx)
        features.update(self._compute_lra_features_pytorch(edges, U_t, s_t, Vt_t, neighbor_mapping, node_to_idx))

        half_rank = max(1, len(s) // 2)
        features.update(self._compute_half_rank_features_pytorch(
            edges, U_t, s_t, Vt_t, half_rank, neighbor_mapping, node_to_idx
        ))
        return features

    # ================================================================
    # Pairwise degree features (edge-level)
    # ================================================================
    @_timing
    def compute_degree_features(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> Dict[str, List[float]]:
        """
        Normalized degree centrality of endpoints for each edge.

        Returns
        -------
        Dict[str, List[float]]: {'deg_cent_u': [...], 'deg_cent_v': [...]}
        """
        edges_list = list(edges)
        if not edges_list:
            return {'deg_cent_u': [], 'deg_cent_v': []}
        degrees = dict(G.degree())
        maxd = max(degrees.values()) if degrees else 1
        u_deg = np.array([degrees.get(u, 0) for u, _ in edges_list], dtype=np.float32)
        v_deg = np.array([degrees.get(v, 0) for _, v in edges_list], dtype=np.float32)
        z = maxd if maxd > 0 else 1.0
        return {'deg_cent_u': (u_deg / z).tolist(), 'deg_cent_v': (v_deg / z).tolist()}

    # ================================================================
    # Numba kernels (no on-disk compilation cache)
    # ================================================================
    @staticmethod
    @njit(fastmath=True)
    def _numba_common_neighbors(adj_data, adj_indices, adj_indptr, u_idx, v_idx):
        """Intersect CSR rows for common neighbors; returns counts for each pair."""
        n = len(u_idx)
        res = np.zeros(n, dtype=np.int32)
        for i in prange(n):
            u = u_idx[i]; v = v_idx[i]
            if u < 0 or v < 0:
                continue
            us = adj_indptr[u]; ue = adj_indptr[u + 1]
            vs = adj_indptr[v]; ve = adj_indptr[v + 1]
            a = us; b = vs; c = 0
            while a < ue and b < ve:
                nu = adj_indices[a]; nv = adj_indices[b]
                if nu == nv:
                    c += 1; a += 1; b += 1
                elif nu < nv:
                    a += 1
                else:
                    b += 1
            res[i] = c
        return res

    @staticmethod
    @njit(fastmath=True)
    def _numba_preferential_attachment(du, dv):
        """Compute deg(u)*deg(v) per edge."""
        n = len(du)
        out = np.zeros(n, dtype=np.float32)
        for i in prange(n):
            out[i] = du[i] * dv[i]
        return out

    @staticmethod
    @njit(fastmath=True)
    def _numba_lhn_index(adj_data, adj_indices, adj_indptr, u_idx, v_idx, du, dv):
        """
        Compute LHN per edge:
            LHN(u,v) = |Γ(u) ∩ Γ(v)| / (deg(u) * deg(v))
        """
        n = len(u_idx)
        out = np.zeros(n, dtype=np.float32)
        for i in prange(n):
            u = u_idx[i]; v = v_idx[i]
            if u < 0 or v < 0:
                continue
            us = adj_indptr[u]; ue = adj_indptr[u + 1]
            vs = adj_indptr[v]; ve = adj_indptr[v + 1]
            a = us; b = vs; c = 0
            while a < ue and b < ve:
                nu = adj_indices[a]; nv = adj_indices[b]
                if nu == nv:
                    c += 1; a += 1; b += 1
                elif nu < nv:
                    a += 1
                else:
                    b += 1
            dp = du[i] * dv[i]
            out[i] = c / dp if dp > 0 else 0.0
        return out

    # ================================================================
    # SVD helpers
    # ================================================================
    def _build_neighbor_mapping(
        self,
        G: nx.Graph,
        node_to_idx: Dict[Any, int],
        max_idx: int,
    ) -> Dict[int, "torch.Tensor"]:
        """Build neighbor index tensors for each node (used in MLRA/mean features)."""
        mapping: Dict[int, "torch.Tensor"] = {}
        for node in G.nodes():
            nbrs = [node_to_idx.get(n, -1) for n in G.neighbors(node)]
            valid = [i for i in nbrs if 0 <= i < max_idx]
            if TORCH_AVAILABLE:
                mapping[node] = torch.tensor(valid, dtype=torch.long) if valid else torch.tensor([], dtype=torch.long)
            else:
                mapping[node] = valid  # type: ignore
        return mapping

    def _compute_svd_dot_pytorch(
        self,
        edges: List[Tuple[Any, Any]],
        Ue: "torch.Tensor",
        Ve: "torch.Tensor",
        node_to_idx: Dict[Any, int],
    ) -> List[float]:
        """Dot product between SVD embeddings of u (from U) and v (from V)."""
        out: List[float] = []
        for u, v in edges:
            iu = node_to_idx.get(u, -1)
            iv = node_to_idx.get(v, -1)
            if 0 <= iu < Ue.shape[0] and 0 <= iv < Ve.shape[0]:
                out.append(float(torch.dot(Ue[iu], Ve[iv]).item()))
            else:
                out.append(0.0)
        return out

    def _compute_svd_mean_pytorch(
        self,
        edges: List[Tuple[Any, Any]],
        Ue: "torch.Tensor",
        Ve: "torch.Tensor",
        neighbor_mapping: Dict[int, "torch.Tensor"],
        node_to_idx: Dict[Any, int],
    ) -> List[float]:
        """Mean dot product between u's embedding and embeddings of v's neighbors."""
        out: List[float] = []
        for u, v in edges:
            iu = node_to_idx.get(u, -1)
            if iu < 0 or iu >= Ue.shape[0]:
                out.append(0.0); continue
            nbr_idx = neighbor_mapping.get(v, torch.tensor([], dtype=torch.long))
            if getattr(nbr_idx, "numel", lambda: 0)() == 0:
                out.append(0.0); continue
            nbr_emb = Ve[nbr_idx]
            dots = torch.mv(nbr_emb, Ue[iu])
            out.append(float(torch.mean(dots).item()))
        return out

    def _compute_lra_features_pytorch(
        self,
        edges: List[Tuple[Any, Any]],
        U: "torch.Tensor",
        s: "torch.Tensor",
        Vt: "torch.Tensor",
        neighbor_mapping: Dict[int, "torch.Tensor"],
        node_to_idx: Dict[Any, int],
    ) -> Dict[str, List[float]]:
        """Compute LRA, DLRA, and MLRA features at full rank."""
        feats: Dict[str, List[float]] = {"lra": [], "dlra": [], "mlra": []}
        for u, v in edges:
            iu = node_to_idx.get(u, -1)
            iv = node_to_idx.get(v, -1)

            # LRA (u_row * s) dot (v_col)
            if 0 <= iu < U.shape[0] and 0 <= iv < Vt.shape[1]:
                u_row = U[iu, :]
                v_col = Vt[:, iv]
                feats["lra"].append(float(torch.dot(u_row * s, v_col).item()))
            else:
                feats["lra"].append(0.0)

            # DLRA (symmetric proxy via A^T u and A v)
            if 0 <= iu < U.shape[0] and 0 <= iv < Vt.shape[1]:
                u_w = U[iu, :] * s
                At_u = torch.mv(Vt.T, u_w)
                v_w = s * Vt[:, iv]
                At_v = torch.mv(U, v_w)
                feats["dlra"].append(float(torch.dot(At_u, At_v).item()))
            else:
                feats["dlra"].append(0.0)

            # MLRA (mean over v's neighbors)
            nbr_idx = neighbor_mapping.get(v, torch.tensor([], dtype=torch.long))
            if 0 <= iu < U.shape[0] and getattr(nbr_idx, "numel", lambda: 0)() > 0:
                valid_nbr_mask = (nbr_idx >= 0) & (nbr_idx < Vt.shape[1])
                valid_nbr_idx = nbr_idx[valid_nbr_mask]
                if len(valid_nbr_idx) > 0:
                    u_w = U[iu, :] * s
                    nbr_vt = Vt[:, valid_nbr_idx]
                    values = torch.mv(nbr_vt.T, u_w)
                    feats["mlra"].append(float(torch.mean(values).item()))
                else:
                    feats["mlra"].append(0.0)
            else:
                feats["mlra"].append(0.0)
        return feats

    def _compute_half_rank_features_pytorch(
        self,
        edges: List[Tuple[Any, Any]],
        U: "torch.Tensor",
        s: "torch.Tensor",
        Vt: "torch.Tensor",
        half_rank: int,
        neighbor_mapping: Dict[int, "torch.Tensor"],
        node_to_idx: Dict[Any, int],
    ) -> Dict[str, List[float]]:
        """Approximate LRA/DLRA/MLRA using half the rank."""
        Uh = U[:, :half_rank]
        sh = s[:half_rank]
        Vth = Vt[:half_rank, :]

        feats: Dict[str, List[float]] = {
            "lra_approx": [], "dlra_approx": [], "mlra_approx": []
        }
        for u, v in edges:
            iu = node_to_idx.get(u, -1)
            iv = node_to_idx.get(v, -1)

            # LRA approx
            if 0 <= iu < Uh.shape[0] and 0 <= iv < Vth.shape[1]:
                u_row = Uh[iu, :]
                v_col = Vth[:, iv]
                feats["lra_approx"].append(float(torch.dot(u_row * sh, v_col).item()))
            else:
                feats["lra_approx"].append(0.0)

            # DLRA approx
            if 0 <= iu < Uh.shape[0] and 0 <= iv < Vth.shape[1]:
                u_w = Uh[iu, :] * sh
                At_u = torch.mv(Vth.T, u_w)
                v_w = sh * Vth[:, iv]
                At_v = torch.mv(Uh, v_w)
                feats["dlra_approx"].append(float(torch.dot(At_u, At_v).item()))
            else:
                feats["dlra_approx"].append(0.0)

            # MLRA approx
            nbr_idx = neighbor_mapping.get(v, torch.tensor([], dtype=torch.long))
            if 0 <= iu < Uh.shape[0] and getattr(nbr_idx, "numel", lambda: 0)() > 0:
                valid_nbr_mask = (nbr_idx >= 0) & (nbr_idx < Vth.shape[1])
                valid_nbr_idx = nbr_idx[valid_nbr_mask]
                if len(valid_nbr_idx) > 0:
                    u_w = Uh[iu, :] * sh
                    nbr_vt = Vth[:, valid_nbr_idx]
                    values = torch.mv(nbr_vt.T, u_w)
                    feats["mlra_approx"].append(float(torch.mean(values).item()))
                else:
                    feats["mlra_approx"].append(0.0)
            else:
                feats["mlra_approx"].append(0.0)
        return feats

    def _get_zero_svd_features(self, edges: List[Tuple[Any, Any]]) -> Dict[str, List[float]]:
        """Return a zero-vector for each SVD feature when rank/libs are unavailable."""
        n = len(edges)
        return {
            "svd_dot": [0.0] * n,
            "svd_mean": [0.0] * n,
            "lra": [0.0] * n,
            "dlra": [0.0] * n,
            "mlra": [0.0] * n,
            "lra_approx": [0.0] * n,
            "dlra_approx": [0.0] * n,
            "mlra_approx": [0.0] * n,
        }

    # ================================================================
    # Public interface
    # ================================================================
    def extract_all_features(self, G: nx.Graph, edges: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """
        Compute all requested features according to the toggles.

        Returns
        -------
        Dict[str, Any]
            - Node-level features as dicts (node -> value)
            - Edge-level features as lists aligned with `edges`
        """
        logger.info(
            f"🚀 Starting feature extraction |V|={G.number_of_nodes()} "
            f"|E|={G.number_of_edges()} |pairs|={len(edges)}"
        )
        features: Dict[str, Any] = {}

        # Node-level
        if self._on("triangles"):           features["triangles"] = self.compute_triangles(G)
        if self._on("clustering"):          features["clustering"] = self.compute_clustering(G)
        if self._on("pagerank"):            features["pagerank"] = self.compute_pagerank(G)
        if self._on("betweenness"):         features["betweenness"] = self.compute_betweenness_centrality(G)
        if self._on("closeness"):           features["closeness"] = self.compute_closeness_centrality(G)
        if self._on("eigenvector"):         features["eigenvector"] = self.compute_eigenvector_centrality(G)
        if self._on("degree_centrality"):   features["degree_centrality"] = self.compute_degree_centrality(G)

        # Edge-level
        if self._on("common_neighbors"):        features["common_neighbors"] = self.compute_common_neighbors(G, edges)
        if self._on("preferential_attachment"): features["preferential_attachment"] = self.compute_preferential_attachment(G, edges)
        if self._on("jaccard_coefficient"):     features["jaccard_coefficient"] = self.compute_jaccard_coefficient(G, edges)
        if self._on("adamic_adar"):             features["adamic_adar"] = self.compute_adamic_adar(G, edges)
        if self._on("resource_allocation"):     features["resource_allocation"] = self.compute_resource_allocation(G, edges)
        if self._on("lhn_index"):               features["lhn_index"] = self.compute_lhn_index(G, edges)
        if self._on("shortest_paths"):          features["shortest_paths"] = self.compute_shortest_paths(G, edges)

        # SVD-based
        svd_keys = [
            "svd_dot", "svd_mean", "lra", "dlra", "mlra",
            "lra_approx", "dlra_approx", "mlra_approx",
        ]
        wanted = [k for k in svd_keys if self._on(k)]
        if wanted:
            svd_all = self.compute_svd_features(G, edges)
            for k in wanted:
                if k in svd_all:
                    features[k] = svd_all[k]

        # Degree-based endpoints (normalized)
        if self._on("deg_cent_u") or self._on("deg_cent_v"):
            deg = self.compute_degree_features(G, edges)
            if self._on("deg_cent_u"): features["deg_cent_u"] = deg["deg_cent_u"]
            if self._on("deg_cent_v"): features["deg_cent_v"] = deg["deg_cent_v"]

        logger.info(f"🎉 Extracted {len(features)} feature types")
        return features

    def compute_feature_df(self, df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
        """
        Compute features and add them to a DataFrame.

        Supports either:
        - DataFrame with columns ['u','v'], or
        - DataFrame with column ['edge'] containing (u,v) tuples (and optionally 'u','v').

        Node-level features (triangles, clustering, etc.) are mapped to
        `<name>_u` and `<name>_v`. Edge-level features are columns with the
        exact feature name.
        """
        # Normalize to have u,v and edges list
        if "u" in df.columns and "v" in df.columns:
            edges = list(zip(df["u"].tolist(), df["v"].tolist()))
        elif "edge" in df.columns:
            uv = np.vstack(df["edge"].apply(lambda e: (e[0], e[1])).values)
            df = df.assign(u=uv[:, 0], v=uv[:, 1])
            edges = list(zip(df["u"].tolist(), df["v"].tolist()))
        else:
            raise ValueError("DataFrame must contain either ['u','v'] or a single 'edge' column of (u,v) tuples.")

        feats = self.extract_all_features(G, edges)

        # node-level dicts → pairwise columns
        node_maps = {
            "triangles", "clustering", "pagerank", "betweenness",
            "closeness", "eigenvector", "degree_centrality",
        }
        for name in node_maps:
            mapping = feats.get(name)
            if isinstance(mapping, dict):
                df[f"{name}_u"] = df["u"].map(mapping).astype(float)
                df[f"{name}_v"] = df["v"].map(mapping).astype(float)

        # edge-level lists (aligned to edges)
        for name, values in feats.items():
            if name in node_maps:
                continue
            if isinstance(values, list):
                if len(values) != len(df):
                    raise ValueError(f"Edge-level feature '{name}' has length {len(values)} != {len(df)}")
                df[name] = values

        return df

# -----------------------------------------------------------------------------
# Public convenience API expected by the pipeline (NO CACHE ARGS)
# -----------------------------------------------------------------------------
def compute_all_features(
    G: nx.Graph,
    uv_df: pd.DataFrame,
    feature_config_path: Optional[str] = None,
    *,
    svd_rank: int = 50,
    feature_overrides: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    Lightweight wrapper used by the dataset builder (no caching).

    Parameters
    ----------
    G : nx.Graph
        Graph on which features are computed.
    uv_df : pd.DataFrame
        Must contain columns ['u','v'] with node pairs to score.
    feature_config_path : str | None
        Optional path to YAML with {feature_name: bool} toggles.
    svd_rank : int
        Target rank for SVD-based features (heavy backend only).
    feature_overrides : dict[str,bool] | None
        Programmatic toggles that override YAML.

    Returns
    -------
    pd.DataFrame
        Feature columns aligned row-wise to uv_df, WITHOUT 'u','v'.
    """
    if not isinstance(uv_df, pd.DataFrame) or not {'u', 'v'}.issubset(uv_df.columns):
        raise ValueError("uv_df must be a DataFrame with columns ['u','v'].")

    # Merge toggles: YAML (if provided) -> overrides (if provided)
    toggles: Dict[str, bool] = {}
    if feature_config_path:
        try:
            toggles.update(load_feature_toggles_from_yaml(feature_config_path))
        except Exception as e:
            logger.warning(f"Failed to load feature YAML {feature_config_path!r}: {e}")
    if feature_overrides:
        toggles.update({str(k): bool(v) for k, v in feature_overrides.items()})

    extractor = GraphFeatureExtractor(
        svd_rank=svd_rank,
        feature_toggles=toggles if toggles else None,
    )

    # compute_feature_df returns uv+features; we drop uv for clean concat upstream
    df_in = uv_df[['u', 'v']].copy()
    feat_df = extractor.compute_feature_df(df_in, G).drop(columns=['u', 'v'], errors='ignore')
    return feat_df
