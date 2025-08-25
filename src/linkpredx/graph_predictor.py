from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union, List
from statistics import variance

import numpy as np
import networkx as nx
import joblib

# ---------------- optional igraph ----------------
try:
    import igraph as ig  # type: ignore
    IG_AVAILABLE = True
except Exception:
    ig = None  # type: ignore
    IG_AVAILABLE = False


# ============================================================
#               Graph conversion & feature extract
# ============================================================

def _nx_nodes_edges_to_indexed(G: nx.Graph) -> Tuple[List[Tuple[int, int]], int]:
    """
    Map arbitrary NetworkX node IDs to contiguous indices [0..n-1]
    and return edges in that index space along with node count.
    (igraph requires 0..n-1 vertex IDs.)
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges_idx = [(node_to_idx[u], node_to_idx[v]) for (u, v) in G.edges()]
    return edges_idx, len(nodes)


def convert_networkx_to_igraph(G: nx.Graph) -> "ig.Graph":
    """Convert a NetworkX graph to igraph with contiguous integer vertex IDs."""
    if IG_AVAILABLE and isinstance(G, ig.Graph):  # already igraph
        return G

    if not IG_AVAILABLE:
        raise ImportError(
            "python-igraph is not installed. Install it or call extract_graph_features(..., prefer_igraph=False)."
        )

    edges_idx, n = _nx_nodes_edges_to_indexed(G)
    if not edges_idx:
        return ig.Graph(n=n, edges=[], directed=False)
    return ig.Graph(n=n, edges=edges_idx, directed=False)


def extract_graph_features_igraph(G: Union[nx.Graph, "ig.Graph"]) -> Dict[str, float]:
    """Optimized graph feature extraction using igraph; falls back to NetworkX if igraph missing."""
    if not IG_AVAILABLE:
        # graceful fallback
        return extract_graph_features_networkx(G if isinstance(G, nx.Graph) else _igraph_to_networkx(G))

    g = convert_networkx_to_igraph(G) if isinstance(G, nx.Graph) else G

    features: Dict[str, float] = {}
    n_nodes = int(g.vcount())
    n_edges = int(g.ecount())

    if n_nodes == 0:
        return {
            "average_clustering": 0.0,
            "average_shortest_path_length": 0.0,
            "degree_assortativity_coefficient": 0.0,
            "number_of_nodes": 0.0,
            "avg_degree": 0.0,
            "Variance": 0.0,
        }

    # Basic size/degree stats
    features["number_of_nodes"] = float(n_nodes)
    degs = g.degree()
    features["avg_degree"] = float(np.mean(degs)) if len(degs) else 0.0
    features["Variance"] = float(variance(degs)) if len(degs) >= 2 else 0.0

    # Clustering (vectorized)
    if n_nodes > 2:
        cl = g.transitivity_local_undirected(mode="zero")
        if cl is None:
            features["average_clustering"] = 0.0
        else:
            # transitivity* can yield None in some edge cases; filter robustly
            valid = [c for c in cl if c is not None]
            features["average_clustering"] = float(np.mean(valid)) if valid else 0.0
    else:
        features["average_clustering"] = 0.0

    # Shortest paths: use average over reachable pairs (unconn=True)
    try:
        if n_nodes > 1:
            features["average_shortest_path_length"] = float(
                g.average_path_length(directed=False, unconn=True)
            )
        else:
            features["average_shortest_path_length"] = 0.0
    except Exception:
        # fallback: largest component only
        try:
            comps = g.connected_components(mode="weak")
            largest = max(comps, key=len)
            if len(largest) > 1:
                sub = g.subgraph(largest)
                features["average_shortest_path_length"] = float(
                    sub.average_path_length(directed=False)
                )
            else:
                features["average_shortest_path_length"] = 0.0
        except Exception:
            features["average_shortest_path_length"] = 0.0

    # Assortativity (robust to NaN)
    try:
        if n_edges > 0 and n_nodes > 1:
            val = g.assortativity_degree(directed=False)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0.0
            features["degree_assortativity_coefficient"] = float(val)
        else:
            features["degree_assortativity_coefficient"] = 0.0
    except Exception:
        features["degree_assortativity_coefficient"] = 0.0

    return features


def _igraph_to_networkx(g: "ig.Graph") -> nx.Graph:
    """Minimal igraph -> NetworkX conversion for fallback paths."""
    G = nx.Graph()
    G.add_nodes_from(range(g.vcount()))
    G.add_edges_from((e.source, e.target) for e in g.es)
    return G


def extract_graph_features_networkx(G: Union[nx.Graph, "ig.Graph"]) -> Dict[str, float]:
    """NetworkX implementation (kept for compatibility and fallback)."""
    if isinstance(G, ig.Graph) and IG_AVAILABLE:
        G = _igraph_to_networkx(G)

    features: Dict[str, float] = {}
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "average_clustering": 0.0,
            "average_shortest_path_length": 0.0,
            "degree_assortativity_coefficient": 0.0,
            "number_of_nodes": 0.0,
            "avg_degree": 0.0,
            "Variance": 0.0,
        }

    deg_map = dict(G.degree())
    deg_vals = list(deg_map.values())

    features["number_of_nodes"] = float(n_nodes)
    features["avg_degree"] = float(np.mean(deg_vals)) if deg_vals else 0.0
    features["Variance"] = float(variance(deg_vals)) if len(deg_vals) >= 2 else 0.0

    try:
        features["average_clustering"] = float(nx.average_clustering(G))
    except Exception:
        features["average_clustering"] = 0.0

    try:
        if n_nodes > 1 and nx.is_connected(G):
            features["average_shortest_path_length"] = float(nx.average_shortest_path_length(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len) if n_nodes > 0 else set()
            if len(largest_cc) > 1:
                sub = G.subgraph(largest_cc)
                features["average_shortest_path_length"] = float(nx.average_shortest_path_length(sub))
            else:
                features["average_shortest_path_length"] = 0.0
    except Exception:
        features["average_shortest_path_length"] = 0.0

    try:
        if n_edges > 0 and n_nodes > 1:
            val = nx.degree_assortativity_coefficient(G)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0.0
            features["degree_assortativity_coefficient"] = float(val)
        else:
            features["degree_assortativity_coefficient"] = 0.0
    except Exception:
        features["degree_assortativity_coefficient"] = 0.0

    return features


def extract_graph_features(G: Union[nx.Graph, "ig.Graph"], prefer_igraph: bool = True) -> Dict[str, float]:
    """
    Hybrid extractor:
      - If prefer_igraph=True and igraph is available → use igraph path
      - Otherwise → NetworkX path
    """
    if prefer_igraph and IG_AVAILABLE:
        return extract_graph_features_igraph(G)
    return extract_graph_features_networkx(G)


# ============================================================
#                     Model-based chooser
# ============================================================

class GraphPredictor:
    """
    High-performance graph meta-predictor:
      - Extracts global topological features (fast igraph by default)
      - Uses pre-trained meta-models to recommend the best downstream model
        for AUC and @K, and to estimate their scores.
    """

    def __init__(self, use_igraph: bool = True, model_dir: str = "Models"):
        self.use_igraph = bool(use_igraph)
        self.model_dir = model_dir
        self._prediction_cache: Dict[str, Dict] = {}
        self.features = [
            "average_clustering",
            "average_shortest_path_length",
            "degree_assortativity_coefficient",
            "number_of_nodes",
            "avg_degree",
            "Variance",
        ]
        self._load_models()

    # --------------------- IO & caching ---------------------

    def _model_path(self, fname: str) -> str:
        return os.path.join(self.model_dir, fname)

    def _load_models(self) -> None:
        """Load all models with clear errors if missing."""
        try:
            self.auc_clf = joblib.load(self._model_path("best_auc_model_classifier.pkl"))
            self.topk_clf = joblib.load(self._model_path("best_topk_model_classifier.pkl"))
            self.le_auc = joblib.load(self._model_path("auc_label_encoder.pkl"))
            self.le_top = joblib.load(self._model_path("topk_label_encoder.pkl"))
            self.auc_reg = joblib.load(self._model_path("auc_score_regressor.pkl"))
            self.topk_reg = joblib.load(self._model_path("topk_score_regressor.pkl"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    @staticmethod
    def _graph_hash(G: Union[nx.Graph, "ig.Graph"]) -> str:
        """Per-process cache key based on simple structure; OK if not stable across runs."""
        if IG_AVAILABLE and isinstance(G, ig.Graph):
            n_nodes, n_edges = G.vcount(), G.ecount()
            edges_t = tuple(sorted((e.source, e.target) for e in G.es))
        else:
            n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
            edges_t = tuple(sorted(G.edges()))
        return f"{n_nodes}_{n_edges}_{hash(edges_t)}"

    def _extract_feature_vector(self, topo_dict: Dict[str, float]) -> np.ndarray:
        try:
            vec = [float(topo_dict[k]) for k in self.features]
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Missing feature '{missing}' in computed topology dict.")
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    # --------------------- public API ---------------------

    def predict(self, G: Union[nx.Graph, "ig.Graph"], use_cache: bool = True) -> Dict:
        """
        Predict best models (by AUC and @K) for the *downstream* link prediction task
        given the high-level topology of G.
        """
        if use_cache:
            key = self._graph_hash(G)
            hit = self._prediction_cache.get(key)
            if hit is not None:
                return hit.copy()

        topo = extract_graph_features(G, prefer_igraph=self.use_igraph)
        x = self._extract_feature_vector(topo)

        try:
            auc_pred_idx = self.auc_clf.predict(x)[0]
            topk_pred_idx = self.topk_clf.predict(x)[0]

            pred_auc_model = self.le_auc.inverse_transform([auc_pred_idx])[0]
            pred_topk_model = self.le_top.inverse_transform([topk_pred_idx])[0]

            auc_score = float(self.auc_reg.predict(x)[0])
            topk_score = float(self.topk_reg.predict(x)[0])
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")

        result = {
            "topological_features": topo,
            "predicted_best_auc_model": pred_auc_model,
            "predicted_best_topk_model": pred_topk_model,
            "predicted_auc_score": round(auc_score, 4),
            "predicted_topk_score": round(topk_score, 4),
        }

        if use_cache:
            self._prediction_cache[key] = result.copy()
            # light cache eviction
            if len(self._prediction_cache) > 1000:
                self._prediction_cache.pop(next(iter(self._prediction_cache)))

        return result

    def predict_batch(self, graphs: List[Union[nx.Graph, "ig.Graph"]]) -> List[Dict]:
        return [self.predict(G) for G in graphs]

    def benchmark_libraries(self, G: Union[nx.Graph, "ig.Graph"], runs: int = 10) -> Dict[str, float]:
        """Rough speed comparison between NX and igraph paths."""
        import time

        # Normalize both representations
        if isinstance(G, nx.Graph):
            nx_graph = G
            ig_graph = convert_networkx_to_igraph(G) if IG_AVAILABLE else None
        else:
            ig_graph = G
            nx_graph = _igraph_to_networkx(G) if IG_AVAILABLE else None

        # NetworkX timing
        t0 = time.time()
        for _ in range(runs):
            extract_graph_features_networkx(nx_graph)
        nx_time = (time.time() - t0) / runs

        # igraph timing
        if IG_AVAILABLE and ig_graph is not None:
            t0 = time.time()
            for _ in range(runs):
                extract_graph_features_igraph(ig_graph)
            ig_time = (time.time() - t0) / runs
        else:
            ig_time = float("inf")

        return {
            "networkx_time": nx_time,
            "igraph_time": ig_time,
            "speedup": (nx_time / ig_time) if (ig_time not in (0.0, float("inf"))) else float("inf"),
        }

    def clear_cache(self) -> None:
        self._prediction_cache.clear()

    def get_cache_stats(self) -> Dict[str, Union[int, bool]]:
        return {"cache_size": len(self._prediction_cache), "max_cache_size": 1000, "using_igraph": self.use_igraph}
