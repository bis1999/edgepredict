# utils.py

import networkx as nx
from pathlib import Path
from typing import List, Tuple, Optional, Union

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def ensure_dir(path: Union[str, Path], exist_ok: bool = True) -> None:
    """Create directory if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        log.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=exist_ok)

def validate_edge_list(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Ensures that the edge list:
    - contains only integer node IDs
    - has no self-loops
    - has no duplicates
    """
    cleaned = []
    seen = set()

    for u, v in edges:
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"Non-integer node ID found: ({u}, {v})")

        if u == v:
            continue  # Remove self-loops

        edge = tuple(sorted((u, v)))
        if edge in seen:
            continue  # Remove duplicates

        seen.add(edge)
        cleaned.append(edge)

    return cleaned

def is_connected(G: nx.Graph) -> bool:
    """Returns True if the graph is fully connected."""
    return nx.is_connected(G)

def largest_connected_component(G: nx.Graph) -> nx.Graph:
    """Returns the largest connected component subgraph."""
    if nx.is_connected(G):
        return G.copy()
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def report_graph_stats(G: nx.Graph, name: str = "Graph") -> None:
    """Logs basic graph statistics."""
    log.info(f"{name} has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    log.info(f"Is connected? {nx.is_connected(G)}")
    log.info(f"Density: {nx.density(G):.6f}")
    log.info(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
