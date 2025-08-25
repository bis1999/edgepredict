# graph_processor.py
import networkx as nx
import random
from typing import Any, List, Set, Tuple

def _normalize_edge(e: Tuple[Any, Any]) -> Tuple[Any, Any]:
    u, v = e
    return (u, v) if u <= v else (v, u)

def build_connected_observed_edges(
    G: nx.Graph,
    n_obs: int,
    rng: random.Random,
) -> List[Tuple[Any, Any]]:
    """
    Build an observed edge set that preserves per-component connectivity.

    Strategy:
    1. Add ALL bridges to ensure no component gets disconnected
    2. Use Union-Find to add non-bridge edges until reaching spanning forest
    3. Add more non-bridges to densify if needed

    Raises
    ------
    ValueError  if n_obs < sum(max(0, |comp|-1)) across components
    RuntimeError if we fail to complete a spanning forest
    """
    all_edges = set(map(_normalize_edge, G.edges()))
    if not all_edges or n_obs <= 0:
        return []

    # Minimum needed for connectivity across all components
    min_needed = sum(max(0, len(comp) - 1) for comp in nx.connected_components(G))
    if n_obs < min_needed:
        raise ValueError(
            f"n_obs too small to preserve connectivity: need at least {min_needed}, got {n_obs}"
        )

    # Step 1: add all bridges
    bridges = set(map(_normalize_edge, nx.bridges(G)))
    obs_edges: Set[Tuple[Any, Any]] = set(bridges)

    # Step 2: spanning forest via union-find
    uf = nx.utils.UnionFind(G.nodes())
    for u, v in obs_edges:
        uf.union(u, v)

    non_bridges = [e for e in all_edges if e not in obs_edges]
    rng.shuffle(non_bridges)

    need = min_needed - len(obs_edges)
    for u, v in non_bridges:
        if need == 0:
            break
        if uf[u] != uf[v]:
            obs_edges.add((u, v))
            uf.union(u, v)
            need -= 1

    if need > 0:
        raise RuntimeError("Could not construct spanning forest.")

    # Step 3: densify with remaining edges
    extras = [e for e in all_edges if e not in obs_edges]
    rng.shuffle(extras)
    for e in extras:
        if len(obs_edges) >= n_obs:
            break
        obs_edges.add(e)

    result = list(obs_edges)
    rng.shuffle(result)
    return result[:n_obs]
