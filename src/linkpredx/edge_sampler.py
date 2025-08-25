from __future__ import annotations
from typing import Any, Iterable, List, Set, Tuple
import random
import networkx as nx

def _sorted_edge(u: Any, v: Any) -> Tuple[Any, Any]:
    return (u, v) if u <= v else (v, u)

def sample_negative_edges(
    G: nx.Graph,
    num_samples: int,
    rng: random.Random,
    avoid_edges: Iterable[Tuple[Any, Any]] | None = None,
    *,
    enforce_exact: bool = False,   # set True to mimic old strict behavior
) -> List[Tuple[Any, Any]]:
    """
    Uniformly sample distinct non-edges from G, avoiding edges in `avoid_edges`.

    - No self-loops.
    - If requested number exceeds availability, returns as many as possible
      unless `enforce_exact=True` (then raises).
    """
    avoid: Set[Tuple[Any, Any]] = set(_sorted_edge(*e) for e in (avoid_edges or []))
    E_sorted = set(_sorted_edge(u, v) for u, v in G.edges())
    avoid |= E_sorted  # never sample an existing edge

    nodes = list(G.nodes())
    n = len(nodes)

    # Upper bound on available non-edges
    max_non_edges = (n * (n - 1)) // 2 - len(E_sorted)
    if max_non_edges <= 0:
        if enforce_exact and num_samples > 0:
            raise RuntimeError("No non-edges available to sample.")
        return []

    target = min(num_samples, max_non_edges)
    out: List[Tuple[Any, Any]] = []
    seen: Set[Tuple[Any, Any]] = set()

    # Rejection sampling
    tries = 0
    cap = target * 20 + 10_000
    while len(out) < target and tries < cap:
        tries += 1
        u, v = rng.sample(nodes, 2)  # guarantees u != v
        e = _sorted_edge(u, v)
        if e in avoid or e in seen:
            continue
        seen.add(e)
        out.append(e)

    if enforce_exact and len(out) < num_samples:
        raise RuntimeError(
            f"Could only sample {len(out)} negative edges (requested {num_samples})."
        )

    return out
