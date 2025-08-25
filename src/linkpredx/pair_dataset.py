"""
Pairwise Feature Dataset Generator
==================================

Builds a labeled pairwise feature table for link prediction.

- Heavy path (Torch + SciPy present): features.compute_all_features
- Light path (fallback): features_light.compute_all_features
"""

from __future__ import annotations

from typing import Any, List, Tuple, Literal, Optional
import pandas as pd
import networkx as nx

# Capability check
HEAVY = False
try:
    import torch  # type: ignore
    import scipy  # type: ignore
    HEAVY = True
except Exception:
    HEAVY = False

if HEAVY:
    from features import compute_all_features
else:
    from features_light import compute_all_features  # same signature

def build_pairwise_feature_dataset(
    G: nx.Graph,
    pos_edges: List[Tuple[Any, Any]],
    neg_edges: List[Tuple[Any, Any]],
    feature_config_path: Optional[str],
    scenario: Literal["simulation", "specific", "discovery"] = "simulation",
    source: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extract features for positive and negative edge pairs.

    Returns
    -------
    pd.DataFrame with:
      ['u','v','label', <features...>, 'scenario', ('source'?)]
    """
    rows = [(u, v, 1) for (u, v) in pos_edges] + [(u, v, 0) for (u, v) in neg_edges]
    df = pd.DataFrame(rows, columns=["u", "v", "label"])

    if verbose:
        print(f"[pair_dataset] Extracting features for {len(df)} pairs "
              f"using {'HEAVY' if HEAVY else 'LIGHT'} backend")

    feat_df = compute_all_features(G, df[["u", "v"]], feature_config_path)
    out = pd.concat([df, feat_df], axis=1)
    out["scenario"] = scenario
    if source:
        out["source"] = source
    return out
