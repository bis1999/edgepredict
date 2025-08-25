from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import yaml

@dataclass
class GeneratorConfig:
    """Configuration for dataset preparation and CV."""
    # Fractions for observed and training subgraphs
    obs_frac: float = 0.8      # |E'| / |E|
    train_frac: float = 0.8    # |E''| / |E'|

    # Randomness / sampling
    random_state: int = 42
    max_negative_samples: int = 100_000  # cap for discovery test size

    # ==== TRAIN negative controls ====
    # If set, use exactly this many negatives for TRAIN (overrides ratio)
    train_neg_samples: Optional[int] = None
    # Otherwise, use: ceil(train_neg_per_pos * #positives)
    train_neg_per_pos: float = 3.0
    # Optional ceiling for negatives in TRAIN (applied after the above)
    train_neg_max_cap: Optional[int] = None

    # ==== SIMULATION holdout negative controls (NEW) ====
    # If set, use exactly this many negatives for SIMULATION holdout (overrides ratio)
    sim_ho_neg_samples: Optional[int] = None
    # Otherwise, use: ceil(sim_ho_neg_per_pos * #positives in holdout)
    sim_ho_neg_per_pos: float = 2.0
    # Optional ceiling for negatives in SIMULATION holdout
    sim_ho_neg_max_cap: Optional[int] = None
    # =================================

    # CV / balancing
    use_stratified_cv: bool = True
    n_folds: int = 5
    balance_classes: bool = True  # oversampling on TRAIN set

    # Features
    k_svd: int = 50  # target rank for SVD-based features
    feature_config_path: Optional[str] = None  # YAML path for feature toggles

    def __post_init__(self):
        if not 0.0 < self.obs_frac <= 1.0:
            raise ValueError(f"obs_frac must be in (0,1], got {self.obs_frac}")
        if not 0.0 < self.train_frac <= 1.0:
            raise ValueError(f"train_frac must be in (0,1], got {self.train_frac}")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2 for CV")
        if self.train_neg_per_pos < 0.0:
            raise ValueError("train_neg_per_pos must be >= 0")
        if self.sim_ho_neg_per_pos < 0.0:
            raise ValueError("sim_ho_neg_per_pos must be >= 0")

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return GeneratorConfig(**raw)

# Backward-compat alias expected in some modules
DatasetConfig = GeneratorConfig
