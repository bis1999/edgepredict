# Link Prediction Dataset Builder

Turn raw graphs into model-ready train/test DataFrames with consistent topology features and cross-validation splits. Heavy features (PyTorch + SciPy + igraph) are used when available; otherwise a lightweight NetworkX backend is used automatically.

## Scenarios

### 1) Simulation (⛳)
Given `G=(V,E)`:
1. Build **observed** graph `G'=(V,E')` by subsampling `obs_frac` of edges with connectivity preserved (bridges are kept, then union-find spans, then densify).
2. **Test** set uses positives `Y = E \ E'` and sampled negatives from `G` (avoid true edges). Features computed **on `G'`**.
3. Build **training** graph `G''=(V,E'')` by subsampling `train_frac` of `E'`.
4. **Train** set uses positives `Y' = E' \ E''` and sampled negatives from `G'`. Features computed **on `G''`**.
5. Stratified K-Fold CV built on the train set (`n_folds`, `use_stratified_cv`).

### 2) Edge Discovery
- **Test**: sample up to `max_negative_samples` non-edges from `G` (unlabeled for scoring) with features **on `G`**.
- **Train**: build `G'` like above; positives `E \ E'` + negatives from `G`; features **on `G'`**.
- CV like Simulation.

### 3) Specific Edge Prediction
- **Test**: user-provided pairs (unlabeled) with features **on `G`**.
- **Train**: same as Discovery’s training. We avoid using any user target pairs as negatives.

## Connectivity-Preserving Observed Edges

We preserve connectivity per component when constructing `E'` and `E''`:
1. Include **all bridges** of the current graph.
2. Use **Union-Find** to add non-bridges until each component has a spanning forest.
3. **Densify** with additional non-bridges until the target size is reached.

If the requested size is too small to preserve connectivity, we raise with a clear message.

## Negative Sampling

Uniformly sample non-edges, avoiding any specified edges (e.g., true edges and user targets).  
If you request more than available, we return **as many as possible** by default; set `enforce_exact=True` to raise instead.

## Features

- **Heavy backend** (`features.py`): node centralities via igraph, pairwise topological measures, and SVD/LRA family via SciPy + PyTorch.  
- **Light backend** (`features_light.py`): NetworkX-only alternatives for common features; SVD and igraph-only features are disabled.
- Feature toggles can be controlled via a YAML file or left to sensible defaults.

Your pipeline calls a single API:  
`compute_all_features(G, uv_df, feature_config_path=None)`  
and automatically picks heavy vs light.

## Output Format

Each `prepare_dataset(scenario, ...)` returns:

```python
results = {
  "df_tr_top": <train DataFrame>,
  "df_ho_top": <test DataFrame>,
  "cv_folds":  [(train_idx, valid_idx), ...],
  "graphs":    {"original": G, "observed": G_prime_or_None, "training": G_dprime_or_None},
  "edge_lists":{"training_edges": [...], "observed_edges": [...],
                "holdout_missing": Y, "training_missing": Y_prime},
  "feature_info": {
    "n_features": <int>,
    "feature_names": [...],
    "computation_graph": "training" | "observed_or_original"
  },
  "metadata": {
    "execution_time": <seconds>,
    "stage_times": {"edge_splitting": ..., "cv_folds": ...},
    "configuration": {obs_frac, train_frac, n_folds, ...},
    "graph_properties": {"original_is_connected": bool, "original_connected_components": int},
    "dataset_statistics": {"training": {...}, "holdout": {...}}
  }
}



# 🚀 Link Prediction System

> A production-ready, end-to-end link prediction framework with advanced data leakage prevention and adaptive performance optimization.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

## 🌟 Key Features

- **🔒 Data Leakage Prevention**: Bridge-safe edge splitting with separate feature graphs
- **⚡ Adaptive Performance**: Graph-size aware feature selection (1K-100K+ nodes)
- **🎯 Three Core Scenarios**: Specific prediction, edge discovery, simulation/evaluation
- **🤖 Dual Model Support**: Traditional ML (RF, XGB, etc.) + Graph Neural Networks
- **🌉 Connectivity Preservation**: Spanning-forest algorithms maintain graph structure
- **📊 Production Ready**: Unified output schema, comprehensive validation, logging

---

## 📖 Table of Contents

- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Three Core Scenarios](#-three-core-scenarios)
- [Model Backends](#-model-backends)
- [Key Innovations](#-key-innovations)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Performance Guide](#-performance-guide)
- [Contributing](#-contributing)

---

## ⚡ Quick Start

```python
from workflow_manager import CompleteLinkPredictionWorkflow
from core_data_functions import DatasetConfig

# Your edge list
edges = [(1, 2), (2, 3), (3, 4), (1, 4), (2, 4)]

# Create workflow
workflow = CompleteLinkPredictionWorkflow(edges)

# Scenario 1: Predict specific edges
target_edges = [(1, 3), (2, 5), (4, 5)]
result = workflow.scenario_1_specific_prediction(target_edges)

# Scenario 2: Discover all potential edges
result = workflow.scenario_2_edge_discovery(max_candidates=1000)

# Scenario 3: Simulation/evaluation
result = workflow.scenario_3_simulation_evaluation(obs_frac=0.8, train_frac=0.8)
```

---

## 🏗️ System Architecture

<details>
<summary><b>📊 Interactive Architecture Diagram</b> (Click to expand)</summary>

```
🎯 Orchestration Layer
├── workflow_manager.py    # Main orchestration & scenario management
└── trainer.py           # Unified training interface for ML/GNN

💾 Core Data Processing  
├── core_data_functions.py # Enhanced data prep with bridge-safe splitting
└── Original workflow.py  # Your original workflow with caching

⚙️ Feature Engineering
└── features.py          # Config-driven feature extraction (50+ features)

🤖 Machine Learning Models
├── ml_models.py         # Traditional ML + hyperparameter optimization  
├── gnn_models.py        # Graph Neural Network architectures
└── gnn_trainer.py       # Two-stage GNN training system

🔧 Utilities & Support
├── metrics.py           # Evaluation metrics (AUC, AP, Precision@K)
├── logger.py            # Centralized logging
└── seed.py              # Reproducibility utilities
```

**🔗 View Full Interactive Architecture**: [Link to hosted visualization or GIF]

</details>

### Core Design Principles

| Principle | Implementation | Benefit |
|-----------|---------------|---------|
| **Data Integrity** | Separate feature graphs for CV | Zero information leakage |
| **Scalability** | Adaptive configuration by graph size | Handles 100+ to 100K+ nodes |
| **Flexibility** | Three scenarios + dual model support | Covers all use cases |
| **Production Ready** | Unified schema + comprehensive validation | Easy integration |

---

## 🎯 Three Core Scenarios

### 1️⃣ Specific Edge Prediction
**Use Case**: Predict likelihood of user-provided edges

```python
# Example: Friend recommendation verification
target_edges = [(user_1, user_2), (user_1, user_3)]
result = workflow.scenario_1_specific_prediction(target_edges)
predictions = result['data']['prediction_data']
```

**Perfect for**: Friend recommendations, collaboration assessment, targeted scoring

### 2️⃣ Edge Discovery  
**Use Case**: Discover and rank all potential new connections (V×V-E)

```python
# Example: Find top 100 most likely new connections
result = workflow.scenario_2_edge_discovery(max_candidates=1000)
top_candidates = result['data']['discovery_data'].head(100)
```

**Perfect for**: Social network growth, research collaboration discovery, market opportunities

### 3️⃣ Simulation & Evaluation
**Use Case**: Split graph temporally and evaluate model performance

```python
# Example: Evaluate model on held-out 20% of edges
result = workflow.scenario_3_simulation_evaluation(obs_frac=0.8)
holdout_performance = result['data']['holdout']
```

**Perfect for**: Model validation, algorithm comparison, hyperparameter optimization

---

## 🤖 Model Backends

<table>
<tr>
<th width="50%">🔢 Traditional Machine Learning</th>
<th width="50%">🧠 Graph Neural Networks</th>
</tr>
<tr>
<td>

**Models Available:**
- Random Forest ⭐ (recommended)
- Logistic Regression
- Support Vector Machines  
- XGBoost (optional)
- LightGBM (optional)

**Advantages:**
- ✅ Interpretable predictions
- ✅ Feature importance insights
- ✅ Fast training & inference
- ✅ Robust to small datasets

</td>
<td>

**Architectures:**
- Graph Convolutional Networks (GCN)
- GraphSAGE
- Configurable MLP predictor

**Advantages:**
- ✅ End-to-end differentiable
- ✅ Captures graph structure  
- ✅ Scalable to large graphs
- ✅ No manual feature engineering

**Innovation:** Two-stage training (hyperparameter tuning on subset → final training on full graph)

</td>
</tr>
</table>

---

## 🌟 Key Innovations

### 🔒 Data Leakage Prevention
**The Problem**: Traditional approaches use the same graph for feature extraction and validation, creating information leakage.

**Our Solution**: 
- Create `G_feature` (without validation edges) for CV features
- Use `G_original` for prediction features  
- Mathematically guaranteed zero leakage

```python
# Traditional approach (WRONG)
features = extract_features(G, validation_edges)  # 🚨 LEAKAGE!

# Our approach (CORRECT) 
G_feature = G.copy()
G_feature.remove_edges_from(validation_edges)
features = extract_features(G_feature, validation_edges)  # ✅ Safe
```

### 🌉 Bridge-Safe Graph Splitting
**The Problem**: Naive edge removal can disconnect the graph, breaking algorithms.

**Our Solution**: Spanning-forest algorithm ensures connectivity preservation:

```python
# Automatically preserves connectivity
observed_edges = build_connected_observed_edges(G, n_obs, rng)
# Result: guaranteed connected subgraph when possible
```

### ⚡ Adaptive Performance Optimization
**The Problem**: Feature extraction doesn't scale to large graphs.

**Our Solution**: Graph-size aware configuration:

| Graph Size | Strategy | Features Used |
|------------|----------|---------------|
| Small (<1K nodes) | Full computation | All centralities, exact algorithms |
| Medium (1K-10K) | Balanced sampling | Sampled betweenness, core features |
| Large (>10K) | Conservative | Essential features only, approximations |

### 🎯 Two-Stage GNN Training
**The Innovation**: Best of both worlds for hyperparameter tuning and optimal embeddings:

1. **Stage 1**: Hyperparameter optimization on feature graph subset
2. **Stage 2**: Retrain with best parameters on full original graph

**Result**: Unbiased hyperparameter selection + optimal node embeddings

---

## 📦 Installation

### Basic Installation
```bash
git clone https://github.com/yourusername/link-prediction-system.git
cd link-prediction-system
pip install -r requirements.txt
```

### With Optional Dependencies
```bash
# For XGBoost/LightGBM support
pip install xgboost lightgbm

# For GNN support  
pip install torch torch-geometric torch-sparse

# For advanced features
pip install pyyaml  # YAML configuration support
```

### Requirements
- Python 3.8+
- NetworkX 2.5+
- NumPy, Pandas, Scikit-learn
- SciPy (for SVD features)
- ImbalancedLearn (for oversampling)

---

## 💻 Usage Examples

<details>
<summary><b>🔍 Complete Example: Social Network Analysis</b></summary>

```python
import networkx as nx
from workflow_manager import CompleteLinkPredictionWorkflow
from trainer import ScenarioTrainer
from core_data_functions import DatasetConfig

# 1. Load your graph data
edges = [
    (1, 2), (2, 3), (3, 4), (1, 4), (2, 4),
    (5, 6), (6, 7), (7, 8), (5, 8), (1, 5)
]

# 2. Configure the system
config = DatasetConfig(
    validation_frac=0.2,
    neg_sample_strategy="equal", 
    cv_folds=5,
    random_state=42
)

# 3. Create workflow
workflow = CompleteLinkPredictionWorkflow(edges, config)

# 4. Run edge discovery
result = workflow.scenario_2_edge_discovery(max_candidates=50)

# 5. Train ML models
trainer = ScenarioTrainer(result, backend="ml")
ml_results = trainer.run()

# 6. Get predictions
best_trainer = ml_results['trainer']
predictions = best_trainer.predict_candidates(result['data']['discovery_data'])

# 7. Show top recommendations  
top_10 = predictions.head(10)
print("Top 10 Link Recommendations:")
print(top_10[['u', 'v', 'prediction_score', 'predicted_label']])
```

</details>

<details>
<summary><b>🧠 GNN Training Example</b></summary>

```python
# Train with Graph Neural Networks
trainer = ScenarioTrainer(result, backend="gnn", device="cuda")

# For simulation scenario with automatic evaluation
if result['mode'] == 'simulation':
    # Get the graphs and test data
    feature_graph = result['meta']['feature_graph'] 
    full_graph = result['meta']['original_graph']
    test_edges = result['data']['holdout']['edge_pairs']
    test_labels = result['data']['holdout']['label']
    
    # Two-stage GNN training
    gnn_results = trainer.train_gnn_simulation(
        feature_graph=feature_graph,
        full_graph=full_graph, 
        test_edges=test_edges,
        test_labels=test_labels,
        max_trials=20,
        stage2_epochs=800
    )
    
    print(f"Best GNN AUC: {gnn_results['best_auc']:.4f}")

# For prediction/discovery scenarios
else:
    model = trainer.train_gnn_simple(full_graph, epochs=600)
    predictions = model.predict(target_edges)
```

</details>

<details>
<summary><b>⚙️ Advanced Configuration</b></summary>

```python
# Custom feature configuration
feature_config = {
    'node': {
        'betw_cent': 200,        # Sample 200 nodes for betweenness
        'clos_cent': False,      # Skip expensive closeness
    },
    'pairwise': {
        'sp': False,             # Skip shortest paths  
        'ppr': 100,              # Personalized PageRank with 100 sources
    },
    'svd': {
        'svd_rank': 50,          # SVD embedding dimension
    }
}

config = DatasetConfig(
    validation_frac=0.15,       # Use 15% for validation
    neg_sample_strategy=2.0,    # 2x negative samples
    cv_folds=10,                # 10-fold CV
    feature_config=feature_config,
    random_state=42
)

# Or load from YAML
config.feature_config = "configs/large_graph_features.yaml"
```

</details>

---

## ⚙️ Configuration

### Dataset Configuration
```python
@dataclass
class DatasetConfig:
    validation_frac: float = 0.2          # Fraction for cross-validation
    neg_sample_strategy: str = "equal"     # "equal", ratio, or count
    random_state: int = 42                 # Reproducibility seed
    k_svd: int = 50                        # SVD embedding dimension
    cv_folds: int = 5                      # Cross-validation folds
    use_oversampling: bool = True          # Balance classes
    feature_config: Optional[Dict] = None  # Custom feature settings
```

### Feature Configuration  
```yaml
# features.yaml
node:
  betw_cent: 200      # Sample 200 nodes for betweenness centrality
  clos_cent: false    # Skip expensive closeness centrality
  
pairwise:
  sp: false           # Skip shortest path computation
  ppr: 100            # Personalized PageRank with max 100 sources

svd:
  svd_rank: 30        # SVD embedding dimension
```

### Performance Tuning by Graph Size
| Nodes | Recommended Config | Expected Runtime |
|-------|-------------------|------------------|
| < 1,000 | Full features | < 1 minute |
| 1K - 10K | Balanced sampling | 1-10 minutes |  
| 10K - 100K | Conservative features | 10-60 minutes |
| > 100K | Minimal features + GNN | 1+ hours |

---

## 📊 Performance Guide

### Benchmarks
| Graph Size | Nodes | Edges | ML Training | GNN Training | Memory Usage |
|------------|-------|-------|-------------|--------------|--------------|
| Small | 500 | 1,000 | 30s | 2min | 100MB |
| Medium | 5,000 | 15,000 | 5min | 10min | 500MB |
| Large | 50,000 | 200,000 | 30min | 45min | 2GB |

### Optimization Tips

**For Large Graphs (>10K nodes):**
```python
# Use conservative feature settings
config.feature_config = {
    'betw_cent': 100,     # Sample fewer nodes
    'clos_cent': False,   # Skip expensive features
    'svd_rank': 20,       # Reduce embedding dimension
}

# Consider GNN for very large graphs
trainer = ScenarioTrainer(result, backend="gnn")
```

**For Real-Time Applications:**
```python
# Precompute features and cache
workflow.save_datasets("cache/", include_graphs=True)

# Use fast feature subset
fast_features = compute_fast_features(graph, edge_df)
```

**Memory Optimization:**
```python
# Process in batches for large candidate sets
result = workflow.scenario_2_edge_discovery(
    max_candidates=10000,  # Limit candidates
)

# Clear cache when needed
workflow.clear_cache()
```

---

## 🔧 API Reference

<details>
<summary><b>📚 Core Classes & Methods</b></summary>

### CompleteLinkPredictionWorkflow
```python
class CompleteLinkPredictionWorkflow:
    def __init__(self, edge_list: List[Tuple[int, int]], 
                 config: DatasetConfig = DatasetConfig())
    
    def scenario_1_specific_prediction(self, 
                                     target_edges: List[Tuple[int, int]],
                                     config: DatasetConfig = None,
                                     include_cv_evaluation: bool = True) -> Dict[str, Any]
    
    def scenario_2_edge_discovery(self,
                                candidate_edges: Optional[List[Tuple[int, int]]] = None,
                                max_candidates: Optional[int] = None,
                                config: DatasetConfig = None,
                                include_cv_evaluation: bool = True) -> Dict[str, Any]
    
    def scenario_3_simulation_evaluation(self,
                                       obs_frac: float = 0.8,
                                       train_frac: float = 0.8,
                                       config: GeneratorConfig = None) -> Dict[str, Any]
```

### ScenarioTrainer
```python
class ScenarioTrainer:
    def __init__(self, workflow_result: Dict[str, Any], 
                 backend: str = "ml", device: str = "cpu")
    
    def run(self, backend: str = None) -> Dict[str, Any]
    
    def train_gnn_simple(self, full_graph: nx.Graph, **kwargs) -> UnifiedGNNLinkPredictor
    
    def train_gnn_simulation(self, feature_graph: nx.Graph, full_graph: nx.Graph,
                           test_edges: List[Tuple], test_labels: List[int]) -> Dict
```

</details>

### Output Schema
All scenarios return a consistent format:
```python
{
    "mode": "prediction" | "discovery" | "simulation",
    "data": {
        # DataFrames with predictions/candidates/train+holdout
    },
    "cv": {
        # Cross-validation data (if applicable)  
    },
    "meta": {
        # Execution info, config, graph stats, timings
    }
}
```

---

## 🧪 Testing & Validation

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_core_functions.py -v
python -m pytest tests/test_workflows.py -v
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Validation Checklist
- ✅ Data leakage prevention verified
- ✅ Bridge-safe splitting tested
- ✅ Connectivity preservation validated  
- ✅ Cross-validation stratification confirmed
- ✅ Reproducibility with random seeds
- ✅ Performance benchmarks updated
- ✅ Memory usage profiled

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/link-prediction-system.git
cd link-prediction-system

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Areas for Contribution
- 🎯 **New Scenarios**: Additional workflow patterns
- 🤖 **Model Backends**: New ML/GNN architectures  
- ⚙️ **Features**: Novel graph feature extractors
- 📊 **Metrics**: Advanced evaluation methods
- 🔧 **Performance**: Optimization and scaling
- 📚 **Documentation**: Examples and tutorials

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NetworkX team for excellent graph algorithms
- PyTorch Geometric for GNN implementations  
- Scikit-learn for robust ML foundations
- The graph neural network research community

---

## 📞 Support

- **Documentation**: [Link to full docs]
- **Issues**: [GitHub Issues](https://github.com/yourusername/link-prediction-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/link-prediction-system/discussions)
- **Email**: your.email@domain.com

---

**⭐ If this project helps you, please give it a star!**
