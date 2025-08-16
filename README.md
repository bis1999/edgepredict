# LinkPredX

**LinkPredX** is a modular, extensible Python package for **link prediction on graphs**. It unifies **traditional ML models** and **GNNs**, with a consistent workflow API, bridge-safe dataset splits, rich topological features, and even a **meta-learner recommender** that suggests the best model for your graph.

---

## ✨ Features

### End-to-End Workflow Manager
Three scenarios with a **normalized schema**:
1. **Specific edge prediction** - Predict likelihood for user-provided edge pairs
2. **Edge discovery** - Rank potential new edges for recommendation
3. **Simulation** - Train/holdout split evaluation for model validation

All scenarios return a consistent output format:
```python
{
    "mode": "prediction" | "discovery" | "simulation",
    "data": {...},   # DataFrames with features and predictions
    "cv": {...}|None,
    "meta": {...}    # timings, graph info, configuration
}
```

### Unified ML & GNN Backend
- **Traditional ML**: Random Forest, Logistic Regression, SVM, XGBoost, LightGBM
- **Graph Neural Networks**: GCN, GraphSAGE with automatic hyperparameter optimization
- **Two-stage GNN training**: Hyperparameter tuning on feature graph + final training on full graph

### Rich Feature Engineering
- **Node-level**: Centrality measures, clustering coefficients, PageRank
- **Pairwise**: Common neighbors, Jaccard, Adamic-Adar, shortest paths
- **SVD-based**: Low-rank approximations, embedding similarities
- **Configurable**: YAML-based feature toggles for performance tuning

### Bridge-Safe Dataset Splits
- Preserves graph connectivity during validation splits
- Strong edge validation with duplicate detection
- Prevents data leakage in temporal scenarios

---

## 🚀 Quick Start

### Installation
```bash
pip install networkx pandas scikit-learn numpy
# Optional dependencies
pip install torch torch-geometric xgboost lightgbm scipy pyyaml
```

### Basic Usage

```python
import networkx as nx
from linkpredx import CompleteLinkPredictionWorkflow

# Create your graph
edge_list = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]
workflow = CompleteLinkPredictionWorkflow(edge_list)

# Scenario 1: Predict specific edges
target_edges = [(0, 2), (1, 4), (2, 4)]
result = workflow.scenario_1_specific_prediction(target_edges)
print(f"Prediction data shape: {result['data']['prediction_data'].shape}")

# Scenario 2: Discover new edges
result = workflow.scenario_2_edge_discovery(max_candidates=1000)
discovery_df = result['data']['discovery_data']
print(f"Top candidates:\n{discovery_df.head()}")

# Scenario 3: Simulation evaluation
result = workflow.scenario_3_simulation_evaluation(obs_frac=0.8, train_frac=0.8)
train_df = result['data']['train']
holdout_df = result['data']['holdout']
print(f"Train: {len(train_df)} samples, Holdout: {len(holdout_df)} samples")
```

### Training Models

```python
from linkpredx import ScenarioTrainer

# ML Backend
trainer = ScenarioTrainer(result, backend="ml")
ml_results = trainer.run()
best_model = ml_results['trainer'].train_best_model()

# GNN Backend  
gnn_model = trainer.train_gnn_simple(workflow.graph, epochs=600)
predictions = gnn_model.predict(target_edges)
```

---

## 📊 Advanced Features

### Feature Configuration
Create a `features.yaml` file to customize feature extraction:

```yaml
node:
  betw_cent: 200      # Sample 200 nodes for betweenness centrality
  clos_cent: false    # Disable expensive closeness centrality
  pagerank: true
  
pairwise:
  sp: false          # Disable shortest path computation
  ppr: 100           # Personalized PageRank with max 100 source nodes
  
svd:
  svd_rank: 30       # Use rank-30 SVD for embedding features
  lra: true          # Enable low-rank approximation features
```

### Custom ML Models
```python
from linkpredx import train_ml_models

custom_grids = {
    "random_forest": {
        "n_estimators": [500, 1000],
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 5, 10]
    }
}

trainer = train_ml_models(cv_data, models=["random_forest"], custom_grids=custom_grids)
```

### GNN Hyperparameter Optimization
```python
from linkpredx import UnifiedGNNLinkPredictor

model = UnifiedGNNLinkPredictor(
    model_type="sage",           # or "gcn"
    hidden_dims=[128, 64],
    dropout=0.3,
    learning_rate=0.01,
    node_feature_type="random"   # or "identity"
)
```

---

## 🏗️ Architecture

### Core Components

- **`workflow_manager.py`** - High-level orchestration of the three scenarios
- **`core_data_functions.py`** - Dataset preparation, bridge-safe splits, validation
- **`features.py`** - Comprehensive feature extraction with runtime optimization
- **`ml_models.py`** - Traditional ML training with hyperparameter search
- **`gnn_trainer.py`** - GNN training with two-stage optimization
- **`trainer.py`** - Unified trainer interface for both ML and GNN backends

### Data Flow
```
Edge List → Graph Processing → Feature Extraction → Model Training → Predictions
    ↓              ↓                  ↓                ↓              ↓
Validation → Bridge-safe Splits → CV Folds → Hyperparameter → Evaluation
```

---

## 🎯 Use Cases

### Academic Research
- **Social Networks**: Friend recommendation, collaboration prediction
- **Biological Networks**: Protein-protein interaction discovery
- **Citation Networks**: Paper recommendation, collaboration patterns

### Industry Applications
- **E-commerce**: Product recommendation, customer similarity
- **Finance**: Transaction pattern detection, risk assessment
- **Knowledge Graphs**: Entity linking, relationship discovery

---

## 📋 Requirements

### Core Dependencies
- `networkx >= 2.5`
- `pandas >= 1.3.0` 
- `numpy >= 1.20.0`
- `scikit-learn >= 1.0.0`
- `imbalanced-learn`

### Optional Dependencies
- `torch >= 1.9.0` + `torch-geometric` (for GNN support)
- `xgboost` (for XGBoost models)
- `lightgbm` (for LightGBM models)
- `scipy` (for SVD features)
- `pyyaml` (for feature configuration)

---

## 🔧 Configuration

### Dataset Configuration
```python
from linkpredx import DatasetConfig

config = DatasetConfig(
    validation_frac=0.2,           # Fraction for cross-validation
    neg_sample_strategy="equal",   # "equal", ratio, or count
    random_state=42,
    k_svd=50,                      # SVD rank for embeddings
    cv_folds=5,
    use_oversampling=True          # Balance classes in CV
)
```

### Simulation Configuration  
```python
from linkpredx import GeneratorConfig

config = GeneratorConfig(
    neg_sample_strategy=5.0,       # 5:1 negative sampling ratio
    obs_frac=0.8,                  # 80% edges observed
    train_frac=0.8,                # 80% of observed for training
    n_splits=5,                    # CV folds
    max_negative_samples=100000    # Memory management
)
```

---

## 🚦 Performance Tips

### For Large Graphs (>10K nodes)
- Set `betw_cent: 200` to sample pivots instead of exact computation
- Disable expensive features: `clos_cent: false`, `load_cent: false`
- Reduce SVD rank: `svd_rank: 20`
- Limit PPR sources: `ppr: 100`

### For Small Graphs (<1K nodes)
- Enable all features for maximum information
- Use `node_feature_type: "identity"` for GNNs
- Increase CV folds for robust validation

### Memory Optimization
- Use `max_negative_samples` to cap memory usage
- Process predictions in batches with `batch_size`
- Consider feature subsets for very high-dimensional graphs

---

## 🤝 Contributing

LinkPredX is designed to be extensible:

1. **Add new features** in `features.py` with proper toggles
2. **Integrate new ML models** in `ml_models.py` 
3. **Extend GNN architectures** in `gnn_models.py`
4. **Create new scenarios** in `workflow_manager.py`

---

## 📖 Citation

If you use LinkPredX in your research, please cite:

```bibtex
@software{linkpredx,
  title={LinkPredX: A Unified Framework for Link Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/linkpredx}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🐛 Issues & Support

- **Bug Reports**: Use GitHub Issues with minimal reproducible examples
- **Feature Requests**: Describe your use case and desired API
- **Performance Issues**: Include graph size and system specifications

---

**Happy Link Predicting! 🔗✨**
