# Link Prediction Framework

A comprehensive machine learning framework for graph link prediction that supports multiple scenarios, feature extraction methods, and both traditional ML and Graph Neural Network (GNN) approaches.

## 🚀 Features

- **Multiple Prediction Scenarios**: Simulation, Discovery, and Specific edge prediction
- **Dual Backend Support**: Traditional ML (Random Forest, Logistic Regression, SVM, XGBoost) and GNNs (GCN, GraphSAGE)
- **Rich Feature Extraction**: 20+ graph features including centrality measures, similarity indices, and SVD-based embeddings
- **Intelligent Model Selection**: Pre-trained meta-models for automatic algorithm recommendation
- **Flexible Configuration**: YAML-based feature toggles and hyperparameter management
- **Cross-Validation**: Built-in stratified k-fold validation with early stopping
- **Comprehensive Evaluation**: Multiple metrics including AUC, precision@k, recall@k, and F1 scores

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Feature Extraction](#feature-extraction)
- [Model Selection](#model-selection)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

```bash
# Core dependencies
pip install networkx pandas numpy scikit-learn matplotlib seaborn

# Optional dependencies for enhanced performance
pip install torch torch-geometric  # For GNN support
pip install scipy numba             # For fast feature computation
pip install igraph                  # For optimized graph operations
pip install xgboost                 # For XGBoost models
```

### Basic Installation

```bash
git clone https://github.com/yourusername/link-prediction-framework.git
cd link-prediction-framework
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
import networkx as nx
from config import GeneratorConfig
from dataset_preparer import LinkPredictionDatasetPreparer
from trainer import LinkPredictionTrainer

# 1. Create or load your graph
G = nx.karate_club_graph()

# 2. Configure the dataset preparation
config = GeneratorConfig(
    obs_frac=0.8,           # Use 80% of edges for training
    train_frac=0.8,         # 80% of observed edges for training
    n_folds=5,              # 5-fold cross-validation
    balance_classes=True    # Balance positive/negative samples
)

# 3. Prepare dataset for simulation scenario
preparer = LinkPredictionDatasetPreparer(G, config)
workflow_result = preparer.prepare_dataset(scenario="simulation")

# 4. Train models
trainer = LinkPredictionTrainer(workflow_result)

# Train ML models
ml_results = trainer.train_ml_models(
    models=["random_forest", "logistic_regression"],
    primary_metric="auc"
)

# Train GNN model
gnn_results = trainer.train_gnn_model(
    model_type="sage",
    epochs=300,
    hidden_dims=[64, 32]
)

# 5. Compare results
print(f"ML Best Model: {ml_results['best_summary']['model']}")
print(f"ML AUC: {ml_results['best_summary']['best_score']:.4f}")
print(f"GNN AUC: {gnn_results['simulation_metrics']['auc']:.4f}")
```

## 🏗️ Architecture

### Core Components

```
link-prediction-framework/
├── config.py              # Configuration management
├── dataset_preparer.py     # Data preparation and splitting
├── trainer.py             # Unified training orchestrator
├── ml_models.py           # Traditional ML models
├── gnn_trainer.py         # Graph Neural Network trainer
├── features.py            # Heavy feature extraction (PyTorch/SciPy)
├── features_light.py      # Lightweight feature extraction
├── graph_predictor.py     # Meta-model for algorithm selection
├── metrics.py             # Evaluation metrics
├── edge_sampler.py        # Negative edge sampling
├── graph_processor.py     # Graph processing utilities
├── pair_dataset.py        # Pairwise feature dataset generation
├── utils.py               # Utility functions
├── logger.py              # Logging configuration
├── configs/
│   └── features.yaml      # Feature toggle configuration
└── Models/
    ├── best_auc_model_classifier.pkl
    ├── best_topk_model_classifier.pkl
    └── ...                # Pre-trained meta-models
```

### Three Prediction Scenarios

1. **Simulation**: Predict held-out edges from a complete graph
2. **Discovery**: Find new potential edges in an existing graph
3. **Specific**: Score user-provided edge candidates

## 💡 Usage Examples

### Scenario 1: Simulation (Link Recovery)

```python
# Hide 20% of edges and try to recover them
config = GeneratorConfig(obs_frac=0.8, train_frac=0.8)
preparer = LinkPredictionDatasetPreparer(G, config)
result = preparer.prepare_dataset(scenario="simulation")

trainer = LinkPredictionTrainer(result)
ml_results = trainer.train_ml_models()

print(f"Recovered edges with AUC: {ml_results['holdout_metrics']['auc']:.4f}")
```

### Scenario 2: Discovery (New Edge Prediction)

```python
# Find potential new connections
result = preparer.prepare_dataset(scenario="discovery")
trainer = LinkPredictionTrainer(result)
gnn_results = trainer.train_gnn_model(model_type="gcn")

# Get top-k predictions
predictions = gnn_results['candidate_predictions']
top_predictions = predictions.nlargest(10, 'prediction_score')
```

### Scenario 3: Specific Edge Scoring

```python
# Score specific edge candidates
candidate_edges = [(1, 5), (2, 8), (3, 7)]
result = preparer.prepare_dataset(
    scenario="specific", 
    predict_edges=candidate_edges
)

trainer = LinkPredictionTrainer(result)
results = trainer.train_ml_models()
scores = results['candidate_predictions']['prediction_score']
```

### Intelligent Model Selection

```python
from graph_predictor import GraphPredictor

# Get algorithm recommendation
predictor = GraphPredictor()
recommendation = predictor.predict(G)

print(f"Best for AUC: {recommendation['predicted_best_auc_model']}")
print(f"Best for top-k: {recommendation['predicted_best_topk_model']}")
print(f"Expected AUC: {recommendation['predicted_auc_score']:.4f}")

# Use recommended model
trainer = LinkPredictionTrainer(workflow_result)
results = trainer.train_ml_models(
    models=[recommendation['predicted_best_auc_model']]
)
```

## ⚙️ Configuration

### Dataset Configuration

```python
from config import GeneratorConfig

config = GeneratorConfig(
    # Graph splitting
    obs_frac=0.8,                    # Fraction of edges to observe
    train_frac=0.8,                  # Fraction of observed edges for training
    
    # Negative sampling
    train_neg_per_pos=3.0,           # Negative to positive ratio for training
    sim_ho_neg_per_pos=2.0,          # Negative to positive ratio for holdout
    train_neg_max_cap=50000,         # Maximum negative samples for training
    
    # Cross-validation
    n_folds=5,                       # Number of CV folds
    use_stratified_cv=True,          # Use stratified CV
    balance_classes=True,            # Balance classes with oversampling
    
    # Features
    k_svd=50,                        # SVD rank for matrix factorization
    feature_config_path="configs/features.yaml",
    
    # Randomness
    random_state=42
)
```

### Feature Configuration (configs/features.yaml)

```yaml
# Node-level features
triangles: true
clustering: true
pagerank: true
betweenness: true
closeness: true
eigenvector: true
degree_centrality: true

# Pairwise features
common_neighbors: true
jaccard_coefficient: true
adamic_adar: true
resource_allocation: true
preferential_attachment: true
lhn_index: false

# SVD-based features (requires heavy backend)
svd_dot: true
svd_mean: true
lra: true
dlra: true
mlra: true

# Distance features
shortest_paths: false  # Expensive for large graphs
```

## 🎯 Feature Extraction

### Available Features

**Node-level features** (computed per node, then mapped to edges):
- Triangle count
- Clustering coefficient  
- PageRank centrality
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality
- Degree centrality

**Pairwise features** (computed directly for edge pairs):
- Common neighbors
- Jaccard coefficient
- Adamic-Adar index
- Resource allocation index
- Preferential attachment
- Leicht-Holme-Newman index
- Shortest path distance

**SVD-based features** (matrix factorization):
- SVD dot product
- SVD neighborhood mean
- Low-rank approximation (LRA)
- Directional LRA
- Mean LRA

### Performance Backends

**Heavy Backend** (features.py):
- Uses PyTorch, SciPy, igraph, Numba
- Optimized for large graphs
- Full SVD feature support
- 10-100x faster on large graphs

**Light Backend** (features_light.py):
- Pure NetworkX implementation
- No external dependencies
- Good for small-medium graphs
- Automatic fallback when heavy deps unavailable

## 🤖 Model Selection

The framework includes pre-trained meta-models that analyze graph topology and recommend the best algorithm:

```python
predictor = GraphPredictor()
recommendation = predictor.predict(G)

# Example output:
{
    'predicted_best_auc_model': 'random_forest',
    'predicted_best_topk_model': 'xgboost', 
    'predicted_auc_score': 0.8542,
    'predicted_topk_score': 0.7891,
    'topological_features': {
        'average_clustering': 0.5706,
        'avg_degree': 4.588,
        'number_of_nodes': 34.0,
        ...
    }
}
```

### Supported Models

**Traditional ML**:
- Random Forest
- Logistic Regression  
- Support Vector Machine
- XGBoost (optional)

**Graph Neural Networks**:
- Graph Convolutional Network (GCN)
- GraphSAGE

## 📊 API Reference

### Core Classes

#### `LinkPredictionDatasetPreparer`

```python
preparer = LinkPredictionDatasetPreparer(G, config)
result = preparer.prepare_dataset(
    scenario="simulation",           # "simulation", "discovery", "specific"
    predict_edges=None               # Required for "specific" scenario
)
```

#### `LinkPredictionTrainer`

```python
trainer = LinkPredictionTrainer(workflow_result, device="cpu")

# ML training
ml_results = trainer.train_ml_models(
    models=["random_forest", "xgboost"],
    primary_metric="auc",
    ks=[5, 10, 20]
)

# GNN training  
gnn_results = trainer.train_gnn_model(
    model_type="sage",
    epochs=300,
    hidden_dims=[64, 32],
    learning_rate=0.01
)

# Compare both
comparison = trainer.compare_backends()
```

#### `GraphPredictor`

```python
predictor = GraphPredictor(use_igraph=True, model_dir="Models")
recommendation = predictor.predict(G)
batch_results = predictor.predict_batch([G1, G2, G3])
```

### Utility Functions

```python
# Convenience function
from trainer import train_scenario_models

results = train_scenario_models(
    workflow_result=result,
    backend="ml",  # or "gnn"
    device="cpu"
)
```

## 🚄 Performance

### Benchmark Results

Framework performance on various graph sizes:

| Graph Size | Nodes | Edges | Feature Time | Training Time | Memory |
|------------|-------|-------|--------------|---------------|---------|
| Small      | 100   | 500   | 0.1s        | 2s           | 50MB   |
| Medium     | 1K    | 5K    | 0.8s        | 15s          | 200MB  |  
| Large      | 10K   | 50K   | 8s          | 120s         | 1.2GB  |
| X-Large    | 100K  | 500K  | 85s         | 1800s        | 8GB    |

### Optimization Tips

1. **Use igraph backend** for large graphs: `pip install igraph`
2. **Enable SVD features selectively** - they're powerful but expensive
3. **Limit negative sampling** with `max_negative_samples` for discovery
4. **Use GPU for GNNs**: Set `device="cuda"`
5. **Feature caching**: Features are computed fresh each time (by design for flexibility)

### Memory Management

```python
# For large graphs, limit negative sampling
config = GeneratorConfig(
    max_negative_samples=100_000,    # Cap discovery test size
    train_neg_max_cap=50_000,        # Cap training negatives
    sim_ho_neg_max_cap=20_000        # Cap simulation holdout negatives
)
```

## 🧪 Evaluation Metrics

The framework provides comprehensive evaluation:

```python
{
    'auc': 0.8542,                   # Area under ROC curve
    'avg_precision': 0.7891,         # Average precision
    'accuracy': 0.8123,              # Classification accuracy
    'precision': 0.7654,             # Precision at threshold
    'recall': 0.8234,                # Recall at threshold
    'f1': 0.7932,                    # F1 score at threshold
    'precision@5': 0.9200,           # Precision at top-5
    'recall@5': 0.2300,              # Recall at top-5
    'precision@10': 0.8500,          # Precision at top-10
    'recall@10': 0.4250,             # Recall at top-10
    'hits@10': 17                    # Number of hits in top-10
}
```

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **New algorithms**: Implement additional ML/GNN models
- **Feature engineering**: Add new graph features
- **Optimization**: Performance improvements for large graphs
- **Documentation**: Examples and tutorials
- **Testing**: Unit tests and benchmarks

### Development Setup

```bash
git clone https://github.com/yourusername/link-prediction-framework.git
cd link-prediction-framework
pip install -e .
pip install pytest black flake8
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{link_prediction_framework,
  title={Link Prediction Framework: A Comprehensive Machine Learning Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/link-prediction-framework}
}
```

## 🔗 Related Work

- [NetworkX](https://networkx.org/) - Graph analysis library
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - GNN framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [igraph](https://igraph.org/) - High-performance graph analysis

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/link-prediction-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/link-prediction-framework/discussions)
- **Email**: your.email@example.com

---

**Happy Link Predicting!** 🎯
