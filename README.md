# Simplified Link Prediction Framework

A clean, modular framework for link prediction that combines traditional machine learning with Graph Neural Networks (GNNs). Designed for simplicity and ease of use while maintaining flexibility for research.

## Key Features

- **Dual Approach**: Both traditional ML with handcrafted features AND GNNs
- **Simplified GNNs**: Uses only identity/random node features (no topological features)
- **Modular Design**: Easy to extend and customize
- **Minimal Dependencies**: Works with standard scientific Python stack
- **Well Documented**: Clear examples and comprehensive documentation

## Quick Start

```python
from link_prediction_framework import run_link_prediction_experiment
import networkx as nx

# Create a sample graph
G = nx.karate_club_graph()
edge_list = list(G.edges())

# Run experiment with both ML and GNN
results = run_link_prediction_experiment(
    edge_list=edge_list,
    method="both",
    verbose=True
)

# Get top predictions
print(results['ml']['top_10'])  # Top ML predictions
print(results['gnn']['top_10']) # Top GNN predictions
```

## Installation

### Basic Requirements
```bash
pip install networkx pandas scikit-learn numpy
```

### For GNN Support
```bash
pip install torch torch-geometric torch-sparse
```

### For XGBoost Support
```bash
pip install xgboost
```

## Framework Components

### 1. Configuration Classes

#### MLConfig
Configuration for traditional ML models:
```python
from link_prediction_framework import MLConfig

config = MLConfig(
    k_svd=50,                              # SVD rank for embeddings
    models_to_try=["random_forest", "xgboost"],
    cv_folds=5,                            # Cross-validation folds
    validation_frac=0.2,                   # Fraction of edges for validation
    neg_sample_strategy="equal"            # Negative sampling strategy
)
```

#### GNNConfig
Configuration for GNN models:
```python
from link_prediction_framework import GNNConfig

config = GNNConfig(
    node_feature_type="random",            # "identity", "random", or "mixed"
    feature_dim=64,                        # Node feature dimension
    hidden_dims=[64, 32],                  # Hidden layer sizes
    model_type="gcn",                      # "gcn" or "sage"
    epochs=100,                            # Training epochs
    learning_rate=0.01
)
```

### 2. Node Feature Types (GNN)

The framework uses simplified node features for GNNs:

- **Identity**: One-hot encoding (or random projection if too many nodes)
- **Random**: Random Gaussian features (normalized)
- **Mixed**: Combination of identity and random features

This approach avoids using topological features in GNNs, making the comparison with traditional ML methods more meaningful.

### 3. Traditional ML Features

For ML models, the framework extracts comprehensive handcrafted features:

#### Node-level Features
- Degree centrality
- PageRank
- Clustering coefficient

#### Pairwise Features
- Common neighbors
- Preferential attachment
- Jaccard coefficient
- Adamic-Adar index

#### Embedding Features
- SVD-based node embeddings
- Dot product similarities
- Cosine similarities

## Usage Examples

### Example 1: Basic Usage
```python
from link_prediction_framework import run_link_prediction_experiment
import networkx as nx

# Load your graph
edge_list = [(0, 1), (1, 2), (2, 3), ...]  # Your edges
# OR: edge_list = list(nx.karate_club_graph().edges())

# Run with default settings
results = run_link_prediction_experiment(edge_list, method="both")
```

### Example 2: Custom Configuration
```python
from link_prediction_framework import MLConfig, GNNConfig, run_link_prediction_experiment

# Custom ML configuration
ml_config = MLConfig(
    k_svd=30,
    models_to_try=["random_forest"],
    cv_folds=3
)

# Custom GNN configuration
gnn_config = GNNConfig(
    node_feature_type="random",
    feature_dim=32,
    hidden_dims=[32, 16],
    epochs=50
)

# Run experiment
results = run_link_prediction_experiment(
    edge_list=edge_list,
    method="both",
    ml_config=ml_config,
    gnn_config=gnn_config
)
```

### Example 3: File Input
```python
# Load edge list from file
def load_edges(filepath):
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                u, v = map(int, line.strip().split()[:2])
                edges.append((u, v))
    return edges

edge_list = load_edges("my_graph.txt")
results = run_link_prediction_experiment(edge_list)
```

## Run Script Usage

The framework includes a comprehensive run script with multiple examples:

```bash
# Basic example
python run_experiments.py --example basic

# Custom configuration example  
python run_experiments.py --example custom

# ML-only example
python run_experiments.py --example ml_only

# Compare GNN node features
python run_experiments.py --example gnn_features

# Load from file
python run_experiments.py --data your_edges.txt

# Scalability benchmark
python run_experiments.py --benchmark

# Create sample data
python run_experiments.py --create_sample
```

## Results Structure

The framework returns a comprehensive results dictionary:

```python
results = {
    'graph_stats': {
        'nodes': 34,
        'edges': 78,
        'density': 0.139
    },
    'ml': {
        'training_results': {...},      # Model training details
        'predictions': DataFrame,       # All predictions with scores
        'top_10': DataFrame            # Top 10 predictions
    },
    'gnn': {
        'training_results': {...},      # GNN training details  
        'predictions': DataFrame,       # All predictions with scores
        'top_10': DataFrame            # Top 10 predictions
    },
    'comparison': {                     # Only if both methods run
        'correlation': 0.85,           # Correlation between methods
        'top_k_overlap': {...}         # Overlap in top-k predictions
    }
}
```

## Advanced Usage

### Individual Components

You can also use individual components:

```python
from link_prediction_framework import DatasetPreparer, MLPredictor, GNNPredictor

# Prepare data
preparer = DatasetPreparer(graph, config)
data = preparer.prepare_data()

# Train ML model
ml_predictor = MLPredictor(ml_config)
ml_results = ml_predictor.fit(data['cv_data'], data['train_graph'])

# Make predictions
predictions = ml_predictor.predict(data['prediction_data'], graph)
```

### Custom Feature Extraction

```python
from link_prediction_framework import extract_features

# Extract features for your edge pairs
edge_pairs = pd.DataFrame({'u': [0, 1, 2], 'v': [1, 2, 3]})
features = extract_features(graph, edge_pairs, k_svd=50)
```

## Evaluation Metrics

The framework automatically computes standard link prediction metrics:

- **AUC**: Area under ROC curve
- **Precision@K**: Precision in top-K predictions
- **Recall@K**: Recall in top-K predictions
- **Top-K Overlap**: Overlap between different methods

## Design Principles

### 1. Simplicity
- Single main function for most use cases
- Sensible defaults that work well
- Clear, readable code structure

### 2. Modularity
- Each component can be used independently
- Easy to extend with new models or features
- Configuration-driven approach

### 3. Fair Comparison
- GNNs use only simple features (no topology)
- ML models use comprehensive handcrafted features
- Same data splits for all methods

### 4. Reproducibility
- All random operations are seeded
- Configuration objects store all parameters
- Results include full provenance information

## Performance Notes

### Graph Size Guidelines
- **Small graphs** (< 1,000 nodes): All features enabled, fast execution
- **Medium graphs** (1,000-10,000 nodes): Balanced feature set, good performance
- **Large graphs** (> 10,000 nodes): Conservative features, may need tuning

### Memory Usage
- ML models: Scales with number of features and edges
- GNN models: Scales with number of nodes and feature dimension
- Feature extraction: Most memory-intensive step

### Speed Optimization Tips
1. Reduce SVD rank for large graphs (`k_svd=20`)
2. Use fewer CV folds (`cv_folds=3`)
3. Limit models to try (`models_to_try=["random_forest"]`)
4. Reduce GNN epochs (`epochs=50`)

## File Formats

### Edge List Format
```
# Comments start with #
# Format: node1 node2 [weight]
0 1
1 2
2 3
0 3
```

### Prediction Output Format
```csv
u,v,prediction_score,predicted_label
0,5,0.95,1
1,4,0.87,1
2,6,0.12,0
```

## Troubleshooting

### Common Issues

#### "PyTorch not available"
GNN functionality requires PyTorch and PyTorch Geometric:
```bash
pip install torch torch-geometric torch-sparse
```

#### "XGBoost not available"
For XGBoost support:
```bash
pip install xgboost
```

#### Memory errors with large graphs
- Reduce `k_svd` parameter
- Use `node_feature_type="random"` with smaller `feature_dim`
- Reduce `validation_frac`

#### Poor GNN performance
- Try different `node_feature_type` values
- Increase `feature_dim` and `hidden_dims`
- Adjust learning rate and epochs

### Error Messages

#### "Graph has no nodes after cleaning"
Your edge list contains only self-loops or invalid edges.

#### "No valid edges found"
Check your edge list format and ensure nodes are integers.

#### "Model not trained"
Call `fit()` before `predict()` when using individual components.

## API Reference

### Main Functions

#### `run_link_prediction_experiment()`
```python
def run_link_prediction_experiment(
    edge_list: List[Tuple[int, int]],
    method: str = "both",                    # "ml", "gnn", or "both"
    ml_config: Optional[MLConfig] = None,
    gnn_config: Optional[GNNConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
```

#### `extract_features()`
```python
def extract_features(
    graph: nx.Graph,
    edge_pairs: pd.DataFrame,
    k_svd: int = 50
) -> pd.DataFrame:
```

### Configuration Classes

#### `MLConfig`
- `validation_frac: float = 0.2` - Fraction of edges for validation
- `k_svd: int = 50` - SVD rank for embeddings  
- `models_to_try: List[str]` - Models to try in grid search
- `cv_folds: int = 5` - Cross-validation folds
- `neg_sample_strategy: str = "equal"` - Negative sampling strategy
- `k_top: int = 10` - Top-k for evaluation metrics

#### `GNNConfig`  
- `node_feature_type: str = "identity"` - Type of node features
- `feature_dim: int = 64` - Node feature dimension
- `hidden_dims: List[int] = [64, 64]` - Hidden layer sizes
- `model_type: str = "gcn"` - GNN architecture type
- `learning_rate: float = 0.01` - Learning rate
- `epochs: int = 100` - Training epochs
- `dropout: float = 0.2` - Dropout rate

### Core Classes

#### `DatasetPreparer`
Handles data preparation and splitting:
```python
preparer = DatasetPreparer(graph, config)
data = preparer.prepare_data()
```

#### `MLPredictor`
Traditional ML predictor:
```python
predictor = MLPredictor(ml_config)
results = predictor.fit(cv_data, train_graph)
predictions = predictor.predict(test_edges, graph)
```

#### `GNNPredictor`  
GNN-based predictor:
```python
predictor = GNNPredictor(gnn_config)
results = predictor.fit(cv_data, train_graph)
predictions = predictor.predict(test_edges, graph)
```

## Examples Gallery

### Research Use Cases

#### 1. Method Comparison Study
```python
# Compare different approaches systematically
methods = ["ml", "gnn"]
datasets = [nx.karate_club_graph(), nx.dolphins_graph()]
results = {}

for dataset_name, graph in zip(["karate", "dolphins"], datasets):
    edge_list = list(graph.edges())
    for method in methods:
        key = f"{dataset_name}_{method}"
        results[key] = run_link_prediction_experiment(
            edge_list, method=method, verbose=False
        )
```

#### 2. Feature Ablation Study
```python
# Test different feature combinations for ML
feature_configs = [
    {"k_svd": 0},      # No embeddings
    {"k_svd": 20},     # Small embeddings  
    {"k_svd": 50},     # Standard embeddings
]

for i, config in enumerate(feature_configs):
    ml_config = MLConfig(**config)
    results = run_link_prediction_experiment(
        edge_list, method="ml", ml_config=ml_config
    )
    print(f"Config {i}: {results['ml']['training_results']}")
```

#### 3. GNN Architecture Study
```python
# Compare GNN architectures and features
configs = [
    {"model_type": "gcn", "node_feature_type": "identity"},
    {"model_type": "gcn", "node_feature_type": "random"},  
    {"model_type": "sage", "node_feature_type": "identity"},
    {"model_type": "sage", "node_feature_type": "random"},
]

for config in configs:
    gnn_config = GNNConfig(**config)
    result = run_link_prediction_experiment(
        edge_list, method="gnn", gnn_config=gnn_config
    )
    print(f"{config}: AUC = {result['gnn']['training_results']['best_val_auc']:.3f}")
```

### Practical Applications

#### 1. Social Network Analysis
```python
# Recommend friend connections
friends_graph = nx.read_edgelist("friendships.txt")
edge_list = list(friends_graph.edges())

results = run_link_prediction_experiment(edge_list, method="both")

# Get top friend recommendations
recommendations = results['ml']['predictions'].head(20)
print("Top friend recommendations:")
for _, row in recommendations.iterrows():
    print(f"User {row['u']} -> User {row['v']}: {row['prediction_score']:.3f}")
```

#### 2. Protein Interaction Prediction
```python
# Predict protein-protein interactions
ppi_graph = nx.read_edgelist("protein_interactions.txt")
edge_list = list(ppi_graph.edges())

# Use ML for interpretable features
ml_config = MLConfig(models_to_try=["random_forest"])
results = run_link_prediction_experiment(
    edge_list, method="ml", ml_config=ml_config
)

# Analyze feature importance (if using sklearn)
# predictor = MLPredictor(ml_config)
# importance = predictor.get_feature_importance()
```

#### 3. Knowledge Graph Completion
```python
# Complete missing relations in knowledge graph
kg_edges = [(entity1, entity2), ...]  # Your KG edges

# Compare both approaches
results = run_link_prediction_experiment(kg_edges, method="both")

# Ensemble predictions
ml_preds = results['ml']['predictions']
gnn_preds = results['gnn']['predictions']

# Simple ensemble: average scores
ensemble = ml_preds.merge(gnn_preds, on=['u', 'v'])
ensemble['ensemble_score'] = (
    ensemble['prediction_score_x'] + ensemble['prediction_score_y']
) / 2
```

## Contributing

The framework is designed to be easily extensible:

### Adding New ML Models
```python
def _get_model_and_grid(self, model_name: str):
    if model_name == "my_new_model":
        model = MyNewModel()
        param_grid = {"param1": [1, 2, 3]}
        return model, param_grid
```

### Adding New Features
```python
def _my_new_features(graph: nx.Graph, edges: List[Tuple[int, int]]):
    features = {}
    features['my_feature'] = [compute_feature(u, v) for u, v in edges]
    return features

# Add to extract_features()
features.update(_my_new_features(graph, edges))
```

### Adding New GNN Architectures
```python
class MyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        # Implementation
        pass
    
    def forward(self, x, edge_index):
        # Implementation
        pass

# Add to GNNPredictor
if self.config.model_type == "my_gnn":
    self.model = MyGNN(...)
```

## License

This framework is provided as-is for research and educational purposes. Feel free to modify and extend for your needs.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{simplified_link_prediction,
  title={Simplified Link Prediction Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Support

For questions, issues, or contributions:
- Check the troubleshooting section above
- Review the examples in `run_experiments.py`
- Examine the source code for implementation details

The framework prioritizes simplicity and clarity over performance optimization, making it ideal for research, education, and rapid prototyping.
