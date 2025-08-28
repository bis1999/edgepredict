# EdgePredict

A comprehensive machine learning framework for **graph link prediction** that supports multiple scenarios, rich feature extraction, and both **traditional ML** and **Graph Neural Network (GNN)** approaches.  
It also includes a **meta-learner** that predicts the best-performing model for a given graph, saving time and avoiding trial-and-error.

> **Paper:** *Meta-learning optimizes predictions of missing links in real-world networks* — Bisman Singh, Lucy Van Kleunen, Aaron Clauset.  
> arXiv: https://arxiv.org/abs/2508.09069
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1emo9Q3MlAAHp0uqtYy1h8bx18M_tXKVj#scrollTo=1f0f001d-696f-408e-acb1-9435a83bda53)


---

## 📦 Dev Setup

You and your users can run everything directly from a clone (no packaging needed).

### 1) Create an environment
**Conda**
```bash
conda create -n edgepredict python=3.11 -y
conda activate edgepredict
```
**OR venv**
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Clone the repo
```bash
git clone https://github.com/bis1999/edgepredict.git
cd edgepredict
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If PyTorch/PyG wheels complain on your platform, install Torch first from the official instructions for your OS/CUDA, then re-run `pip install -r requirements.txt`.

### 4) Imports (no packaging required)
When you run from the **repo root**, Python can import `link_prediction` directly.  
In notebooks or scripts launched from the repo root, you can do:

```python
from link_prediction.models.downloader import ensure_models
```

If you run from a different working directory, add the repo path:

```python
import os, sys
sys.path.append(os.path.abspath("/path/to/edgepredict"))
```

---

## 🔽 Download pretrained models

Pretrained meta-learner models are stored in **`meta_learner_models/`** at the repo root.

### Option A — Python API (recommended)

```python
from link_prediction.models.downloader import ensure_models

# Downloads release v0.1.0 assets (with resume/retries) into ./meta_learner_models/
models_dir = ensure_models("meta_learner_models")
print("Models ready at:", models_dir)
```

✔️ Resume support • ✔️ SHA-256 verification • ✔️ Uses `GITHUB_TOKEN` if available

### Option B — wget (Jupyter-/shell-friendly)
Paste this as a **`%%bash`** cell in Jupyter, or run in your shell:

```bash
%%bash
set -euo pipefail
TAG="v0.1.0"
OWNER="bis1999"
REPO="edgepredict"
BASE="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}"

mkdir -p meta_learner_models
cd meta_learner_models

# Download with resume (-c)
wget -c "${BASE}/auc_label_encoder.pkl"
wget -c "${BASE}/auc_score_regressor.pkl"
wget -c "${BASE}/best_auc_model_classifier.pkl"
wget -c "${BASE}/best_topk_model_classifier.pkl"
wget -c "${BASE}/topk_label_encoder.pkl"
wget -c "${BASE}/topk_score_regressor.pkl"

# Verify SHA-256 checksums
cat > checksums.sha256 <<'EOF'
41be0890f06d446db2085d7cde664ff1b47ca8e87985b34f9a7e06af41a0eb34  auc_label_encoder.pkl
2bdb2f204e371d748aeea85a3a8c046afe914b3917b98e870b4fe1a4d7bf99fe  auc_score_regressor.pkl
28013ab22d9111e24180ab17d9176ba661cf67ac566a8c820ee3dbb8baba506d  best_auc_model_classifier.pkl
36b54a5e171991e1b1b312427bdaefc000c90bd5387a6e4af4fffc7a50840ada  best_topk_model_classifier.pkl
178ce76129e643263c22faee77f675f0941ff10c119eb41b74e74a017c9b8616  topk_label_encoder.pkl
9fe6ddd7f36a95e3e54a5a2f887494560a744213e6d2c8f2e8da6f1a54e799f8  topk_score_regressor.pkl
EOF

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c checksums.sha256
else
  shasum -a 256 -c checksums.sha256
fi
```

---

## ⚙️ Configuration

Feature extraction is controlled via YAML (see `configs/feature.yaml`). Toggle features on/off with booleans. Typical groups include:

- **Classical**: Common Neighbors, Jaccard, Adamic–Adar, Resource Allocation, LHN  
- **Centralities**: Degree, Closeness, Betweenness, Eigenvector, PageRank  
- **Clustering & triangles**  
- **SVD-based**: LRA / dLRA / mLRA variants

Example:
```yaml
classical:
  common_neighbors: true
  jaccard: true
  adamic_adar: true
  resource_allocation: true
  lhn: false

centrality:
  degree: true
  pagerank: true

svd:
  enabled: true
  k: 64
```

---

## 🧠 Core ideas & scenarios 

We support **three evaluation scenarios** matching common use cases.  
**Mantra:** compute features on the **graph available at prediction time** to avoid look-ahead leakage.

### 1) Simulation
**Goal:** re-discover held-out true edges.

- Build an observed graph `G'` by sampling edges from `G` (keep it connected).
- Test positives: `Y = E - E'`; add the same number of negatives sampled from non-edges of `G`.
- Compute **test features** on `G'` (because those edges weren’t visible when predicting).
- Build a training support graph `G''` that is a connected subgraph of `G'`.
- Train positives: `Y' = E' - E''`; add negatives sampled from non-edges of `G'`.
- Compute **train features** on `G''`.
- Optional: apply class balancing **only on train**. Stratified cross-validation is supported.

### 2) Discovery
**Goal:** score new non-edges in the current graph.

- Test set = a capped set of non-edges of `G` (unlabeled).
- Training uses an observed graph `G'` that is a subset of `G` to avoid look-ahead leakage.
- Train positives: `E - E'`; negatives from non-edges of `G`.
- Compute **test features** on `G` and **train features** on `G'`.

### 3) Specific
**Goal:** score a user-supplied set of pairs.

- Test set = the provided pairs; compute features on `G`.
- Training uses a connected subset `G_train` of `G`.
- Train positives: `E - E_train`; negatives from non-edges of `G`.
- Compute **train features** on `G_train`.
---

## 🚀 Quickstart (programmatic)

```python
# 1) Prepare a small graph (edge list)
from link_prediction.core.dataset_preparer import LinkPredictionDatasetPreparer, DatasetConfig
edges = [(1,2), (2,3), (3,4), (1,4), (2,4)]
cfg = DatasetConfig(validation_frac=0.2, cv_folds=5, random_state=42)
prep = LinkPredictionDatasetPreparer(edges, cfg)

# 2) Choose a scenario
train_data, test_data = prep.prepare_simulation()        # or:
# train_data, test_data = prep.prepare_discovery()
# train_data, test_data = prep.prepare_specific(pairs=[(1,5), (2,5)])

# 3) Train baselines
from link_prediction.training.trainer import LinkPredictionTrainer
trainer = LinkPredictionTrainer(train_data, test_data)
results = trainer.run_all()   # returns metrics tables
print(results)
```

---

## 🧠 Meta-learner usage

Use graph-level statistics to predict the best algorithm and expected performance (AUC / Top-K) **before** training:

```python
# 1) Ensure pretrained models are present
from link_prediction.utils.graph_predictor import GraphPredictor
from link_prediction.models.downloader import ensure_models
ensure_models("meta_learner_models")  # downloads + verifies
gp = GraphPredictor(models_path="meta_learner_models")

# Example: run on an edgelist file
pred = gp.predict_from_graph("example_networks/facebook.edgelist")
print("Best model:", pred["best_model"])
print("Predicted AUC:", pred["auc"])
print("Predicted Top-K:", pred["topk"])
```

---

## 📁 Repository layout 

```
edgepredict/
├── configs/
├── example_networks/
├── link_prediction/
│   ├── core/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── utils/
├── meta_learner_models/
├── Link Prediction Demo Notebook.ipynb
├── requirements.txt
├── LICENSE
├── README.md
├── .gitattributes
└── .gitignore
```

### Example `src/` layout 

```
src/
├── link_prediction/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── dataset_preparer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py
│   │   ├── gnn_trainer.py
│   │   └── ml_models.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   └── features_light.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── edge_sampler.py
│   │   ├── graph_processor.py
│   │   ├── graph_predictor.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── pair_dataset.py
│   │   └── utils.py
```

---

## ✍️ Authors & Links

- **Paper:** *Meta-learning optimizes predictions of missing links in real-world networks*  
  Bisman Singh, Lucy Van Kleunen, Aaron Clauset — arXiv: https://arxiv.org/abs/2508.09069
- **Author:** Bisman Singh — Email: <singh.bisman7@gmail.com> — LinkedIn: https://www.linkedin.com/in/bisman-singh1999/

---

## 🤝 Contributing

- Open issues for bugs or feature requests.  
- PRs welcome (please include tests where applicable).

---

## 📄 License

MIT License © 2025 Bisman Singh
