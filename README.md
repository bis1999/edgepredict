# EdgePredict

A comprehensive machine learning framework for **graph link prediction** that supports multiple scenarios, rich feature extraction, and both **traditional ML** and **Graph Neural Network (GNN)** approaches.  
It also includes a **meta-learner** that predicts the best-performing model for a given graph, saving time and avoiding trial-and-error.

---

## 📦 Installation

Install from source:

```bash
git clone https://github.com/bis1999/edgepredict.git
cd edgepredict
pip install -e .
```

Optional extras for a smoother download experience:

```bash
pip install "edgepredict[download]"
```

> Python 3.10–3.12 recommended. See `requirements.txt` for pinned versions used during development.

---

## 🔽 Download pretrained models

Pretrained meta-learner models are required. By default, they are stored in **`meta_learner_models/`** at the project root.

### Option A — Python API (recommended)

```python
from linkpredx.git_link_script.link_prediction.models import ensure_models

# Downloads release v0.1.0 assets (with resume/retries) into ./meta_learner_models/
models_dir = ensure_models()
print("Models ready at:", models_dir)
```

✔️ Resume support  
✔️ SHA-256 checksum verification  
✔️ Uses `GITHUB_TOKEN` if available (avoids throttling)

### Option B — wget (Jupyter-friendly)

Paste this as a single **`%%bash`** cell in Jupyter, or run in your shell:

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

Feature extraction is controlled via a simple YAML file (e.g., `configs/feature.yaml`). Toggle features on/off with booleans. Typical groups include:

- **Classical**: Common Neighbors, Jaccard, Adamic–Adar, Resource Allocation, LHN
- **Centralities**: Degree, Closeness, Betweenness, Eigenvector, PageRank
- **Clustering & triangles**
- **SVD-based**: LRA / dLRA / mLRA variants

Example snippet:

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

We support **three evaluation scenarios**, each matching a real-world use case.  
**Mantra:** features for a label set are computed on the **graph available at prediction time**.

### 1) Simulation
**Goal:** “Can we re-discover held-out true edges?”  
- Build observed graph \( G' \) by sampling edges from \( G \) while keeping it connected.  
- Holdout/test positives \( Y = E \setminus E' \); add an equal number of negatives from non-edges of \( G \).  
- Compute **test features** on \( G' \) (those edges weren’t visible at prediction time).  
- Build training support graph \( G'' \subset G' \) (connected).  
- Train positives \( Y' = E' \setminus E'' \) plus negatives from non-edges of \( G' \).  
- Compute **train features** on \( G'' \).  
- (Optional) apply oversampling **on train only**. Cross-validation folds are supported.

### 2) Discovery
**Goal:** “Score new non-edges in the current graph.”  
- Test = a (capped) set of non-edges of \( G \) (unlabeled).  
- Training uses an observed \( G' \subset G \) to avoid look-ahead leakage.  
- Train positives \( E \setminus E' \), negatives from non-edges of \( G \).  
- **Test features** on \( G \); **Train features** on \( G' \).

### 3) Specific
**Goal:** “Score a user-supplied set of pairs.”  
- Test = provided pairs; features computed on \( G \).  
- Training uses a connected subset \( G_{\text{train}} \subset G \).  
- Train positives \( E \setminus E_{\text{train}} \), negatives from non-edges of \( G \).  
- **Train features** on \( G_{\text{train}} \).

---

## 🚀 Quickstart (programmatic)

```python
# 1) Ensure pretrained meta-learner models are present
from linkpredx.git_link_script.link_prediction.models import ensure_models
ensure_models()  # downloads into ./meta_learner_models if needed

# 2) Prepare a simple graph
from linkpredx.git_link_script.core.dataset_preparer import LinkPredictionDatasetPreparer, DatasetConfig
edges = [(1,2), (2,3), (3,4), (1,4), (2,4)]
cfg = DatasetConfig(validation_frac=0.2, cv_folds=5, random_state=42)
prep = LinkPredictionDatasetPreparer(edges, cfg)

# 3) Choose a scenario
train_data, test_data = prep.prepare_simulation()        # or:
# train_data, test_data = prep.prepare_discovery()
# train_data, test_data = prep.prepare_specific(pairs=[(1,5), (2,5)])

# 4) Train baselines (RF/XGB/LogReg; GNNs if enabled)
from linkpredx.git_link_script.trainer import LinkPredictionTrainer
trainer = LinkPredictionTrainer(train_data, test_data)
results = trainer.run_all()   # returns metrics tables
print(results)
```

---

## 🧠 Meta-learner usage

Use graph-level statistics to predict the best algorithm and expected performance (AUC / Top-K) **before** running heavy training:

```python
from linkpredx.git_link_script.graph_predictor import GraphPredictor
from linkpredx.git_link_script.link_prediction.models import ensure_models

ensure_models()  # makes sure meta_learner_models/ is populated
gp = GraphPredictor(models_path="meta_learner_models")

# Run on an edgelist file (or supply a NetworkX graph in your implementation)
pred = gp.predict_from_graph("example_networks/facebook.edgelist")

print("Best model:", pred["best_model"])
print("Predicted AUC:", pred["auc"])
print("Predicted Top-K:", pred["topk"])
```

---

## 📂 Repository layout

```
git_link_script/
 ├── configs/                 # YAML configs (feature toggles, etc.)
 ├── example_networks/        # small example graphs
 ├── Link Prediction Demo Notebook.ipynb
 ├── link_prediction/
 │    ├── core/               # dataset preparation & orchestration
 │    ├── features/           # feature extraction modules
 │    ├── models/             # pretrained model downloader (ensure_models)
 │    ├── prediction/         # meta-learner & predictors
 │    ├── utils/              # logging & helpers
 ├── meta_learner_models/     # pretrained meta-learner models (downloaded here)
 └── requirements.txt
```

---

## 🤝 Contributing

- Open issues for bugs or feature requests.  
- PRs welcome (please include tests where applicable).

---

## 📄 License

MIT License © 2025 Bisman Singh
