## 🧠 Core ideas & scenarios (plain Markdown — no LaTeX)

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
