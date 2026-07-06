---
name: use-ddrtree
description: Use when you need to reduce a numeric data matrix to a low-dimensional embedding while jointly learning an explicit principal tree graph and a soft clustering onto that graph — e.g. the dimensionality-reduction backbone behind Monocle 2 pseudotime/trajectory inference.
---

# ddrtree

Python port of the R `DDRTree` package (Discriminative Dimensionality
Reduction via learning a Tree, Mao et al. KDD'15). It is a single-call
tool: `ddrtree.DDRTree(X, ...)` takes a `(D, N)` numeric matrix (D features
x N samples, R's convention) and returns a `DDRTreeResult` with the
reduced-dimension embedding, the principal-graph node coordinates, and the
minimum-spanning-tree topology connecting them — all computed together in
one alternating-optimisation loop. There is no persistent state object and
no multi-step API: everything downstream (`pca_projection`, `sq_dist`,
`get_major_eigenvalue`) is an internal helper `DDRTree` calls itself, not a
separate step a caller chains manually.

## When to use / when not to use

Use it when you need an *explicit tree graph* over the reduced points, not
just an embedding — e.g. as the dimensionality-reduction step feeding a
downstream pseudotime/branching-trajectory pipeline. Do not use it for a
plain embedding with no graph structure (PCA/UMAP/t-SNE instead), and do
not feed it raw, unprocessed single-cell counts — `DDRTree` expects an
already-preprocessed numeric matrix, it does not normalize or QC data.

## Entry points

- `ddrtree.DDRTree(X, dimensions=2, ncenter=..., backend='numpy'|'torch', ...)`
  — the one function most users need.
- `ddrtree.DDRTreeResult` — the dataclass it returns (`.W`, `.Z`, `.Y`,
  `.stree`, `.X`, `.objective_vals`, `.history`).
- `ddrtree.pca_projection`, `ddrtree.sq_dist`, `ddrtree.get_major_eigenvalue`
  — public but low-level; call these directly only when reimplementing or
  inspecting a piece of the algorithm, not as part of normal usage.

## Quick reference

```python
import numpy as np
from ddrtree import DDRTree

# X is a D x N matrix (features x samples), matching the R convention.
rng = np.random.default_rng(0)
X = rng.standard_normal((10, 200))

res = DDRTree(X, dimensions=2, max_iter=20, sigma=1e-3, lambda_=None,
              ncenter=50, gamma=10.0, tol=1e-3, verbose=False)

res.Z      # 2 x 200  reduced-dimension embedding
res.Y      # 2 x 50   principal-graph node coordinates
res.stree  # 200 x 200 scipy.sparse MST weights (leading 50 x 50 block populated)
```

GPU execution: pass `backend='torch', device='cuda'` (optionally
`dtype='float32'`); requires the `torch` extra installed.

For more: `biobabel.describe_package(import_name="ddrtree")`.
