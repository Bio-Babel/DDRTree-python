# DDRTree-python

A pure NumPy/SciPy port of the [DDRTree](https://cran.r-project.org/package=DDRTree) R
package — **Discriminative Dimensionality Reduction via learning a Tree** — from the
KDD'15 paper by Qi Mao, Li Wang, Steve Goodison and Yijun Sun.

DDRTree simultaneously:

1. Reduces high-dimensional data to a low-dimensional latent space `Z`,
2. Learns an explicit smooth principal tree graph embedded in that space, and
3. Obtains a soft clustering of points onto the tree nodes.

It is the dimensionality-reduction backbone used by Monocle 2 for single-cell
pseudotime / branching-trajectory inference, but is a general-purpose algorithm
for any data with a tree-like intrinsic structure.

## Installation

```
pip install ddrtree             # NumPy backend only
pip install ddrtree[torch]      # + PyTorch backend (CPU / CUDA)
```

The core depends on `numpy`, `scipy`, and `scikit-learn`. The optional
`torch` extra enables the GPU-friendly backend — no C/C++ extensions are
built either way.

## Quick start

```python
import numpy as np
from ddrtree import DDRTree

# X is a D x N matrix (features x samples), matching the R convention.
rng = np.random.default_rng(0)
X = rng.standard_normal((10, 200))

res = DDRTree(X, dimensions=2, max_iter=20,
              sigma=1e-3, lambda_=None, ncenter=50,
              gamma=10.0, tol=1e-3, verbose=False)

res.Z      # 2 x 200  reduced-dimension embedding
res.Y      # 2 x 50   principal-graph node coordinates
res.W      # 10 x 2   orthogonal projection basis
res.stree  # N x N scipy.sparse MST weights (first K x K block populated)
res.objective_vals  # objective at each iteration
```

## Backends

`DDRTree` dispatches to one of two computational backends via the
`backend` argument. The public function signature is otherwise unchanged.
Backend selection is always explicit — there is no auto-detection.

| `backend` | Executes on | Typical use                                  |
| --------- | ----------- | -------------------------------------------- |
| `"numpy"` | CPU (NumPy) | **Default.** Reference path, aligned with R. |
| `"torch"` | CPU / CUDA  | GPU acceleration (Borůvka MST, fast BLAS).   |

```python
# GPU path — requires torch with CUDA
res = DDRTree(X, ncenter=500, backend="torch", device="cuda")

# Half precision (torch backend only). Memory ½, throughput ~1.5–2× on
# CUDA, ~1e-3 relative drift in the converged embedding.
res = DDRTree(X, ncenter=500, backend="torch", device="cuda", dtype="float32")
```

Torch backend runs a **pure-torch parallel Borůvka** (`O(log K)` rounds)
for the MST step — no host round trip per iteration. Explicit
`mst_algorithm="prim"` or `"kruskal"` routes MST through NumPy / SciPy
for strict parity testing against the R gold standard; see
`tests/test_boruvka_integration.py`.

### Known differences between backends

* **PCA initialisation.** NumPy uses `scipy.sparse.linalg.svds` (iterative,
  Lanczos, matches R's `irlba`). Torch uses a direct `torch.linalg.svd`
  for the truncated branch — identical subspace, small per-iteration
  numerical drift.
* **K-means.** Both backends call `sklearn.cluster.KMeans` on CPU (small K,
  negligible overhead). No GPU K-means yet.
* **Cholesky fallback.** When the `tmp_M` system drifts non-PSD both
  backends fall back to LU and emit a `RuntimeWarning`. The Y-update
  Cholesky is strict in both backends — the `(λ/γ)L + Γ` system is PSD
  by construction, any failure there is surfaced rather than masked.

## Numerical parity with the R package

The test suite runs the same inputs through the R `DDRTree` package and compares
the results, for both backends. Eigen-vector sign flips (inherent to
eigen-decompositions) are handled in tests. See
`tests/scripts/generate_gold_standard.R`, `tests/test_ddrtree.py` (NumPy)
and `tests/test_backend_torch_gold.py` (Torch).

## Reference

Qi Mao, Li Wang, Steve Goodison, Yijun Sun.
*Dimensionality Reduction via Graph Structure Learning*.
KDD'15. https://dl.acm.org/doi/10.1145/2783258.2783309
