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
pip install ddrtree
```

Only depends on `numpy`, `scipy`, and `scikit-learn` — no C/C++ extensions.

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

## Numerical parity with the R package

The test suite runs the same inputs through the R `DDRTree` package and compares
the results. Eigen-vector sign flips (inherent to eigen-decompositions) are
handled in tests. See `tests/scripts/generate_gold_standard.R` and
`tests/test_ddrtree.py`.

## Reference

Qi Mao, Li Wang, Steve Goodison, Yijun Sun.
*Dimensionality Reduction via Graph Structure Learning*.
KDD'15. https://dl.acm.org/doi/10.1145/2783258.2783309
