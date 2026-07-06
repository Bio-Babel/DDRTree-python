"""Smoke test for ddrtree._biobabel.

Exercises the one real path through the public API on a tiny synthetic
matrix: no network, no GPU, no on-disk data.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    from ddrtree import DDRTree, DDRTreeResult

    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 20))  # (D, N) = 4 features x 20 samples

    res = DDRTree(X, dimensions=2, max_iter=3, ncenter=5, tol=1e-3)

    assert isinstance(res, DDRTreeResult)
    assert res.W.shape == (4, 2)
    assert res.Z.shape == (2, 20)
    assert res.Y.shape == (2, 5)
    assert res.stree.shape == (20, 20)
    assert len(res.objective_vals) >= 1

    print("ddrtree smoke check OK:", res.Z.shape, res.Y.shape)


if __name__ == "__main__":
    main()
