"""Borůvka end-to-end integration with the torch backend (P3).

When ``backend="torch"`` (and ``mst_algorithm`` is left unset) the torch
backend runs the GPU-friendly Borůvka MST rather than routing through
host-side NumPy Prim. Because DDRTree's Y-center distances are
continuous, the MST edge set under Borůvka and Prim must be identical —
a strong correctness gate for the GPU path.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree import DDRTree


def _X(seed: int = 0, N: int = 50, D: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, N))
    X[0] += np.linspace(-1, 1, N)
    return X


def test_torch_default_is_boruvka_and_matches_prim_path() -> None:
    """Torch default (Borůvka) must produce the same MST edge set as torch
    with explicit Prim. Continuous weights → unique MST → strict match."""
    X = _X(0)
    kw = dict(dimensions=2, ncenter=15, max_iter=10, tol=1e-6, backend="torch")
    res_boruvka = DDRTree(X, **kw)                         # default
    res_prim = DDRTree(X, **kw, mst_algorithm="prim")
    res_explicit_boruvka = DDRTree(X, **kw, mst_algorithm="boruvka")

    # Default == explicit boruvka (same code path).
    np.testing.assert_array_equal(
        res_boruvka.stree.toarray(), res_explicit_boruvka.stree.toarray()
    )

    # Edge set under Borůvka == edge set under Prim (unique weights).
    e_b = set(zip(*np.nonzero(res_boruvka.stree.toarray())))
    e_p = set(zip(*np.nonzero(res_prim.stree.toarray())))
    assert e_b == e_p


def test_numpy_rejects_boruvka_request() -> None:
    """Requesting Borůvka from the NumPy backend must fail loudly rather
    than silently fall back to Prim — the user explicitly asked for a
    different algorithm."""
    X = _X(0)
    with pytest.raises(ValueError, match=r"boruvka.*only available.*torch"):
        DDRTree(X, ncenter=10, backend="numpy", mst_algorithm="boruvka")


def test_numpy_default_still_prim_and_unchanged() -> None:
    """The NumPy backend's default MST is still Prim; passing
    mst_algorithm=None yields the same result as ``mst_algorithm="prim"``
    and the same as not passing mst_algorithm at all."""
    X = _X(1)
    kw = dict(dimensions=2, ncenter=8, max_iter=8)
    a = DDRTree(X, **kw)
    b = DDRTree(X, **kw, mst_algorithm=None)
    c = DDRTree(X, **kw, mst_algorithm="prim")
    np.testing.assert_array_equal(a.stree.toarray(), b.stree.toarray())
    np.testing.assert_array_equal(a.stree.toarray(), c.stree.toarray())


def test_torch_explicit_prim_bypasses_boruvka() -> None:
    """With mst_algorithm='prim' the torch backend must route through the
    NumPy Prim, producing results identical to the numpy backend's MST."""
    X = _X(2)
    kw = dict(dimensions=2, ncenter=12, max_iter=8, tol=1e-6)
    res_np = DDRTree(X, **kw, backend="numpy")
    res_torch_prim = DDRTree(X, **kw, backend="torch", mst_algorithm="prim")

    # MST edge set and weights should match bit-for-bit: both use the
    # same NumPy _prim_mst on the same K×K distance matrix.
    np.testing.assert_allclose(
        res_np.stree.toarray(), res_torch_prim.stree.toarray(),
        atol=1e-8, rtol=1e-8,
    )


@pytest.mark.parametrize("K", [5, 10, 25])
def test_boruvka_end_to_end_tree_has_Kminus1_edges(K: int) -> None:
    X = _X(0, N=max(40, 3 * K))
    res = DDRTree(X, dimensions=2, ncenter=K, max_iter=8, backend="torch")
    tree_block = res.stree.toarray()[:K, :K]
    # Symmetric, K-1 unique edges → 2*(K-1) non-zero entries.
    np.testing.assert_allclose(tree_block, tree_block.T)
    assert np.count_nonzero(tree_block) == 2 * (K - 1)


def test_torch_invalid_mst_algorithm_message_mentions_boruvka() -> None:
    X = _X(0)
    with pytest.raises(ValueError) as exc:
        DDRTree(X, ncenter=5, backend="torch", mst_algorithm="nope")
    msg = str(exc.value)
    assert "prim" in msg and "kruskal" in msg and "boruvka" in msg
