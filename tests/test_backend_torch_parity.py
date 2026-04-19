"""Parity tests for ``backend="torch"`` (CPU path, P2).

Baseline is the NumPy backend — the torch backend must converge to the
*same* principal graph at roughly the same rate, even if intermediate
floats drift slightly (sources of drift: ``torch.linalg.svd`` vs
``scipy.sparse.linalg.svds`` in PCA init, ``cholesky_solve`` vs
NumPy triangular solves, and BLAS vendor differences).

Tolerances are deliberately tight enough to catch real bugs (a missing
term or a transposed matrix would blow them out by orders of magnitude).
The tests are *not* allowed to silently mask discrepancies — they assert
positive equalities (shapes, MST edge topology, iteration count) and use
``np.testing.assert_allclose`` with explicit ``atol``/``rtol``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree import DDRTree


def _align_signs(P: np.ndarray, ref: np.ndarray, pc_axis: int) -> np.ndarray:
    """Flip signs along ``pc_axis`` so each PC best aligns with ``ref``.

    Eigenvector / singular-vector sign is not unique: SVD implementations
    freely choose ``v`` or ``-v``. DDRTree's objective is invariant under
    coupled sign flips of W/Z/Y along matching components, so two backends
    can legitimately converge to solutions that differ only in per-component
    signs. This helper neutralises that ambiguity before numerical
    comparison. ``pc_axis=1`` for W (D, d); ``pc_axis=0`` for Z, Y (d, ·).
    """
    assert P.shape == ref.shape
    non_pc_axis = 1 - pc_axis
    signs = np.sign(np.sum(P * ref, axis=non_pc_axis))
    signs[signs == 0] = 1.0
    shape = [1, 1]
    shape[pc_axis] = -1
    return P * signs.reshape(shape)


def _iris_like(seed: int = 0, N: int = 50, D: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, N))
    # Add a little structure so DDRTree has something to recover.
    X[0] += np.linspace(-1, 1, N)
    X[1] += np.sin(np.linspace(0, 3.14, N))
    return X


@pytest.mark.parametrize(
    "ncenter,dim,seed",
    [
        (None, 2, 0),   # ncenter=N (no KMeans)
        (10, 2, 0),     # modest K
        (15, 2, 3),     # different K and seed
        (8,  1, 2),     # 1-D embedding
    ],
)
def test_torch_matches_numpy_on_small_problem(
    ncenter: int, dim: int, seed: int
) -> None:
    X = _iris_like(seed)
    kw = dict(dimensions=dim, max_iter=15, tol=1e-6)
    if ncenter is not None:
        kw["ncenter"] = ncenter

    a = DDRTree(X, **kw, backend="numpy")
    b = DDRTree(X, **kw, backend="torch")

    # Shapes and iteration count must match exactly.
    assert a.W.shape == b.W.shape
    assert a.Z.shape == b.Z.shape
    assert a.Y.shape == b.Y.shape
    assert len(a.objective_vals) == len(b.objective_vals)

    # Objective is invariant to sign flips of principal components, so we
    # compare it directly — this is the strongest sign-free parity check.
    np.testing.assert_allclose(
        a.objective_vals, b.objective_vals, atol=1e-3, rtol=1e-6
    )

    # Reconstruction W @ Z is the sign-invariant view on W and Z.
    recon_a = a.W @ a.Z
    recon_b = b.W @ b.Z
    np.testing.assert_allclose(recon_a, recon_b, atol=1e-4, rtol=1e-4)

    # After sign alignment, W/Z/Y themselves should agree tightly.
    W_b = _align_signs(b.W, a.W, pc_axis=1)
    Z_b = _align_signs(b.Z, a.Z, pc_axis=0)
    Y_b = _align_signs(b.Y, a.Y, pc_axis=0)
    np.testing.assert_allclose(a.W, W_b, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(a.Z, Z_b, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(a.Y, Y_b, atol=1e-4, rtol=1e-4)

    # MST structure. Since P2 still uses the NumPy Prim under the hood
    # for both backends, the edge set must be identical. (P3 introduces
    # Borůvka; that test moves to test_boruvka.py.)
    a_edges = set(zip(*np.nonzero(a.stree.toarray())))
    b_edges = set(zip(*np.nonzero(b.stree.toarray())))
    assert a_edges == b_edges, "MST edge set differs between backends"


def test_torch_returns_numpy_arrays_not_tensors() -> None:
    """The public contract is that results are NumPy arrays regardless of
    backend; downstream code relying on ``res.W.shape``, ``res.W @ other``
    etc. must keep working without special-casing."""
    X = _iris_like(0)
    res = DDRTree(X, dimensions=2, ncenter=6, max_iter=5, backend="torch")
    assert isinstance(res.W, np.ndarray)
    assert isinstance(res.Z, np.ndarray)
    assert isinstance(res.Y, np.ndarray)
    assert isinstance(res.X, np.ndarray)
    # stree is scipy sparse in both backends
    import scipy.sparse as sp

    assert sp.issparse(res.stree)


def test_torch_backend_respects_lambda_default() -> None:
    """When ``lambda_`` is not set, both backends should default to 5*N and
    produce matching objective_vals[0] within tight tolerance."""
    X = _iris_like(1)
    a = DDRTree(X, dimensions=2, ncenter=8, max_iter=1, backend="numpy")
    b = DDRTree(X, dimensions=2, ncenter=8, max_iter=1, backend="torch")
    assert abs(a.objective_vals[0] - b.objective_vals[0]) < 1e-4


def test_torch_backend_mst_kruskal_path() -> None:
    """``mst_algorithm="kruskal"`` must work under the torch backend too."""
    X = _iris_like(4)
    res = DDRTree(
        X, dimensions=2, ncenter=8, max_iter=5,
        mst_algorithm="kruskal", backend="torch",
    )
    # K-1 edges in the K×K block, symmetric
    tree = res.stree.toarray()
    K = 8
    mst_block = tree[:K, :K]
    np.testing.assert_allclose(mst_block, mst_block.T)
    assert np.count_nonzero(mst_block) == 2 * (K - 1)


def test_torch_backend_invalid_mst_algorithm_raises() -> None:
    X = _iris_like(0)
    with pytest.raises(ValueError, match=r"mst_algorithm must be"):
        DDRTree(X, ncenter=5, mst_algorithm="bogus", backend="torch")


def test_torch_backend_invalid_X_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"must be 2D"):
        DDRTree(np.zeros(10), backend="torch")


def test_torch_backend_ncenter_too_large_raises() -> None:
    X = _iris_like(0, N=20)
    with pytest.raises(ValueError, match=r"ncenter must be"):
        DDRTree(X, ncenter=21, backend="torch")


def test_torch_backend_initial_method_is_called() -> None:
    """``initial_method`` callback is used to seed Z — same semantics as R."""
    X = _iris_like(0, N=30)

    calls = []

    def init(X_arg, alpha=0):
        calls.append(alpha)
        rng = np.random.default_rng(alpha)
        return rng.standard_normal((X_arg.shape[1], 3))  # (N, d_out >= 2)

    res = DDRTree(
        X, dimensions=2, ncenter=10, max_iter=3,
        initial_method=init, alpha=7, backend="torch",
    )
    assert calls == [7]
    assert res.Z.shape == (2, 30)
