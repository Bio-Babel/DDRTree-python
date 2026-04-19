"""Unit tests for torch backend internals.

These tests verify the small helpers in ``ddrtree._backends._torch`` in
isolation — important because they are the only pieces of the torch path
that diverge algorithmically from the NumPy reference. Bugs here would
surface as subtle per-iteration drift that a top-level parity test might
mask behind a loose tolerance. We keep assertions strict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree._backends._torch import (
    _torch_cholesky_solve,
    _torch_get_major_eigenvalue,
    _torch_pca_projection,
    _torch_sq_dist,
)
from ddrtree._utils import get_major_eigenvalue, pca_projection, sq_dist


# ----------------------------- sq_dist -------------------------------------


@pytest.mark.parametrize("Da,Na,Nb", [(3, 4, 5), (1, 10, 1), (6, 1, 8)])
def test_torch_sq_dist_matches_numpy(Da: int, Na: int, Nb: int) -> None:
    rng = np.random.default_rng(Da * 31 + Na + Nb)
    a = rng.standard_normal((Da, Na))
    b = rng.standard_normal((Da, Nb))
    out_np = sq_dist(a, b)
    out_t = _torch_sq_dist(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(out_t, out_np, atol=1e-12, rtol=1e-12)


def test_torch_sq_dist_row_mismatch_raises() -> None:
    a = torch.zeros((3, 4))
    b = torch.zeros((5, 4))
    with pytest.raises(ValueError, match=r"row counts differ"):
        _torch_sq_dist(a, b)


# ----------------------------- pca_projection ------------------------------


def test_torch_pca_projection_full_branch_matches_subspace() -> None:
    """L >= min(n, p): compare subspaces (column spans) not raw matrices,
    since eig results can differ in ordering of degenerate eigenvalues and
    per-column sign."""
    rng = np.random.default_rng(0)
    C = rng.standard_normal((4, 4))
    C = (C + C.T) / 2.0                          # symmetric
    L = 4                                        # triggers full-eigen branch
    W_np = pca_projection(C, L)
    W_t = _torch_pca_projection(torch.from_numpy(C), L).numpy()
    # Principal-subspace distance via ||P - P'||_F where P = W W^T (L==d so
    # this is the identity projector; we just need the span to match).
    P_np = W_np @ W_np.T
    P_t = W_t @ W_t.T
    np.testing.assert_allclose(P_t, P_np, atol=1e-8)


def test_torch_pca_projection_truncated_branch_matches_subspace() -> None:
    """L < min(n, p): truncated SVD branch. Compare subspaces; per-column
    sign and ordering of equal singular values are implementation-defined."""
    rng = np.random.default_rng(1)
    # Make C low-rank with clear gap so top-L subspace is well-defined.
    U = rng.standard_normal((10, 3))
    V = rng.standard_normal((10, 3))
    s = np.diag([5.0, 3.0, 1.5])
    C = U @ s @ V.T + 1e-4 * rng.standard_normal((10, 10))
    L = 2
    W_np = pca_projection(C, L)
    W_t = _torch_pca_projection(torch.from_numpy(C), L).numpy()
    P_np = W_np @ np.linalg.pinv(W_np)           # rank-L projector
    P_t = W_t @ np.linalg.pinv(W_t)
    np.testing.assert_allclose(P_t, P_np, atol=1e-5)


def test_torch_pca_projection_output_shape_square() -> None:
    """In DDRTree, ``pca_projection`` is only ever called on square (D,D) or
    (K,K) matrices. For square ``C`` the output is always ``(D, L)``
    regardless of branch."""
    C = torch.randn((7, 7), dtype=torch.float64)
    for L in (1, 2, 5, 7):
        out = _torch_pca_projection(C, L)
        assert out.shape == (7, L)


# --------------------------- get_major_eigenvalue --------------------------


def test_torch_get_major_eigenvalue_full_branch() -> None:
    """When L >= min(n,p): spectral norm squared of C."""
    rng = np.random.default_rng(2)
    C = rng.standard_normal((4, 4))
    exp = get_major_eigenvalue(C, 4)
    got = _torch_get_major_eigenvalue(torch.from_numpy(C), 4)
    assert abs(got - exp) < 1e-10


def test_torch_get_major_eigenvalue_truncated_branch() -> None:
    """When L < min(n,p): the R quirk — max abs entry of top-L right
    singular vectors. Both NumPy and torch use direct SVD in this branch
    so results should be tight."""
    rng = np.random.default_rng(3)
    C = rng.standard_normal((6, 8))
    exp = get_major_eigenvalue(C, 2)
    got = _torch_get_major_eigenvalue(torch.from_numpy(C), 2)
    assert abs(got - exp) < 1e-10


# --------------------------- Cholesky / LU fallback -----------------------


def test_torch_cholesky_solve_psd_matches_direct_solve() -> None:
    """Happy path: PSD A → Cholesky succeeds → result matches linalg.solve."""
    rng = np.random.default_rng(4)
    A_np = rng.standard_normal((5, 5))
    A_np = A_np @ A_np.T + np.eye(5)             # SPD
    B_np = rng.standard_normal((5, 3))
    A_t = torch.from_numpy(A_np)
    B_t = torch.from_numpy(B_np)
    X_got = _torch_cholesky_solve(A_t, B_t, iter_index=0).numpy()
    X_exp = np.linalg.solve(A_np, B_np)
    np.testing.assert_allclose(X_got, X_exp, atol=1e-10, rtol=1e-10)


def test_torch_cholesky_solve_non_psd_falls_back_to_lu() -> None:
    """When A is symmetric but indefinite (non-PSD), Cholesky must fail
    cleanly via info>0, we warn, and LU recovers the exact solution."""
    # Construct a symmetric indefinite, well-conditioned matrix.
    A_np = np.array(
        [[1.0, 0.0, 0.5], [0.0, -2.0, 0.1], [0.5, 0.1, 3.0]], dtype=np.float64
    )
    # Sanity: eigenvalues span both signs.
    w = np.linalg.eigvalsh(A_np)
    assert (w < 0).any() and (w > 0).any()
    B_np = np.array([[1.0], [2.0], [3.0]])
    A_t = torch.from_numpy(A_np)
    B_t = torch.from_numpy(B_np)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_got = _torch_cholesky_solve(A_t, B_t, iter_index=7).numpy()
    assert any(
        issubclass(w_.category, RuntimeWarning) and "iteration 7" in str(w_.message)
        for w_ in caught
    ), "fallback did not emit the expected RuntimeWarning"
    X_exp = np.linalg.solve(A_np, B_np)
    np.testing.assert_allclose(X_got, X_exp, atol=1e-10, rtol=1e-10)


def test_torch_cholesky_solve_singular_lu_raises() -> None:
    """If even LU cannot decompose (truly singular A), the fallback must
    surface the failure rather than return garbage."""
    # Singular: rank 1 matrix.
    A_np = np.array([[1.0, 2.0], [2.0, 4.0]])
    B_np = np.array([[1.0], [1.0]])
    A_t = torch.from_numpy(A_np)
    B_t = torch.from_numpy(B_np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we do not care about the PSD warning here
        with pytest.raises(np.linalg.LinAlgError):
            _torch_cholesky_solve(A_t, B_t, iter_index=0)
