"""Per-function R-vs-Python parity tests for the three exported helpers.

Gold-standard outputs are dumped by ``tests/scripts/generate_utility_gold.R``
so we can verify our ports of ``sq_dist``, ``pca_projection`` and
``get_major_eigenvalue`` against R for matrices that span both code paths
(full-eigen vs irlba branches).
"""

from pathlib import Path

import numpy as np
import pytest

from ddrtree import get_major_eigenvalue, pca_projection, sq_dist


GOLD = Path(__file__).parent / "data" / "utils"


def _load(name: str) -> np.ndarray:
    return np.loadtxt(GOLD / f"{name}.tsv", delimiter="\t",
                      dtype=np.float64, ndmin=2)


def _load_scalar(name: str) -> float:
    return float(np.loadtxt(GOLD / f"{name}.txt", dtype=np.float64))


# ---------------------------------------------------------------------------
# sq_dist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", [1, 2, 3, 4],
                         ids=["square_DxN", "rect_DxNa_DxNb",
                              "single_col_each", "wide_low_D"])
def test_sq_dist_matches_r_gold(idx):
    a = _load(f"sqdist_a{idx}")
    b = _load(f"sqdist_b{idx}")
    out_r = _load(f"sqdist_out{idx}")
    out_py = sq_dist(a, b)
    assert out_py.shape == out_r.shape
    np.testing.assert_allclose(out_py, out_r, atol=1e-12, rtol=1e-12)


def test_sq_dist_self_is_symmetric_and_zero_diagonal():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 7))
    out = sq_dist(a, a)
    np.testing.assert_allclose(out, out.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-12)


def test_sq_dist_dtype_promotion_from_int():
    """Integer input should be promoted to float64 internally."""
    a = np.array([[0, 1], [0, 0]], dtype=np.int64)
    b = np.array([[0, 0], [0, 1]], dtype=np.int64)
    out = sq_dist(a, b)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.array([[0.0, 1.0], [1.0, 2.0]]), atol=1e-12)


def test_sq_dist_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="row counts differ"):
        sq_dist(np.zeros((3, 4)), np.zeros((4, 4)))


def test_sq_dist_handles_single_column():
    a = np.array([[1.0], [0.0], [0.0]])
    b = np.array([[0.0], [1.0], [0.0]])
    out = sq_dist(a, b)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# pca_projection
# ---------------------------------------------------------------------------


def _columns_span_same_subspace(A: np.ndarray, B: np.ndarray,
                                tol: float = 1e-6) -> bool:
    """A and B span the same column subspace iff ||AA^T - BB^T||_F is small.

    Both should have orthonormal columns of the same shape.
    """
    return float(np.linalg.norm(A @ A.T - B @ B.T)) < tol


def test_pca_projection_full_path_4x4_L4():
    """L >= min(dim(C)): R uses full eigen(); ours uses eigh. Subspace match."""
    C = _load("pca_C1")
    W_r = _load("pca_W1_L4")
    W_py = pca_projection(C, 4)
    assert W_py.shape == W_r.shape == (4, 4)
    assert _columns_span_same_subspace(W_py, W_r)


@pytest.mark.parametrize("L", [2, 3], ids=["L2", "L3"])
def test_pca_projection_irlba_path_6x6(L):
    """L < min(dim(C)): R uses irlba (approximate); we use eigh (exact).

    They span the same top-L subspace (within irlba's converged accuracy).
    """
    C = _load("pca_C2")
    W_r = _load(f"pca_W2_L{L}")
    W_py = pca_projection(C, L)
    assert W_py.shape == W_r.shape == (6, L)
    # irlba has limited precision for small matrices, so allow a looser bound.
    assert _columns_span_same_subspace(W_py, W_r, tol=1e-3)


def test_pca_projection_10x10_L4():
    C = _load("pca_C3")
    W_r = _load("pca_W3_L4")
    W_py = pca_projection(C, 4)
    assert W_py.shape == W_r.shape == (10, 4)
    assert _columns_span_same_subspace(W_py, W_r, tol=1e-3)


def test_pca_projection_returns_orthonormal_columns():
    rng = np.random.default_rng(123)
    M = rng.standard_normal((8, 8))
    C = M @ M.T
    W = pca_projection(C, 3)
    np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-10)


def test_pca_projection_eigvals_descending():
    """Top columns should correspond to descending eigenvalues."""
    rng = np.random.default_rng(7)
    M = rng.standard_normal((6, 6))
    C = M @ M.T
    W = pca_projection(C, 4)
    rayleigh = np.array([w @ C @ w for w in W.T])
    assert np.all(np.diff(rayleigh) <= 1e-9), f"non-descending: {rayleigh}"


# ---------------------------------------------------------------------------
# get_major_eigenvalue (with R's quirky irlba branch)
# ---------------------------------------------------------------------------


def test_get_major_eigenvalue_full_branch_matches_r():
    """L >= min(dim): R returns norm(C, '2')^2. We must match exactly."""
    C = _load("gme_C1")
    v_r = _load_scalar("gme_v1_L4")
    v_py = get_major_eigenvalue(C, 4)
    assert v_py == pytest.approx(v_r, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("idx,L", [(2, 2), (3, 3)],
                         ids=["6x8_L2", "10x12_L3"])
def test_get_major_eigenvalue_irlba_branch_matches_r(idx, L):
    """L < min(dim): R returns max(abs(eigen_res$v)).

    Both irlba and our SVD give numerically equivalent right singular vectors
    for these well-conditioned random matrices, so max|V| should agree.
    """
    C = _load(f"gme_C{idx}")
    v_r = _load_scalar(f"gme_v{idx}_L{L}")
    v_py = get_major_eigenvalue(C, L)
    # R's irlba is iterative; allow modest slack on the irlba path.
    assert v_py == pytest.approx(v_r, rel=1e-4, abs=1e-4)


def test_get_major_eigenvalue_full_branch_is_spectral_norm_sq():
    rng = np.random.default_rng(31)
    C = rng.standard_normal((5, 5))
    val = get_major_eigenvalue(C, 5)
    assert val == pytest.approx(float(np.linalg.norm(C, ord=2)) ** 2,
                                rel=1e-12, abs=1e-12)
