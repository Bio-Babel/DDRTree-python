"""Unit tests for utility helpers against small hand-computable references."""

import numpy as np
import pytest

from ddrtree import pca_projection, sq_dist, get_major_eigenvalue


def test_sq_dist_matches_pairwise_sqeuclidean():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5))
    b = rng.standard_normal((4, 3))
    out = sq_dist(a, b)
    # Reference via explicit loops.
    ref = np.zeros((5, 3))
    for i in range(5):
        for j in range(3):
            ref[i, j] = np.sum((a[:, i] - b[:, j]) ** 2)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_sq_dist_self_diagonal_near_zero():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 6))
    out = sq_dist(a, a)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_pca_projection_symmetric_matches_eigh_top_vectors():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 6))
    C = A @ A.T                                   # symmetric PSD
    L = 2
    W = pca_projection(C, L)
    assert W.shape == (6, L)
    # Columns are eigenvectors: verify via C @ W aligns with W scaled by eigenvalues.
    vals, vecs = np.linalg.eigh(C)
    top = vecs[:, np.argsort(vals)[::-1][:L]]
    # sign-invariant comparison via |W^T W_ref| ≈ I
    prod = np.abs(W.T @ top)
    np.testing.assert_allclose(prod, np.eye(L), atol=1e-10)


def test_get_major_eigenvalue_full_branch_is_spectral_norm_sq():
    rng = np.random.default_rng(3)
    C = rng.standard_normal((4, 4))
    L = 4  # triggers full branch
    val = get_major_eigenvalue(C, L)
    ref = float(np.linalg.norm(C, ord=2)) ** 2
    assert val == pytest.approx(ref, rel=1e-10, abs=1e-10)


def test_get_major_eigenvalue_irlba_branch_is_max_abs_v():
    rng = np.random.default_rng(4)
    C = rng.standard_normal((6, 8))
    L = 2
    val = get_major_eigenvalue(C, L)
    # Reference: top-L right singular vectors → max abs entry.
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    ref = float(np.max(np.abs(vt[:L, :].T)))
    assert val == pytest.approx(ref, rel=1e-10, abs=1e-10)
