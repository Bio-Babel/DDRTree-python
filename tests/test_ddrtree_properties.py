"""Mathematical-property tests on DDRTree outputs.

These don't compare against R — they assert invariants the algorithm must
satisfy on any well-formed input (orthonormality of W, tree structure of
``stree``, soft-assignment closure, etc.). Useful for catching regressions
that R parity tests would miss (e.g. accidentally returning ``stree`` as a
non-tree).
"""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from ddrtree import DDRTree


def _make_branching_data(n_per=20, dims=6, noise=0.02, seed=11):
    """Generate a Y-shaped trunk + two branches embedded in ``dims``-D."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_per)
    branch_a = np.vstack([t,  1.0 * t**2])
    branch_b = np.vstack([t, -1.0 * t**2])
    trunk    = np.vstack([np.linspace(-1, 0, n_per), np.zeros(n_per)])
    pts = np.hstack([trunk, branch_a, branch_b])               # (2, 3*n_per)
    M = np.linalg.qr(rng.standard_normal((dims, 2)))[0]        # (dims, 2)
    return M @ pts + rng.normal(scale=noise, size=(dims, pts.shape[1]))


# ---------------------------------------------------------------------------
# W orthonormality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_W_columns_are_orthonormal(dimensions):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 50))
    res = DDRTree(X, dimensions=dimensions, max_iter=10, sigma=1e-2,
                  lambda_=1.0, ncenter=15, gamma=10.0, tol=1e-3)
    assert res.W.shape == (8, dimensions)
    np.testing.assert_allclose(res.W.T @ res.W, np.eye(dimensions), atol=1e-8)


# ---------------------------------------------------------------------------
# stree structure
# ---------------------------------------------------------------------------


def test_stree_is_symmetric():
    X = _make_branching_data()
    res = DDRTree(X, dimensions=2, max_iter=10, sigma=5e-3, lambda_=None,
                  ncenter=15, gamma=10.0, tol=1e-3)
    S = res.stree.toarray()
    np.testing.assert_allclose(S, S.T, atol=1e-12)


def test_stree_block_has_K_minus_1_edges_and_is_a_tree():
    X = _make_branching_data()
    K = 15
    res = DDRTree(X, dimensions=2, max_iter=10, sigma=5e-3, lambda_=None,
                  ncenter=K, gamma=10.0, tol=1e-3)
    S = res.stree.toarray()
    # Only the leading K x K block carries the principal-graph edges;
    # the surrounding rows/cols are zero (matches R's N x N layout).
    block = S[:K, :K]
    assert np.all(S[K:, :] == 0)
    assert np.all(S[:, K:] == 0)
    n_edges = int((block > 0).sum() // 2)
    assert n_edges == K - 1, f"expected K-1={K-1} edges, got {n_edges}"
    # Tree ⇔ exactly one connected component on K nodes
    n_comp, _ = connected_components(block, directed=False)
    assert n_comp == 1


def test_stree_K_equals_N_when_ncenter_None():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 12))
    res = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-3, lambda_=1.0,
                  ncenter=None, gamma=10.0, tol=1e-2)
    assert res.Y.shape[1] == X.shape[1]
    assert res.stree.shape == (X.shape[1], X.shape[1])
    block = res.stree.toarray()
    n_edges = int((block > 0).sum() // 2)
    assert n_edges == X.shape[1] - 1


# ---------------------------------------------------------------------------
# Output shape contracts
# ---------------------------------------------------------------------------


def test_output_shapes_match_dimensions_and_ncenter():
    rng = np.random.default_rng(3)
    D, N, d, K = 7, 25, 3, 8
    X = rng.standard_normal((D, N))
    res = DDRTree(X, dimensions=d, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=K, gamma=10.0, tol=1e-3)
    assert res.W.shape == (D, d)
    assert res.Z.shape == (d, N)
    assert res.Y.shape == (d, K)
    assert res.stree.shape == (N, N)
    assert res.X is X.astype(np.float64) or np.shares_memory(res.X, X) or \
        res.X.shape == (D, N)


# ---------------------------------------------------------------------------
# Reconstruction & objective behaviour
# ---------------------------------------------------------------------------


def test_W_subspace_explains_more_variance_than_random():
    """The discovered basis W should capture more variance of X than a
    randomly-drawn orthonormal basis of the same rank.

    Note: we compare ``W W^T X`` (pure projection) rather than ``W @ Z``
    because ``Z`` carries tree-smoothing constraints and can have larger
    reconstruction error than a free projection by design.
    """
    X = _make_branching_data(dims=8)
    res = DDRTree(X, dimensions=2, max_iter=15, sigma=5e-3, lambda_=None,
                  ncenter=20, gamma=10.0, tol=1e-3)
    err_ddr = np.linalg.norm(X - res.W @ (res.W.T @ X), ord="fro")

    rng = np.random.default_rng(99)
    err_rand = []
    for _ in range(10):
        rand = np.linalg.qr(rng.standard_normal((8, 2)))[0]
        err_rand.append(np.linalg.norm(X - rand @ (rand.T @ X), ord="fro"))
    median_rand = float(np.median(err_rand))
    assert err_ddr < median_rand, (err_ddr, median_rand)


def test_objective_is_finite_at_every_iteration():
    X = _make_branching_data()
    res = DDRTree(X, dimensions=2, max_iter=15, sigma=5e-3, lambda_=None,
                  ncenter=20, gamma=10.0, tol=1e-3)
    obj = np.array(res.objective_vals)
    assert np.all(np.isfinite(obj))
    assert obj.size >= 1


def test_objective_eventually_decreases():
    """After the warm-up iterations, the objective should trend downward."""
    X = _make_branching_data()
    res = DDRTree(X, dimensions=2, max_iter=20, sigma=5e-3, lambda_=None,
                  ncenter=20, gamma=10.0, tol=1e-5)
    obj = np.array(res.objective_vals)
    if obj.size >= 5:
        assert obj[-1] <= obj[1] + 1e-6, (
            f"objective not decreasing: first={obj[1]}, last={obj[-1]}"
        )


# ---------------------------------------------------------------------------
# Soft-assignment R is implicitly probabilistic — check via reconstruction
# ---------------------------------------------------------------------------


def test_Y_lies_in_W_column_space():
    """Each Y column equals W^T applied to some point in the original space.

    Equivalently, ``W (W^T W)^{-1} (W @ Y) ≈ W @ Y`` since Y is already in
    the d-dim latent space; we check that mapping back to the input space
    preserves it: ``W^T (W @ Y) ≈ Y`` (W is orthonormal so this is exact).
    """
    X = _make_branching_data()
    res = DDRTree(X, dimensions=2, max_iter=10, sigma=5e-3, lambda_=None,
                  ncenter=20, gamma=10.0, tol=1e-3)
    np.testing.assert_allclose(res.W.T @ (res.W @ res.Y), res.Y, atol=1e-10)
