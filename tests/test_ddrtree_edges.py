"""Edge-case / parameter-sweep tests for ``DDRTree``.

These exercise corners that neither the R-parity suite nor the property
suite hit: single-iter runs, tight/loose ``tol``, 1-D embeddings, the
``initial_method`` callback, ``lambda_=None`` defaulting to ``5*N``, a
``verbose=True`` smoke run, and input coercion.
"""

import io
import contextlib

import numpy as np
import pytest

from ddrtree import DDRTree


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


def test_dimensions_equals_one():
    """1-D embedding should work — corresponds to a purely linear pseudotime."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 30))
    res = DDRTree(X, dimensions=1, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=10, gamma=10.0, tol=1e-3)
    assert res.W.shape == (5, 1)
    assert res.Z.shape == (1, 30)
    assert res.Y.shape == (1, 10)
    # 1D W is a unit vector.
    assert np.linalg.norm(res.W) == pytest.approx(1.0, abs=1e-10)


def test_dimensions_equals_input_D():
    """d == D is allowed: W becomes a D×D orthogonal matrix."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 30))
    res = DDRTree(X, dimensions=4, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=10, gamma=10.0, tol=1e-3)
    assert res.W.shape == (4, 4)
    np.testing.assert_allclose(res.W.T @ res.W, np.eye(4), atol=1e-8)


# ---------------------------------------------------------------------------
# ncenter boundaries
# ---------------------------------------------------------------------------


def test_ncenter_equals_N_via_explicit_arg():
    rng = np.random.default_rng(2)
    N = 20
    X = rng.standard_normal((5, N))
    res = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=N, gamma=10.0, tol=1e-3)
    assert res.Y.shape[1] == N


def test_ncenter_equals_2_small_tree():
    """Smallest meaningful tree: a single edge between two centers."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 20))
    res = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=2, gamma=10.0, tol=1e-3)
    assert res.Y.shape == (2, 2)
    block = res.stree.toarray()[:2, :2]
    n_edges = int((block > 0).sum() // 2)
    assert n_edges == 1


def test_ncenter_greater_than_N_raises():
    # Non-zero input so pca_projection doesn't short-circuit on a degenerate
    # all-zeros ARPACK path before the ncenter validation fires.
    rng = np.random.default_rng(31)
    X = rng.standard_normal((3, 5))
    with pytest.raises(ValueError):
        DDRTree(X, dimensions=2, ncenter=10)


# ---------------------------------------------------------------------------
# lambda_ default
# ---------------------------------------------------------------------------


def test_lambda_none_defaults_to_5N():
    """``lambda_=None`` should behave identically to ``lambda_ = 5*N``."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((5, 18))
    r_auto = DDRTree(X, dimensions=2, max_iter=8, sigma=1e-2,
                     lambda_=None, ncenter=8, gamma=10.0, tol=1e-3)
    r_manual = DDRTree(X, dimensions=2, max_iter=8, sigma=1e-2,
                       lambda_=5.0 * X.shape[1], ncenter=8, gamma=10.0,
                       tol=1e-3)
    np.testing.assert_allclose(r_auto.W, r_manual.W, atol=1e-12)
    np.testing.assert_allclose(r_auto.Z, r_manual.Z, atol=1e-12)
    np.testing.assert_allclose(r_auto.Y, r_manual.Y, atol=1e-12)
    assert r_auto.objective_vals == r_manual.objective_vals


# ---------------------------------------------------------------------------
# max_iter / tol
# ---------------------------------------------------------------------------


def test_max_iter_1_runs_single_iteration():
    """With max_iter=1 we should get exactly one objective value and no
    convergence check is triggered (which lives in ``iter >= 1``)."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 15))
    res = DDRTree(X, dimensions=2, max_iter=1, sigma=1e-2, lambda_=1.0,
                  ncenter=5, gamma=10.0, tol=1e-3)
    assert len(res.objective_vals) == 1


def test_tol_zero_forces_all_iterations():
    """``tol=0`` means no early stop (delta is never strictly less than 0)."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((4, 20))
    res = DDRTree(X, dimensions=2, max_iter=12, sigma=1e-2, lambda_=1.0,
                  ncenter=7, gamma=10.0, tol=0.0)
    assert len(res.objective_vals) == 12


def test_large_tol_exits_after_two_iterations():
    """A huge ``tol`` triggers early stop the first time delta is evaluated
    (at iter=1), so we see exactly 2 objective values recorded."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 20))
    res = DDRTree(X, dimensions=2, max_iter=10, sigma=1e-2, lambda_=1.0,
                  ncenter=7, gamma=10.0, tol=1e9)
    assert len(res.objective_vals) == 2


# ---------------------------------------------------------------------------
# initial_method callback
# ---------------------------------------------------------------------------


def test_initial_method_callback_is_used_for_Z_init():
    """When ``initial_method`` is supplied the init Z comes from it rather
    than PCA, so that direction should show up in the output compared to the
    default run on the same data."""
    rng = np.random.default_rng(8)
    D, N = 5, 30
    X = rng.standard_normal((D, N))

    # A deliberately-bad custom init: random orthonormal basis of shape (N, d)
    # returned per the API contract.
    def custom_init(Xin):
        q = np.linalg.qr(rng.standard_normal((Xin.shape[1], 2)))[0]
        return q  # (N, 2)

    r_def = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                    ncenter=5, gamma=10.0, tol=1e-3)
    r_alt = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                    ncenter=5, gamma=10.0, tol=1e-3,
                    initial_method=custom_init)

    # Any meaningful change should land us at a different objective trajectory
    # within the first iteration.
    assert r_def.objective_vals[0] != r_alt.objective_vals[0]


# ---------------------------------------------------------------------------
# verbose / input coercion
# ---------------------------------------------------------------------------


def test_verbose_true_smoke_runs_without_error_and_prints():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((4, 12))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        DDRTree(X, dimensions=2, max_iter=3, sigma=1e-2, lambda_=1.0,
                ncenter=4, gamma=10.0, tol=1e-3, verbose=True)
    # Should have emitted per-iteration diagnostics.
    assert "iter" in buf.getvalue().lower()


def test_integer_input_promoted_to_float():
    rng = np.random.default_rng(10)
    X_int = (rng.standard_normal((4, 15)) * 10).astype(np.int64)
    # Must not crash and must return float64 outputs.
    res = DDRTree(X_int, dimensions=2, max_iter=3, sigma=1.0,
                  lambda_=1.0, ncenter=5, gamma=10.0, tol=1e-3)
    assert res.W.dtype == np.float64
    assert res.Z.dtype == np.float64


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2D"):
        DDRTree(np.zeros(10), dimensions=2)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_runs():
    """Identical inputs produce identical outputs (no RNG leaking in)."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((5, 25))
    kwargs = dict(dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=6, gamma=10.0, tol=1e-3)
    r1 = DDRTree(X, **kwargs)
    r2 = DDRTree(X, **kwargs)
    np.testing.assert_array_equal(r1.W, r2.W)
    np.testing.assert_array_equal(r1.Z, r2.Z)
    np.testing.assert_array_equal(r1.Y, r2.Y)
    assert r1.objective_vals == r2.objective_vals
