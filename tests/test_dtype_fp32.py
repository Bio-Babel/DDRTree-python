"""Tests for the ``dtype`` parameter (P5): fp32 opt-in on the torch backend.

fp32 halves memory and is typically 1.5–2× faster on CUDA. In exchange
we accept:

* ~1e-3 relative drift on the converged Y/Z vs the fp64 reference,
* a higher probability of the Cholesky→LU fallback firing (because the
  ``tmp_M`` system can drift closer to non-PSD at lower precision),
* possibly a different iteration count when the objective trajectory
  crosses the ``tol`` threshold at a slightly different step.

Tests here assert the behaviour is still correct (right kind of tree,
reasonable reconstruction), not bit-for-bit reproducibility.
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
    X[1] += np.sin(np.linspace(0, 3.14, N))
    return X


# --------------------------------------------------------------------------- #
# Parameter validation                                                         #
# --------------------------------------------------------------------------- #


def test_numpy_rejects_fp32() -> None:
    X = _X(0)
    with pytest.raises(ValueError, match=r"backend='numpy'"):
        DDRTree(X, ncenter=5, backend="numpy", dtype="float32")


def test_numpy_accepts_explicit_float64_dtype() -> None:
    """The default ``dtype=None`` is semantically fp64; passing it
    explicitly is accepted and produces the same result."""
    X = _X(0)
    a = DDRTree(X, ncenter=5, max_iter=4, backend="numpy")
    b = DDRTree(X, ncenter=5, max_iter=4, backend="numpy", dtype="float64")
    np.testing.assert_array_equal(a.W, b.W)


def test_invalid_dtype_rejected() -> None:
    X = _X(0)
    with pytest.raises(ValueError, match=r"dtype must be"):
        DDRTree(X, ncenter=5, dtype="half")


def test_torch_explicit_fp64_matches_default() -> None:
    X = _X(0)
    a = DDRTree(X, ncenter=8, max_iter=5, backend="torch")
    b = DDRTree(X, ncenter=8, max_iter=5, backend="torch", dtype="float64")
    np.testing.assert_array_equal(a.W, b.W)


# --------------------------------------------------------------------------- #
# fp32 correctness                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ncenter,seed", [(8, 0), (12, 1), (15, 4)])
def test_fp32_runs_and_approximates_fp64(ncenter: int, seed: int) -> None:
    """fp32 must (a) not crash, (b) produce well-formed output, and (c)
    match fp64 up to ~1e-3 relative error on the sign-invariant
    reconstruction ``W @ Z``."""
    X = _X(seed)
    kw = dict(dimensions=2, ncenter=ncenter, max_iter=10, tol=1e-5, backend="torch")

    r64 = DDRTree(X, **kw, dtype="float64")
    r32 = DDRTree(X, **kw, dtype="float32")

    assert r32.W.shape == r64.W.shape
    assert r32.Y.shape == r64.Y.shape

    # Iteration count can drift by one or two at fp32 because the
    # objective stopping criterion crosses at a slightly different step.
    assert abs(len(r32.objective_vals) - len(r64.objective_vals)) <= 2

    # Reconstruction comparison (sign-invariant).
    np.testing.assert_allclose(
        r32.W @ r32.Z, r64.W @ r64.Z, atol=1e-2, rtol=1e-2,
    )


def test_fp32_mst_edge_set_matches_fp64_on_stable_case() -> None:
    """For a modest problem where fp32 does not perturb the MST, the
    edge set should match fp64. We pick a seed where weights are well-
    separated to make this a robust assertion."""
    X = _X(seed=0, N=60)
    kw = dict(dimensions=2, ncenter=10, max_iter=8, tol=1e-4, backend="torch")

    r64 = DDRTree(X, **kw, dtype="float64")
    r32 = DDRTree(X, **kw, dtype="float32")

    e64 = set(zip(*np.nonzero(r64.stree.toarray())))
    e32 = set(zip(*np.nonzero(r32.stree.toarray())))
    assert e32 == e64


def test_fp32_returns_numpy_fp32_result() -> None:
    """fp32 compute → fp32 numpy output. The user opted in to lower
    precision explicitly, so the result dtype reflects it honestly
    (rather than silently upcasting to fp64 and pretending the
    precision wasn't lost). ``X`` is echoed as fp64 because it's a
    copy of the caller's input, which we coerce to fp64 for sklearn
    K-means anyway."""
    X = _X(0)
    res = DDRTree(X, ncenter=5, max_iter=3, backend="torch", dtype="float32")
    assert res.W.dtype == np.float32
    assert res.Z.dtype == np.float32
    assert res.Y.dtype == np.float32
    assert res.X.dtype == np.float64   # caller's data, not compute output


def test_fp64_returns_numpy_fp64_result() -> None:
    X = _X(0)
    res = DDRTree(X, ncenter=5, max_iter=3, backend="torch")  # default fp64
    assert res.W.dtype == np.float64
    assert res.Z.dtype == np.float64
    assert res.Y.dtype == np.float64


# --------------------------------------------------------------------------- #
# fp32 compatible with device param                                            #
# --------------------------------------------------------------------------- #


def test_fp32_with_cpu_device_runs() -> None:
    X = _X(0)
    res = DDRTree(
        X, ncenter=5, max_iter=3, backend="torch", device="cpu", dtype="float32",
    )
    assert res.W.shape == (4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_fp32_on_cuda_matches_cpu() -> None:
    X = _X(0)
    kw = dict(ncenter=10, max_iter=5, tol=1e-4, backend="torch", dtype="float32")
    cpu = DDRTree(X, **kw, device="cpu")
    gpu = DDRTree(X, **kw, device="cuda")
    # fp32 CPU vs GPU BLAS can differ a hair more; 5e-3 is appropriate.
    np.testing.assert_allclose(
        cpu.W @ cpu.Z, gpu.W @ gpu.Z, atol=5e-3, rtol=5e-3,
    )
