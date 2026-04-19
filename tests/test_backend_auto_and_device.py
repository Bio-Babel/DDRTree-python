"""Tests for ``backend="auto"`` dispatch and the ``device`` parameter (P4).

On a CUDA host these tests exercise the end-to-end GPU path. On a
CPU-only host they cover the fallback logic and the explicit
CPU-device paths. The CUDA-specific cases use ``skipif`` guards so the
suite stays green across environments.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree import DDRTree
from ddrtree._core import _resolve_auto_backend


# --------------------------------------------------------------------------- #
# _resolve_auto_backend unit tests (pure logic — no real backend executed)    #
# --------------------------------------------------------------------------- #


def test_auto_prefers_numpy_when_no_cuda() -> None:
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert _resolve_auto_backend(None) == "numpy"
        assert _resolve_auto_backend("cpu") == "numpy"


def test_auto_prefers_torch_when_cuda_available() -> None:
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert _resolve_auto_backend(None) == "torch"


def test_auto_honours_explicit_cpu_request() -> None:
    """When CPU is explicitly requested, 'auto' stays on numpy even on
    hosts where CUDA is available — torch CPU rarely beats numpy."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert _resolve_auto_backend("cpu") == "numpy"


def test_auto_refuses_cuda_without_cuda_runtime() -> None:
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match=r"CUDA"):
            _resolve_auto_backend("cuda")


def test_auto_accepts_cuda_when_available() -> None:
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert _resolve_auto_backend("cuda") == "torch"


# --------------------------------------------------------------------------- #
# Auto dispatch integrated into the public API                                #
# --------------------------------------------------------------------------- #


def _X(seed: int = 0, N: int = 40, D: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, N))
    X[0] += np.linspace(-1, 1, N)
    return X


def test_auto_backend_runs_without_crashing_cpu_only() -> None:
    """On a CPU-only host ``backend="auto"`` must resolve to numpy and
    return the same result as ``backend="numpy"``."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        X = _X(0)
        a = DDRTree(X, dimensions=2, ncenter=8, max_iter=5, backend="numpy")
        b = DDRTree(X, dimensions=2, ncenter=8, max_iter=5, backend="auto")
        np.testing.assert_array_equal(a.W, b.W)
        np.testing.assert_array_equal(a.Z, b.Z)


# --------------------------------------------------------------------------- #
# device parameter validation                                                  #
# --------------------------------------------------------------------------- #


def test_numpy_rejects_cuda_device() -> None:
    X = _X(0)
    with pytest.raises(ValueError, match=r"backend='numpy'"):
        DDRTree(X, backend="numpy", device="cuda")


def test_numpy_accepts_cpu_device_as_noop() -> None:
    """The dispatcher accepts ``device='cpu'`` under the NumPy backend
    (it's a no-op); the result must match ``device=None``."""
    X = _X(0)
    a = DDRTree(X, ncenter=5, max_iter=4, backend="numpy")
    b = DDRTree(X, ncenter=5, max_iter=4, backend="numpy", device="cpu")
    np.testing.assert_array_equal(a.W, b.W)


def test_torch_cpu_device_runs() -> None:
    X = _X(0)
    res = DDRTree(X, ncenter=5, max_iter=4, backend="torch", device="cpu")
    # Result is always materialised back to numpy regardless of device.
    assert isinstance(res.W, np.ndarray)


def test_torch_cuda_device_without_cuda_raises() -> None:
    X = _X(0)
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match=r"CUDA"):
            DDRTree(X, ncenter=5, max_iter=4, backend="torch", device="cuda")


# --------------------------------------------------------------------------- #
# CUDA real-hardware tests (skipif-guarded)                                    #
# --------------------------------------------------------------------------- #


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)


@cuda_only
def test_torch_cuda_matches_cpu_end_to_end() -> None:
    """CUDA and CPU execution must produce numerically equivalent results
    on fp64 (small drift only from non-deterministic GPU BLAS reductions).
    """
    X = _X(0)
    cpu = DDRTree(X, ncenter=10, max_iter=8, tol=1e-6, backend="torch", device="cpu")
    gpu = DDRTree(X, ncenter=10, max_iter=8, tol=1e-6, backend="torch", device="cuda")
    np.testing.assert_allclose(
        cpu.W @ cpu.Z, gpu.W @ gpu.Z, atol=1e-5, rtol=1e-5,
    )
    # MST edge set identical (Borůvka deterministic under unique weights).
    e_cpu = set(zip(*np.nonzero(cpu.stree.toarray())))
    e_gpu = set(zip(*np.nonzero(gpu.stree.toarray())))
    assert e_cpu == e_gpu


@cuda_only
def test_auto_on_cuda_host_picks_torch_cuda() -> None:
    X = _X(0)
    res = DDRTree(X, ncenter=10, max_iter=5, backend="auto")
    # We cannot introspect which device was used post-hoc (results are
    # converted to numpy). The strongest assertion is that the result
    # matches an explicit torch+cuda run.
    ref = DDRTree(X, ncenter=10, max_iter=5, backend="torch", device="cuda")
    np.testing.assert_allclose(res.W @ res.Z, ref.W @ ref.Z, atol=1e-5, rtol=1e-5)


@cuda_only
def test_cuda_device_index_parses() -> None:
    X = _X(0)
    res = DDRTree(X, ncenter=5, max_iter=3, backend="torch", device="cuda:0")
    assert isinstance(res.W, np.ndarray)
