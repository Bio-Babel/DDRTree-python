"""Tests for the ``device`` parameter.

On a CUDA host these tests exercise the end-to-end GPU path. On a
CPU-only host they cover the explicit CPU-device paths and the
cross-backend validation of ``device``. CUDA-specific cases use
``skipif`` guards so the suite stays green across environments.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree import DDRTree


def _X(seed: int = 0, N: int = 40, D: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, N))
    X[0] += np.linspace(-1, 1, N)
    return X


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
def test_cuda_device_index_parses() -> None:
    X = _X(0)
    res = DDRTree(X, ncenter=5, max_iter=3, backend="torch", device="cuda:0")
    assert isinstance(res.W, np.ndarray)
