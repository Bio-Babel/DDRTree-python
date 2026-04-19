"""Torch backend vs R gold standard.

Loads the same ``iris_k3`` / ``iris_full`` / ``branch_k20`` fixtures as
``test_ddrtree.py`` and runs them through ``backend="torch"``. Tolerances
are relaxed relative to the NumPy backend (which is tuned tightly against
R) because the torch path uses a direct SVD for ``pca_projection`` in the
truncated branch — this contributes a small per-iteration drift on top of
what NumPy already exhibits.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddrtree import DDRTree


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _load_matrix(fname: str) -> np.ndarray:
    return np.loadtxt(os.path.join(DATA, fname), dtype=np.float64)


def _load_objectives(fname: str) -> np.ndarray:
    return np.loadtxt(os.path.join(DATA, fname), dtype=np.float64)


def _align_signs(P: np.ndarray, ref: np.ndarray, pc_axis: int) -> np.ndarray:
    assert P.shape == ref.shape
    non_pc_axis = 1 - pc_axis
    signs = np.sign(np.sum(P * ref, axis=non_pc_axis))
    signs[signs == 0] = 1.0
    shape = [1, 1]
    shape[pc_axis] = -1
    return P * signs.reshape(shape)


def _subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Chordal subspace distance ||P_A - P_B||_F where P = QQ^T."""
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    return float(np.linalg.norm(QA @ QA.T - QB @ QB.T, ord="fro"))


# Tolerances vs R gold, slightly looser than the NumPy backend's tolerances.
# Drift is dominated by the SVD algorithm choice in pca_projection; all
# other ops are backend-BLAS-identical at fp64 on CPU.
CASES = [
    dict(
        name="iris_k3",
        kwargs=dict(dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                    ncenter=3, gamma=10.0, tol=1e-2),
        atol=1e-4, rtol=1e-4,
        obj_atol=1e-3, obj_rtol=1e-4,
    ),
    dict(
        name="iris_full",
        kwargs=dict(dimensions=2, max_iter=5, sigma=1e-3, lambda_=1.0,
                    ncenter=None, gamma=10.0, tol=1e-2),
        atol=1e-4, rtol=1e-4,
        obj_atol=1e-3, obj_rtol=1e-4,
    ),
    dict(
        name="branch_k20",
        kwargs=dict(dimensions=2, max_iter=20, sigma=5e-3, lambda_=None,
                    ncenter=20, gamma=10.0, tol=1e-3),
        atol=5e-3, rtol=5e-3,
        obj_atol=8.0, obj_rtol=3e-2,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_torch_matches_r_gold_standard(case):
    name = case["name"]
    kwargs = case["kwargs"]
    atol = case["atol"]
    rtol = case["rtol"]
    obj_atol = case["obj_atol"]
    obj_rtol = case["obj_rtol"]

    X = _load_matrix(f"{name}_X.tsv")
    W_r = _load_matrix(f"{name}_W.tsv")
    Z_r = _load_matrix(f"{name}_Z.tsv")
    Y_r = _load_matrix(f"{name}_Y.tsv")
    stree_r = _load_matrix(f"{name}_stree.tsv")
    obj_r = _load_objectives(f"{name}_obj.txt")

    res = DDRTree(X, backend="torch", **kwargs)

    # Reconstruction (sign-invariant)
    recon_py = res.W @ res.Z
    recon_r = W_r @ Z_r
    np.testing.assert_allclose(recon_py, recon_r, atol=atol, rtol=rtol)

    # W subspace span
    assert _subspace_distance(res.W, W_r) < max(1e-4, atol)

    # After sign alignment, Z and Y agree
    Z_py_aligned = _align_signs(res.Z, Z_r, pc_axis=0)
    Y_py_aligned = _align_signs(res.Y, Y_r, pc_axis=0)
    np.testing.assert_allclose(Z_py_aligned, Z_r, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Y_py_aligned, Y_r, atol=atol, rtol=rtol)

    # MST topology — edge set must be identical. The torch backend routes
    # MST through the NumPy Prim in P2, so this is a strict check. P3 may
    # loosen to weight-sum equality when Borůvka becomes default.
    stree_py = res.stree.toarray()
    edges_py = set(zip(*np.nonzero(stree_py)))
    edges_r = set(zip(*np.nonzero(stree_r)))
    assert edges_py == edges_r, (
        f"MST edge set differs: py\\r = {edges_py - edges_r}, "
        f"r\\py = {edges_r - edges_py}"
    )
    np.testing.assert_allclose(stree_py, stree_r, atol=atol, rtol=rtol)

    # Iteration count match
    assert len(res.objective_vals) == len(obj_r), (
        f"Iteration count mismatch: py={len(res.objective_vals)}, "
        f"r={len(obj_r)}"
    )
    np.testing.assert_allclose(
        np.array(res.objective_vals), obj_r, atol=obj_atol, rtol=obj_rtol
    )
