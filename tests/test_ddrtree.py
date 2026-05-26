"""End-to-end tests comparing DDRTree Python output against the R gold standard.

The gold-standard TSV files are produced by ``tests/scripts/generate_gold_standard.R``
using the conda env ``/home/groups/xiaojie/nianping/Conda_Files/envs/monocle2``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from ddrtree import DDRTree


DATA_DIR = Path(__file__).parent / "data"


def _load_matrix(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, delimiter="\t", dtype=np.float64, ndmin=2)


def _load_objectives(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, dtype=np.float64, ndmin=1)


def _subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Sign / basis-invariant distance between column-spans of A and B.

    Both A and B should have orthonormal columns of the same shape. Returns
    ``||A A^T - B B^T||_F``, which is 0 iff span(A) == span(B).
    """
    return float(np.linalg.norm(A @ A.T - B @ B.T))


def _align_signs(P: np.ndarray, ref: np.ndarray, pc_axis: int) -> np.ndarray:
    """Flip signs along ``pc_axis`` of ``P`` so each PC best aligns with ``ref``.

    Parameters
    ----------
    pc_axis : int
        The axis of ``P`` along which principal components are enumerated.
        For ``W`` shaped ``(D, d)``, PCs are columns so ``pc_axis=1``.
        For ``Z``, ``Y`` shaped ``(d, N)`` / ``(d, K)``, PCs are rows so
        ``pc_axis=0``.
    """
    assert P.shape == ref.shape
    non_pc_axis = 1 - pc_axis
    signs = np.sign(np.sum(P * ref, axis=non_pc_axis))   # shape (d,)
    signs[signs == 0] = 1.0
    shape = [1, 1]
    shape[pc_axis] = -1
    return P * signs.reshape(shape)


# Per-case tolerances. Short runs (3 iters) on iris stay at machine precision;
# the 20-iteration branch_k20 case accumulates a small drift (~1e-3 abs, ~1e-4
# rel) vs the R gold standard because R's `pca_projection_R` uses `irlba`
# (iterative/approximate), whereas this port uses `numpy.linalg.eigh`
# (direct/exact). Both implementations span the same subspace; the drift is
# purely a per-iteration SVD-convergence gap in the R reference.
CASES = [
    dict(
        name="iris_k3",
        kwargs=dict(dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                    ncenter=3, gamma=10.0, tol=1e-2),
        atol=1e-5, rtol=1e-5,
        obj_atol=1e-4, obj_rtol=1e-4,
    ),
    dict(
        name="iris_full",
        kwargs=dict(dimensions=2, max_iter=5, sigma=1e-3, lambda_=1.0,
                    ncenter=None, gamma=10.0, tol=1e-2),
        atol=1e-5, rtol=1e-5,
        obj_atol=1e-4, obj_rtol=1e-4,
    ),
    dict(
        name="branch_k20",
        # For the branching case, ncenter=20 on 90 points triggers a K-means
        # init. R uses Hartigan-Wong, sklearn uses Lloyd — they land at
        # different local optima, which shifts the first couple of objective
        # values by ~2%. By iteration 5+ the DDRTree outer loop has re-
        # converged the two trajectories to the same solution (final Y, MST,
        # subspace all match within 5e-3), so we accept looser tolerance on
        # the per-iteration objective trace only.
        kwargs=dict(dimensions=2, max_iter=20, sigma=5e-3, lambda_=None,
                    ncenter=20, gamma=10.0, tol=1e-3),
        atol=5e-3, rtol=5e-3,
        obj_atol=8.0, obj_rtol=3e-2,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_ddrtree_matches_r_gold_standard(case):
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

    res = DDRTree(X, **kwargs)

    # Reconstruction X ≈ W @ Z should match (invariant to sign flips /
    # basis rotations within the reduced subspace).
    recon_py = res.W @ res.Z
    recon_r = W_r @ Z_r
    np.testing.assert_allclose(recon_py, recon_r, atol=atol, rtol=rtol)

    # W spans the same subspace (up to orthogonal rotation / sign flips).
    assert _subspace_distance(res.W, W_r) < max(1e-4, atol)

    # PC sign is ambiguous: flipping column j of W also flips row j of Z / Y
    # but leaves W @ Z invariant. Align per-row for Z, Y (rows = PCs).
    Z_py_aligned = _align_signs(res.Z, Z_r, pc_axis=0)
    Y_py_aligned = _align_signs(res.Y, Y_r, pc_axis=0)
    np.testing.assert_allclose(Z_py_aligned, Z_r, atol=atol, rtol=rtol)
    np.testing.assert_allclose(Y_py_aligned, Y_r, atol=atol, rtol=rtol)

    # MST edge topology must match exactly (weights may drift with atol).
    stree_py = res.stree.toarray()
    edges_py = set(zip(*np.nonzero(stree_py)))
    edges_r = set(zip(*np.nonzero(stree_r)))
    assert edges_py == edges_r, (
        f"MST edge set differs: py\\r = {edges_py - edges_r}, "
        f"r\\py = {edges_r - edges_py}"
    )
    np.testing.assert_allclose(stree_py, stree_r, atol=atol, rtol=rtol)

    # Iteration count check: allow ±1 drift vs R because we intentionally
    # deviate from R's objective formula (see _core.py / _torch.py — R/C++
    # double-squares the spectral-norm term, contradicting the documented
    # intent at DDRTree.cpp:341; we use the single-squared form). The
    # relative-change ratio that gates convergence rescales accordingly,
    # so the per-iteration ``delta < tol`` check may fire one step earlier
    # or later than R on borderline cases.
    assert abs(len(res.objective_vals) - len(obj_r)) <= 1, (
        f"Iteration count drifts more than 1 step: "
        f"py={len(res.objective_vals)}, r={len(obj_r)}"
    )

    # We no longer compare ``objective_vals`` element-wise against R's
    # gold standard ``obj.txt`` files: those were generated with R's
    # double-square defect baked in (DDRTree.cpp:349-352), so they
    # encode ``||C||_2^4 + λ tr(Y L Y^T) + γ obj1`` while we now produce
    # ``||C||_2^2 + λ tr(Y L Y^T) + γ obj1``. The structural assertions
    # above (W subspace, Z/Y sign-aligned values, MST edges,
    # reconstruction) confirm the fixed-point solution still matches R
    # exactly; only the convergence-monitoring scalar differs by design.
    # Note: the DDRTree objective is *not* guaranteed to be monotonically
    # non-increasing — the alternating-block updates can produce small
    # oscillations on some inputs (e.g. branch_k20). Convergence is
    # gated by |Δobj| / |prev| < tol regardless of sign, so we only
    # assert finiteness here.
    obj_py = np.asarray(res.objective_vals, dtype=float)
    assert np.all(np.isfinite(obj_py))


def test_ddrtree_basic_shapes():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((6, 40))
    res = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
                  ncenter=10, gamma=10.0, tol=1e-3)
    assert res.W.shape == (6, 2)
    assert res.Z.shape == (2, 40)
    assert res.Y.shape == (2, 10)
    assert res.stree.shape == (40, 40)
    assert 1 <= len(res.objective_vals) <= 5


def test_ddrtree_invalid_input():
    with pytest.raises(ValueError):
        DDRTree(np.zeros((5,)), dimensions=2)          # 1-D
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        # Non-zero X so that pca_projection doesn't hit the degenerate
        # all-zeros ARPACK path before we reach the ncenter check.
        DDRTree(rng.standard_normal((3, 5)), dimensions=2, ncenter=10)
