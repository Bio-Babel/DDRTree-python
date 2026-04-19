"""Focused tests for the four R-parity fixes in DDRTree-python:

1. ``**kwargs`` are forwarded to ``initial_method`` (R's ``...`` semantics).
2. ``DDRTreeResult.history`` exists and is ``None`` (R always returns NULL).
3. The inner MST defaults to Prim (matches Boost.Graph) and ``"kruskal"``
   is available as an explicit opt-in that agrees with Prim when edge
   weights are unique.
4. Extra kwargs are silently dropped when ``initial_method`` is not given,
   matching R's behaviour of silently swallowing unused ``...`` args.
"""

from dataclasses import fields

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from ddrtree import DDRTree, DDRTreeResult
from ddrtree._core import _prim_mst


# ---------------------------------------------------------------------------
# 1. kwargs forwarding to initial_method
# ---------------------------------------------------------------------------


def test_kwargs_are_forwarded_to_initial_method():
    """Captured kwargs inside initial_method must match what we passed in."""
    captured = {}

    def fake_init(X, **kw):
        captured.update(kw)
        # Return a trivial init that satisfies the (<=N, <=D) contract.
        return np.zeros((X.shape[1], 2))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 20))
    DDRTree(X, dimensions=2, initial_method=fake_init, max_iter=2,
            sigma=1e-2, lambda_=1.0, ncenter=4, gamma=10.0, tol=1e-3,
            nn_k=8, embed="graphlap", custom=3.14)

    assert captured == {"nn_k": 8, "embed": "graphlap", "custom": 3.14}


def test_initial_method_keyword_matches_r_signature():
    """``initial_method(X, ...)`` should receive exactly ``X`` plus kwargs,
    positionally-first just like R's ``initial_method(X, ...)`` call."""
    seen = []

    def fake_init(X, *args, **kw):
        seen.append((X.shape, args, kw))
        return np.zeros((X.shape[1], 2))

    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 12))
    DDRTree(X, dimensions=2, initial_method=fake_init, max_iter=1,
            sigma=1e-2, lambda_=1.0, ncenter=3, gamma=10.0, tol=1e-3,
            alpha=0.5)
    assert len(seen) == 1
    Xshape, args, kw = seen[0]
    assert Xshape == (4, 12)
    assert args == ()
    assert kw == {"alpha": 0.5}


# ---------------------------------------------------------------------------
# 2. history field
# ---------------------------------------------------------------------------


def test_result_has_history_field_always_None():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 15))
    res = DDRTree(X, dimensions=2, max_iter=3, sigma=1e-2, lambda_=1.0,
                  ncenter=5, gamma=10.0, tol=1e-3)
    # Attribute exists
    assert hasattr(res, "history")
    # Always None — matches R's return list convention
    assert res.history is None


def test_DDRTreeResult_dataclass_fields_match_R_return_list():
    """The exposed fields should match R's return list names (with history)."""
    names = {f.name for f in fields(DDRTreeResult)}
    expected = {"W", "Z", "Y", "stree", "X", "objective_vals", "history"}
    assert names == expected, (names, expected)


# ---------------------------------------------------------------------------
# 3. MST default = Prim, and Prim correctness
# ---------------------------------------------------------------------------


def test_prim_default_matches_kruskal_on_unique_weights():
    """On a generic random problem (distinct squared distances to fp precision)
    Prim and Kruskal must return identical trees — differing only by tie-
    breaking, and for float coordinates ties are vanishingly rare."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 60))
    common = dict(dimensions=2, max_iter=6, sigma=5e-3, lambda_=1.0,
                  ncenter=12, gamma=10.0, tol=1e-3)
    r_prim = DDRTree(X, **common, mst_algorithm="prim")
    r_krus = DDRTree(X, **common, mst_algorithm="kruskal")
    # Tree edge-sets identical
    ep = set(zip(*np.nonzero(r_prim.stree.toarray())))
    ek = set(zip(*np.nonzero(r_krus.stree.toarray())))
    assert ep == ek
    # Numerical outputs identical (no tie-break divergence on float data).
    np.testing.assert_allclose(r_prim.W, r_krus.W, atol=1e-10)
    np.testing.assert_allclose(r_prim.Y, r_krus.Y, atol=1e-10)


def test_prim_rejects_invalid_algorithm_string():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4, 10))
    with pytest.raises(ValueError, match="mst_algorithm"):
        DDRTree(X, dimensions=2, ncenter=3, mst_algorithm="boruvka")


def test__prim_mst_produces_a_spanning_tree():
    """K-1 edges, connected, symmetric, weights taken from input matrix."""
    rng = np.random.default_rng(5)
    K = 12
    pts = rng.standard_normal((K, 3))
    # Build a full squared-distance matrix
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    mst = _prim_mst(d2)
    assert mst.shape == (K, K)
    np.testing.assert_allclose(mst, mst.T, atol=1e-12)
    assert int((mst > 0).sum() // 2) == K - 1
    n_comp, _ = connected_components(mst > 0, directed=False)
    assert n_comp == 1
    # Non-zero weights equal the input distances.
    mask = mst > 0
    np.testing.assert_allclose(mst[mask], d2[mask], atol=1e-12)


def test__prim_mst_matches_kruskal_total_weight():
    """Both algorithms are MSTs, so total edge weight must agree exactly."""
    rng = np.random.default_rng(6)
    K = 15
    pts = rng.standard_normal((K, 4))
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    mst_prim = _prim_mst(d2)
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst_kr = minimum_spanning_tree(d2).toarray()
    mst_kr = mst_kr + mst_kr.T
    # Sum over upper-triangle.
    w_prim = float(mst_prim[np.triu_indices(K, 1)].sum())
    w_kr = float(mst_kr[np.triu_indices(K, 1)].sum())
    assert w_prim == pytest.approx(w_kr, rel=1e-10, abs=1e-12)


def test__prim_mst_degenerate_K_equals_one():
    mst = _prim_mst(np.zeros((1, 1)))
    assert mst.shape == (1, 1)
    assert np.all(mst == 0)


# ---------------------------------------------------------------------------
# 4. R-style "..." silently swallowed when no initial_method
# ---------------------------------------------------------------------------


def test_extra_kwargs_silently_ignored_when_no_initial_method():
    """R drops unused `...` args without warning; Python must do the same."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 15))
    # These kwargs have nothing to attach to — they should be accepted silently.
    res = DDRTree(X, dimensions=2, max_iter=2, sigma=1e-2, lambda_=1.0,
                  ncenter=4, gamma=10.0, tol=1e-3,
                  unused_arg=42, another=[1, 2, 3])
    assert res.W.shape == (4, 2)
