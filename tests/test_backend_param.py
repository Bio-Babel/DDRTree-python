"""Tests for the ``backend`` parameter added to the public ``DDRTree`` API.

These tests verify the dispatcher contract:

* ``backend="numpy"`` (the default today) is the reference path and must
  produce bit-for-bit the same result as calling ``DDRTree`` without
  specifying ``backend`` at all.
* Unknown backend values raise ``ValueError`` with a useful message.
* Backend values that are reserved for future phases but not yet
  implemented must be rejected too — we refuse to silently fall back.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddrtree import DDRTree


def _make_small_X(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, 40))


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_backend_numpy_matches_default(seed: int) -> None:
    """``backend="numpy"`` must be bit-for-bit identical to the default."""
    X = _make_small_X(seed)
    a = DDRTree(X, dimensions=2, ncenter=5, max_iter=10)
    b = DDRTree(X, dimensions=2, ncenter=5, max_iter=10, backend="numpy")

    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.Z, b.Z)
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.stree.toarray(), b.stree.toarray())
    assert a.objective_vals == b.objective_vals


def test_backend_unknown_value_raises() -> None:
    X = _make_small_X()
    with pytest.raises(ValueError, match=r"backend must be one of"):
        DDRTree(X, backend="elephant")


@pytest.mark.parametrize("not_yet", ["torch", "auto"])
def test_backend_reserved_values_rejected_in_p1(not_yet: str) -> None:
    """Backends reserved for later phases must be actively rejected now.

    We refuse to silently fall back to ``"numpy"`` because that would hide
    a user's intent to exercise a specific backend when one is available.
    """
    X = _make_small_X()
    with pytest.raises(ValueError, match=r"backend must be one of"):
        DDRTree(X, backend=not_yet)


def test_backend_is_case_sensitive() -> None:
    X = _make_small_X()
    with pytest.raises(ValueError, match=r"backend must be one of"):
        DDRTree(X, backend="NumPy")


def test_backend_error_message_lists_valid_options() -> None:
    X = _make_small_X()
    with pytest.raises(ValueError) as exc:
        DDRTree(X, backend="gpu")
    assert "numpy" in str(exc.value)
