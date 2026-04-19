"""Targeted coverage for small branches that are hard to exercise
through the top-level DDRTree() dispatcher.

These tests don't add new behaviour — they just keep the overall
coverage honest so that an actual bug in these paths would show up as
a coverage regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddrtree import DDRTree


def test_torch_backend_verbose_prints(capsys: pytest.CaptureFixture[str]) -> None:
    """``verbose=True`` must route through the per-iteration print
    branches in the torch backend."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 20))
    DDRTree(X, dimensions=2, ncenter=4, max_iter=3, verbose=True, backend="torch")
    captured = capsys.readouterr()
    assert "[DDRTree/torch] iter" in captured.out
    assert "objective" in captured.out


def test_torch_backend_direct_invalid_dtype_raises() -> None:
    """The dispatcher validates dtype, but calling the torch backend
    directly with a bad value must still fail with the same message."""
    pytest.importorskip("torch")
    from ddrtree._backends._torch import ddrtree_torch

    X = np.zeros((3, 10))
    with pytest.raises(ValueError, match=r"dtype must be"):
        ddrtree_torch(X, ncenter=3, dtype="bfloat16")


def test_torch_initial_method_shape_too_big_raises() -> None:
    """Mirror of the numpy-side check: initial_method cannot return more
    rows than N or more columns than D."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 10))

    def too_big(X_arg):
        return rng.standard_normal((X_arg.shape[1] + 5, X_arg.shape[0]))

    with pytest.raises(ValueError, match=r"initial_method must return"):
        DDRTree(X, initial_method=too_big, dimensions=2, backend="torch")
