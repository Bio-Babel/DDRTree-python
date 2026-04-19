"""Unit tests for the torch-native Borůvka MST.

These tests use SciPy's MST (Kruskal) as an independent oracle for the
total weight, and the pure-NumPy Prim (already validated against R gold
standard) for edge-set equality under unique weights.

We intentionally avoid relaxed assertions: an MST with the wrong edge
count, with cycles, or with a different total weight is a bug — not
acceptable drift.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.sparse.csgraph import minimum_spanning_tree as _scipy_mst

from ddrtree._core import _prim_mst
from ddrtree._mst._boruvka_torch import boruvka_mst


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _random_symmetric(K: int, seed: int, *, unique: bool = True) -> np.ndarray:
    """Generate a symmetric non-negative weight matrix with zero diagonal.

    When ``unique=True`` we add vertex-indexed perturbations that guarantee
    distinct edge weights, so MST edge set is unambiguous."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 10.0, (K, K))
    W = (W + W.T) / 2.0
    if unique:
        # Small vertex-pair specific perturbations break any ties.
        idx_i = np.arange(K)[:, None]
        idx_j = np.arange(K)[None, :]
        W = W + 1e-6 * (idx_i + idx_j + idx_i * idx_j)
        W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


def _mst_edge_set(mst: np.ndarray) -> set[frozenset[int]]:
    """Unordered edge set from a (possibly non-symmetric) weighted MST."""
    K = mst.shape[0]
    edges = set()
    for i in range(K):
        for j in range(K):
            if mst[i, j] != 0:
                edges.add(frozenset((i, j)))
    return edges


def _total_weight(mst: np.ndarray) -> float:
    """Sum of weights of tree edges (symmetric matrix: halve the sum)."""
    return float(mst.sum()) / 2.0


# --------------------------------------------------------------------------- #
# Happy-path correctness                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("K", [2, 3, 4, 5, 10, 25, 64, 100])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_boruvka_weight_matches_scipy(K: int, seed: int) -> None:
    W = _random_symmetric(K, seed, unique=True)
    mst_t = boruvka_mst(torch.from_numpy(W)).numpy()
    mst_scipy = _scipy_mst(W).toarray()
    w_scipy = _total_weight(mst_scipy + mst_scipy.T)
    w_t = _total_weight(mst_t)
    assert abs(w_t - w_scipy) < 1e-10, (
        f"Borůvka total weight {w_t} != scipy {w_scipy} for K={K}, seed={seed}"
    )


@pytest.mark.parametrize("K", [3, 5, 10, 25, 64])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_boruvka_edge_set_matches_prim_on_unique_weights(K: int, seed: int) -> None:
    """When edge weights are unique the MST is unique, so Borůvka and
    Prim must pick exactly the same edges."""
    W = _random_symmetric(K, seed, unique=True)
    mst_b = boruvka_mst(torch.from_numpy(W)).numpy()
    mst_p = _prim_mst(W)
    assert _mst_edge_set(mst_b) == _mst_edge_set(mst_p), (
        f"edge set mismatch (K={K}, seed={seed}): "
        f"B\\P={_mst_edge_set(mst_b) - _mst_edge_set(mst_p)}, "
        f"P\\B={_mst_edge_set(mst_p) - _mst_edge_set(mst_b)}"
    )


@pytest.mark.parametrize("K", [2, 3, 5, 10, 25])
def test_boruvka_tree_properties(K: int) -> None:
    """Result must be (1) symmetric, (2) have exactly 2*(K-1) non-zeros
    (each tree edge listed twice), (3) spanning tree (connected), and
    (4) acyclic (trivially: K-1 edges + connected ⇒ tree)."""
    W = _random_symmetric(K, seed=3, unique=True)
    mst = boruvka_mst(torch.from_numpy(W)).numpy()
    np.testing.assert_allclose(mst, mst.T)
    assert np.count_nonzero(mst) == 2 * (K - 1)

    # Connectivity: BFS/DFS from vertex 0 must reach all vertices.
    adj = (mst > 0)
    visited = np.zeros(K, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in np.nonzero(adj[u])[0]:
            if not visited[v]:
                visited[v] = True
                stack.append(int(v))
    assert visited.all(), "MST is not connected"


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


def test_boruvka_K_equals_one_returns_empty() -> None:
    out = boruvka_mst(torch.zeros((1, 1), dtype=torch.float64))
    assert out.shape == (1, 1)
    assert float(out.sum()) == 0.0


def test_boruvka_K_equals_two_returns_single_edge() -> None:
    W = np.array([[0.0, 2.5], [2.5, 0.0]])
    mst = boruvka_mst(torch.from_numpy(W)).numpy()
    assert mst[0, 1] == 2.5 and mst[1, 0] == 2.5
    assert np.count_nonzero(mst) == 2


def test_boruvka_all_equal_weights_gives_valid_tree() -> None:
    """Pathological: all off-diagonal weights are identical. Any K-1
    edges forming a tree is a valid MST. Result must still be a tree."""
    K = 8
    W = np.ones((K, K)) - np.eye(K)
    mst = boruvka_mst(torch.from_numpy(W)).numpy()
    # Tree structure
    assert np.count_nonzero(mst) == 2 * (K - 1)
    # Weight
    np.testing.assert_allclose(_total_weight(mst), K - 1)
    # Connectivity
    adj = (mst > 0)
    visited = np.zeros(K, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in np.nonzero(adj[u])[0]:
            if not visited[v]:
                visited[v] = True
                stack.append(int(v))
    assert visited.all()


def test_boruvka_non_square_raises() -> None:
    with pytest.raises(ValueError, match=r"square 2D"):
        boruvka_mst(torch.zeros((3, 4)))


def test_boruvka_1d_input_raises() -> None:
    with pytest.raises(ValueError, match=r"square 2D"):
        boruvka_mst(torch.zeros((5,)))


def test_boruvka_nonfinite_entries_raise() -> None:
    """A graph with +inf off-diagonal entries is not connectable beyond
    those edges; for the complete-graph contract we refuse to silently
    produce a partial MST."""
    K = 5
    W = _random_symmetric(K, seed=0, unique=True)
    W[0, 4] = np.inf
    W[4, 0] = np.inf
    # Isolate vertex 0 entirely from the others.
    W[0, 1:] = np.inf
    W[1:, 0] = np.inf
    with pytest.raises(RuntimeError, match=r"edges"):
        boruvka_mst(torch.from_numpy(W))


# --------------------------------------------------------------------------- #
# Dtype / device preservation                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_boruvka_preserves_dtype(dtype: torch.dtype) -> None:
    W = torch.from_numpy(_random_symmetric(6, seed=0, unique=True)).to(dtype)
    out = boruvka_mst(W)
    assert out.dtype == dtype


def test_boruvka_preserves_cpu_device() -> None:
    W = torch.from_numpy(_random_symmetric(6, seed=0, unique=True))
    out = boruvka_mst(W)
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_boruvka_runs_on_cuda() -> None:
    W = torch.from_numpy(_random_symmetric(32, seed=0, unique=True)).cuda()
    out = boruvka_mst(W)
    assert out.device.type == "cuda"
    # Verify correctness on device too — transfer back and compare.
    mst_scipy = _scipy_mst(W.cpu().numpy()).toarray()
    w_cuda = _total_weight(out.cpu().numpy())
    w_scipy = _total_weight(mst_scipy + mst_scipy.T)
    assert abs(w_cuda - w_scipy) < 1e-8
