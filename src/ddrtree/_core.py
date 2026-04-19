"""DDRTree iteration core.

Numerical convention matches the R package: data matrix ``X`` is ``(D, N)``
with D = features, N = samples. The optimisation alternates the updates of

    W ∈ R^{D × d},  Z ∈ R^{d × N},  Y ∈ R^{d × K},  B (MST, K × K),  R (N × K)

as described in Mao et al., KDD'15, and implemented in ``src/DDRTree.cpp``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans

from ._utils import get_major_eigenvalue, pca_projection, sq_dist

_VALID_BACKENDS = ("numpy", "torch", "auto")


def _resolve_auto_backend(device: Optional[str]) -> str:
    """Pick the best available backend for ``backend="auto"``.

    Heuristic (deliberately simple and predictable):

    * If the caller asked for a CUDA device, only ``torch`` can satisfy
      it — missing ``torch`` or CUDA raises here rather than silently
      demoting to NumPy on CPU.
    * Otherwise, prefer ``torch`` on CUDA when available (the speed
      payoff is meaningful); fall back to NumPy (torch on CPU rarely
      beats NumPy on the problem sizes DDRTree targets).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        if device is not None and str(device).startswith("cuda"):
            raise RuntimeError(
                "backend='auto' with device='cuda' requires PyTorch; "
                "install with `pip install ddrtree[torch]`."
            )
        return "numpy"
    import torch as _torch

    if device is not None and str(device).startswith("cuda"):
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "backend='auto' with device='cuda' requires a CUDA-capable "
                "PyTorch build and an available GPU."
            )
        return "torch"

    if device == "cpu":
        return "numpy"  # torch CPU rarely beats numpy at DDRTree sizes

    # device is None: prefer CUDA torch when available, else NumPy.
    if _torch.cuda.is_available():
        return "torch"
    return "numpy"


@dataclass
class DDRTreeResult:
    """Container for DDRTree outputs, mirroring the R return list.

    The R ``DDRTree`` function returns a list with exactly these seven names:
    ``W, Z, stree, Y, X, objective_vals, history``. ``history`` is always
    ``NULL`` in R (it's an unfinished placeholder) — we echo it as ``None``
    so that user code that iterates R-style field names doesn't break.
    """
    W: np.ndarray                      # (D, d) orthonormal basis
    Z: np.ndarray                      # (d, N) reduced-dimension coords
    Y: np.ndarray                      # (d, K) principal graph node coords
    stree: sp.csr_matrix               # (N, N) MST weights, K×K block filled
    X: np.ndarray                      # (D, N) input matrix (echoed, as in R)
    objective_vals: List[float]        # objective per iteration
    history: None = None               # placeholder, always None — matches R


def DDRTree(
    X: np.ndarray,
    dimensions: int = 2,
    initial_method: Optional[Callable[..., np.ndarray]] = None,
    max_iter: int = 20,
    sigma: float = 1e-3,
    lambda_: Optional[float] = None,
    ncenter: Optional[int] = None,
    gamma: float = 10.0,
    tol: float = 1e-3,
    verbose: bool = False,
    mst_algorithm: Optional[str] = None,
    backend: str = "numpy",
    device: Optional[str] = None,
    **kwargs,
) -> DDRTreeResult:
    """Perform DDRTree principal-graph learning.

    Dispatches to the requested computational ``backend``. The ``"numpy"``
    backend (default) is the reference implementation and mirrors the R
    package's ``src/DDRTree.cpp`` line-by-line; it is the target of
    gold-standard parity tests. Other backends (``"torch"``, ``"auto"``) are
    introduced by later phases and share the same public signature and
    return contract.

    ``mst_algorithm`` picks the minimum-spanning-tree algorithm used each
    iteration. When left ``None`` each backend uses its natural default:
    the NumPy backend runs dense Prim (matches R), and the torch backend
    runs a GPU-friendly parallel Borůvka. Explicit values are
    ``"prim"`` / ``"kruskal"`` (both backends) and ``"boruvka"`` (torch
    only). All choices yield the same tree when edge weights are unique;
    they may differ only in tie-breaking, which DDRTree's continuous
    squared-distance weights never trigger in practice.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS!r}; got {backend!r}"
        )
    if backend == "auto":
        backend = _resolve_auto_backend(device)

    if backend == "numpy":
        # NumPy backend has no device concept. Accept only None or "cpu"
        # so that ``backend="auto"`` can hand us ``device="cpu"`` without
        # friction, while any GPU request is rejected clearly.
        if device not in (None, "cpu"):
            raise ValueError(
                f"device={device!r} is not supported by backend='numpy'. "
                "Use backend='torch' for GPU/CUDA execution."
            )
        # Resolve the per-backend MST default. NumPy natively runs dense
        # Prim (matches R's src/DDRTree.cpp). "boruvka" is not offered by
        # the NumPy backend — we refuse to accept it here rather than
        # silently switching algorithms.
        if mst_algorithm is None:
            mst_algorithm = "prim"
        elif mst_algorithm == "boruvka":
            raise ValueError(
                "mst_algorithm='boruvka' is only available with "
                "backend='torch'."
            )
        return _ddrtree_numpy(
            X=X,
            dimensions=dimensions,
            initial_method=initial_method,
            max_iter=max_iter,
            sigma=sigma,
            lambda_=lambda_,
            ncenter=ncenter,
            gamma=gamma,
            tol=tol,
            verbose=verbose,
            mst_algorithm=mst_algorithm,
            **kwargs,
        )
    # backend == "torch": lazy import so installations without PyTorch
    # keep working for the default NumPy path. Torch's natural default is
    # the GPU-friendly parallel Borůvka.
    if mst_algorithm is None:
        mst_algorithm = "boruvka"
    from ._backends._torch import ddrtree_torch

    return ddrtree_torch(
        X=X,
        dimensions=dimensions,
        initial_method=initial_method,
        max_iter=max_iter,
        sigma=sigma,
        lambda_=lambda_,
        ncenter=ncenter,
        gamma=gamma,
        tol=tol,
        verbose=verbose,
        mst_algorithm=mst_algorithm,
        device=device,
        **kwargs,
    )


def _ddrtree_numpy(
    X: np.ndarray,
    dimensions: int = 2,
    initial_method: Optional[Callable[..., np.ndarray]] = None,
    max_iter: int = 20,
    sigma: float = 1e-3,
    lambda_: Optional[float] = None,
    ncenter: Optional[int] = None,
    gamma: float = 10.0,
    tol: float = 1e-3,
    verbose: bool = False,
    mst_algorithm: str = "prim",
    **kwargs,
) -> DDRTreeResult:
    """NumPy reference implementation of DDRTree.

    Parameters match the R ``DDRTree`` function (with Python-idiomatic
    renames: ``maxIter`` → ``max_iter``, ``param.gamma`` → ``gamma``,
    ``lambda`` → ``lambda_`` because ``lambda`` is reserved).

    Parameters
    ----------
    X : ndarray of shape (D, N)
        Input data; columns are samples. Follows R's convention.
    dimensions : int
        Target reduced dimensionality ``d``.
    initial_method : callable, optional
        If provided, called as ``initial_method(X, **kwargs)`` (matching R's
        ``initial_method(X, ...)`` semantics). Expected to return an array
        of shape ``(N, d_out)`` with ``d_out >= dimensions``; used to
        initialise ``Z`` in place of the default PCA initialisation.
    max_iter : int
    sigma : float
        Bandwidth for the soft-assignment (softmax).
    lambda_ : float, optional
        Regularisation strength for the reversed-graph-embedding term.
        Defaults to ``5 * N`` when ``None`` (matches R default).
    ncenter : int, optional
        Number of principal-graph nodes ``K``. ``None`` → ``K = N`` (no
        K-means initialisation).
    gamma : float
        Regularisation strength for the K-means-like term.
    tol : float
        Relative change in objective below which we stop early.
    verbose : bool
        Print iteration-level diagnostics.
    mst_algorithm : {"prim", "kruskal"}
        Which minimum-spanning-tree algorithm to run on the K×K
        inter-center squared distances each iteration. ``"prim"`` (default)
        mirrors Boost.Graph's ``prim_minimum_spanning_tree`` used in the R
        package's ``src/DDRTree.cpp`` and is recommended for strict numerical
        parity with R. ``"kruskal"`` dispatches to
        ``scipy.sparse.csgraph.minimum_spanning_tree``. Both give identical
        trees whenever edge weights are unique; they can differ only in
        tie-breaking.
    **kwargs
        Passed through to ``initial_method(X, **kwargs)`` when that callback
        is supplied; silently ignored otherwise (matches R's ``...`` which
        is only consumed by ``initial_method`` and dropped when absent).

    Returns
    -------
    DDRTreeResult
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    D, N = X.shape

    # ---- Initialisation (mirrors DDRTree.R) ------------------------------
    W = pca_projection(X @ X.T, dimensions)          # (D, d)
    if initial_method is None:
        Z = W.T @ X                                  # (d, N)
    else:
        tmp = np.asarray(initial_method(X, **kwargs), dtype=np.float64)
        if tmp.shape[0] > N or tmp.shape[1] > D:
            raise ValueError(
                "initial_method must return (<=N) x (<=D); got "
                f"{tmp.shape}"
            )
        Z = tmp[:, :dimensions].T                    # (d, N)

    if ncenter is None:
        K = N
        Y = Z[:, :K].copy()
    else:
        K = int(ncenter)
        if K > Z.shape[1]:
            raise ValueError(
                "ncenter must be <= ncol(X); "
                f"got ncenter={K}, ncol(X)={Z.shape[1]}"
            )
        # Match R exactly: kmeans on t(Z) with initial centers taken as
        # t(Z)[seq(1, ncol(Z), length.out=K), ]. R's vector subsetting
        # truncates non-integer indices toward zero, so the 0-based indices
        # are `floor(linspace(1, N, K)) - 1`.
        idx = np.linspace(1, Z.shape[1], num=K).astype(int) - 1
        init_centers = Z.T[idx, :]
        km = KMeans(n_clusters=K, init=init_centers, n_init=1, max_iter=10,
                    algorithm="lloyd", tol=1e-4).fit(Z.T)
        Y = km.cluster_centers_.T                    # (d, K)

    if lambda_ is None:
        lambda_ = 5.0 * N

    if mst_algorithm not in ("prim", "kruskal"):
        raise ValueError(
            f"mst_algorithm must be 'prim' or 'kruskal'; got {mst_algorithm!r}"
        )

    return _ddrtree_reduce_dim(
        X=X, Z=Z, Y=Y, W=W,
        dimensions=dimensions, max_iter=max_iter, K=K,
        sigma=sigma, lambda_=lambda_, gamma=gamma, tol=tol,
        verbose=verbose, mst_algorithm=mst_algorithm,
    )


def _ddrtree_reduce_dim(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    dimensions: int,
    max_iter: int,
    K: int,
    sigma: float,
    lambda_: float,
    gamma: float,
    tol: float,
    verbose: bool,
    mst_algorithm: str = "prim",
) -> DDRTreeResult:
    """Inner loop — a line-by-line port of ``DDRTree_reduce_dim_cpp``."""
    D, N = X.shape
    objective_vals: List[float] = []

    # Pre-allocate — not strictly necessary in numpy but keeps shapes
    # explicit and helps mental mapping back to the C++ reference.
    last_tree = np.zeros((K, K), dtype=np.float64)     # last distance-weighted MST
    B = np.zeros((K, K), dtype=np.float64)             # MST adjacency (0/1)

    for it in range(max_iter):
        if verbose:
            print(f"[DDRTree] iter {it}")

        # --- MST over current Y centers -------------------------------
        distsqMU = sq_dist(Y, Y)                       # (K, K)
        if mst_algorithm == "prim":
            # Default. Matches Boost.Graph's prim_minimum_spanning_tree
            # (start vertex 0) used in R's src/DDRTree.cpp.
            mst = _prim_mst(distsqMU)
        else:  # "kruskal"
            # SciPy's Kruskal-style MST. Zero off-diagonal entries are
            # dropped, which matches our need to skip self-loops
            # (diagonal ≈ 0).
            mst_sp = minimum_spanning_tree(distsqMU)
            mst = mst_sp.toarray()
            mst = mst + mst.T                          # symmetrise
        B = (mst > 0).astype(np.float64)               # (K, K) 0/1 adjacency
        last_tree = mst                                # keep weights for output

        # Laplacian L = diag(B @ 1) - B
        L_mat = np.diag(B.sum(axis=0)) - B             # (K, K)

        # --- Soft assignment R (N, K) ---------------------------------
        distZY = sq_dist(Z, Y)                         # (N, K)
        min_dist = distZY.min(axis=1, keepdims=True)   # (N, 1)
        tmp_distZY = distZY - min_dist                 # (N, K)
        tmp_R = np.exp(-tmp_distZY / sigma)            # (N, K)
        row_sums = tmp_R.sum(axis=1, keepdims=True)    # (N, 1)
        R_mat = tmp_R / row_sums                       # (N, K) softmax

        # Gamma = diag(colSums(R))
        Gamma = np.diag(R_mat.sum(axis=0))             # (K, K)

        # --- Objective (mirrors C++ exactly, quirks and all) ----------
        # obj1 = -sigma * sum(log(rowSum(exp(-tmp_distZY/sigma))) - min_dist[:,0]/sigma)
        x1 = np.log(np.sum(np.exp(-tmp_distZY / sigma), axis=1))
        obj1 = -sigma * float(np.sum(x1 - min_dist[:, 0] / sigma))

        # obj2 = get_major_eigenvalue(X - W @ Z, d)^2 + lambda * trace(Y L Y^T) + gamma * obj1
        major_eig = get_major_eigenvalue(X - W @ Z, dimensions)
        obj2 = major_eig * major_eig
        obj2 += lambda_ * float(np.trace(Y @ L_mat @ Y.T))
        obj2 += gamma * obj1
        objective_vals.append(obj2)

        if verbose:
            print(f"[DDRTree]   objective = {obj2:.8f}")

        # --- Convergence check ----------------------------------------
        # Matches R/C++: `delta /= |prev|` unconditionally. If prev is zero
        # the division yields +Inf, which is never < tol, so iteration
        # continues — exactly as in the C++ reference.
        if it >= 1:
            delta = abs(objective_vals[-1] - objective_vals[-2])
            delta /= abs(objective_vals[-2])
            if verbose:
                print(f"[DDRTree]   delta = {delta:.3e}")
            if delta < tol:
                break

        # --- Updates --------------------------------------------------
        # tmp = ((Gamma + (lambda/gamma) * L) * (gamma+1)/gamma) - R^T R
        tmp_M = (Gamma + (lambda_ / gamma) * L_mat) * ((gamma + 1.0) / gamma)
        tmp_M = tmp_M - R_mat.T @ R_mat                # (K, K)
        # tmp_dense = solve(tmp_M, R^T).T              (N, K)
        # Mirrors the C++ path in src/DDRTree.cpp: try Cholesky first,
        # fall back to LU on failure. When the fallback fires we emit a
        # RuntimeWarning so the non-PSD signal doesn't stay invisible —
        # the C++ reference prints "Error!" in the same situation.
        try:
            Lc = np.linalg.cholesky(tmp_M)
            sol = _cho_solve(Lc, R_mat.T)              # (K, N)
        except np.linalg.LinAlgError:
            warnings.warn(
                "DDRTree: Cholesky decomposition of tmp_M failed at "
                f"iteration {it}; falling back to LU. This usually means "
                "the soft-assignment R has collapsed or parameters "
                "(lambda, gamma, sigma) have driven the system non-PSD.",
                RuntimeWarning,
                stacklevel=2,
            )
            sol = np.linalg.solve(tmp_M, R_mat.T)      # (K, N)
        tmp_dense = sol.T                              # (N, K)

        # Q_eff  (D, N)  =  (X + (X @ tmp_dense) @ R^T) / (gamma+1)
        # This is X · Q_paper in abbreviated form — avoids materialising N×N Q.
        Q_eff = (X + (X @ tmp_dense) @ R_mat.T) / (gamma + 1.0)

        # tmp1 = Q_eff @ X^T   (D, D)
        tmp1 = Q_eff @ X.T

        # W = top-d eigenvectors of (tmp1 + tmp1^T)/2
        W = pca_projection((tmp1 + tmp1.T) / 2.0, dimensions)          # (D, d)

        # Z = W^T @ C, where C = Q_eff in this optimised formulation
        Z = W.T @ Q_eff                                                # (d, N)

        # Y = solve(lambda/gamma * L + Gamma, (Z R)^T)^T
        # (λ/γ)·L + Γ is PSD by construction (L is a graph Laplacian,
        # Γ is diagonal with non-negative entries), so Cholesky must succeed.
        # We mirror the unconditional LLT.solve in R's src/DDRTree.cpp:
        # any failure here is a genuine numerical breakdown that should
        # surface rather than be masked by an LU fallback.
        Y_lhs = (lambda_ / gamma) * L_mat + Gamma
        Lc2 = np.linalg.cholesky(Y_lhs)
        Y = _cho_solve(Lc2, (Z @ R_mat).T).T                           # (d, K)

    # ---- Final stree as N × N sparse, K × K block populated --------------
    stree = sp.lil_matrix((N, N), dtype=np.float64)
    stree[:K, :K] = last_tree
    stree = stree.tocsr()

    return DDRTreeResult(
        W=W, Z=Z, Y=Y, stree=stree, X=X,
        objective_vals=objective_vals,
    )


def _cho_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve ``A @ X = B`` given Cholesky factor ``L`` such that ``A = L L^T``."""
    y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, y)


def _prim_mst(dist_sq: np.ndarray) -> np.ndarray:
    """Minimum spanning tree via Prim's algorithm starting at vertex 0.

    Mirrors Boost.Graph's ``prim_minimum_spanning_tree(g, &spanning_tree[0])``
    as used in R's ``src/DDRTree.cpp``: a complete undirected graph over K
    vertices with weights ``dist_sq[i, j]`` and the default start vertex 0.

    Parameters
    ----------
    dist_sq : ndarray of shape (K, K)
        Symmetric non-negative weight matrix. Diagonal is ignored (self-loops
        are never chosen).

    Returns
    -------
    mst : ndarray of shape (K, K)
        Symmetric weighted adjacency of the MST. Entry ``mst[i, j]`` equals
        ``dist_sq[i, j]`` when the edge ``(i, j)`` is in the tree, else 0.

    Notes
    -----
    The implementation is the classical O(K²) dense Prim using vectorised
    relaxations; appropriate for K up to a few thousand — the regime Monocle 2
    operates in. For unique edge weights the resulting tree is identical to
    Kruskal's. Only tie-breaking differs, where Prim prefers the vertex with
    the smaller index as returned by ``np.argmin``.
    """
    K = dist_sq.shape[0]
    mst = np.zeros((K, K), dtype=np.float64)
    if K <= 1:
        return mst

    in_tree = np.zeros(K, dtype=bool)
    min_edge_w = np.full(K, np.inf, dtype=np.float64)
    min_edge_from = np.full(K, -1, dtype=np.int64)

    # Seed: add vertex 0 to the tree and relax its outgoing edges.
    in_tree[0] = True
    min_edge_w[1:] = dist_sq[0, 1:]
    min_edge_from[1:] = 0

    for _ in range(K - 1):
        # Pick the non-tree vertex with the smallest incoming edge weight.
        candidate = np.where(in_tree, np.inf, min_edge_w)
        u = int(np.argmin(candidate))
        if not np.isfinite(candidate[u]):
            # No finite-weight edge connects the remaining vertices to the
            # growing tree. Either the input has non-finite weights
            # (e.g. Y collapsed to NaN) or the graph is disconnected. Both
            # signal a real problem — surface it rather than quietly
            # returning a partial MST.
            raise ValueError(
                "Prim's MST could not grow: remaining vertices have no "
                "finite-weight edge to the tree (NaN/Inf in weights, or "
                "a disconnected graph)."
            )
        p = int(min_edge_from[u])
        w = float(dist_sq[u, p])
        mst[u, p] = w
        mst[p, u] = w
        in_tree[u] = True
        # Relax edges from the newly-added vertex u.
        row = dist_sq[u]
        improved = (~in_tree) & (row < min_edge_w)
        min_edge_w = np.where(improved, row, min_edge_w)
        min_edge_from = np.where(improved, u, min_edge_from)

    return mst
