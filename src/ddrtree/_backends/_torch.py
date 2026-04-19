"""PyTorch backend for DDRTree (CPU path introduced in P2).

This module mirrors the structure of ``ddrtree._core._ddrtree_numpy`` so the
two implementations can be diffed side-by-side. All tensors are ``float64``
by default; P5 will introduce an ``fp32`` opt-in. MST still routes through
the NumPy Prim implementation in this phase — P3 replaces it with a pure
``torch`` Borůvka that also runs on CUDA.

Design notes
------------

* ``torch`` is imported at module load time. ``_core`` only imports this
  module when ``backend="torch"`` is requested, so installations without
  PyTorch are unaffected.
* ``pca_projection`` uses ``torch.linalg.svd`` (direct) for the truncated
  branch rather than an iterative Lanczos solver. This is a deliberate
  deviation from the NumPy path's ``scipy.sparse.linalg.svds``: ``torch``
  has no ARPACK-with-``v0`` equivalent, and a direct SVD is both
  deterministic and numerically sufficient at the matrix sizes DDRTree
  operates on.
* Cholesky uses ``torch.linalg.cholesky_ex``: if ``info > 0`` (non-PSD),
  we fall back to ``torch.linalg.lu_factor_ex`` + ``lu_solve`` and emit a
  ``RuntimeWarning``. This matches the NumPy backend's Cholesky → LU
  fallback — but via explicit info codes rather than a broad ``try/except``,
  so unrelated failures propagate untouched.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

from .._core import DDRTreeResult, _prim_mst
from .._mst._boruvka_torch import boruvka_mst
from scipy.sparse.csgraph import minimum_spanning_tree as _scipy_mst


# ---------------------------------------------------------------------------
# Small torch-native helpers. Intentionally private: the public helpers in
# ``ddrtree._utils`` remain NumPy-only. Users writing torch-native code can
# import these via ``ddrtree._backends._torch`` but we don't promote them.
# ---------------------------------------------------------------------------


def _torch_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance; mirrors ``_utils.sq_dist``.

    ``a`` is ``(D, Na)``, ``b`` is ``(D, Nb)``; returns ``(Na, Nb)``.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"_torch_sq_dist: row counts differ (a: {a.shape[0]}, b: {b.shape[0]})"
        )
    aa = (a * a).sum(dim=0)
    bb = (b * b).sum(dim=0)
    ab = a.T @ b
    return (aa[:, None] + bb[None, :] - 2.0 * ab).abs()


def _torch_pca_projection(C: torch.Tensor, L: int) -> torch.Tensor:
    """Top-``L`` eigen/right-singular vectors of ``C``; mirrors ``pca_projection``.

    * ``L >= min(dim(C))`` → ``torch.linalg.eig`` (non-symmetric algorithm,
      matching R's default ``eigen()``).
    * ``L <  min(dim(C))`` → direct ``torch.linalg.svd`` top-L. Differs from
      the NumPy backend's iterative ``svds`` path; acceptable drift source,
      documented and covered by relaxed parity tests.
    """
    n, p = C.shape
    if L >= min(n, p):
        vals, vecs = torch.linalg.eig(C)
        order = torch.argsort(vals.real, descending=True)
        return vecs[:, order[:L]].real.contiguous()
    _, _, Vh = torch.linalg.svd(C, full_matrices=False)
    # singular values are sorted descending, so Vh[:L] are the top-L
    return Vh[:L, :].T.contiguous()


def _torch_get_major_eigenvalue(C: torch.Tensor, L: int) -> float:
    """Quirk-faithful port of ``_utils.get_major_eigenvalue``."""
    n, p = C.shape
    if L >= min(n, p):
        s_max = torch.linalg.matrix_norm(C, ord=2)
        return float(s_max * s_max)
    _, _, Vh = torch.linalg.svd(C, full_matrices=False)
    v_topL = Vh[:L, :].T
    return float(v_topL.abs().max())


def _torch_cholesky_solve(
    A: torch.Tensor, B: torch.Tensor, *, iter_index: int
) -> torch.Tensor:
    """Solve ``A @ X = B`` via Cholesky, LU fallback when A is non-PSD.

    Semantics match the NumPy backend's Cholesky → LU fallback in the
    ``tmp_M`` system: when the soft-assignment collapses or parameters
    drive the system non-PSD, we warn loudly and continue via LU rather
    than bailing out — the C++ reference behaves the same way.

    The fallback is guarded on ``info`` from ``cholesky_ex`` rather than a
    bare ``try/except``: a non-PSD signal from BLAS returns ``info > 0``
    cleanly, while any other unrelated error still propagates.
    """
    Lc, info = torch.linalg.cholesky_ex(A)
    if int(info) == 0:
        return torch.cholesky_solve(B, Lc)
    warnings.warn(
        "DDRTree[torch]: Cholesky decomposition of tmp_M failed at "
        f"iteration {iter_index}; falling back to LU. This usually means "
        "the soft-assignment R has collapsed or parameters "
        "(lambda, gamma, sigma) have driven the system non-PSD.",
        RuntimeWarning,
        stacklevel=3,
    )
    LU, piv, info2 = torch.linalg.lu_factor_ex(A)
    if int(info2) != 0:
        raise np.linalg.LinAlgError(
            f"DDRTree[torch]: both Cholesky (info={int(info)}) and LU "
            f"(info={int(info2)}) failed on tmp_M."
        )
    return torch.linalg.lu_solve(LU, piv, B)


# ---------------------------------------------------------------------------
# MST dispatch. P2 routes Torch through the NumPy Prim (via host copy) so
# the MST edge set is bit-for-bit identical to the NumPy backend. P3 will
# introduce a GPU-friendly Borůvka implementation and make it the torch
# default.
# ---------------------------------------------------------------------------


def _mst_via_numpy(distsqMU: torch.Tensor, mst_algorithm: str) -> torch.Tensor:
    """Host-side MST fallback (Prim or Kruskal). For strict parity testing."""
    dist_np = distsqMU.detach().cpu().numpy()
    if mst_algorithm == "prim":
        mst_np = _prim_mst(dist_np)
    else:  # "kruskal"
        mst_sp = _scipy_mst(dist_np)
        mst_np = mst_sp.toarray()
        mst_np = mst_np + mst_np.T
    return torch.from_numpy(mst_np).to(device=distsqMU.device, dtype=distsqMU.dtype)


def _compute_mst(distsqMU: torch.Tensor, mst_algorithm: str) -> torch.Tensor:
    """Dispatch MST computation.

    * ``"boruvka"`` stays entirely on-device (torch). On CUDA this is the
      fast path — no host round trip.
    * ``"prim"`` and ``"kruskal"`` transfer to NumPy / SciPy for bit-for-bit
      parity with the reference NumPy backend; intended for strict parity
      testing rather than production runs.
    """
    if mst_algorithm == "boruvka":
        return boruvka_mst(distsqMU)
    return _mst_via_numpy(distsqMU, mst_algorithm)


# ---------------------------------------------------------------------------
# Public entry point for the torch backend. Signature and semantics match
# ``_ddrtree_numpy``; the ``_core`` dispatcher forwards kwargs unchanged.
# ---------------------------------------------------------------------------


def ddrtree_torch(
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
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs,
) -> DDRTreeResult:
    """Torch-based DDRTree.

    Input ``X`` may be a NumPy array or any array-like convertible via
    ``np.ascontiguousarray``. The result is returned as NumPy arrays to
    preserve backward compatibility with existing downstream code.

    Parameters
    ----------
    device : str or None
        Target torch device (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        When ``None`` we default to ``"cpu"`` — if the caller wants GPU
        execution they must opt in explicitly, either via ``device="cuda"``
        here or through ``backend="auto"``.
    dtype : {"float32", "float64"} or None
        Compute precision for the iteration tensors. ``None`` (default)
        is ``"float64"`` — the R-aligned reference precision. ``"float32"``
        halves memory and typically gains a further 1.5–2× on CUDA, at
        the cost of ~1e-3 relative drift in the converged Y/Z and a
        slightly higher chance of the Cholesky→LU fallback firing when
        the ``tmp_M`` system drifts near non-PSD.
    """
    X_np = np.ascontiguousarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_np.shape}")
    D, N = X_np.shape

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} was requested but PyTorch reports "
            "CUDA is not available. Install a CUDA-capable build or "
            "choose device='cpu'."
        )
    if dtype in (None, "float64"):
        t_dtype = torch.float64
    elif dtype == "float32":
        t_dtype = torch.float32
    else:
        raise ValueError(
            f"dtype must be None, 'float32' or 'float64'; got {dtype!r}"
        )
    dtype = t_dtype

    X_t = torch.from_numpy(X_np).to(device=resolved_device, dtype=dtype)

    # ---- Initialisation (mirrors _ddrtree_numpy / DDRTree.R) ----------------
    W = _torch_pca_projection(X_t @ X_t.T, dimensions)            # (D, d)
    if initial_method is None:
        Z = W.T @ X_t                                             # (d, N)
    else:
        tmp_np = np.asarray(initial_method(X_np, **kwargs), dtype=np.float64)
        if tmp_np.shape[0] > N or tmp_np.shape[1] > D:
            raise ValueError(
                "initial_method must return (<=N) x (<=D); got "
                f"{tmp_np.shape}"
            )
        tmp = torch.from_numpy(np.ascontiguousarray(tmp_np)).to(
            device=resolved_device, dtype=dtype
        )
        Z = tmp[:, :dimensions].T                                 # (d, N)

    if ncenter is None:
        K = N
        Y = Z[:, :K].clone()
    else:
        K = int(ncenter)
        if K > Z.shape[1]:
            raise ValueError(
                "ncenter must be <= ncol(X); "
                f"got ncenter={K}, ncol(X)={Z.shape[1]}"
            )
        # K-means init goes through sklearn on CPU; for typical K this is
        # cheap, and it keeps the initialisation identical between backends.
        Z_np = Z.detach().cpu().numpy()
        idx = np.linspace(1, Z_np.shape[1], num=K).astype(int) - 1
        init_centers = Z_np.T[idx, :]
        km = KMeans(
            n_clusters=K,
            init=init_centers,
            n_init=1,
            max_iter=10,
            algorithm="lloyd",
            tol=1e-4,
        ).fit(Z_np.T)
        Y = torch.from_numpy(
            np.ascontiguousarray(km.cluster_centers_.T, dtype=np.float64)
        ).to(device=resolved_device, dtype=dtype)

    if lambda_ is None:
        lambda_ = 5.0 * N

    if mst_algorithm not in ("prim", "kruskal", "boruvka"):
        raise ValueError(
            "mst_algorithm must be 'prim', 'kruskal' or 'boruvka'; "
            f"got {mst_algorithm!r}"
        )

    return _ddrtree_reduce_dim_torch(
        X_t=X_t, X_np=X_np,
        Z=Z, Y=Y, W=W,
        dimensions=dimensions, max_iter=max_iter, K=K,
        sigma=sigma, lambda_=lambda_, gamma=gamma, tol=tol,
        verbose=verbose, mst_algorithm=mst_algorithm,
    )


def _ddrtree_reduce_dim_torch(
    X_t: torch.Tensor,
    X_np: np.ndarray,
    Z: torch.Tensor,
    Y: torch.Tensor,
    W: torch.Tensor,
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
    """Inner loop — torch port of ``_ddrtree_reduce_dim``."""
    D, N = X_t.shape
    objective_vals: List[float] = []
    dtype = X_t.dtype
    device = X_t.device

    last_tree = torch.zeros((K, K), dtype=dtype, device=device)

    for it in range(max_iter):
        if verbose:
            print(f"[DDRTree/torch] iter {it}")

        # --- MST over current Y centers ---------------------------------
        distsqMU = _torch_sq_dist(Y, Y)                        # (K, K)
        mst = _compute_mst(distsqMU, mst_algorithm)
        B = (mst > 0).to(dtype=dtype)                          # (K, K) 0/1
        last_tree = mst

        L_mat = torch.diag(B.sum(dim=0)) - B                   # (K, K)

        # --- Soft assignment R (N, K) ------------------------------------
        distZY = _torch_sq_dist(Z, Y)                          # (N, K)
        min_dist, _ = distZY.min(dim=1, keepdim=True)          # (N, 1)
        tmp_distZY = distZY - min_dist
        tmp_R = torch.exp(-tmp_distZY / sigma)
        row_sums = tmp_R.sum(dim=1, keepdim=True)
        R_mat = tmp_R / row_sums                               # (N, K)

        Gamma = torch.diag(R_mat.sum(dim=0))                   # (K, K)

        # --- Objective (mirrors C++ exactly) -----------------------------
        x1 = torch.log(torch.sum(torch.exp(-tmp_distZY / sigma), dim=1))
        obj1 = -sigma * float((x1 - min_dist[:, 0] / sigma).sum())

        major_eig = _torch_get_major_eigenvalue(X_t - W @ Z, dimensions)
        obj2 = major_eig * major_eig
        obj2 += lambda_ * float(torch.trace(Y @ L_mat @ Y.T))
        obj2 += gamma * obj1
        objective_vals.append(obj2)

        if verbose:
            print(f"[DDRTree/torch]   objective = {obj2:.8f}")

        if it >= 1:
            delta = abs(objective_vals[-1] - objective_vals[-2])
            delta /= abs(objective_vals[-2])
            if verbose:
                print(f"[DDRTree/torch]   delta = {delta:.3e}")
            if delta < tol:
                break

        # --- Updates -----------------------------------------------------
        tmp_M = (Gamma + (lambda_ / gamma) * L_mat) * ((gamma + 1.0) / gamma)
        tmp_M = tmp_M - R_mat.T @ R_mat                        # (K, K)
        sol = _torch_cholesky_solve(tmp_M, R_mat.T, iter_index=it)
        tmp_dense = sol.T                                      # (N, K)

        Q_eff = (X_t + (X_t @ tmp_dense) @ R_mat.T) / (gamma + 1.0)
        tmp1 = Q_eff @ X_t.T
        W = _torch_pca_projection((tmp1 + tmp1.T) / 2.0, dimensions)
        Z = W.T @ Q_eff                                        # (d, N)

        # Y = solve((λ/γ)L + Γ, (Z R)^T)^T — the LHS is PSD by construction
        # (Laplacian + non-negative diagonal), so a Cholesky failure here
        # would signal a genuine numerical breakdown. We surface it instead
        # of masking it with an LU fallback.
        Y_lhs = (lambda_ / gamma) * L_mat + Gamma
        Lc2, info2 = torch.linalg.cholesky_ex(Y_lhs)
        if int(info2) != 0:
            raise np.linalg.LinAlgError(
                "DDRTree[torch]: Cholesky on (lambda/gamma)*L + Gamma "
                f"failed at iter {it} (info={int(info2)}); this indicates "
                "numerical breakdown in the Y update."
            )
        Y = torch.cholesky_solve((Z @ R_mat).T, Lc2).T         # (d, K)

    # ---- Final stree as N × N sparse, K × K block populated ----------------
    last_tree_np = last_tree.detach().cpu().numpy()
    stree = sp.lil_matrix((N, N), dtype=np.float64)
    stree[:K, :K] = last_tree_np
    stree = stree.tocsr()

    return DDRTreeResult(
        W=W.detach().cpu().numpy(),
        Z=Z.detach().cpu().numpy(),
        Y=Y.detach().cpu().numpy(),
        stree=stree,
        X=X_np,
        objective_vals=objective_vals,
    )
