"""Utility helpers ported from R/DDRTree.R and src/DDRTree.cpp.

Matrix convention matches the R package: data matrices are stored as
``(D, N)`` — rows are features/dimensions, columns are samples.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import norm


def sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance between columns of ``a`` and ``b``.

    Mirrors ``sqdist_R`` in the R package: inputs are ``(D, Na)`` and
    ``(D, Nb)``; output is ``(Na, Nb)``.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"sq_dist: row counts differ (a: {a.shape[0]}, b: {b.shape[0]})"
        )
    aa = np.sum(a * a, axis=0)                      # (Na,)
    bb = np.sum(b * b, axis=0)                      # (Nb,)
    ab = a.T @ b                                    # (Na, Nb)
    dist = aa[:, None] + bb[None, :] - 2.0 * ab
    return np.abs(dist)


def pca_projection(C: np.ndarray, L: int) -> np.ndarray:
    """Return the top-``L`` eigen / right-singular vectors of ``C``.

    A faithful port of ``pca_projection_R`` from ``R/DDRTree.R``:

    * ``L >= min(dim(C))`` → full ``eigen(C)`` in R. We use
      ``numpy.linalg.eig`` and take the real part, matching the fact that R's
      ``eigen()`` (without ``symmetric=TRUE``) uses the general algorithm
      even on symmetric input.
    * ``L <  min(dim(C))`` → ``irlba(C, nv=L)`` in R, returning the matrix of
      right singular vectors ``$v``. We use ``scipy.sparse.linalg.svds``
      (ARPACK-based Lanczos, in the same family as irlba) to remain close
      to R's iterative behaviour rather than jumping to a direct ``eigh``.

    Output is a ``(nrow(C), L)`` matrix with columns ordered by descending
    eigenvalue / singular value.
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    n, p = C.shape

    if L >= min(n, p):
        # Mirror R's `eigen(C)` (general algorithm even for symmetric input).
        vals, vecs = np.linalg.eig(C)
        order = np.argsort(vals.real)[::-1]
        return np.ascontiguousarray(vecs[:, order[:L]].real)

    # Mirror R's `irlba(C, nv=L, v=initial_v)` — Lanczos-style truncated SVD
    # with the deterministic starting vector R uses:
    #   initial_v <- qnorm(1:(ncol(C)+1)/(ncol(C)+1))[1:ncol(C)]
    # scipy's svds dispatches to ARPACK's eigsh on the (min(m,n))×(min(m,n))
    # Gram matrix, so v0 has length min(n, p).
    v0 = norm.ppf(np.arange(1, min(n, p) + 1) / (min(n, p) + 1))
    _, s, vt = svds(C, k=L, which="LM", v0=v0)
    order = np.argsort(s)[::-1]
    return np.ascontiguousarray(vt.T[:, order])


def get_major_eigenvalue(C: np.ndarray, L: int) -> float:
    """Quirk-faithful port of ``get_major_eigenvalue``.

    The R implementation has two branches:

    * ``L >= min(dim(C))`` → returns ``norm(C, '2') ** 2`` (spectral norm²).
    * ``L <  min(dim(C))`` → returns ``max(abs(eigen_res$v))``, i.e. the
      largest absolute value in the right singular vector matrix. This is
      almost certainly a bug (it ignores the singular values), but it only
      feeds into the objective value that drives convergence checks, so we
      mirror it exactly for numerical parity with the R gold standard.
    """
    C = np.asarray(C, dtype=np.float64)
    n, p = C.shape

    if L >= min(n, p):
        return float(np.linalg.norm(C, ord=2)) ** 2

    # Match the R path: irlba(C, nv=L) → max(abs(v)), the max abs entry of
    # the top-L right singular vectors. Compute via truncated SVD.
    # Use numpy.linalg.svd on the full matrix (sizes in practice are small
    # enough, and this avoids the stochastic convergence paths of ARPACK).
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    v_topL = vt[:L, :].T          # (p, L), matches irlba $v
    return float(np.max(np.abs(v_topL)))


