"""Python port of the DDRTree principal-graph algorithm.

See ``ddrtree.DDRTree`` for the main entry point.
"""

from ._core import DDRTree, DDRTreeResult
from ._utils import pca_projection, sq_dist, get_major_eigenvalue
from ._version import __version__

__all__ = [
    "DDRTree",
    "DDRTreeResult",
    "pca_projection",
    "sq_dist",
    "get_major_eigenvalue",
    "__version__",
]
