"""Minimum spanning tree algorithms used by DDRTree backends.

The NumPy backend uses the classical dense-Prim implementation that lives
in ``ddrtree._core`` (kept there for historical reasons and to minimise
diff against the pre-refactor code). This subpackage hosts GPU-friendly
variants that the torch backend dispatches to.
"""
