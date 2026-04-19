"""Computational backends for DDRTree.

Each backend module implements the full DDRTree iteration end-to-end. The
public ``ddrtree.DDRTree`` dispatcher selects a backend by name. Backend
modules are intentionally loaded lazily so that optional dependencies
(``torch``) are only imported when the corresponding backend is requested.
"""
