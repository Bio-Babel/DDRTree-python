"""Parallel Borůvka MST in PyTorch.

Given a K×K symmetric weight matrix of a *complete* graph, return the
minimum spanning tree as a symmetric weighted adjacency matrix. All
operations are device-agnostic ``torch`` ops, so the function runs on CPU
or CUDA without branching.

Algorithm
---------
Each round in Borůvka's algorithm does, in parallel:

1. **Find roots.** Compress the union-find forest with ``O(log K)``
   pointer-jumping steps so every vertex points directly at its current
   component root.
2. **Mask intra-component edges.** Set ``D[i, j] = +inf`` where
   ``root[i] == root[j]`` so they can never win a min reduction.
3. **Per-node min outgoing edge.** ``(min_val, min_dst) = D_masked.min(dim=1)``.
4. **Per-component min.** Segmented reduction: for each component id
   ``c``, take ``min`` over all nodes whose ``root == c``. Implemented via
   ``scatter_reduce_("amin")`` — native on GPU.
5. **Determine the component's chosen winner edge.** A node is "its
   component's winner" iff its local min equals the component min; tie
   is broken by choosing the smallest node id in the component.
6. **Mirror-pair resolution.** If components ``A`` and ``B`` both pick
   each other's winner edge, keep only the direction where
   ``root[src] < root[dst]``. This is a classical Borůvka trick to avoid
   2-cycles; it preserves the ``O(log K)`` round bound because at least
   one edge per component is still retained in the chosen forest (the
   discarded mirror's partner still unions the same pair).
7. **Record edges + union.** Write ``D[src, dst]`` into the symmetric MST
   matrix; link the root of ``src`` to the root of ``dst``.

Correctness & determinism
-------------------------
For a complete graph with strictly unique edge weights, the resulting
edge set is identical to the one produced by any other MST algorithm
(Prim, Kruskal). When weights tie, the tie-break on smallest node id
yields a deterministic — but not necessarily Prim-equivalent — edge set
whose total weight is still minimal. DDRTree weights are continuous
squared distances between real-valued principal-graph nodes, so the
tie-free case is what we hit in practice.
"""

from __future__ import annotations

import math

import torch


def boruvka_mst(D: torch.Tensor) -> torch.Tensor:
    """Minimum spanning tree of a complete weighted graph via Borůvka.

    Parameters
    ----------
    D : torch.Tensor, shape (K, K)
        Symmetric non-negative weight matrix. Must be finite on the
        off-diagonal; diagonal is ignored. Dtype may be float32 or
        float64; output preserves dtype and device.

    Returns
    -------
    mst : torch.Tensor, shape (K, K)
        Symmetric adjacency matrix: entry ``(i, j)`` equals ``D[i, j]``
        iff ``(i, j)`` is a tree edge, else 0. Same device and dtype as
        ``D``.

    Raises
    ------
    ValueError
        If ``D`` is not square 2D.
    RuntimeError
        If the algorithm finishes with fewer than ``K - 1`` edges (which
        can only happen when ``D`` contains non-finite entries off the
        diagonal or the graph is not actually complete).
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"boruvka_mst: D must be square 2D; got shape {tuple(D.shape)}"
        )
    K = D.shape[0]
    device = D.device
    dtype = D.dtype
    mst = torch.zeros((K, K), dtype=dtype, device=device)
    if K <= 1:
        return mst

    INF = float("inf")

    # Working copy with +inf on diagonal so self-loops never win a min.
    D_work = D.clone()
    diag_idx = torch.arange(K, device=device)
    D_work[diag_idx, diag_idx] = INF

    parent = torch.arange(K, device=device)           # union-find pointer
    node_idx = torch.arange(K, device=device)         # for winner lookups

    # Pointer-jumping depth: ceil(log2(K)) + 1 suffices because the union-
    # find tree depth is bounded by the number of completed rounds and is
    # halved by each pointer-jumping iteration.
    jump_depth = max(1, int(math.ceil(math.log2(K))) + 1)

    # Outer round bound: with unique weights Borůvka finishes in O(log K)
    # rounds. Pathological tied-weight inputs can reduce merges per round
    # down to a single edge (star-pattern caused by argmin tie-breaking),
    # so the worst-case upper bound is ``K - 1`` rounds. Normal runs exit
    # early via the ``edges_added >= edges_needed`` check; this bound is a
    # safety ceiling.
    max_rounds = K - 1

    edges_needed = K - 1
    edges_added = 0
    sentinel = K   # larger than any real node id; used to mark "no winner"

    for _ in range(max_rounds):
        # 1. Find roots via pointer jumping, then path-compress `parent`.
        root = parent.clone()
        for _ in range(jump_depth):
            root = parent[root]
        parent = root.clone()

        prev_edges = edges_added

        # Early exit when only one component remains.
        if torch.unique(root).numel() <= 1:
            break

        # 2. Mask intra-component edges.
        same = root.unsqueeze(1) == root.unsqueeze(0)
        D_masked = torch.where(same, torch.tensor(INF, dtype=dtype, device=device), D_work)

        # 3. Per-node best outgoing edge.
        min_val, min_dst = D_masked.min(dim=1)            # (K,), (K,)

        # 4. Per-component best min_val via segmented amin on root buckets.
        comp_best_val = torch.full(
            (K,), INF, dtype=dtype, device=device,
        )
        comp_best_val.scatter_reduce_(
            0, root, min_val, reduce="amin", include_self=True,
        )

        # 5. Winner selection. A node is its component's winner if its
        #    local min equals the component best. On ties pick smallest id.
        is_winner = min_val == comp_best_val[root]
        candidate = torch.where(
            is_winner,
            node_idx,
            torch.full_like(node_idx, sentinel),
        )
        comp_winner = torch.full(
            (K,), sentinel, dtype=node_idx.dtype, device=device,
        )
        comp_winner.scatter_reduce_(
            0, root, candidate, reduce="amin", include_self=True,
        )

        # A source node contributes iff (a) it is the smallest-id winner
        # of its component, and (b) its component has any outgoing edge
        # at all. The second clause guards the degenerate case where a
        # component is already saturated (shouldn't happen for a complete
        # graph with K > 1, but keeps the contract safe).
        picked = (comp_winner[root] == node_idx) & is_winner
        picked = picked & torch.isfinite(comp_best_val[root])

        sel_src = node_idx[picked]
        if sel_src.numel() == 0:
            break
        sel_dst = min_dst[sel_src]
        sel_src_root = root[sel_src]
        sel_dst_root = root[sel_dst]

        # 6. Mirror-pair resolution: keep only when src root id is
        #    strictly less than dst root id.
        keep = sel_src_root < sel_dst_root
        new_src = sel_src[keep]
        new_dst = sel_dst[keep]
        if new_src.numel() == 0:
            # Can only occur if every picked edge was a mirror where
            # src_root > dst_root, which would itself imply that the
            # mirror partner with src_root < dst_root was also picked.
            # In that case ``keep`` cannot be all-False. This branch is
            # a defensive guard for non-complete inputs.
            break

        # 7. Record edges (symmetric, using ORIGINAL D values).
        w = D[new_src, new_dst]
        mst[new_src, new_dst] = w
        mst[new_dst, new_src] = w
        edges_added += int(new_src.numel())

        # 8. Union: parent[src_root] = dst_root.
        new_src_root = root[new_src]
        new_dst_root = root[new_dst]
        parent[new_src_root] = new_dst_root

        if edges_added >= edges_needed:
            break

        # Progress guard: if a round added zero edges we are stuck. This
        # can only happen on ill-formed inputs (disconnected / non-finite)
        # — bail out cleanly rather than spin K-1 rounds in silence.
        if edges_added == prev_edges:
            break

    if edges_added != edges_needed:
        raise RuntimeError(
            f"boruvka_mst: produced {edges_added}/{edges_needed} edges. "
            "Input is not a valid complete graph with finite off-diagonal "
            "weights."
        )

    return mst
