"""Closer diff for branch_k20: find which entries / iterations diverge."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from ddrtree import DDRTree

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

X = np.loadtxt(os.path.join(DATA, "branch_k20_X.tsv"), delimiter="\t")
Z_r = np.loadtxt(os.path.join(DATA, "branch_k20_Z.tsv"), delimiter="\t")
Y_r = np.loadtxt(os.path.join(DATA, "branch_k20_Y.tsv"), delimiter="\t")
W_r = np.loadtxt(os.path.join(DATA, "branch_k20_W.tsv"), delimiter="\t")
stree_r = np.loadtxt(os.path.join(DATA, "branch_k20_stree.tsv"), delimiter="\t")
obj_r = np.loadtxt(os.path.join(DATA, "branch_k20_obj.txt"))

res = DDRTree(X, dimensions=2, max_iter=20, sigma=5e-3, lambda_=None,
              ncenter=20, gamma=10.0, tol=1e-3)
np.set_printoptions(precision=5, suppress=True, linewidth=160)

print("Iterations  py vs r :", len(res.objective_vals), len(obj_r))
print("obj_py :", np.array(res.objective_vals))
print("obj_r  :", obj_r)

# MST edge set
stree_py = res.stree.toarray()
ep = set(zip(*np.nonzero(stree_py)))
er = set(zip(*np.nonzero(stree_r)))
print("\nMST edges equal:", ep == er)
print("  py \\ r:", ep - er)
print("  r \\ py:", er - ep)

# Row-wise sign align for Y
def align_rows(P, ref):
    s = np.sign((P * ref).sum(axis=1)); s[s == 0] = 1
    return P * s.reshape(-1, 1)

Yp_al = align_rows(res.Y, Y_r)
Zp_al = align_rows(res.Z, Z_r)
print("\nY max abs diff:", np.max(np.abs(Yp_al - Y_r)))
print("Z max abs diff:", np.max(np.abs(Zp_al - Z_r)))
print("W max abs diff (per-col sign):", np.max(np.abs(
    res.W * np.sign((res.W * W_r).sum(axis=0)) - W_r)))
