"""Dump Python-side initialization state before iteration starts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from ddrtree._utils import pca_projection, sq_dist
from sklearn.cluster import KMeans

np.set_printoptions(precision=8, suppress=True, linewidth=160)

X = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "data", "iris_k3_X.tsv"),
               delimiter="\t", dtype=np.float64)
print("X\n", X)
dimensions = 2; ncenter = 3

W = pca_projection(X @ X.T, dimensions)
print("W init\n", W)
Z = W.T @ X
print("Z init\n", Z)

K = ncenter
# Mirror R: seq(1, ncol(Z), length.out=K), truncated toward 0 (R as.integer)
real_idx = np.linspace(1, Z.shape[1], num=K)
print("real idx (1-based)\n", real_idx)
int_idx = real_idx.astype(int) - 1   # truncate then 0-base
print("int idx (0-based)\n", int_idx)
init_centers = Z.T[int_idx, :]
print("init centers\n", init_centers)

km = KMeans(n_clusters=K, init=init_centers, n_init=1, max_iter=10).fit(Z.T)
print("kmeans labels\n", km.labels_)
print("kmeans centers\n", km.cluster_centers_)
Y = km.cluster_centers_.T
print("Y init\n", Y)

print("sq_dist(Y,Y)\n", sq_dist(Y, Y))
print("sq_dist(Z,Y)\n", sq_dist(Z, Y))
