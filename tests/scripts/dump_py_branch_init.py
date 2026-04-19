import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from ddrtree._utils import pca_projection
from sklearn.cluster import KMeans

X = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "data", "branch_k20_X.tsv"), delimiter="\t")
dimensions = 2; ncenter = 20
W = pca_projection(X @ X.T, dimensions)
Z = W.T @ X
K = ncenter
idx = np.linspace(1, Z.shape[1], K).astype(int) - 1
init_centers = Z.T[idx, :]
km = KMeans(n_clusters=K, init=init_centers, n_init=1, max_iter=10).fit(Z.T)
np.set_printoptions(precision=6, suppress=True, linewidth=160)
print("W row 0 :", W[0])
print("Z col 0 :", Z[:, 0])
print("Y (km.centers.T) :")
print(km.cluster_centers_.T)
print("iters:", km.n_iter_)
print("inertia:", km.inertia_)
