"""Print internal state of the Python DDRTree run on iris_k3 for diffing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from ddrtree import DDRTree

X = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "data", "iris_k3_X.tsv"),
               delimiter="\t", dtype=np.float64)
np.set_printoptions(precision=8, suppress=True, linewidth=160)
print("X shape", X.shape)
res = DDRTree(X, dimensions=2, max_iter=5, sigma=1e-2, lambda_=1.0,
              ncenter=3, gamma=10.0, tol=1e-2, verbose=True)
print("obj", res.objective_vals)
print("W", res.W)
print("Z", res.Z)
print("Y", res.Y)
print("stree", res.stree.toarray())
