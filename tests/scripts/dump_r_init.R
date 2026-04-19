suppressPackageStartupMessages(library(DDRTree))

data("iris")
X <- as.matrix(t(iris[c(1, 2, 52, 103), 1:4]))
D <- nrow(X); N <- ncol(X)
dimensions <- 2
ncenter <- 3

cat("=== X ===\n"); print(X)

# Mirror DDRTree.R initialization
W <- pca_projection_R(X %*% t(X), dimensions)
cat("=== W (init, from pca_projection_R(X X^T, 2)) ===\n"); print(W)

Z <- t(W) %*% X
cat("=== Z (init = W^T X) ===\n"); print(Z)

K <- ncenter
idx_real <- seq(1, ncol(Z), length.out = K)
cat("=== seq(1, ncol(Z), length.out=K) ===\n"); print(idx_real)
idx_int <- as.integer(idx_real)
cat("=== as.integer of that (R subset indices) ===\n"); print(idx_int)
centers <- t(Z)[idx_real, ]
cat("=== initial centers (t(Z)[idx_real, ]) ===\n"); print(centers)

km <- kmeans(t(Z), K, centers = centers)
cat("=== kmeans cluster ===\n"); print(km$cluster)
cat("=== kmeans centers ===\n"); print(km$centers)

Y <- t(km$centers)
cat("=== Y (init) ===\n"); print(Y)

# Also dump distZY and the first objective
library(DDRTree)
dist_mu <- sqdist_R(Y, Y); cat("=== sq_dist(Y, Y) ===\n"); print(dist_mu)
dist_zy <- sqdist_R(Z, Y); cat("=== sq_dist(Z, Y) ===\n"); print(dist_zy)
