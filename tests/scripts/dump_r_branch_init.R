suppressPackageStartupMessages(library(DDRTree))
X <- as.matrix(read.table(file.path(commandArgs(trailingOnly=TRUE)[1], "branch_k20_X.tsv"), sep="\t"))
dimensions <- 2; ncenter <- 20
W <- pca_projection_R(X %*% t(X), dimensions)
Z <- t(W) %*% X
K <- ncenter
centers <- t(Z)[seq(1, ncol(Z), length.out=K), ]
km <- kmeans(t(Z), K, centers=centers)
cat("=== W init first row ===\n"); print(W[1, ])
cat("=== Z init first col ===\n"); print(Z[, 1])
cat("=== kmeans centers (transposed = Y) ===\n"); print(t(km$centers))
cat("=== kmeans iter ===\n"); print(km$iter)
cat("=== kmeans totss-betweenss ===\n"); print(c(km$totss, km$betweenss, km$tot.withinss))
