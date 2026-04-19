suppressPackageStartupMessages({
    library(DDRTree)
})

set.seed(42)

run_case <- function(name, X, dimensions, maxIter, sigma, lambda, ncenter, gamma, tol, out_dir) {
    res <- DDRTree(X,
                   dimensions = dimensions,
                   maxIter   = maxIter,
                   sigma     = sigma,
                   lambda    = lambda,
                   ncenter   = ncenter,
                   param.gamma = gamma,
                   tol       = tol,
                   verbose   = FALSE)
    write.table(res$W,    file=file.path(out_dir, paste0(name, "_W.tsv")),    sep="\t", row.names=FALSE, col.names=FALSE)
    write.table(res$Z,    file=file.path(out_dir, paste0(name, "_Z.tsv")),    sep="\t", row.names=FALSE, col.names=FALSE)
    write.table(res$Y,    file=file.path(out_dir, paste0(name, "_Y.tsv")),    sep="\t", row.names=FALSE, col.names=FALSE)
    write.table(as.matrix(res$stree), file=file.path(out_dir, paste0(name, "_stree.tsv")),
                sep="\t", row.names=FALSE, col.names=FALSE)
    writeLines(as.character(res$objective_vals),
               con=file.path(out_dir, paste0(name, "_obj.txt")))
    write.table(X, file=file.path(out_dir, paste0(name, "_X.tsv")),
                sep="\t", row.names=FALSE, col.names=FALSE)
    cat(sprintf("[%s] D=%d N=%d K=%s iters=%d obj_last=%.6f\n",
                name, nrow(X), ncol(X),
                ifelse(is.null(ncenter), "NULL", as.character(ncenter)),
                length(res$objective_vals),
                tail(res$objective_vals, 1)))
}

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Case 1: iris subset, ncenter=3 (matches DDRTree.Rd example)
data("iris")
X1 <- as.matrix(t(iris[c(1, 2, 52, 103), 1:4]))
run_case("iris_k3",
         X = X1, dimensions = 2, maxIter = 5,
         sigma = 1e-2, lambda = 1, ncenter = 3, gamma = 10, tol = 1e-2,
         out_dir = out_dir)

# Case 2: iris subset, ncenter=NULL (K=N)
run_case("iris_full",
         X = X1, dimensions = 2, maxIter = 5,
         sigma = 1e-3, lambda = 1, ncenter = NULL, gamma = 10, tol = 1e-2,
         out_dir = out_dir)

# Case 3: synthetic two-branch tree, ncenter < N
set.seed(7)
n_per <- 30
t_axis <- seq(0, 1, length.out = n_per)
branch_a <- rbind(t_axis,                       1.0 * t_axis^2)
branch_b <- rbind(t_axis,                      -1.0 * t_axis^2)
trunk    <- rbind(seq(-1, 0, length.out = n_per), rep(0, n_per))
pts <- cbind(trunk, branch_a, branch_b)
# Embed into 8-D via random orthogonal lift, add small noise
set.seed(11)
M <- qr.Q(qr(matrix(rnorm(8 * 2), 8, 2)))
X3 <- M %*% pts + matrix(rnorm(8 * ncol(pts), sd = 0.02), 8, ncol(pts))
run_case("branch_k20",
         X = X3, dimensions = 2, maxIter = 20,
         sigma = 5e-3, lambda = NULL, ncenter = 20, gamma = 10, tol = 1e-3,
         out_dir = out_dir)

cat("done\n")
