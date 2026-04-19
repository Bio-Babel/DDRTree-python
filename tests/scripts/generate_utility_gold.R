# Dump R-side outputs of DDRTree's three exported helpers across a small
# zoo of inputs, so the Python port can be verified call-for-call.
suppressPackageStartupMessages(library(DDRTree))

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Deterministic seed shared with the Python loader.
set.seed(2024)

dump_mat <- function(name, m) {
    write.table(m, file.path(out_dir, paste0(name, ".tsv")),
                sep = "\t", row.names = FALSE, col.names = FALSE)
}

dump_scalar <- function(name, x) {
    writeLines(format(x, digits = 17), file.path(out_dir, paste0(name, ".txt")))
}

# -------------------------------------------------------------------------
# sq_dist matrix zoo
# -------------------------------------------------------------------------
# Square: a = b, D=3, N=5
a1 <- matrix(rnorm(15), 3, 5); b1 <- a1
dump_mat("sqdist_a1", a1); dump_mat("sqdist_b1", b1)
dump_mat("sqdist_out1", sqdist_R(a1, b1))

# Rectangular: D=4, Na=6, Nb=3
a2 <- matrix(rnorm(24), 4, 6); b2 <- matrix(rnorm(12), 4, 3)
dump_mat("sqdist_a2", a2); dump_mat("sqdist_b2", b2)
dump_mat("sqdist_out2", sqdist_R(a2, b2))

# Single column on each side (degenerate corner case): D=5, Na=1, Nb=1
a3 <- matrix(rnorm(5), 5, 1); b3 <- matrix(rnorm(5), 5, 1)
dump_mat("sqdist_a3", a3); dump_mat("sqdist_b3", b3)
dump_mat("sqdist_out3", sqdist_R(a3, b3))

# Wide, low-D: D=2, Na=10, Nb=4
a4 <- matrix(rnorm(20), 2, 10); b4 <- matrix(rnorm(8), 2, 4)
dump_mat("sqdist_a4", a4); dump_mat("sqdist_b4", b4)
dump_mat("sqdist_out4", sqdist_R(a4, b4))

# -------------------------------------------------------------------------
# pca_projection_R zoo
# -------------------------------------------------------------------------
# Symmetric PSD, full path (L >= min(dim)): 4x4 with L=4
M1 <- matrix(rnorm(16), 4, 4); C1 <- M1 %*% t(M1)
dump_mat("pca_C1", C1)
dump_mat("pca_W1_L4", pca_projection_R(C1, 4))

# Symmetric PSD, irlba path (L < min(dim)): 6x6 with L=2
M2 <- matrix(rnorm(36), 6, 6); C2 <- M2 %*% t(M2)
dump_mat("pca_C2", C2)
dump_mat("pca_W2_L2", pca_projection_R(C2, 2))

# Symmetric PSD, irlba path with L=3 (still less than 6)
dump_mat("pca_W2_L3", pca_projection_R(C2, 3))

# Larger symmetric PSD 10x10 with L=4 — exercises a stable irlba run.
M3 <- matrix(rnorm(100), 10, 10); C3 <- M3 %*% t(M3)
dump_mat("pca_C3", C3)
dump_mat("pca_W3_L4", pca_projection_R(C3, 4))

# -------------------------------------------------------------------------
# get_major_eigenvalue zoo
# -------------------------------------------------------------------------
# Full branch: L >= min(dim). 4x4 matrix, L=4 → returns norm(C, '2')^2.
G1 <- matrix(rnorm(16), 4, 4)
dump_mat("gme_C1", G1)
dump_scalar("gme_v1_L4", get_major_eigenvalue(G1, 4))

# irlba branch: L < min(dim). 6x8 matrix, L=2 → returns max(abs(v)).
G2 <- matrix(rnorm(48), 6, 8)
dump_mat("gme_C2", G2)
dump_scalar("gme_v2_L2", get_major_eigenvalue(G2, 2))

# irlba branch on 10x12 with L=3.
G3 <- matrix(rnorm(120), 10, 12)
dump_mat("gme_C3", G3)
dump_scalar("gme_v3_L3", get_major_eigenvalue(G3, 3))

cat("done\n")
