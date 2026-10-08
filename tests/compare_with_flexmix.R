#!/usr/bin/env Rscript
# Fit a finite mixture of linear regressions with flexmix, as an external check
# on the MoP EM implementation in src/mop/.
#
# The model is   y_k = x_k^T u_{c_k} + eps_k,  eps_k ~ N(0, phi_{c_k}),
# which is the MoP Gaussian variant with g the identity. flexmix fits it by EM
# under maximum likelihood. Its sigma is the standard deviation, so phi is
# sigma squared.
#
# Usage:
#   Rscript tests/compare_with_flexmix.R <data.csv> <out.json> <n_components> <seed>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: compare_with_flexmix.R <data.csv> <out.json> <n_components> <seed>")
}
data_csv <- args[[1]]
out_json <- args[[2]]
k <- as.integer(args[[3]])
seed <- as.integer(args[[4]])

suppressPackageStartupMessages(library(flexmix))

d <- read.csv(data_csv)
xcols <- grep("^x", names(d), value = TRUE)
d$xmat <- I(as.matrix(d[, xcols]))
y <- d$y

set.seed(seed)
m <- flexmix(y ~ 0 + xmat, data = d, k = k,
             model = FLXMRglm(family = "gaussian"),
             control = list(tol = 1e-10, iter.max = 5000, verbose = 0))

coefs <- lapply(seq_len(k), function(p)
  as.numeric(m@components[[p]][[1]]@parameters$coef))
sigma <- sapply(seq_len(k), function(p)
  as.numeric(m@components[[p]][[1]]@parameters$sigma))

json <- paste0(
  '{"k":', k,
  ',"prior":[', paste(sprintf("%.12g", m@prior), collapse = ","), ']',
  ',"u":[', paste(sapply(coefs, function(v)
       paste0("[", paste(sprintf("%.12g", v), collapse = ","), "]")), collapse = ","), ']',
  ',"sigma":[', paste(sprintf("%.12g", sigma), collapse = ","), ']',
  ',"loglik":', sprintf("%.12g", as.numeric(logLik(m))),
  ',"converged":', as.integer(m@converged),
  ',"niter":', as.integer(m@iter),
  '}')
writeLines(json, out_json)
cat(json, "\n")
