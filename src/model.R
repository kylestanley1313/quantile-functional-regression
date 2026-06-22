library(MASS)


## ---------- Mean Covariance Model ---------- ##

fit_mean_cov <- function(Z, ridge = 0) {
  K <- ncol(Z)
  list(
    mu    = colMeans(Z),
    Sigma = cov(Z) + ridge * diag(K)
  )
}

## Draw n iid samples from N(fit$mu, fit$Sigma), returned as a length-n list of
## K-vectors (matches the input shape decode_z_to_Qi expects).
draw_mean_cov <- function(n, fit) {
  K <- length(fit$mu)
  draws <- MASS::mvrnorm(n, mu = fit$mu, Sigma = fit$Sigma)
  if (n == 1) draws <- matrix(draws, nrow = 1)
  asplit(draws, MARGIN = 1) |> lapply(as.numeric)
}

