library(MASS)

source('src/model/utils.R')
source('src/model/binary.R')
source('src/model/amm.R')


## ---------- Mean Covariance Model ---------- ##

fit_mean_cov <- function(Z, ridge = 0) {
  K <- ncol(Z)
  list(
    mu    = colMeans(Z),
    Sigma = cov(Z) + ridge * diag(K)
  )
}

## Since fit_mean_cov is unconditional, `X` (the sample_fun contract's 
## optional covariate argument) is accepted and ignored.
sample_mean_cov <- function(n, fit, X = NULL) {
  K <- length(fit$mu)
  draws <- MASS::mvrnorm(n, mu = fit$mu, Sigma = fit$Sigma)
  if (n == 1) draws <- matrix(draws, nrow = 1)
  asplit(draws, MARGIN = 1) |> lapply(as.numeric)
}