library(coda)
library(rjags)


## ---------- Mean Model

mean_model_string <- "
model {

  ## Likelihood
  for (i in 1:N) {
    Z[i,1:K] ~ dmnorm(mu[1:K], Omega[1:K,1:K])
  }

  ## Mean vector
  for (k in 1:K) {
    mu[k] ~ dnorm(0.0, tau_mu)
  }
  tau_mu ~ dgamma(0.001, 0.001)

  ## Covariance/Precision
  Omega[1:K,1:K] ~ dwish(R[1:K,1:K], nu)
  Sigma[1:K,1:K] <- inverse(Omega[1:K,1:K])

}
"

fit_mean_model <- function(
    Z, 
    n_chains = 1, 
    burn_in = 2000, 
    thin_interval = 5, 
    n_samps = 5000
) {
  
  ## Globals
  N <- nrow(Z)
  K <- ncol(Z)  # really K_plus = K + 1 since we append vertical constant
  n_iter <- thin_interval * n_samps
  
  ## Wishart hyperparams
  R <- diag(K)        # scale matrix
  nu <- K + 1         # minimally informative
  
  ## Prepare data for JAGS
  data_jags <- list(
    Z = Z,
    N = N,
    K = K,
    R = R,
    nu = nu
  )
  
  ## Initial values
  inits_fun <- function() {
    list(
      mu = colMeans(Z),
      Omega = diag(K)
    )
  }
  
  ## Run JAGS
  jags_mod <- jags.model(
    textConnection(mean_model_string),
    data = data_jags,
    inits = inits_fun,
    n.chains = n_chains,
    quiet = FALSE
  )
  update(jags_mod, n.iter = burn_in)
  samps <- coda.samples(
    jags_mod,
    variable.names = c("mu", "Sigma"),
    n.iter = n_iter,
    thin = thin_interval
  )
  
  do.call(rbind, samps)
}
