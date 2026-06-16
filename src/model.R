library(coda)
library(rjags)


## ---------- Mean Model ---------- ##

model_string_mean <- "
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

fit_model_mean <- function(
    Z, 
    n_chains = 1, 
    burn_in = 2000, 
    thin_interval = 5, 
    n_samps = 5000,
    R = NULL,
    nu = NULL,
    seed = 12345,
    quiet = FALSE
) {
  N <- nrow(Z)
  K <- ncol(Z)  # really K_plus = K + 1 since we append vertical constant
  n_iter <- thin_interval * n_samps
  
  ## Wishart hyperparams
  R <- R %||% diag(K)  # scale matrix
  nu <- nu %||% K + 1  # minimally informative
  
  ## Prepare data for JAGS
  data_jags <- list(
    Z = Z,
    N = N,
    K = K,
    R = R,
    nu = nu
  )
  
  ## Initial values
  inits_fun <- function(chain) {
    list(
      mu = colMeans(Z),
      Omega = diag(K),
      .RNG.name = "base::Wichmann-Hill",
      .RNG.seed = seed + chain
    )
  }
  
  ## Run JAGS
  jags_mod <- jags.model(
    textConnection(model_string_mean),
    data = data_jags,
    inits = inits_fun,
    n.chains = n_chains,
    quiet = quiet
  )
  progress_bar <- if (quiet) "none" else "text"
  update(jags_mod, n.iter = burn_in, progress.bar = progress_bar)
  samps <- coda.samples(
    jags_mod,
    variable.names = c("mu", "Sigma"),
    n.iter = n_iter,
    thin = thin_interval,
    progress.bar = progress_bar
  )
  
  do.call(rbind, samps)
}


## ---------- Mean Model with Binary Predictor ---------- ##

model_string_bin_pred <- "
model {

  ## Likelihood
  for (i in 1:N) {
    Z[i,1:K] ~ dmnorm(mu_i[i,1:K], Omega[1:K,1:K])
    for (k in 1:K) {
      mu_i[i,k] <- mu[k] + beta[k] * x[i]
    }
  }
  ## Intercept (baseline mean)
  for (k in 1:K) {
    mu[k] ~ dnorm(0.0, tau_mu)
  }
  tau_mu ~ dgamma(0.001, 0.001)
  
  ## Regression coefficients for binary predictor
  for (k in 1:K) {
    beta[k] ~ dnorm(0.0, tau_beta)
  }
  tau_beta ~ dgamma(0.001, 0.001)
  
  ## Covariance/Precision
  Omega[1:K,1:K] ~ dwish(R[1:K,1:K], nu)
  Sigma[1:K,1:K] <- inverse(Omega[1:K,1:K])
  
}
"

fit_model_bin_pred <- function(
    Z, 
    x,
    n_chains = 1, 
    burn_in = 2000, 
    thin_interval = 5, 
    n_samps = 5000,
    R = NULL,
    nu = NULL,
    seed = 12345,
    quiet = FALSE
) {

  ## Globals
  N <- nrow(Z)
  K <- ncol(Z)
  n_iter <- thin_interval * n_samps
  
  ## Check predictor
  if(length(x) != N)
    stop("x must have length N")
  if(!all(x %in% c(0, 1)))
    stop("x must be binary (0 or 1)")
  
  ## Wishart hyperparams
  R <- R %||% diag(K)  # scale matrix
  nu <- nu %||% K + 1  # minimally informative
  
  ## Prepare data for JAGS
  data_jags <- list(
    Z = Z,
    x = x,
    N = N,
    K = K,
    R = R,
    nu = nu
  )
  
  ## Initial values
  inits_fun <- function(chain) {
    list(
      mu = colMeans(Z),
      beta = rep(0, K),
      Omega = diag(K),
      .RNG.name = "base::Wichmann-Hill",
      .RNG.seed = seed + chain
    )
  }
  
  ## Run JAGS
  jags_mod <- jags.model(
    textConnection(model_string_bin_pred),
    data = data_jags,
    inits = inits_fun,
    n.chains = n_chains,
    quiet = quiet
  )
  progress_bar <- if (quiet) "none" else "text"
  update(jags_mod, n.iter = burn_in)
  samps <- coda.samples(
    jags_mod,
    variable.names = c("mu", "beta", "Sigma"),
    n.iter = n_iter,
    thin = thin_interval,
    progress.bar = progress_bar
  )
  
  do.call(rbind, samps)
}
