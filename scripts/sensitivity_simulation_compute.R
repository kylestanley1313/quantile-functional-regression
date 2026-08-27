# Sensitivity simulation -- compute stage ------------------------------------
# Computes Hotelling p-values and generativity scores across K_levels for a
# single (mu_alpha_2, n_2, sim_rep) row of the sweep. All parameters and the
# row identity (LSB_JOBINDEX) are supplied via environment variables by
# batch/submit_sensitivity_simulation.sh -- see that script for how a row
# index maps to a (sim_rep, mu_alpha_2, n_2) triple. Output goes to
# <ART_DIR>/b-<LSB_JOBINDEX>/; scripts/sensitivity_simulation_aggregate.R
# later scans every such directory to combine and plot results.
# NOTE: run with the working directory set to the repo root (as with all
# scripts/*.R that use src/cot.R).
library(Hotelling)
library(yaml)
source("src/cot.R")

# Row parameters: read from env -----------------------------------------------
require_env <- function(name, convert = identity) {
  v <- Sys.getenv(name, unset = NA)
  if (is.na(v)) stop("Missing required env var: ", name)
  convert(v)
}

b          <- require_env("LSB_JOBINDEX", as.integer)
sim_rep    <- require_env("SIM_REP", as.integer)
mu_alpha_2 <- require_env("MU_ALPHA_2", as.numeric)
n_2        <- require_env("N_2", as.integer)

mu_mu_1           <- require_env("MU_MU_1", as.numeric)
sigma_mu_1        <- require_env("SIGMA_MU_1", as.numeric)
mu_log_sigma_1    <- require_env("MU_LOG_SIGMA_1", as.numeric)
sigma_log_sigma_1 <- require_env("SIGMA_LOG_SIGMA_1", as.numeric)
sigma_alpha_2     <- require_env("SIGMA_ALPHA_2", as.numeric)
nobs              <- require_env("NOBS", as.integer)
n_1               <- require_env("N_1", as.integer)
K_max             <- require_env("K_MAX", as.integer)
dir_art           <- require_env("ART_DIR")

K_bounds <- as.integer(strsplit(require_env("K_LEVELS"), ":")[[1]])
K_levels <- K_bounds[1]:K_bounds[2]

# b-specific RNG stream so results are reproducible independent of which
# node/process computes this row or in what order rows are submitted.
sim_seed_base <- 12345
set.seed(sim_seed_base + b)
print(str_glue("===== b = {b} (sim_rep = {sim_rep}, mu_alpha_2 = {mu_alpha_2}, n_2 = {n_2}) ====="))

dir_row <- file.path(dir_art, str_glue("b-{b}"))
dir.create(dir_row, showWarnings = FALSE, recursive = TRUE)
path_flow_b <- file.path(dir_row, "flow_tmp.pth")

# Data-generating helpers ------------------------------------------------------
sample_normal <- function(mu_mu, sigma_mu, mu_log_sigma, sigma_log_sigma, nobs = 500) {
  mu <- rnorm(n = 1, mean = mu_mu, sd = sigma_mu)
  sigma <- exp(rnorm(n = 1, mean = mu_log_sigma, sd = sigma_log_sigma))

  x <- rnorm(n = nobs, mean = mu, sd = sigma)
  q <- sort(x)
  list(x = x, q = q)
}

sample_skewnormal <- function(mu_mu,
                              sigma_mu,
                              mu_log_sigma,
                              sigma_log_sigma,
                              mu_alpha,
                              sigma_alpha,
                              nobs = 500) {
  # random hierarchical parameters
  mu <- rnorm(1, mean = mu_mu, sd = sigma_mu)
  sigma <- exp(rnorm(1, mean = mu_log_sigma, sd = sigma_log_sigma))

  m <- mu_alpha
  v <- sigma_alpha^2

  sigma_log_sq <- log(1 + v / m^2)
  sigma_log <- sqrt(sigma_log_sq)
  mu_log <- log(m) - 0.5 * sigma_log_sq

  alpha <- exp(rnorm(1, mean = mu_log, sd = sigma_log))
  delta <- alpha/ sqrt(1 + alpha^2)

  omega_sq <- sigma^2 / (1 - ((2 * delta^2) / pi))
  omega <- sqrt(omega_sq)

  xi <- mu - omega * delta * sqrt(2/pi)

  x <- sn::rsn(n = nobs, xi = xi, omega = omega, alpha = alpha)

  list(
    mu = mu,
    sigma = sigma,
    alpha = alpha,
    x = x,
    q = sort(x)
  )
}

# Simulate this row's data -----------------------------------------------------
group <- c(rep(1, times = n_1), rep(2, times = n_2))
Q_1 <- sapply(seq_len(n_1),
  FUN = function(x) {
    sample_normal(
      nobs = nobs,
      mu_mu = mu_mu_1,
      sigma_mu = sigma_mu_1,
      mu_log_sigma = mu_log_sigma_1,
      sigma_log_sigma = sigma_log_sigma_1
    )$q
  }
)
Q_2 <- sapply(seq_len(n_2),
  FUN = function(x) {
    sample_skewnormal(
      nobs = nobs,
      mu_mu = mu_mu_1,
      sigma_mu = sigma_mu_1,
      mu_log_sigma = mu_log_sigma_1,
      sigma_log_sigma = sigma_log_sigma_1,
      mu_alpha = mu_alpha_2,
      sigma_alpha = sigma_alpha_2
    )$q
  }
)

# Encoder-decoder input: one raw-Y vector per subject (already sorted, which
# stage_eqf_sgrid's sort() leaves unchanged). asplit() would retain a "dim"
# attribute on each 1-d slice (tripping up pracma::spinterp's is.vector()
# check downstream), so go through a data.frame instead.
y_list <- unname(c(as.list(as.data.frame(Q_1)), as.list(as.data.frame(Q_2))))
p_grid <- p_grid_fun(
  breaks = c(1 / (1 + nobs), nobs / (1 + nobs)),
  interval_counts = c(nobs)
)

# Standard PCA baseline: PCA run directly on the quantile functions, bypassing
# the encoder-decoder entirely. Fit once (K-independent); PC scores are
# sliced to the first K columns per K below.
Q_mat <- t(cbind(Q_1, Q_2))
pca_out <- prcomp(Q_mat, center = TRUE, scale. = FALSE)
scores_pca <- pca_out$x

# Result containers -------------------------------------------------------
k_names <- paste0("K_", K_levels)
p_vals_by_k     <- setNames(rep(NA_real_, length(K_levels)), k_names)
p_vals_by_k_pca <- p_vals_by_k
gen_uncond_by_k <- p_vals_by_k
gen_cond1_by_k  <- p_vals_by_k
gen_cond2_by_k  <- p_vals_by_k

# S/R kept small: evaluate_pipeline_generativity() runs once unconditionally
# and once per group for every K below, so this stays cheap per row.
S_gen <- 2
R_gen <- 5

# n_cores hardcoded to 1: each row is a single LSF core, so generativity
# evaluation is plain sequential -- no future::plan() pool needed.
mean_log_norm_generativity <- function(pipeline, y_list_g, context) {
  tryCatch(
    {
      gen <- evaluate_pipeline_generativity(
        pipeline, y_list_g,
        S = S_gen, R = R_gen, n_cores = 1
      )
      mean(unlist(lapply(gen$splits, `[[`, "log_norm")))
    },
    error = function(e) {
      message(str_glue("Generativity evaluation failed ({context}): {conditionMessage(e)}"))
      NA_real_
    }
  )
}

# K-loop: fit the pipeline once per K and reuse it for the Hotelling p-value
# plus the three generativity scores. A K cell stays NA in all matrices if
# the pipeline/encode step itself fails.
for (k_idx in seq_along(K_levels)) {
  K <- K_levels[k_idx]

  # PCA baseline p-value: Hotelling's T^2 on the first K PCA scores.
  # Independent of the encoder-decoder fit below, so computed even if that
  # fit fails for this K.
  K_pca <- min(K, ncol(scores_pca))
  p_vals_by_k_pca[k_idx] <- tryCatch(
    Hotelling::hotelling.test(
      scores_pca[group == 1, seq_len(K_pca), drop = FALSE],
      scores_pca[group == 2, seq_len(K_pca), drop = FALSE],
      var.equal = FALSE
    )$pval,
    error = function(e) {
      if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
        return(NA_real_)
      }
      stop(e)
    }
  )

  fit_out <- tryCatch(
    {
      pipeline <- construct_pipeline(
        stages = list(
          stage_eqf_sgrid(),
          stage_eqf_cgrid(p_grid = p_grid),
          stage_wame(K = K, K_max = K_max, order_by = 'importance'),
          stage_flow(path = path_flow_b, n_layers = 8, hidden = 64)
        ),
        seed = gen_seed()
      )
      pipeline <- fit(pipeline, y_list)
      y_ctx <- new_context(
        payload = y_list,
        cache = pipeline$training$cache,
        meta = list()
      )
      z_ctx <- encode(pipeline, y_ctx, from = 0, to = pipeline$n_stages)
      list(pipeline = pipeline, Z_mat = do.call(rbind, z_ctx$payload))
    },
    error = function(e) {
      if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
        return(NULL)
      }
      stop(e)
    }
  )
  if (is.null(fit_out)) next

  pipeline <- fit_out$pipeline
  Z_mat <- fit_out$Z_mat

  p_vals_by_k[k_idx] <- tryCatch(
    Hotelling::hotelling.test(
      Z_mat[group == 1, , drop = FALSE],
      Z_mat[group == 2, , drop = FALSE],
      var.equal = FALSE
    )$pval,
    error = function(e) {
      if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
        return(NA_real_)
      }
      stop(e)
    }
  )

  gen_uncond_by_k[k_idx] <- mean_log_norm_generativity(
    pipeline, y_list,
    context = str_glue("b={b}, K={K}, unconditional")
  )
  gen_cond1_by_k[k_idx] <- mean_log_norm_generativity(
    pipeline, y_list[group == 1],
    context = str_glue("b={b}, K={K}, group=1")
  )
  gen_cond2_by_k[k_idx] <- mean_log_norm_generativity(
    pipeline, y_list[group == 2],
    context = str_glue("b={b}, K={K}, group=2")
  )
}

# This row's flow model is no longer needed once its K-loop finishes.
unlink(path_flow_b)

# Save row output ---------------------------------------------------------
saveRDS(
  list(
    K_levels        = K_levels,
    p_vals_by_k     = p_vals_by_k,
    p_vals_by_k_pca = p_vals_by_k_pca,
    gen_uncond_by_k = gen_uncond_by_k,
    gen_cond1_by_k  = gen_cond1_by_k,
    gen_cond2_by_k  = gen_cond2_by_k
  ),
  file.path(dir_row, "results.rds")
)
write_yaml(
  list(b = b, sim_rep = sim_rep, mu_alpha_2 = mu_alpha_2, n_2 = n_2),
  file.path(dir_row, "settings.yml")
)
