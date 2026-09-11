# Power simulation (Setting 4), inferential arm -- compute stage ------------
# For a single (regime, effect, psi, sim_rep) row of the sweep: simulate
# group-structured distributional data, fit a fresh SGM encoder, destroy a
# proportion psi of the group structure by relabeling, then record (a) the
# two-sample Hotelling p-value on the relabeled embeddings and (b) the
# frozen-reference conditional-generativity p-value for test group V_TEST under
# the omnibus (Wasserstein) statistic and the directed statistic matching
# REGIME (skewness or mean) -- the other directed statistic is not computed
# and is left NA. As psi grows Hotelling power falls while generativity
# rejection rises (see the README in this directory).
#
# All parameters and the row identity (LSB_JOBINDEX) come from environment
# variables set by batch/submit_power_simulation_inferential.sh -- see that
# script for how a row index maps to a (sim_rep, effect, psi) tuple (REGIME is
# fixed for the whole run, not swept). Output goes to
# <ART_DIR>/replicates/b-<LSB_JOBINDEX>/;
# scripts/power_simulations/power_simulation_inferential_aggregate.R later
# scans every such directory to combine and plot results.
# NOTE: run with the working directory set to the repo root (as with all
# scripts/*.R that use src/cot.R).
library(Hotelling)
library(yaml)
source("src/cot.R")
source("scripts/power_simulations/utils.R")

# Row parameters: read from env ----------------------------------------------
require_env <- function(name, convert = identity) {
  v <- Sys.getenv(name, unset = NA)
  if (is.na(v)) stop("Missing required env var: ", name)
  convert(v)
}

b       <- require_env("LSB_JOBINDEX", as.integer)
sim_rep <- require_env("SIM_REP", as.integer)
regime  <- require_env("REGIME")
effect  <- require_env("EFFECT", as.numeric)
psi     <- require_env("PSI", as.numeric)

nobs            <- require_env("NOBS", as.integer)
n_0             <- require_env("N_0", as.integer)
n_1             <- require_env("N_1", as.integer)
n_anchor        <- require_env("N_ANCHOR", as.integer)
K               <- require_env("K", as.integer)
K_max           <- require_env("K_MAX", as.integer)
m_mult          <- require_env("M_MULT", as.numeric)
b_null          <- require_env("B_NULL", as.integer)
alpha           <- require_env("ALPHA", as.numeric)
v_test          <- require_env("V_TEST", as.integer)
ridge           <- require_env("RIDGE", as.numeric)
mu_mu           <- require_env("MU_MU", as.numeric)
sigma_mu        <- require_env("SIGMA_MU", as.numeric)
mu_log_sigma    <- require_env("MU_LOG_SIGMA", as.numeric)
sigma_log_sigma <- require_env("SIGMA_LOG_SIGMA", as.numeric)
mu_alpha        <- require_env("MU_ALPHA", as.numeric)
sigma_alpha     <- require_env("SIGMA_ALPHA", as.numeric)
dir_art         <- require_env("ART_DIR")

stopifnot(regime %in% c("mean", "skewness"), v_test %in% c(0L, 1L))

# Fixed internals (not swept -- kept out of the env surface, as with S_gen/R_gen
# in scripts/sensitivity_simulation_compute.R). j_aug matches the default in
# evaluate_pipeline_generativity(); with nobs >= j_aug the augmented grid is
# p_grid plus its {0, 1} endpoints, same as that function produces.
sim_seed_base <- 12345
j_aug         <- 500L

# b-specific RNG stream so results are reproducible independent of which
# node/process computes this row or in what order rows are submitted.
set.seed(sim_seed_base + b)
print(str_glue(
  "===== b = {b} (sim_rep = {sim_rep}, regime = {regime}, effect = {effect}, psi = {psi}) ====="
))

dir_row <- file.path(dir_art, "replicates", str_glue("b-{b}"))
dir.create(dir_row, showWarnings = FALSE, recursive = TRUE)
path_flow_b <- file.path(dir_row, "flow_tmp.pth")

save_row <- function(p_hotelling, p_gen_wass, p_gen_skew, p_gen_mean) {
  saveRDS(
    list(
      regime      = regime,
      psi         = psi,
      effect      = effect,
      K           = K,
      p_hotelling = p_hotelling,
      p_gen_wass  = p_gen_wass,
      p_gen_skew  = p_gen_skew,
      p_gen_mean  = p_gen_mean
    ),
    file.path(dir_row, "results.rds")
  )
  write_yaml(
    list(b = b, sim_rep = sim_rep, regime = regime, psi = psi, effect = effect),
    file.path(dir_row, "settings.yml")
  )
}

# Data-generating helpers ---------------------------------------------------
# Borrowed from scripts/sensitivity_simulation_compute.R. Group 1 is either a
# location shift (mean regime) or a moment-matched skew-normal that differs from
# group 0 only in skewness (skewness regime).
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
  delta <- alpha / sqrt(1 + alpha^2)

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

sample_group_0 <- function() {
  sample_normal(
    nobs = nobs,
    mu_mu = mu_mu,
    sigma_mu = sigma_mu,
    mu_log_sigma = mu_log_sigma,
    sigma_log_sigma = sigma_log_sigma
  )$q
}

sample_group_1 <- function() {
  if (regime == "mean") {
    sample_normal(
      nobs = nobs,
      mu_mu = mu_mu + effect,
      sigma_mu = sigma_mu,
      mu_log_sigma = mu_log_sigma,
      sigma_log_sigma = sigma_log_sigma
    )$q
  } else {
    sample_skewnormal(
      nobs = nobs,
      mu_mu = mu_mu,
      sigma_mu = sigma_mu,
      mu_log_sigma = mu_log_sigma,
      sigma_log_sigma = sigma_log_sigma,
      mu_alpha = mu_alpha + effect,
      sigma_alpha = sigma_alpha
    )$q
  }
}

sample_group_v <- if (v_test == 0L) sample_group_0 else sample_group_1

# Simulate this row's data -------------------------------------------------
x <- c(rep(0L, times = n_0), rep(1L, times = n_1))
Q_0 <- sapply(seq_len(n_0), function(i) sample_group_0())
Q_1 <- sapply(seq_len(n_1), function(i) sample_group_1())

# See scripts/sensitivity_simulation_compute.R for why this goes through
# as.data.frame() rather than asplit().
y_list <- unname(c(as.list(as.data.frame(Q_0)), as.list(as.data.frame(Q_1))))
p_grid <- p_grid_fun(
  breaks = c(1 / (1 + nobs), nobs / (1 + nobs)),
  interval_counts = c(nobs)
)

# Fit a fresh encoder-decoder for this row -------------------------------
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

if (is.null(fit_out)) {
  message(str_glue("Pipeline fit failed for b={b}; recording NA row."))
  unlink(path_flow_b)
  save_row(NA_real_, NA_real_, NA_real_, NA_real_)
  quit(save = "no", status = 0)
}

pipeline <- fit_out$pipeline
Z_mat    <- fit_out$Z_mat

p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(j_aug - length(p_grid)))))
w_aug      <- get_quadrature_weights(p_grid_aug)

# Observed anchor for the generativity test: a fresh draw from the true
# data-generating model at x = V_TEST (true labels), never used to fit the SGM,
# encoded to EQFs on the augmented grid.
Q_anchor    <- sapply(seq_len(n_anchor), function(i) sample_group_v())
y_anchor    <- unname(as.list(as.data.frame(Q_anchor)))
obs_qf_list <- encode_y_to_Qi_aug(pipeline, y_anchor, p_grid_aug)

# Destroy a proportion psi of the group structure by reassigning fresh random
# labels (not flipping -- flipping at psi = 1 is inversion, i.e. intact
# structure with reversed sign).
n_re  <- round(psi * length(x))
label <- x
if (n_re > 0) {
  idx_re <- sample.int(length(x), n_re)
  label[idx_re] <- sample(c(0L, 1L), n_re, replace = TRUE)
}

# Hotelling curve point: two-sample T^2 on the relabeled embeddings.
p_hotelling <- tryCatch(
  Hotelling::hotelling.test(
    Z_mat[label == 0, , drop = FALSE],
    Z_mat[label == 1, , drop = FALSE],
    var.equal = TRUE
  )$pval,
  error = function(e) {
    if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
      return(NA_real_)
    }
    stop(e)
  }
)

# Conditional-generativity curve points: fit the latent model to the RELABELED
# embeddings for group V_TEST, then run the frozen-reference test against the
# true-label anchor for each statistic.
fit_v <- fit_latent_conditional(
  Z_mat, x = label, v = v_test, ridge = ridge, shared_cov = TRUE
)
draw_ref_qf <- make_decoder_sampler(pipeline, fit_v, p_grid_aug, p_grid)

run_gen <- function(statistic) {
  tryCatch(
    generativity_frozen_ref_test(
      obs_qf_list, draw_ref_qf, w_aug,
      statistic = statistic,
      M       = round(m_mult * n_anchor),
      B       = b_null,
      alpha   = alpha,
      n_cores = 1,
      seed    = gen_seed()
    )$p_value,
    error = function(e) {
      message(str_glue(
        "Generativity test failed (b={b}, stat={statistic}): {conditionMessage(e)}"
      ))
      NA_real_
    }
  )
}

# Only the directed statistic matching REGIME is computed; the other is left
# NA (see power_simulation_inferential_aggregate.R, which picks the matching
# one out).
p_gen_wass <- run_gen("wasserstein")
p_gen_skew <- if (regime == "skewness") run_gen("skewness") else NA_real_
p_gen_mean <- if (regime == "mean") run_gen("mean") else NA_real_

# Save row output -------------------------------------------------------
unlink(path_flow_b)
save_row(p_hotelling, p_gen_wass, p_gen_skew, p_gen_mean)
