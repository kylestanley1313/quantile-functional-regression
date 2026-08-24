# Load packages and functions: --------------------------------------------
# NOTE: run with the working directory set to the repo root (as with all
# scripts/*.R that use src/cot.R) since cot.R sources src/utils.R and
# src/config.yml via paths relative to the repo root, not this script.
library(ggplot2)
library(Hotelling)
library(data.table)
library(transport)
library(parallel)
source("src/cot.R")

# Helpers: -------------------------------------------------

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

# Encoder-decoder settings: -------------------------------------------------
# Default pipeline settings, to be tuned later. K is swept explicitly below
# (stage_wame's K forces a fixed embedding dimension, bypassing its
# auto-selection); stage_flow keeps its own stock defaults.
dir_art <- file.path("artifacts", "sensitivity-K-1")
dir.create(dir_art, showWarnings = FALSE, recursive = TRUE)
path_flow <- file.path(dir_art, "flow.pth")

# Generativity assessment settings: evaluate_pipeline_generativity() (see
# src/utils.R) is called once unconditionally (on the full two-group sample)
# and once per group (conditionally) for every (b, K) pipeline fit below, so
# S/R are kept small for this first pass.
S_gen <- 2
R_gen <- 5
n_cores_gen <- min(4, R_gen)

# Persistent worker pool reused across every evaluate_pipeline_generativity()
# call below (~450 calls across the nested b x K loop); evaluate_pipeline_
# generativity() detects this pre-existing non-sequential plan and won't
# tear it down between calls, avoiding repeated pool-respawn/Python-reinit
# cost. Per-call n_cores_gen arguments are effectively inert once this pool
# is active -- the pool's fixed worker count controls actual parallelism.
future::plan(future::multisession, workers = n_cores_gen)

# Setting: ----------------------------------------------------------------
nsim_per_setting <- 20 # MAKE BIGGER
mu_mu_1 <- 0
sigma_mu_1 <- 5
mu_log_sigma_1 <- 0
sigma_log_sigma_1 <- 1
mu_alpha_2_levels <- c(1.0, 1.25, 1.5, 1.75, 2.0)# c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5) # c(2, 3, 4) # c(2, 4, 6, 8, 10)
sigma_alpha_2 <- 3
nobs <- 500
n_1 <- 100 # 100
n_2_levels <- c(100) # c(50, 100, 150)
K_levels <- 1:30 # capped low: each K refits the whole encoder-decoder
K_max <- 50

simulation_settings <- expand.grid(
  sim_rep = seq_len(nsim_per_setting),
  mu_alpha_2 = mu_alpha_2_levels,
  n_2 = n_2_levels
)
nsim <- nrow(simulation_settings)
sd_vec <- mu_vec <- array(NA, c(nobs, 2, nsim))
p_vals_by_k <- matrix(
  NA_real_,
  nrow = nsim,
  ncol = length(K_levels),
  dimnames = list(NULL, paste0("K_", K_levels))
)
gen_uncond_by_k <- gen_cond1_by_k <- gen_cond2_by_k <- p_vals_by_k_pca <- p_vals_by_k

# Fit a mean-covariance model to y_list_g's Z-embeddings, decode synthetic
# draws, and score them against the real data via evaluate_pipeline_generativity()
# (src/utils.R), returning the mean log-normalized generativity across all
# S * R splits/replicates. Errors are caught per call (rather than per (b, K)
# pipeline) so one group's failure doesn't blank out the others.
mean_log_norm_generativity <- function(pipeline, y_list_g, S, R, n_cores, context) {
  tryCatch(
    {
      gen <- evaluate_pipeline_generativity(
        pipeline, y_list_g,
        S = S, R = R, n_cores = n_cores
      )
      mean(unlist(lapply(gen$splits, `[[`, "log_norm")))
    },
    error = function(e) {
      message(stringr::str_glue("Generativity evaluation failed ({context}): {conditionMessage(e)}"))
      NA_real_
    }
  )
}

set.seed(1)
start <- Sys.time()
for (b in 1:nsim) {
  print(str_glue("===== b = {b} ====="))
  n_2 <- simulation_settings[b, "n_2"]
  group <- c(rep(1, times = n_1), rep(2, times = n_2))
  mu_alpha_2 <- simulation_settings[b, "mu_alpha_2"]
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

  mu_vec[, 1, b] <- apply(Q_1, MARGIN = 1, mean)
  mu_vec[, 2, b] <- apply(Q_2, MARGIN = 1, mean)

  sd_vec[, 1, b] <- apply(Q_1, MARGIN = 1, sd)
  sd_vec[, 2, b] <- apply(Q_2, MARGIN = 1, sd)

  # Encoder-decoder input: one raw-Y vector per subject (already sorted,
  # which stage_eqf_sgrid's sort() leaves unchanged). asplit() would retain a
  # "dim" attribute on each 1-d slice (tripping up pracma::spinterp's
  # is.vector() check downstream), so go through a data.frame instead.
  y_list <- unname(c(as.list(as.data.frame(Q_1)), as.list(as.data.frame(Q_2))))
  p_grid <- p_grid_fun(
    breaks = c(1 / (1 + nobs), nobs / (1 + nobs)),
    interval_counts = c(nobs)
  )

  # Standard PCA baseline: PCA run directly on the quantile functions,
  # bypassing the encoder-decoder entirely. Q_1/Q_2 are both nobs-length
  # order statistics of equal-size draws, so they already sit on a common
  # probability grid and can be stacked (subjects x nobs) without a separate
  # grid-alignment stage. PCA is fit once per b (K-independent); PC scores
  # are then sliced to the first K columns per K below.
  Q_mat <- t(cbind(Q_1, Q_2))
  pca_out <- prcomp(Q_mat, center = TRUE, scale. = FALSE)
  scores_pca <- pca_out$x

  ###################################
  ###################################
  # K <- 3
  # pipeline <- construct_pipeline(
  #   stages = list(
  #     stage_eqf_sgrid(),
  #     stage_eqf_cgrid(p_grid = p_grid),
  #     stage_wame(K = K, K_max = 20, order_by = 'importance'),
  #     stage_flow(path = path_flow, n_layers = 8, hidden = 64, val_frac = 0, max_epoch = 500)
  #   ),
  #   seed = gen_seed()
  # )
  # pipeline <- fit(pipeline, y_list)

  # ## Encode/Decode
  # y_ctx <- new_context(
  #   payload = y_list,
  #   cache = pipeline$training$cache,
  #   meta = list()
  # )
  # Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
  # Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
  # c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
  # z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
  # c_ctx_ <- decode(pipeline, z_ctx, from = 4, to = 3)
  # Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
  # Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
  # y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)

  # ## Plotting
  # col_1 <- rgb(0, 0, 0, alpha = 0.5)
  # col_2 <- rgb(1, 0, 0, alpha = 0.5) 
  # k_to <- 1
  # k_from <- 3
  # colors <- c(
  #   rep(col_1, n_1), 
  #   rep(col_2, n_2)
  # )
  # grid_list <- rep(list(p_grid), n_1 + n_2)
  # C_mat <- cbind(C_mat, do.call(rbind, c_ctx$payload$Q_star_list))
  # Z_mat <- do.call(rbind, z_ctx$payload)
  # plot_funs(Q_ctx$payload, grid_list, colors = colors, path = path_plot_tmp)
  # plot_embeddings(C_mat[,k_to:k_from], colors = colors, path = path_plot_tmp)
  # plot_embeddings(Z_mat[,k_to:k_from], colors = colors, path = path_plot_tmp)

  # ## Synthetic
  # model <- fit_mean_cov(Z_mat)
  # z_draws <- draw_mean_cov(nrow(Z_mat), model)
  # Qi_draws <- decode_z_to_Qi(pipeline, z_draws, Ji = nobs)

  # ## Plotting
  # col_real <- rgb(0, 0, 0, alpha = 0.5)
  # col_synth <- rgb(0, 1, 0, alpha = 0.5)
  # fun_list <- c(Qi_ctx$payload, Qi_draws)
  # colors <- c(rep(col_real, n_1 + n_2), rep(col_synth, n_1 + n_2))
  # shuff <- sample(length(fun_list))
  # fun_list <- fun_list[shuff]
  # grid_list <- grid_list <- rep(list(p_grid), length(fun_list))
  # plot_funs(fun_list, grid_list, colors = colors, path = path_plot_tmp)
  ###################################
  ###################################

  # For each K: fit the pipeline once and reuse it for the Hotelling p-value
  # plus the three generativity scores. (b, K) cells stay NA in all four
  # matrices if the pipeline/encode step itself fails.
  for (k_idx in seq_along(K_levels)) {
    K <- K_levels[k_idx]

    # PCA baseline p-value: Hotelling's T^2 on the first K PCA scores.
    # Independent of the encoder-decoder fit below, so it's computed even
    # if that fit fails for this (b, K).
    K_pca <- min(K, ncol(scores_pca))
    p_vals_by_k_pca[b, k_idx] <- tryCatch(
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

    # fit_out <- tryCatch(
    #   {
    #     pipeline <- construct_pipeline(
    #       stages = list(
    #         stage_eqf_sgrid(),
    #         stage_eqf_cgrid(p_grid = p_grid),
    #         stage_wame(K = K, K_max = K_max, order_by = 'importance'),
    #         stage_flow(path = path_flow, n_layers = 8, hidden = 64)
    #       ),
    #       seed = gen_seed()
    #     )
    #     pipeline <- fit(pipeline, y_list)
    #     y_ctx <- new_context(
    #       payload = y_list,
    #       cache = pipeline$training$cache,
    #       meta = list()
    #     )
    #     z_ctx <- encode(pipeline, y_ctx, from = 0, to = pipeline$n_stages)
    #     list(pipeline = pipeline, Z_mat = do.call(rbind, z_ctx$payload))
    #   },
    #   error = function(e) {
    #     if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
    #       return(NULL)
    #     }
    #     stop(e)
    #   }
    # )
    # if (is.null(fit_out)) next

    # pipeline <- fit_out$pipeline
    # Z_mat <- fit_out$Z_mat

    # p_vals_by_k[b, k_idx] <- tryCatch(
    #   Hotelling::hotelling.test(
    #     Z_mat[group == 1, , drop = FALSE],
    #     Z_mat[group == 2, , drop = FALSE],
    #     var.equal = FALSE
    #   )$pval,
    #   error = function(e) {
    #     if (grepl("computationally singular", conditionMessage(e), fixed = TRUE)) {
    #       return(NA_real_)
    #     }
    #     stop(e)
    #   }
    # )

    # gen_uncond_by_k[b, k_idx] <- mean_log_norm_generativity(
    #   pipeline, y_list, S_gen, R_gen, n_cores_gen,
    #   context = stringr::str_glue("b={b}, K={K}, unconditional")
    # )
    # gen_cond1_by_k[b, k_idx] <- mean_log_norm_generativity(
    #   pipeline, y_list[group == 1], S_gen, R_gen, n_cores_gen,
    #   context = stringr::str_glue("b={b}, K={K}, group=1")
    # )
    # gen_cond2_by_k[b, k_idx] <- mean_log_norm_generativity(
    #   pipeline, y_list[group == 2], S_gen, R_gen, n_cores_gen,
    #   context = stringr::str_glue("b={b}, K={K}, group=2")
    # )
  }
}
future::plan(future::sequential)  # release the generativity worker pool
saveRDS(p_vals_by_k, file.path(dir_art, 'p_vals_by_k.rds'))
saveRDS(p_vals_by_k_pca, file.path(dir_art, 'p_vals_by_k_pca.rds'))
saveRDS(gen_uncond_by_k, file.path(dir_art, 'gen_uncond_by_k.rds'))
saveRDS(gen_cond1_by_k, file.path(dir_art, 'gen_cond1_by_k.rds'))
saveRDS(gen_cond2_by_k, file.path(dir_art, 'gen_cond2_by_k.rds'))
stop <- Sys.time()
tot_time <- stop - start
print(str_glue("Total Time: {tot_time}"))

## NOTES
##  (p_vals_by_k_1) mu_alpha = c(0.5, 1, 1.5), K = 1:20
##  (p_vals_by_k_2) mu_alpha = c(1.1, 1.2, 1.3, 1.4), K = 1:20
##  (p_vals_by_k_3) mu_alpha = c(1.0, 1.1, 1.2, 1.3, 1.4, 1.5), K = 1:30

# Examine Results: --------------------------------------------------------
## helpers ------------------------------------------------------------------

# Reshape a (sim x K) result matrix into long form, one row per
# (sim_rep, mu_alpha_2, n_2, K).
melt_by_k <- function(mat, simulation_settings) {
  dt <- data.table(simulation_settings, mat)
  dt_lng <- melt.data.table(dt,
    id.vars = c('sim_rep', 'mu_alpha_2', 'n_2'),
    variable.name = 'K', variable.factor = FALSE,
    value.name = 'value'
  )
  dt_lng[, K := as.numeric(stringr::str_remove(K, 'K_'))]
  dt_lng
}

plot_metric_vs_k <- function(summ, y_lab, title_str) {
  ggplot(data = summ) +
    aes(x = K, y = value, colour = factor(mu_alpha_2, ordered = TRUE),
        pch = factor(n_2)) +
    geom_point() +
    geom_line() +
    labs(colour = expression(mu[alpha]~":"),
         linetype = expression(n[2]~":"),
         pch = expression(n[2]~":"),
         y = y_lab,
         x = "Retained K",
         title = title_str,
         subtitle = expression("Across swept"~K)) +
    theme(legend.position = 'bottom',
          text = element_text(size = 15, hjust = 0.5),
          plot.title = element_text(size = 15, hjust = 0),
          axis.title = element_text(size = 15))
}

## power ------------------------------------------------------------------
power_summ <- melt_by_k(p_vals_by_k, simulation_settings)[
  , .(value = mean(value < 0.05, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(power_summ, "Power", "Power using encoder-decoder Z-embeddings")

## p-values ------------------------------------------------------------------
pval_summ <- melt_by_k(p_vals_by_k, simulation_settings)[
  , .(value = mean(value, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(pval_summ, "P-Value", "P-Value using encoder-decoder Z-embeddings")

## power (PCA baseline) -----------------------------------------------------
power_summ_pca <- melt_by_k(p_vals_by_k_pca, simulation_settings)[
  , .(value = mean(value < 0.05, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(power_summ_pca, "Power", "Power using standard PCA on quantile functions")

## p-values (PCA baseline) ---------------------------------------------------
pval_summ_pca <- melt_by_k(p_vals_by_k_pca, simulation_settings)[
  , .(value = mean(value, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(pval_summ_pca, "P-Value", "P-Value using standard PCA on quantile functions")


## generativity -------------------------------------------------------------
gen_uncond_summ <- melt_by_k(gen_uncond_by_k, simulation_settings)[
  , .(value = mean(value, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(gen_uncond_summ, "Log normalized generativity", "Unconditional generativity vs K")

gen_cond1_summ <- melt_by_k(gen_cond1_by_k, simulation_settings)[
  , .(value = mean(value, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(gen_cond1_summ, "Log normalized generativity", "Conditional generativity (group 1) vs K")

gen_cond2_summ <- melt_by_k(gen_cond2_by_k, simulation_settings)[
  , .(value = mean(value, na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)
]
plot_metric_vs_k(gen_cond2_summ, "Log normalized generativity", "Conditional generativity (group 2) vs K")
