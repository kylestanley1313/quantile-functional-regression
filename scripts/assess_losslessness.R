library(purrr)
library(yaml)

source('src/cot.R')
source('src/utils.R')


## ---------- Utilities ---------- ##

## Computes cross-validated reconstruction losses (validation only) across a
## grid of K values. `stages_fun(K)` must return a list of unfit stages in
## which only the LAST stage depends on K (true of stages_qg_pca/_q_pca/
## _g_pca above) -- everything before it is fit once per CV fold and reused
## across the whole K sweep, since fold membership doesn't depend on K.
compute_cv_losses <- function(
    y_list,
    stages_fun,
    supp_Y = NULL,
    p_star = 0,
    y_star = NULL,
    y_min = NULL,
    pipeline_seed = 12345,
    K_from = 1,
    K_to = 50,
    K_by = 1,
    V = 5,
    seed = gen_seed(),
    verbose = TRUE
) {

  ## Set parameters
  N <- length(y_list)
  Ks <- seq(K_from, K_to, K_by)
  n_stages <- length(stages_fun(Ks[1]))
  loss_fun <- loss_to_fun[[stages_fun(Ks[1])[[n_stages]]$state$loss]]

  valid_losses_by_K <- setNames(
    replicate(length(Ks), numeric(N), simplify = FALSE),
    as.character(Ks)
  )

  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))

  for (v in 1:V) {
    message(str_glue("---------- Fold {v} of {V} ----------"))

    idx_train <- which(folds != v)
    idx_valid <- which(folds == v)

    ## Fit the K-independent prefix (all stages but the last) ONCE per fold.
    prefix_stages <- stages_fun(Ks[1])[1:(n_stages - 1)]
    prefix <- construct_pipeline(
      stages = prefix_stages,
      supp_Y = supp_Y,
      p_star = p_star,
      y_star = y_star,
      y_min = y_min,
      seed = pipeline_seed
    )
    prefix <- fit(prefix, y_list[idx_train])

    ## Encode training data through the prefix once; reused to fit the
    ## K-dependent final stage for every K below.
    ctx_train <- encode(
      prefix,
      new_context(payload = y_list[idx_train], cache = prefix$training$cache),
      from = 0, to = n_stages - 1
    )

    ## Encode validation data through the prefix once: keep the Qi-space
    ## loss target (right after the first stage) and the prefix output
    ## (input to the final stage).
    ctx0_valid <- new_context(payload = y_list[idx_valid], cache = prefix$training$cache)
    ctx1_valid <- encode(prefix, ctx0_valid, from = 0, to = 1)
    data_list_valid <- ctx1_valid$payload
    ctx_valid <- encode(prefix, ctx1_valid, from = 1, to = n_stages - 1)

    for (K in Ks) {
      message(str_glue("---------- K = {K} ----------"))
      K_name <- as.character(K)

      ## Fit only the K-dependent final stage on the (already-encoded) training data.
      final_stage <- stages_fun(K)[[n_stages]]
      final_stage <- fit(final_stage, ctx_train)

      ## Encode/decode only the validation subjects.
      ctx_v <- encode(final_stage, ctx_valid)
      ctx_v <- decode(final_stage, ctx_v)
      ctx_v <- decode(prefix, ctx_v, from = n_stages - 1, to = 1)
      reco_list_valid <- ctx_v$payload

      for (j in seq_along(idx_valid)) {
        i <- idx_valid[j]
        dp <- 1 / (length(data_list_valid[[j]]) + 1)
        valid_losses_by_K[[K_name]][i] <- loss_fun(data_list_valid[[j]], reco_list_valid[[j]], dp)
      }
    }
  }

  valid_losses_by_K
}


plot_losslessness <- function(
    valid_losses_by_K,
    train_losses_by_K = NULL,
    jitter_width = 0.01,
    epsilon = 0.01,
    alpha = 0.05,
    plot_mean = TRUE,
    ylim = NULL,
    ylab = "Loss",
    xlab = "K",
    main = "Loss by K",
    K_star = NULL,
    pairwise_distances = NULL,
    pairwise_percentiles = c(50, 25, 10, 1),
    ylab2 = "Pairwise Distance Percentile",
    concordances = NULL,
    concordance_box_position = "topright",
    concordance_box_cex = 0.8,
    path = NULL,
    width = 1200,
    height = 900,
    res = 150
) {
  ## If saving, open appropriate graphics device
  if (!is.null(path)) {
    ext <- tolower(tools::file_ext(path))
    if (ext == "png") {
      png(path, width = width, height = height, res = res)
    } else if (ext %in% c("jpg", "jpeg")) {
      jpeg(path, width = width, height = height, quality = 100)
    } else {
      stop("Unsupported extension: use png, jpg, or jpeg.")
    }
  }

  ## Secondary y-axis: give the right margin the same room as the left
  if (!is.null(pairwise_distances)) {
    mar <- par("mar")
    mar[4] <- max(mar[4], mar[2])
    par(mar = mar)
  }

  Ks <- as.numeric(names(valid_losses_by_K))
  xlim_pad <- 2 * jitter_width

  y_vals <- unlist(valid_losses_by_K)
  if (!is.null(train_losses_by_K)) {
    y_vals <- c(y_vals, unlist(train_losses_by_K))
  }

  plot(
    NULL,
    xlim = c(min(Ks) - xlim_pad, max(Ks) + xlim_pad),
    ylim = ylim %||% c(0, max(y_vals)),
    xaxt = "n", xlab = xlab, ylab = ylab,
    main = main
  )
  axis(1, at = Ks, labels = Ks)
  
  v_means <- v_quants <- rep(NA_real_, length(Ks))
  t_means <- t_quants <- rep(NA_real_, length(Ks))
  
  for (i in seq_along(Ks)) {
    v_losses <- valid_losses_by_K[[i]]
    v_means[i]  <- mean(v_losses)
    v_quants[i] <- quantile(v_losses, probs = 1 - alpha)
    
    xvals_v <- jitter(rep(Ks[i], length(v_losses)), amount = jitter_width)
    points(xvals_v, v_losses, pch = 16, col = rgb(0,0,0,0.25), cex = 0.5)
    
    if (!is.null(train_losses_by_K)) {
      t_losses <- train_losses_by_K[[i]]
      t_means[i]  <- mean(t_losses)
      t_quants[i] <- quantile(t_losses, probs = 1 - alpha)
    }
  }
  
  col_valid <- rgb(0, 0.75, 0)
  
  if (plot_mean) {
    points(Ks, v_means,  col = col_valid, pch = 19)
    lines(Ks,  v_means,  col = col_valid)
  }
  
  points(Ks, v_quants, col = col_valid, pch = 17)
  lines(Ks,  v_quants, col = col_valid)
  
  if (!is.null(train_losses_by_K)) {
    points(Ks, t_means,  col = "blue", pch = 19)
    lines(Ks,  t_means,  col = "blue")
    
    points(Ks, t_quants, col = "blue", pch = 17)
    lines(Ks,  t_quants, col = "blue")
  }
  
  for (e in epsilon) {
    abline(h = e, lty = "dashed", col = "red")
  }
  
  if (!is.null(K_star)) {
    abline(v = K_star, lty = "dashed", col = 'red')
  }

  ## Secondary y-axis: pairwise-distance percentiles
  if (!is.null(pairwise_distances)) {
    ticks_y <- quantile(
      pairwise_distances,
      probs = pairwise_percentiles / 100,
      names = FALSE
    )
    yr <- par("usr")[3:4]
    in_range <- ticks_y >= yr[1] & ticks_y <= yr[2]
    axis(
      side = 4,
      at = ticks_y[in_range],
      labels = pairwise_percentiles[in_range],
      las = 1
    )
    ## Match the left axis title's position and size
    mtext(ylab2, side = 4, line = par("mgp")[1],
          cex = par("cex") * par("cex.lab"))
  }

  ## Concordance summary box. Reports the alpha-quantile of per-subject
  ## reconstruction concordances at K_star -- so (1 - alpha) of subjects
  ## have concordance above the quoted value.
  if (!is.null(concordances) &&
      !is.null(K_star) && !isTRUE(is.na(K_star)) &&
      !is.null(alpha)) {
    q_lower <- quantile(concordances, probs = alpha,
                        names = FALSE, na.rm = TRUE)
    pct <- round(100 * (1 - alpha))
    box_text <- c(
      sprintf("K = %g", K_star),
      sprintf("%d%% of subjects have", pct),
      sprintf("concordance > %.3f", q_lower)
    )
    legend(
      concordance_box_position,
      legend  = box_text,
      bty     = "o",
      bg      = rgb(1, 1, 1, 0.85),
      cex     = concordance_box_cex,
      inset   = 0.02,
      x.intersp = -0.5
    )
  }

  ## Close device only if we opened it
  if (!is.null(path)) dev.off()
}


## ---------- Pipeline Fitting Functions ---------- ##

## Stage-list builders. Only the LAST stage in each list depends on K --
## compute_cv_losses() relies on this to split off a K-independent "prefix"
## it can fit once per CV fold and reuse across the whole K sweep.
stages_qg_pca <- function(K, K_max = 20, p_grid = NULL, lambda = NULL, order_by = 'variance', loss = 'wasserstein') {
  list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_wame(K = K, K_max = K_max, lambda = lambda, order_by = order_by, loss = loss)
  )
}


stages_q_pca <- function(K, p_grid = NULL, loss = 'wasserstein') {
  list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_q_pca(K = K, loss = loss)
  )
}


stages_g_pca <- function(K, p_grid = NULL, loss = 'wasserstein') {
  list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_lqd(),
    stage_g_pca(K = K, loss = loss)
  )
}


fit_pipeline_qg_pca <- function(
    y_list, K,
    p_grid = NULL,
    K_max = 20,
    supp_Y = NULL,
    p_star = 0,
    y_star = NULL,
    y_min = NULL,
    loss = 'wasserstein',
    seed = 12345,
    lambda = NULL,
    order_by = 'variance'
) {

  pipeline <- construct_pipeline(
    stages = stages_qg_pca(K, K_max = K_max, p_grid = p_grid, lambda = lambda, order_by = order_by, loss = loss),
    supp_Y = supp_Y,
    p_star = p_star,
    y_star = y_star,
    y_min = y_min,
    seed = seed
  )
  pipeline <- fit(pipeline, y_list)

  pipeline
}


fit_pipeline_q_pca <- function(
    y_list, K,
    p_grid = NULL,
    supp_Y = NULL,
    y_star = NULL,
    y_min = NULL,
    loss = 'wasserstein',
    seed = 12345
) {

  pipeline <- construct_pipeline(
    stages = stages_q_pca(K, p_grid = p_grid, loss = loss),
    supp_Y = supp_Y,
    y_star = y_star,
    y_min = y_min,
    seed = seed
  )
  pipeline <- fit(pipeline, y_list)

  pipeline
}

fit_pipeline_g_pca <- function(
    y_list, K,
    p_grid = NULL,
    supp_Y = NULL,
    p_star = 0,
    y_star = NULL,
    y_min = NULL,
    loss = 'wasserstein',
    seed = 12345
) {

  pipeline <- construct_pipeline(
    stages = stages_g_pca(K, p_grid = p_grid, loss = loss),
    supp_Y = supp_Y,
    p_star = p_star,
    y_star = y_star,
    y_min = y_min,
    seed = seed
  )
  pipeline <- fit(pipeline, y_list)

  pipeline
}



## ---------- EXECUTION ---------- ##


## ----- Globals 

dir_art <- file.path('artifacts', 'demo_nhanes_v5', 'losslessness')


## ----- Create YAML configs

## Dataset specs
datasets <- list(
  # sgrow2 = list(
  #   path = file.path('data', 'processed', 'chop-mims_v1.rds'),
  #   p_grid = p_grid_fun(
  #     breaks = c(1/(1200 + 1), 0.95, 1200/(1200 + 1)),
  #     interval_counts = c(51, 50)
  #   ),
  #   K_max = 20,
  #   supp_Y = c(0, 0.6, seq(0.601, 500, by = 0.001)),
  #   y_star = 0,
  #   y_min = 0,
  #   lambda = 0,
  #   order_by = 'importance',
  #   pairwise_samp_rate = 1.0
  # )
  nhanes = list(
    path = file.path('data', 'processed', 'nhanes-fda-mims_v2.rds'),
    p_grid = p_grid_fun(
      breaks = c(1/(7350 + 1), 0.95, 7350/(7350 + 1)),
      interval_counts = c(51, 50)
    ),
    K_max = 20,
    supp_Y = c(0, seq(0.006, 500, by = 0.001)),
    y_star = 0,
    y_min = 0,
    lambda = 0, # sqrt(10)^-1,
    order_by = 'importance',
    pairwise_samp_rate = 0.1,
  )
)


## Create configs
models <- c('qg_pca')
i <- 1
for (dataset in names(datasets)) {
  for (model in models) {
    path <- file.path(dir_art, str_glue('config-{i}.yml'))
    config <- list(
      dataset = dataset,
      model_type = model,
      path_loss_valid = file.path(dir_art, str_glue('config-{i}_loss-valid.rds')),
      path_pairwise_dist_norm = file.path(dir_art, str_glue('config-{i}_pairwise-dist-norm.rds')),
      path_concordances = file.path(dir_art, str_glue('config-{i}_concordances.rds'))
    )
    write_yaml(config, path)
    i <- i + 1  
  }
}
n_configs <- i - 1



## ----- Process Configs

set.seed(12345)
for (i in 1:n_configs) {
  message(str_glue("========== CONFIG {i} of {n_configs} =========="))
  
  ## Load config
  path <- file.path(dir_art, str_glue('config-{i}.yml'))
  config <- read_yaml(path)
  
  ## Load data
  y_list <- readRDS(datasets[[config$dataset]]$path)
  
  ## Create fit_pipeline function (used below for the one-off pairwise-distance fit)
  ## and stages_fun (used by compute_cv_losses for the fast CV sweep).
  if (config$model_type == 'qg_pca') {
    fit_pipeline <- partial(
      fit_pipeline_qg_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      K_max = datasets[[config$dataset]]$K_max,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min,
      lambda = datasets[[config$dataset]]$lambda,
      order_by = datasets[[config$dataset]]$order_by %||% 'variance',
      loss = 'wasserstein'
    )
    stages_fun <- partial(
      stages_qg_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      lambda = datasets[[config$dataset]]$lambda,
      order_by = datasets[[config$dataset]]$order_by %||% 'variance',
      loss = 'wasserstein'
    )
  } else if (config$model_type == 'q_pca') {
    fit_pipeline <- partial(
      fit_pipeline_q_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min
    )
    stages_fun <- partial(
      stages_q_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      loss = 'wasserstein'
    )
  } else if (config$model_type == 'g_pca') {
    fit_pipeline <- partial(
      fit_pipeline_g_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min
    )
    stages_fun <- partial(
      stages_g_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      loss = 'wasserstein'
    )
  } else {
    stop(str_glue("Config {i} has invalid model_type!"))
  }

  out <- compute_cv_losses(
    y_list = y_list,
    stages_fun = stages_fun,
    supp_Y = datasets[[config$dataset]]$supp_Y,
    y_star = datasets[[config$dataset]]$y_star,
    y_min = datasets[[config$dataset]]$y_min,
    K_from = 1,
    K_to = 10,
    K_by = 1,
    V = 5,
    seed = gen_seed(),
    verbose = FALSE
  )
  saveRDS(out, config$path_loss_valid)

  ## Compute normalized pairwise distances on a subsample of the training data
  ## TOOD: This no longer works given new pairwise distance computation!
  pipe       <- fit_pipeline(y_list, K = 1)
  Qi_list_p  <- pipe$training$meta$Qi_list
  Ji_vec_p   <- pipe$training$meta$Ji_vec
  loss_fun_p <- loss_to_fun[[pipe$stages[[pipe$n_stages]]$state$loss]]
  p_grid_p   <- pipe$training$cache$p_grid
  supp_Y_p  <- pipe$training$cache$supp_Y
  samp_rate  <- datasets[[config$dataset]]$pairwise_samp_rate
  set.seed(12345)
  idx_samp   <- sample(seq_along(Qi_list_p), size = floor(samp_rate * length(Qi_list_p)))
  d <- compute_pwds(
    Qi_list      = Qi_list_p[idx_samp],
    loss_fun     = loss_fun_p,
    p_grid_aug   = p_grid_p,
    supp_Y       = supp_Y_p
  )
  saveRDS(d, config$path_pairwise_dist_norm)
}


## ----- Concordance Computation

## Per-subject reconstruction concordance at K_stars[i]. Computed on the
## full y_list via a single non-CV fit at K_star: encode through the
## pipeline, decode back to Qi-space (stage_from = 2), then compute
## per-subject concordance correlation between original and reconstructed
## Qi. Saved as length-N numeric vectors keyed by config.
##
## K_stars[i] non-NULL / non-NA is required: NULL or NA entries are
## skipped (no file written), and plot_losslessness will fall back to
## drawing no annotation box for that config.

K_stars <- c(5)
for (i in 1:n_configs) {
  message(str_glue("========== Concordance for CONFIG {i} of {n_configs} =========="))

  ## Pull K_star defensively (handles both numeric-vector and list forms).
  K_star <- if (is.list(K_stars)) K_stars[[i]] else K_stars[i]
  if (is.null(K_star) || isTRUE(is.na(K_star))) {
    message(str_glue("  K_stars[{i}] is NULL/NA -- skipping concordance."))
    next
  }

  ## Load config + data
  path_config <- file.path(dir_art, str_glue('config-{i}.yml'))
  config      <- read_yaml(path_config)
  y_list      <- readRDS(datasets[[config$dataset]]$path)

  ## Reconstruct fit_pipeline (same partial as the fitting loop).
  if (config$model_type == 'qg_pca') {
    fit_pipeline <- partial(
      fit_pipeline_qg_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min,
      lambda = datasets[[config$dataset]]$lambda,
      order_by = datasets[[config$dataset]]$order_by %||% 'variance',
      loss = 'wasserstein'
    )
  } else if (config$model_type == 'q_pca') {
    fit_pipeline <- partial(
      fit_pipeline_q_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min
    )
  } else if (config$model_type == 'g_pca') {
    fit_pipeline <- partial(
      fit_pipeline_g_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min
    )
  } else {
    stop(str_glue("Config {i} has invalid model_type!"))
  }

  ## Fit at K_star and reconstruct.
  pipeline   <- fit_pipeline(y_list, K = K_star)
  stage_from <- 1
  stage_to   <- pipeline$n_stages

  ctx <- new_context(
    payload = y_list,
    cache   = pipeline$training$cache,
    meta    = list()
  )
  ctx       <- encode(pipeline, ctx, from = 0,          to = stage_from)
  data_list <- ctx$payload
  ctx       <- encode(pipeline, ctx, from = stage_from, to = stage_to)
  ctx       <- decode(pipeline, ctx, from = stage_to,   to = stage_from)
  reco_list <- ctx$payload

  ## Per-subject concordance correlation in Qi-space (length-N vector).
  N_           <- length(data_list)
  concordances <- numeric(N_)
  for (k in seq_len(N_)) {
    Q1 <- data_list[[k]]
    Q2 <- reco_list[[k]]
    w  <- get_quadrature_weights(pi_grid_fun(length(Q1)))
    concordances[k] <- sqrt(1 - one_minus_sqconc(Q1, Q2, w))
  }
  saveRDS(concordances, config$path_concordances)
}


## ----- Plot

## Paper figure spec (copied from demo_nhanes.R): render at final printed
## size so fig_pointsize is the printed text size
fig_res       <- 600
fig_pointsize <- 7
fig_cex       <- 1
fig_cex_lab   <- 1.0
fig_cex_main  <- 1.2 * fig_cex_lab
fig_cex_axis  <- 0.8 * fig_cex_lab
fig_png <- function(path, width_in, height_in, pointsize = fig_pointsize) {
  png(path, width = width_in, height = height_in, units = "in",
      res = fig_res, pointsize = pointsize)
}

## Set plotting parameters
path <- file.path(dir_art, 'config-1_pairwise-dist-norm.rds')
pairwise_dist_norm <- readRDS(path)
epsilon <- quantile(pairwise_dist_norm, c(0.05))

set.seed(12345)
for (i in 1:n_configs) {
  print(i)
  path_config <- file.path(dir_art, str_glue('config-{i}.yml'))
  path_plot <- file.path(dir_art, str_glue('config-{i}.png'))
  
  ## Extract losses
  config <- read_yaml(path_config)
  loss_valid <- readRDS(config$path_loss_valid)
  pairwise_dist_norm <- readRDS(config$path_pairwise_dist_norm)
  concordances <- if (!is.null(config$path_concordances) &&
                      file.exists(config$path_concordances)) {
    readRDS(config$path_concordances)
  } else NULL

  ## Plot losslessness
  fig_png(path_plot, 4.3, 3.25)   # TODO: final printed size
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
  plot_losslessness(
    loss_valid, NULL,
    jitter_width = 0.2,
    epsilon = epsilon,
    alpha = 0.05,
    plot_mean = FALSE,
    ylim = c(0, 20),
    ylab = "Cross-Validated Wasserstein Error",
    ylab2 = "Pairwise Wasserstein Distance Percentile",
    xlab = "K",
    K_star = K_stars[i],
    pairwise_distances = pairwise_dist_norm,
    concordances = concordances,
    concordance_box_position = "topright",
    # main = str_glue('{config$dataset} | {config$model_type}'),
    main = "NHANES Reconstructions",
    concordance_box_cex = fig_cex_axis
  )
  dev.off()

}


##### Concordance Computation for Jeff #####
# path <- file.path(dir_art, 'losses', str_glue('config-1_concordances.rds'))
# concordances <- readRDS(path)

# mean(concordances > 0.99)
# mean(concordances > 0.997)
############################################
