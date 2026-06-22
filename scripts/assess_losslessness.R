library(purrr)
library(yaml)

source('src/cot.R')
source('src/utils.R')


## ---------- Utilities ---------- ##

assess_losslessness <- function(
    y_list, 
    fit_pipeline,
    K_from = 1, 
    K_to = 50, 
    K_by = 1, 
    stage_from = 0,
    stage_to = NULL,
    V = 5,
    seed = gen_seed(),
    quantile_levels = c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999),
    verbose = TRUE
) {
  
  ## Set parameters
  N <- length(y_list)
  Ks <- seq(K_from, K_to, K_by)
  valid_losses_by_K <- list()
  train_losses_by_K <- list()

  ## Initialize per-quantile residual container: q -> K -> length-N vector
  valid_quantile_resid_by_K <- NULL
  if (!is.null(quantile_levels)) {
    valid_quantile_resid_by_K <- vector("list", length(quantile_levels))
    names(valid_quantile_resid_by_K) <- as.character(quantile_levels)
    for (q_name in names(valid_quantile_resid_by_K)) {
      valid_quantile_resid_by_K[[q_name]] <- list()
    }
  }

  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))

  for (K in Ks) {
    message(str_glue("---------- K = {K} ----------"))

    valid_losses <- vector(mode = "numeric", length = N)
    train_losses <- matrix(nrow = N, ncol = V)
    valid_q_resid <- NULL
    if (!is.null(quantile_levels)) {
      valid_q_resid <- matrix(
        NA_real_, nrow = N, ncol = length(quantile_levels),
        dimnames = list(NULL, as.character(quantile_levels))
      )
    }
    for (v in 1:V) {
      
      ## Get train/valid indices
      idx_train <- which(folds != v)
      idx_valid <- which(folds == v)
      
      ## Fit pipeline
      pipeline <- fit_pipeline(y_list[idx_train], K)
      stage_to <- stage_to %||% pipeline$n_stages
      loss_fun <- loss_to_fun[[pipeline$stages[[pipeline$n_stages]]$state$loss]]

      ## Fast forward to stage_from data (get data_list)
      ctx <- new_context(
        payload = y_list,
        cache = pipeline$training$cache,
        meta = list()
      )
      if (stage_from > 0) {
        ctx <- encode(pipeline, ctx, from = 0, to = stage_from)
      } 
      data_list <- ctx$payload
      
      ## Encode/Decode (get reco_list)
      ctx <- encode(pipeline, ctx, from = stage_from, to = stage_to)
      ctx <- decode(pipeline, ctx, from = stage_to, to = stage_from)
      reco_list <- ctx$payload
      
      ## Compute train/valid losses
      for (i in 1:N) {
        dp <- 1 / (length(data_list[[i]]) + 1)
        loss_i <- loss_fun(data_list[[i]], reco_list[[i]], dp)
        if (folds[i] == v) {
          valid_losses[i] <- loss_i
          if (!is.null(quantile_levels)) {
            q_data <- quantile(data_list[[i]], probs = quantile_levels, names = FALSE)
            q_reco <- quantile(reco_list[[i]], probs = quantile_levels, names = FALSE)
            valid_q_resid[i, ] <- q_data - q_reco
          }
        } else {
          train_losses[i,v] <- loss_i
        }
      }
    }

    valid_losses_by_K[[as.character(K)]] <- valid_losses
    train_losses_by_K[[as.character(K)]] <- train_losses

    if (!is.null(quantile_levels)) {
      K_name <- as.character(K)
      for (q_idx in seq_along(quantile_levels)) {
        q_name <- as.character(quantile_levels[q_idx])
        valid_quantile_resid_by_K[[q_name]][[K_name]] <- valid_q_resid[, q_idx]
      }
    }

  }

  list(
    valid_losses_by_K = valid_losses_by_K,
    train_losses_by_K = train_losses_by_K,
    valid_quantile_resid_by_K = valid_quantile_resid_by_K
  )
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

  ## Bump right margin if drawing a secondary y-axis
  if (!is.null(pairwise_distances)) {
    par(mar = c(5, 4, 4, 5) + 0.1)
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
    points(xvals_v, v_losses, pch = 19, col = rgb(0,0,0,0.25), cex = 0.5)
    
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
    mtext(ylab2, side = 4, line = 3)
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


plot_quantile_residuals <- function(
    valid_resid_by_K,
    quantile_level = NULL,
    jitter_width = 0.01,
    epsilon = NULL,
    alpha = 0.05,
    plot_mean = TRUE,
    ylim = NULL,
    ylab = "Residual",
    xlab = "K",
    main = NULL,
    K_star = NULL,
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

  Ks <- as.numeric(names(valid_resid_by_K))
  xlim_pad <- 2 * jitter_width

  y_vals <- unlist(valid_resid_by_K)
  y_max  <- max(abs(y_vals))
  if (!is.null(epsilon)) y_max <- max(y_max, max(abs(epsilon)))

  if (is.null(main)) {
    main <- if (is.null(quantile_level)) {
      "Residuals by K"
    } else {
      str_glue("Residuals by K  (q = {quantile_level})")
    }
  }

  plot(
    NULL,
    xlim = c(min(Ks) - xlim_pad, max(Ks) + xlim_pad),
    ylim = ylim %||% c(-y_max, y_max),
    xaxt = "n", xlab = xlab, ylab = ylab,
    main = main
  )
  axis(1, at = Ks, labels = Ks)

  v_means <- v_lo <- v_hi <- rep(NA_real_, length(Ks))

  for (i in seq_along(Ks)) {
    r <- valid_resid_by_K[[i]]
    v_means[i] <- mean(r)
    if (!is.null(alpha)) {
      v_lo[i]    <- quantile(r, probs = alpha / 2)
      v_hi[i]    <- quantile(r, probs = 1 - alpha / 2)
    }

    xvals <- jitter(rep(Ks[i], length(r)), amount = jitter_width)
    points(xvals, r, pch = 19, col = rgb(0, 0, 0, 0.25), cex = 0.5)
  }

  col_valid <- rgb(0, 0.75, 0)

  if (plot_mean) {
    points(Ks, v_means, col = col_valid, pch = 19)
    lines(Ks,  v_means, col = col_valid)
  }

  if (!is.null(alpha)) {
    points(Ks, v_lo, col = col_valid, pch = 17)
    lines(Ks,  v_lo, col = col_valid)
    points(Ks, v_hi, col = col_valid, pch = 17)
    lines(Ks,  v_hi, col = col_valid)
  }

  abline(h = 0, lty = "dashed", col = "gray40")

  if (!is.null(epsilon)) {
    for (e in epsilon) {
      abline(h =  e, lty = "dashed", col = "red")
      abline(h = -e, lty = "dashed", col = "red")
    }
  }

  if (!is.null(K_star)) {
    abline(v = K_star, lty = "dashed", col = "black")
  }

  ## Close device only if we opened it
  if (!is.null(path)) dev.off()
}





## ---------- Pipeline Fitting Functions ---------- ##


fit_pipeline_qg_pca <- function(
    y_list, K,
    p_grid = NULL,
    supp_Y = NULL,
    p_star = 0,
    y_star = NULL,
    y_min = NULL,
    loss = 'wasserstein',
    seed = 12345,
    lambda = NULL
) {

  pipeline <- construct_pipeline(
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_wame(K = K, lambda = lambda, loss = loss)
    ),
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
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_q_pca(K = K, loss = loss)
    ),
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
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_g_pca(K = K, loss = loss)
    ),
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

####################################################
# ## Q: What percentage of pairwise distances have concordance > 0.99?
# path <- 'artifacts/demo_nhanes/pipe_nhanes.rds'
# pipeline <- readRDS(path)

# seed <- 12347
# samp_rate <- 0.05
# Qi_list <- pipeline$training$meta$Qi_list
# N <- length(Qi_list)
# set.seed(seed)
# N_samp <- floor(samp_rate * N)
# idx_samp <- sample(1:N, size = N_samp)
# Qi_list_samp <- Qi_list[idx_samp]
# pw_dists <- list()
# cnt <- 1
# start <- Sys.time()
# for (i in 1:(N_samp-1)) {
#   for (j in (i+1):N_samp) {
#     pw_dists[[cnt]] <- sqrt(1 - one_minus_sqconc(Qi_list_samp[[i]], Qi_list_samp[[j]]))
#     cnt <- cnt + 1
#   }
# }
# stop <- Sys.time()
# print(stop - start)

# pw_dists <- unlist(pw_dists)
# hist(pw_dists)
# print(mean(pw_dists > 0.99))
####################################################

## ----- Globals 

dir_sim <- file.path('simulations', 'nhanes-baseline')


## ----- Create YAML configs

## Dataset specs
datasets <- list(
  # multi_250 = list(
  #   path = file.path('data', 'processed', 'multinomial-250.rds'),
  #   p_grid_fun = partial(p_grid_fun, J = 100),
  #   supp_Y = 1:10,
  #   y_star = 1,
  #   y_min = 1,
  #   y_trans = 'identity',
  #   y_shift = 0
  # )#,
  # multi_500 = list(
  #   path = file.path('data', 'processed', 'multinomial-500.rds'),
  #   p_grid_fun = partial(p_grid_fun, J = 100),
  #   supp_Y = 1:10,
  #   y_star = 1,
  #   y_min = 1,
  #   y_trans = 'identity',
  #   y_shift = 0
  # ),
  # multi_1000 = list(
  #   path = file.path('data', 'processed', 'multinomial-1000.rds'),
  #   p_grid_fun = partial(p_grid_fun, J = 100),
  #   supp_Y = 1:10,
  #   y_star = 1,
  #   y_min = 1,
  #   y_trans = 'identity',
  #   y_shift = 0
  # ),
  # multi_5000 = list(
  #   path = file.path('data', 'processed', 'multinomial-5000.rds'),
  #   p_grid_fun = partial(p_grid_fun, J = 100),
  #   supp_Y = 1:10,
  #   y_star = 1,
  #   y_min = 1,
  #   y_trans = 'identity',
  #   y_shift = 0
  # ),
  # enmo = list(
  #   path = file.path('data', 'processed', 'chop-enmo-5_v1.rds'),
  #   p_grid_fun = partial(
  #     p_grid_fun, J = 150,
  #     p_left = 0.05, J_left = 50,
  #     p_right = 0.95, J_right = 50
  #   ),
  #   supp_Y = seq(0, 12, by = 0.0001),
  #   y_star = 0,
  #   y_min = 0,
  #   y_trans = 'boxcox',
  #   y_shift = 1e-4
  # ),
  # mims = list(
  #   path = file.path('data', 'processed', 'chop-mims_v1.rds'),
  #   p_grid = p_grid_fun(
  #     breaks = c(1/10001, 0.95, 10000/10001),
  #     interval_counts = c(51, 50)
  #   ),
  #   supp_Y = c(0, 0.6, seq(0.601, 2000, by = 0.001)),
  #   y_star = 0,
  #   y_min = 0,
  #   y_trans = 'identity',
  #   y_shift = 0,
  #   loss_scale_samp_rate = 1.0,
  #   lambda = 1e-6
  # )
  # tean = list(
  #   path = file.path('data', 'processed', 'tean_v2.rds'),
  #   # p_grid_fun = partial(
  #   #   p_grid_fun, J = 100,
  #   #   p_right = 0.95, J_right = 50
  #   # ),
  #   p_grid_fun = partial(
  #     p_grid_fun, J = 100,
  #     p_right = 0.95, J_right = 50
  #   ),
  #   supp_Y = 0:50000,
  #   y_star = 0,
  #   y_min = 0,
  #   y_trans = 'log',
  #   y_shift = 1,
  #   loss_scale_samp_rate = 0.1,
  #   lambda = 0
  # ),
  # nhanes_1000 = list(
  #   path = file.path('data', 'processed', 'nhanes_v1_nofilter_N-1000.rds'),
  #   p_grid_fun = partial(
  #     p_grid_fun, J = 100,
  #     p_right = 0.95, J_right = 50
  #   ),
  #   supp_Y = c(0, seq(0.006, 400, by = 0.001)),
  #   y_star = -0.01,
  #   y_min = -0.01,
  #   y_trans = 'log',
  #   y_shift = 100,
  #   loss_scale_samp_rate = 0.1,
  #   lambda = 0
  # ),
  # nhanes = list(
  #   path = file.path('data', 'processed', 'nhanes_v1_nofilter.rds'),
  #   p_grid = p_grid_fun(
  #     breaks = c(1/(10080 + 1), 0.95, 10080/(10080 + 1)),
  #     interval_counts = c(51, 50)
  #   ),
  #   supp_Y = c(0, seq(0.006, 400, by = 0.001)),
  #   y_star = 0,
  #   y_min = 0,
  #   y_trans = 'identity', #'identity',
  #   y_shift = 0,
  #   loss_scale_samp_rate = 1.0, # 0.05,
  #   p_scale = NULL, # 0.025,
  #   lambda = 0
  # ),
  nhanes_2 = list(
    path = file.path('data', 'processed', 'nhanes_v1_nofilter_N-1000.rds'),
    p_grid = p_grid_fun(
      breaks = c(1/(10080 + 1), 0.95, 10080/(10080 + 1)),
      interval_counts = c(51, 50)
    ),
    supp_Y = c(0, seq(0.006, 400, by = 0.001)),
    y_star = 0,
    y_min = 0,
    pairwise_samp_rate = 1.0, # 0.05,
    lambda = 0.1 # 0
  )
)


## Create configs
models <- c('qg_pca')
i <- 1
for (dataset in names(datasets)) {
  for (model in models) {
    path <- file.path(dir_sim, 'configs', str_glue('config-{i}.yml'))
    config <- list(
      dataset = dataset,
      model_type = model,
      path_loss_valid = file.path(dir_sim, 'losses', str_glue('config-{i}_loss-valid.rds')),
      path_loss_train = file.path(dir_sim, 'losses', str_glue('config-{i}_loss-train.rds')),
      path_quantile_resid_valid = file.path(dir_sim, 'losses', str_glue('config-{i}_quantile-resid-valid.rds')),
      path_pairwise_dist_norm = file.path(dir_sim, 'losses', str_glue('config-{i}_pairwise-dist-norm.rds')),
      path_concordances = file.path(dir_sim, 'losses', str_glue('config-{i}_concordances.rds'))
    )
    write_yaml(config, path)
    i <- i + 1  
  }
}
n_configs <- i - 1



## ----- Process Configs

set.seed(12346)
for (i in 1:n_configs) {
  message(str_glue("========== CONFIG {i} of {n_configs} =========="))
  
  ## Load config
  path <- file.path(dir_sim, 'configs', str_glue('config-{i}.yml'))
  config <- read_yaml(path)
  
  ## Load data
  y_list <- readRDS(datasets[[config$dataset]]$path)
  
  ## Create fit_pipeline function
  if (config$model_type == 'qg_pca') {
    fit_pipeline <- partial(
      fit_pipeline_qg_pca,
      p_grid = datasets[[config$dataset]]$p_grid,
      supp_Y = datasets[[config$dataset]]$supp_Y,
      y_star = datasets[[config$dataset]]$y_star,
      y_min = datasets[[config$dataset]]$y_min,
      lambda = datasets[[config$dataset]]$lambda,
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
  
  out <- assess_losslessness(
    y_list = y_list,
    fit_pipeline = fit_pipeline,
    K_from = 1,
    K_to = 4,
    K_by = 1,
    stage_from = 1,
    V = 5,
    seed = gen_seed(),
    verbose = FALSE
  )
  saveRDS(out$valid_losses_by_K, config$path_loss_valid)
  saveRDS(out$train_losses_by_K, config$path_loss_train)
  saveRDS(out$valid_quantile_resid_by_K, config$path_quantile_resid_valid)

  ## Compute normalized pairwise distances on a subsample of the training data
  pipe       <- fit_pipeline(y_list, K = 1)
  Qi_list_p  <- pipe$training$meta$Qi_list
  Ji_vec_p   <- pipe$training$meta$Ji_vec
  loss_fun_p <- loss_to_fun[[pipe$stages[[pipe$n_stages]]$state$loss]]
  p_grid_p   <- pipe$training$cache$p_grid
  supp_Y_p  <- pipe$training$cache$supp_Y
  samp_rate  <- datasets[[config$dataset]]$pairwise_samp_rate
  set.seed(12345)
  idx_samp   <- sample(seq_along(Qi_list_p), size = floor(samp_rate * length(Qi_list_p)))
  d <- pairwise_distance(
    Qi_list      = Qi_list_p[idx_samp],
    loss_fun     = loss_fun_p,
    pi_grid_list = lapply(Ji_vec_p[idx_samp], pi_grid_fun),
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

K_stars <- c(4)
for (i in 1:n_configs) {
  message(str_glue("========== Concordance for CONFIG {i} of {n_configs} =========="))

  ## Pull K_star defensively (handles both numeric-vector and list forms).
  K_star <- if (is.list(K_stars)) K_stars[[i]] else K_stars[i]
  if (is.null(K_star) || isTRUE(is.na(K_star))) {
    message(str_glue("  K_stars[{i}] is NULL/NA -- skipping concordance."))
    next
  }

  ## Load config + data
  path_config <- file.path(dir_sim, 'configs', str_glue('config-{i}.yml'))
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

set.seed(12345)
for (i in 1:n_configs) {
  print(i)
  path_config <- file.path(dir_sim, 'configs', str_glue('config-{i}.yml'))
  path_plot <- file.path(dir_sim, 'plots', str_glue('config-{i}.png'))
  
  ## Extract losses
  config <- read_yaml(path_config)
  loss_valid <- readRDS(config$path_loss_valid)
  loss_train <- readRDS(config$path_loss_train)
  loss_train <- lapply(loss_train, function(mat) rowMeans(mat, na.rm = TRUE))  ## avg. loss
  quantile_resid_valid <- readRDS(config$path_quantile_resid_valid)
  pairwise_dist_norm <- readRDS(config$path_pairwise_dist_norm)
  concordances <- if (!is.null(config$path_concordances) &&
                      file.exists(config$path_concordances)) {
    readRDS(config$path_concordances)
  } else NULL

  ## Plot losslessness
  plot_losslessness(
    loss_valid, NULL,
    jitter_width = 0.2,
    epsilon = c(1.25),
    alpha = 0.05,
    plot_mean = FALSE,
    # ylim = c(0, 15), #c(0, 20),
    ylab = "Cross-Validated Wasserstein Error",
    ylab2 = "Pairwise Wasserstein Distance Percentile",
    xlab = "K",
    K_star = K_stars[i],
    pairwise_distances = pairwise_dist_norm,
    concordances = concordances,
    concordance_box_position = "topright",
    # main = str_glue('{config$dataset} | {config$model_type}'),
    main = "NHANES Validation",
    path = path_plot
  )

  ## Plot quantile residuals
  # for (q_name in names(quantile_resid_valid)) {
  #   max_abs_resid <- max(abs(unlist(quantile_resid_valid)))
  #   path_plot_q <- file.path(
  #     dir_sim, 'plots',
  #     str_glue('config-{i}_quantile-resid-{q_name}.png')
  #   )
  #   plot_quantile_residuals(
  #     quantile_resid_valid[[q_name]],
  #     quantile_level = as.numeric(q_name),
  #     jitter_width   = 0.2,
  #     alpha          = NULL,
  #     plot_mean      = FALSE,
  #     K_star         = NULL,
  #     ylim           = c(-max_abs_resid, max_abs_resid),
  #     main           = str_glue(
  #       '{config$dataset} | {config$model_type} | q = {q_name}'
  #     ),
  #     path           = path_plot_q
  #   )
  # }

}


##### Concordance Computation for Jeff #####
# path <- file.path(dir_sim, 'losses', str_glue('config-1_concordances.rds'))
# concordances <- readRDS(path)

# mean(concordances > 0.99)
# mean(concordances > 0.997)
############################################
