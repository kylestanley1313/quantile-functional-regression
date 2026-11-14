library(reticulate)

source('src/model.R')


## ---------- Miscellaneous ---------- ##

gen_seed <- function() {
  sample.int(.Machine$integer.max, 1)
}


## ---------- Y-Axis Transforms ---------- ##

## Standalone y-axis transforms, retained for axis labelling / manual use after
## the Y-Axis stage was removed from the encoder-decoder pipeline.

identity_transform <- function(x, shift = 0, inverse = FALSE) {
  if (inverse) {
    x - shift
  } else {
    x + shift
  }
}

log_transform <- function(x, shift = 0, inverse = FALSE) {
  if (inverse) {
    exp(x) - shift
  } else {
    log(x + shift)
  }
}

loglog_transform <- function(x, shift = 0, inverse = FALSE) {
  if (inverse) {
    log_transform(log_transform(x, shift, TRUE), shift, TRUE)
  } else {
    log_transform(log_transform(x, shift), shift)
  }
}

sqrt_transform <- function(x, shift = 0, inverse = FALSE) {
  if (inverse) {
    x^2 - shift
  } else {
    sqrt(x + shift)
  }
}

boxcox_transform <- function(x, shift = 0, lambda = 0, inverse = FALSE) {
  if (inverse) {
    if (abs(lambda) < 1e-8) {
      exp(x) - shift
    } else {
      (lambda*x + 1)^(1/lambda) - shift
    }
  } else {
    if (abs(lambda) < 1e-8) {
      log(x + shift)
    } else {
      ((x + shift)^lambda - 1) / lambda
    }
  }
}

y_trans_to_fun <- list(
  'identity' = identity_transform,
  'log' = log_transform,
  'loglog' = loglog_transform,
  'sqrt' = sqrt_transform,
  'boxcox' = boxcox_transform
)


## ---------- Python ---------- ##

init_python <- function() {
  config <- read_yaml(file.path("src", "config.yml"))
  use_python(config$python$path, required = TRUE)
  py_run_string(glue("
  import sys
  path = r'{config$python$module_path}'
  if path not in sys.path:
      sys.path.insert(0, path)
  "))
}

reload_py_pkg <- function() {
  importlib <- import("importlib")
  python <- import("python", convert = TRUE)
  flow <- import("python.flow", convert = TRUE)
  importlib$reload(flow)
  importlib$reload(python)
  python
}


## ---------- P-Grid ---------- ##

pi_grid_fun <- function(J) {
  1:J / (1 + J)
}

p_grid_fun <- function(breaks, interval_counts) {
  n_intervals <- length(interval_counts)
  if (length(breaks) != n_intervals + 1) {
    stop("Length of breaks must be one greater than length of interval_counts")
  }
  grid <- c()
  for (i in 1:n_intervals) {
    grid <- c(grid, seq(breaks[i], breaks[i+1], length.out = interval_counts[i]))
  }
  grid <- c(grid, breaks[n_intervals+1])
  grid <- sort(unique(grid))
  grid
}

get_quadrature_weights <- function(p_grid) {
  J <- length(p_grid)
  dp <- diff(p_grid)
  w <- numeric(J)
  w[1] <- 0.5 * dp[1]
  w[J] <- 0.5 * dp[J - 1]
  w[2:(J-1)] <- 0.5 * (dp[1:(J-2)] + dp[2:(J-1)])
  w / sum(w)
}

## ---------- Loss Functions ---------- ##

one_minus_sqcor <- function(Q1, Q2, w = NULL) {
  if (is.null(w)) {
    1 - cor(Q1, Q2)^2
  } else {
    w <- w / sum(w)
    mu1 <- sum(w * Q1)
    mu2 <- sum(w * Q2)
    s1_sq <- sum(w * (Q1 - mu1)^2)
    s2_sq <- sum(w * (Q2 - mu2)^2)
    s12   <- sum(w * (Q1 - mu1) * (Q2 - mu2))
    1 - s12^2 / (s1_sq * s2_sq)
  }
}

one_minus_sqconc <- function(Q1, Q2, w = NULL) {
  if (is.null(w)) {
    mu1  <- mean(Q1)
    mu2  <- mean(Q2)
    s1_sq <- var(Q1)
    s2_sq <- var(Q2)
    s12   <- cov(Q1, Q2)
  } else {
    w     <- w / sum(w)
    mu1   <- sum(w * Q1)
    mu2   <- sum(w * Q2)
    s1_sq <- sum(w * (Q1 - mu1)^2)
    s2_sq <- sum(w * (Q2 - mu2)^2)
    s12   <- sum(w * (Q1 - mu1) * (Q2 - mu2))
  }
  c <- 2 * s12 / (s1_sq + s2_sq + (mu1 - mu2)^2)
  1 - c^2
}

wasserstein <- function(Q1, Q2, w) {
  sqrt(sum(w * (Q1 - Q2)^2))
}

loss_to_fun <- list(
  'one_minus_sqcor' = one_minus_sqcor,
  'one_minus_sqconc' = one_minus_sqconc,
  'wasserstein' = wasserstein
)


compute_sampled_pwds <- function(
    y_list,
    pipeline,
    loss,
    p_grid_aug,
    N_samp = NULL,
    seed = gen_seed()
) {
  set.seed(seed)
  w_aug <- get_quadrature_weights(p_grid_aug)
  loss_fun <- loss_to_fun[[loss]]

  ## Get indices for samping
  N <- length(y_list)
  if (is.null(N_samp)) {
    idx_samp <- 1:N
  } else {
    idx_samp <- sample(1:N, size = N_samp)
    N <- length(idx_samp)
  }
  distances <- numeric(N * (N - 1) / 2)

  ## Get list of pi_grids
  pi_grid_list <- lapply(lengths(y_list[idx_samp]), pi_grid_fun)

  ## Encode y to Qi_aug
  Qi_aug_list <- encode_y_to_Qi_aug(pipeline, y_list[idx_samp], p_grid_aug)

  ## Compute pairwise distances
  idx <- 1
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      p_low  <- max(pi_grid_list[[i]][1], pi_grid_list[[j]][1])
      p_high <- min(tail(pi_grid_list[[i]], 1), tail(pi_grid_list[[j]], 1))
      mask   <- (p_grid_aug >= p_low) & (p_grid_aug <= p_high)
      w_ij   <- w_aug[mask] / sum(w_aug[mask])
      distances[idx] <- loss_fun(Qi_aug_list[[i]][mask], Qi_aug_list[[j]][mask], w_ij)
      idx <- idx + 1
    }
  }

  distances
}



compute_pwds <- function(
    Qi_list,
    loss_fun,
    p_grid_aug,
    supp_Y = NULL
) {
  w_aug <- get_quadrature_weights(p_grid_aug)

  N <- length(Qi_list)
  distances <- numeric(N * (N - 1) / 2)

  ## Per-subject grids and interpolation onto the shared augmented grid
  pi_grid_list <- lapply(lengths(Qi_list), pi_grid_fun)
  Qi_aug_list <- vector("list", N)
  for (i in seq_len(N)) {
    Qi_aug_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], pi_grid_list[[i]],
      p_grid_aug, supp_Y
    )
  }

  ## Compute pairwise distances over the shared p-support of each pair
  idx <- 1
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      p_low  <- max(pi_grid_list[[i]][1], pi_grid_list[[j]][1])
      p_high <- min(tail(pi_grid_list[[i]], 1), tail(pi_grid_list[[j]], 1))
      mask   <- (p_grid_aug >= p_low) & (p_grid_aug <= p_high)
      w_ij   <- w_aug[mask] / sum(w_aug[mask])
      distances[idx] <- loss_fun(Qi_aug_list[[i]][mask], Qi_aug_list[[j]][mask], w_ij)
      idx <- idx + 1
    }
  }

  distances
}



## ---------- Encoding/Decoding ---------- ##

## NOTE: The hard-coded from=/to= stage indices below assume the canonical
## pipeline order
##   eqf_sgrid(1) -> eqf_cgrid(2) -> wame(3) -> flow(4) -> [pca_rotation(5)]
## where `c` is the wame output (stage 3), `z` is the flow latent (stage 4) and
## `z_rot` is the pca_rotation output (stage 5). The LQD and QG-PCA steps are
## fused inside the single `wame` stage, so there is no separate G/G_Q_star
## boundary. Update these indices if the stage layout changes.
##
## Downstream models (binary, amm) can treat `z_rot`, `z` or `c` as the latent
## representation; see the "Latent-space selection" block below.

decode_z_to_Qi <- function(pipeline, z_list, Ji) {
  z_ctx <- new_context(
    payload = z_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_list)))
  )
  decode(pipeline, z_ctx, from = 4, to = 1)$payload
}

decode_z_to_Qi_aug <- function(pipeline, z_list, p_grid_aug, p_grid) {
  N <- length(z_list)
  z_ctx <- new_context(
    payload = z_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(1000, length.out = length(z_list)))
  )
  Qi_list <- decode(pipeline, z_ctx, from = 4, to = 2)$payload

  supp_Y <- pipeline$training$cache$supp_Y
  for (i in seq_len(N)) {
    Qi_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], p_grid,
      p_grid_aug, supp_Y
    )
  }
  Qi_list
}

decode_z_rot_to_Qi <- function(pipeline, z_rot_list, Ji) {
  z_rot_ctx <- new_context(
    payload = z_rot_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_rot_list)))
  )
  decode(pipeline, z_rot_ctx, from = 5, to = 1)$payload
}


## ---------- c-embedding sign convention ---------- ##

## Orientation of the c-embeddings as seen by everything outside the pipeline
## (models, plots, clustering). The sign of each wame basis function is
## arbitrary, so this only fixes a convention.
##
## Unlike set_rotation_signs(), this does NOT touch any stage's state. c sits
## mid-pipeline: the flow was trained on c in its fitted orientation, so
## negating basis columns would feed it mirrored inputs and change z and z_rot.
## The flips are instead applied at the boundary, in exactly two places --
## c_ctx_to_matrix() on the way out and .c_payload() on the way in -- so the
## stages, the flow, and every z / z_rot result are untouched.
##
## Consequently, never read c_ctx$payload$c_list directly for analysis: use
## c_ctx_to_matrix() (or encode_y_to_c()), and decode with decode_c_to_Qi*().
## Mixing a flipped c with an unflipped decode gives wrong QFs with no error.
set_c_signs <- function(pipeline, flip = integer(0)) {
  flip <- as.integer(flip)
  if (anyNA(flip) || any(flip < 1) || anyDuplicated(flip)) {
    stop("`flip` must be distinct positive component indices.")
  }
  pipeline$c_flip <- flip
  pipeline
}

## Length-K vector of +1/-1; all +1 when no convention has been set.
.c_signs <- function(pipeline, K) {
  flip <- pipeline$c_flip
  if (length(flip) > 0 && max(flip) > K) {
    stop(str_glue("c sign flip refers to component {max(flip)}, but K = {K}."))
  }
  s <- rep(1, K)
  s[flip] <- -1
  s
}

## N x K matrix of c-embeddings, in the pipeline's sign convention, from a
## context at the wame stage output. Note the payload is
## list(c_list, Q_star_list), hence the $c_list.
c_ctx_to_matrix <- function(pipeline, c_ctx) {
  C <- do.call(rbind, c_ctx$payload$c_list)
  sweep(C, 2, .c_signs(pipeline, ncol(C)), `*`)
}


## The wame stage emits list(c_list, Q_star_list), not a bare list of vectors,
## so decoding from C requires rebuilding both slots. Q_star is carried in the
## cache (fixed by the pipeline's y_star), never inside the c-vectors.
##
## NOTE: when cache$Q_star is NULL, encode_fun_flow appends a per-subject Q_star
## column to c_mat (see src/cot.R), making the flow latent K+1 rather than K. A
## K-dimensional c is then not the whole latent, so the helpers below refuse
## that case rather than silently dropping the Q_star coordinate.
.c_payload <- function(pipeline, c_list) {
  Q_star <- pipeline$training$cache$Q_star
  if (is.null(Q_star)) {
    stop("latent = 'c' requires a fixed Q_star (non-NULL y_star) in the cache.")
  }
  ## Incoming c is in the pipeline's sign convention; undo it before the
  ## stages see it (see set_c_signs).
  list(
    c_list = lapply(
      c_list,
      function(v) as.numeric(v) * .c_signs(pipeline, length(v))
    ),
    Q_star_list = rep(list(Q_star), length(c_list))
  )
}

decode_c_to_Qi <- function(pipeline, c_list, Ji) {
  c_ctx <- new_context(
    payload = .c_payload(pipeline, c_list),
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(c_list)))
  )
  decode(pipeline, c_ctx, from = 3, to = 1)$payload
}

decode_c_to_Qi_aug <- function(pipeline, c_list, p_grid_aug, p_grid) {
  N <- length(c_list)
  c_ctx <- new_context(
    payload = .c_payload(pipeline, c_list),
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(1000, length.out = length(c_list)))
  )
  Qi_list <- decode(pipeline, c_ctx, from = 3, to = 2)$payload

  supp_Y <- pipeline$training$cache$supp_Y
  for (i in seq_len(N)) {
    Qi_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], p_grid,
      p_grid_aug, supp_Y
    )
  }
  Qi_list
}


## Encode y_list to the rotated latent space (z) and stack into an N x K matrix.
encode_y_to_z <- function(pipeline, y_list) {
  y_ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  z_ctx <- encode(pipeline, y_ctx, from = 0, to = 4)
  do.call(rbind, z_ctx$payload)
}

## Encode y_list to the wame coefficient space (c) and stack into an N x K
## matrix, in the pipeline's sign convention (see set_c_signs).
encode_y_to_c <- function(pipeline, y_list) {
  y_ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  c_ctx <- encode(pipeline, y_ctx, from = 0, to = 3)
  c_ctx_to_matrix(pipeline, c_ctx)
}


## ---------- Latent-space selection ---------- ##

## `pipeline$latent` selects which stage output the downstream models (binary,
## amm, generativity) treat as the latent representation:
##   'z_rot' (default) -> stage 5, the PCA-rotated flow output
##   'z'               -> stage 4, the unrotated flow output
##   'c'               -> stage 3, the wame coefficients
##
## Set it on the pipeline object (pipeline$latent <- 'c'); every consumer reads
## it from there, so no model or plotting signature changes.
##
## NOTE: the 'z_rot' encode branch deliberately stops at stage 4 (z), not stage
## 5. stage_pca_rotation is a centering plus an orthogonal rotation, so this is
## immaterial for fit_binary, and stopping at 4 reproduces the pre-existing
## encode_y_to_z behaviour exactly. Do not "fix" it without re-running the
## z_rot baselines.
.latent_of <- function(pipeline) pipeline$latent %||% 'z_rot'

encode_y_to_latent <- function(pipeline, y_list) {
  switch(
    .latent_of(pipeline),
    z_rot = encode_y_to_z(pipeline, y_list),
    z     = encode_y_to_z(pipeline, y_list),
    c     = encode_y_to_c(pipeline, y_list),
    stop(str_glue("Unknown pipeline$latent: '{.latent_of(pipeline)}'"))
  )
}

decode_latent_to_Qi <- function(pipeline, latent_list, Ji) {
  switch(
    .latent_of(pipeline),
    z_rot = decode_z_rot_to_Qi(pipeline, latent_list, Ji),
    z     = decode_z_to_Qi(pipeline, latent_list, Ji),
    c     = decode_c_to_Qi(pipeline, latent_list, Ji),
    stop(str_glue("Unknown pipeline$latent: '{.latent_of(pipeline)}'"))
  )
}

decode_latent_to_Qi_aug <- function(pipeline, latent_list, p_grid_aug, p_grid) {
  switch(
    .latent_of(pipeline),
    z_rot = decode_z_to_Qi_aug(pipeline, latent_list, p_grid_aug, p_grid),
    z     = decode_z_to_Qi_aug(pipeline, latent_list, p_grid_aug, p_grid),
    c     = decode_c_to_Qi_aug(pipeline, latent_list, p_grid_aug, p_grid),
    stop(str_glue("Unknown pipeline$latent: '{.latent_of(pipeline)}'"))
  )
}

## Encode y_list to Qi on the augmented p-grid.
encode_y_to_Qi_aug <- function(pipeline, y_list, p_grid_aug) {
  N <- length(y_list)
  Ji_vec <- lengths(y_list)

  y_ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)

  Qi_list <- Qi_ctx$payload
  supp_Y <- pipeline$training$cache$supp_Y
  for (i in seq_len(N)) {
    pi_grid <- pi_grid_fun(Ji_vec[i])
    Qi_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], pi_grid,
      p_grid_aug, supp_Y
    )
  }
  Qi_list
}


## ---------- Losslessness ---------- ##

compute_cv_losses <- function(
    y_list, 
    fit_pipeline,
    K_from = 1, 
    K_to = 50, 
    K_by = 1, 
    stage_from = 0,
    stage_to = NULL,
    V = 5,
    seed = gen_seed(),
    verbose = TRUE
) {
  
  ## Set parameters
  N <- length(y_list)
  Ks <- seq(K_from, K_to, K_by)
  valid_losses_by_K <- list()

  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))

  for (K in Ks) {
    message(str_glue("---------- K = {K} ----------"))

    valid_losses <- vector(mode = "numeric", length = N)
    for (v in 1:V) {
      
      ## Get train/valid indices
      idx_train <- which(folds != v)
      idx_valid <- which(folds == v)
      
      ## Fit pipeline
      pipeline <- fit_pipeline(y_list[idx_train], K = K)
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
      
      ## Compute valid losses
      for (i in 1:N) {
        dp <- 1 / (length(data_list[[i]]) + 1)
        loss_i <- loss_fun(data_list[[i]], reco_list[[i]], dp)
        if (folds[i] == v) {
          valid_losses[i] <- loss_i
        }
      }
    }

    valid_losses_by_K[[as.character(K)]] <- valid_losses
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


## Joint dimension-selection figure: two stacked frames sharing the K axis.
##   - bottom (heights[2] share): per-subject cross-validated reconstruction
##     errors by K (jittered points), the (1 - alpha) quantile curve, the
##     epsilon threshold, and a right axis of pairwise-distance percentiles.
##   - top (heights[1] share): median relative generativity by K, with a
##     reference line at 0.
## The frames abut (the top frame's x axis is the bottom frame's top edge), and
## a vertical line at K_star runs through both.
## valid_losses_by_K: named list (names = K) of per-subject losses.
## gen_scores_by_K:   named list (names = K) of relative generativity
##                    scores (pooled over splits; nested lists are unlisted).
## epsilon: threshold for the dashed line; defaults to the epsilon_percentile
##          percentile of pairwise_distances.
## path = NULL draws to the caller's device (e.g. a fig_png() paper figure with
## fig_par() set); with a path, opens a png at the paper figure spec.
plot_dimension_selection <- function(valid_losses_by_K,
                                     gen_scores_by_K,
                                     pairwise_distances,
                                     K_star             = NULL,
                                     epsilon            = NULL,
                                     epsilon_percentile = 2,
                                     alpha              = 0.05,
                                     pairwise_percentiles = c(50, 25, 10, 1),
                                     heights            = c(1, 3),
                                     jitter_width       = 0.2,
                                     ylim_loss          = NULL,
                                     xlab               = "K",
                                     ylab_loss          = "Cross-Validated Wasserstein Error",
                                     ylab_loss2         = "Pairwise Wasserstein Distance Percentile",
                                     ylab_gen           = "Median rel.\ngenerativity",
                                     col_quant          = rgb(0, 0.75, 0),
                                     col_gen            = "black",
                                     col_ref            = "red",
                                     pt_col             = rgb(0, 0, 0, 0.25),
                                     pt_cex             = 0.5,
                                     path               = NULL,
                                     width              = 3.25,
                                     height             = 3.6,
                                     pointsize          = 7,
                                     res                = 600) {
  Ks <- as.numeric(names(valid_losses_by_K))
  stopifnot(setequal(as.character(Ks), names(gen_scores_by_K)))
  gen_med <- vapply(as.character(Ks), function(k) {
    median(unlist(gen_scores_by_K[[k]], use.names = FALSE), na.rm = TRUE)
  }, numeric(1))
  v_quants <- vapply(valid_losses_by_K, quantile, numeric(1),
                     probs = 1 - alpha, names = FALSE)
  if (is.null(epsilon)) {
    epsilon <- quantile(pairwise_distances, epsilon_percentile / 100,
                        names = FALSE)
  }

  if (!is.null(path)) {
    png(path, width = width, height = height, units = "in", res = res,
        pointsize = pointsize)
    fig_par(cex_main = 1.2, cex_lab = 1, cex = 1, cex_axis = 0.8)
  }
  old_par <- par(no.readonly = TRUE)
  on.exit(if (is.null(path)) par(old_par) else dev.off(), add = TRUE)

  ## Shared horizontal geometry: same xlim and left / right margins in both
  ## frames so the K positions line up. The right margin fits the percentile
  ## tick labels (horizontal) plus the secondary axis label.
  xlim  <- range(Ks) + c(-2, 2) * jitter_width
  mar   <- par("mar")
  mgp   <- par("mgp")
  ## Left margin also fits a multi-line generativity label
  n_lines <- length(strsplit(ylab_gen, "\n", fixed = TRUE)[[1]])
  mar_l <- max(mar[2], mgp[1] + n_lines + 0.4)
  mar_r <- mgp[1] + 1.6
  layout(matrix(1:2, ncol = 1), heights = heights)

  ## ----- Top frame: median generativity
  par(mar = c(0, mar_l, 0.5, mar_r))
  y_rng <- range(c(gen_med, 0), finite = TRUE)
  ylim_gen <- y_rng + c(-0.12, 0.12) * diff(y_rng)
  plot(NULL, xlim = xlim, ylim = ylim_gen, xaxt = "n", yaxt = "n",
       xlab = "", ylab = "")
  axis(2, at = pretty(y_rng, n = 3))
  ## Multi-line label: keep the innermost line on the usual label line so it
  ## clears the tick labels, stacking further lines outward
  mtext(ylab_gen, side = 2, line = mgp[1] + 0.5 * (n_lines - 1),
        cex = par("cex.lab"))
  abline(h = 0, lty = "dashed", col = "gray50")
  if (!is.null(K_star)) abline(v = K_star, lty = 3, lwd = 1.5, col = col_ref)
  lines(Ks, gen_med, col = col_gen)
  points(Ks, gen_med, pch = 19, col = col_gen, cex = 0.9)

  ## ----- Bottom frame: cross-validated reconstruction errors
  par(mar = c(mar[1], mar_l, 0, mar_r))
  y_all <- unlist(valid_losses_by_K, use.names = FALSE)
  plot(NULL, xlim = xlim, ylim = ylim_loss %||% c(0, max(y_all)),
       xaxt = "n", xlab = xlab, ylab = ylab_loss)
  ## Small gap.axis so no K label is dropped at narrow widths
  axis(1, at = Ks, labels = Ks, gap.axis = 0.1)
  for (i in seq_along(Ks)) {
    v <- valid_losses_by_K[[i]]
    points(jitter(rep(Ks[i], length(v)), amount = jitter_width), v,
           pch = 19, col = pt_col, cex = pt_cex)
  }
  abline(h = epsilon, lty = "dashed", col = col_ref)
  ## Label epsilon on the left axis, in the reference colour
  axis(2, at = epsilon, labels = sprintf("%.2f", epsilon), col.axis = col_ref,
       tick = FALSE)
  if (!is.null(K_star)) abline(v = K_star, lty = 3, lwd = 1.5, col = col_ref)
  lines(Ks, v_quants, col = col_quant, lwd = 1.2)
  points(Ks, v_quants, col = col_quant, pch = 17)

  ## Secondary y-axis: pairwise-distance percentiles
  ticks_y <- quantile(pairwise_distances, pairwise_percentiles / 100,
                      names = FALSE)
  yr <- par("usr")[3:4]
  ok <- ticks_y >= yr[1] & ticks_y <= yr[2]
  axis(4, at = ticks_y[ok], labels = pairwise_percentiles[ok], las = 1)
  mtext(ylab_loss2, side = 4, line = mar_r - 1, cex = par("cex.lab"))

  invisible(list(gen_median = gen_med, loss_quantile = v_quants,
                 epsilon = epsilon))
}


## Group-wise jittered scatter of per-subject losses (one column per group),
## styled after plot_losslessness. Each entry in losses_list is a numeric
## vector; group_labels are the x-axis tick labels.
## path = NULL draws to the caller's device (e.g. a fig_png() paper figure with
## fig_par() set). With a path, opens a png at the paper figure spec of
## demo_nhanes.R: width / height in inches at res dpi, pointsize the printed
## text size, and fig_par()'s baseline title / label / tick sizes. The default
## margins follow the current par, trimmed for an omitted title / axis label
## (.trimmed_mar); mar overrides.
plot_losslessness_groups <- function(losses_list,
                                      group_labels,
                                      epsilon      = NULL,
                                      annotate_idx = NULL,
                                      jitter_width = 0.1,
                                      ylab         = "Loss",
                                      xlab         = "",
                                      main         = "",
                                      path         = NULL,
                                      width        = 3.25,
                                      height       = 2.15,
                                      pointsize    = 7,
                                      res          = 600,
                                      mar          = NULL,
                                      pt_col       = rgb(0, 0, 0, 0.25),
                                      pt_cex       = 0.5,
                                      legend_cex   = 1) {
  L <- length(losses_list)
  stopifnot(length(group_labels) == L)

  if (!is.null(path)) {
    png(path, width = width, height = height, units = "in", res = res,
        pointsize = pointsize)
    fig_par(cex_main = 1.2, cex_lab = 1, cex = 1, cex_axis = 0.8)
  }
  if (is.null(mar)) {
    mar <- .trimmed_mar(par("mar"), main, xlab, ylab)
    ## Without an x label only the group names sit below the axis: fit them
    if (.is_blank(xlab)) mar[1] <- par("mgp")[2] + 1.2 * par("cex.axis") + 0.1
    if (!is.null(group_sublabels)) mar[1] <- mar[1] + 1.2 * par("cex.axis")
  }
  old_par <- par(mar = mar)
  ## Restore only the caller's device; our own png is closed below
  if (is.null(path)) on.exit(par(old_par), add = TRUE)

  y_all <- unlist(losses_list, use.names = FALSE)
  y_max <- max(c(y_all, epsilon), na.rm = TRUE)

  plot(NULL,
       xlim = c(0.5, L + 0.5),
       ylim = c(0, y_max),
       xaxt = "n",
       xlab = xlab, ylab = ylab, main = main)
  axis(1, at = seq_len(L), labels = group_labels)

  for (i in seq_len(L)) {
    v <- losses_list[[i]]
    xvals <- jitter(rep(i, length(v)), amount = jitter_width)
    points(xvals, v, pch = 16, cex = pt_cex, col = pt_col)
  }

  if (!is.null(epsilon)) {
    for (e in epsilon) abline(h = e, col = "red", lty = 3, lwd = 1.2)
  }

  if (!is.null(annotate_idx)) {
    eps <- epsilon[1]
    txts <- vapply(annotate_idx, function(g) {
      pct <- 100 * mean(losses_list[[g]] < eps)
      sprintf("%.1f%% of %s < %.2f", pct, group_labels[g], eps)
    }, character(1))
    legend("topleft", legend = txts, bty = "n", cex = legend_cex)
  }

  if (!is.null(path)) dev.off()
  invisible(NULL)
}


## ---------- Generativity ---------- ##

q_moments <- function(Q_list, w) {
  M  <- do.call(rbind, Q_list)          # N x J
  m  <- as.numeric(M %*% w)             # mean per row
  Mc <- sweep(M, 1, m, '-')             # row-centered
  v  <- as.numeric((Mc^2) %*% w)        # variance per row
  m3 <- as.numeric((Mc^3) %*% w)        # third central moment per row
  tol  <- .Machine$double.eps^0.5
  skew <- ifelse(v > tol, m3 / v^1.5, 0)
  list(mean = m, var = v, skew = skew)
}

## Computes N_synth x N_real matrix of squared Wasserstein distances
costs_wasserstein <- function(Q_synth_list, Q_real_list, w) {
  M1 <- do.call(rbind, Q_synth_list)
  M2 <- do.call(rbind, Q_real_list)
  M1w <- sweep(M1, 2, w, '*')
  sq1 <- rowSums(M1w * M1)
  M2w <- sweep(M2, 2, w, '*')
  sq2 <- rowSums(M2w * M2)
  cross <- M1w %*% t(M2)
  cost_mat <- outer(sq1, sq2, '+') - 2 * cross
  cost_mat <- pmax(cost_mat, 0)  # guard tiny negative roundoff
  cost_mat
}

## Computes N_synth x N_real matrix of squared differences between means
costs_mean <- function(Q_synth_list, Q_real_list, w) {
  a <- q_moments(Q_synth_list, w)$mean
  b <- q_moments(Q_real_list,  w)$mean
  outer(a, b, '-')^2
}

## Computes N_synth x N_real matrix of squared differences between variances
costs_variance <- function(Q_synth_list, Q_real_list, w) {
  a <- q_moments(Q_synth_list, w)$var
  b <- q_moments(Q_real_list,  w)$var
  outer(a, b, '-')^2
}

## Computes N_synth x N_real matrix of squared differences  between skewnesses
costs_skewness <- function(Q_synth_list, Q_real_list, w) {
  a <- q_moments(Q_synth_list, w)$skew
  b <- q_moments(Q_real_list,  w)$skew
  outer(a, b, '-')^2
}  

resolve_costs_fun <- function(cost_type = c("wasserstein", "mean",
                                            "variance", "skewness")) {
  cost_type <- match.arg(cost_type)
  switch(cost_type,
    wasserstein = costs_wasserstein,
    mean        = costs_mean,
    variance    = costs_variance,
    skewness    = costs_skewness
  )
}

## Compute the (per-real-observation) generativity costs and scalar score
## sqrt(sum(g)) by solving balanced OT between two empirical measures on a
## shared grid. Returns the cost vector, scalar score, and the raw flow plan.
compute_generativity <- function(
    Q_synth_list,
    Q_real_list,
    w,
    cost_type
) {
  N_synth <- length(Q_synth_list)
  N <- length(Q_real_list)

  ## N_synth x N cost matrix of costs
  costs_fun <- resolve_costs_fun(cost_type)
  cost_mat <- costs_fun(Q_synth_list, Q_real_list, w)

  ## Optimal transport via network simplex
  result <- transport::transport(
    rep(1/N_synth, N_synth),
    rep(1/N, N),
    costm = cost_mat,
    method = "networkflow"
  )

  ## Per-synthetic-observation cost g_i = sum_j pi_ij * c(i, j)
  edge_cost <- result$mass * cost_mat[cbind(result$from, result$to)]
  g <- numeric(N_synth)
  g_grouped <- tapply(edge_cost, result$from, sum)
  g[as.integer(names(g_grouped))] <- g_grouped

  list(
    costs = g,
    score = sqrt(sum(g)),
    plan  = result
  )
}

## Scalar wrapper around compute_generativity for callers that only need the
## score.
compute_generativity_score <- function(Q_a, Q_b, w, cost_type) {
  compute_generativity(Q_a, Q_b, w, cost_type)$score
}


## Data-split relative generativity assessment.
##
## Generativity is a property of both the encoder-decoder and the latent
## model, so it is classified along two axes:
##
##   latent model    covariate-blind: P_theta(.) ignores covariates
##                   covariate-aware: P_theta(. | x); synthetic data are drawn
##                     at the observed covariates of the reference subjects
##   reference data  unconditional: the full held-out data D
##                   conditional:   the stratum D_S = {Q_i in D : x_i in S}
##
## The arguments select the type:
##
##                    | ref_mask = NULL        | ref_mask = stratum mask
##   -----------------+------------------------+--------------------------
##   fit_fun = NULL   | unconditional, blind   | conditional, blind
##   fit_fun, X given | unconditional, aware   | conditional, aware
##
## For each of S splits: partition y_list into train/val (optionally
## subsampling the pool to pool_subsample_n first), fit the latent model on
## the FULL training half, restrict both halves to the reference set
## (ref_mask), compute the baseline real-vs-val score T_tv from the reference
## training subjects, then for each of R replicates draw synthetic latents of
## size matching the reference validation set, decode, and score
## synth-vs-val (T_sv). The reported quantity per replicate is the relative
## generativity, stored as log_norm = log(T_sv / T_tv) (field name kept so
## saved results still load).
##
## The split is drawn before (and independently of) ref_mask, so calls that
## differ only in ref_mask use the same splits and the same fitted models per
## split: their scores are directly comparable (e.g. unconditional vs each
## stratum). The model is always fit on the full training half (not the
## stratum), so a covariate-aware model sees every stratum.
##
## fit_fun / sample_fun contract (covariate-aware scoring):
##   fit_fun(Z, X)    -> a fitted latent-model object ("fit"). Z is the split's
##                       entire training-half latent matrix and X the
##                       corresponding rows of X. Model configuration
##                       (hyperparameters, which column of X is what) is closed
##                       over by the caller.
##   sample_fun(n, fit, X) -> a length-n list of numeric K-vectors, the i-th
##                       drawn from p(Z | X_i). X is the n-row covariate subset
##                       of the reference validation subjects being scored (same
##                       rows, same order as idx_val_grp).
## With fit_fun = sample_fun = NULL (covariate-blind), the latent model is the
## unconditional mean-covariance model (fit_mean_cov with ridge,
## sample_mean_cov) and X is never passed to it.
##
## val_subsample_n subsamples the reference validation set only (scoring cost /
## cross-stratum comparability); pool_subsample_n also shrinks the fit.
##
## keep_draws = TRUE additionally stores, per split, the reference training /
## validation indices (idx_tr_grp, idx_val_grp) and each replicate's synthetic
## latents (z_draws: a length-R list of n_val x K matrices), so any scored
## replicate can be rebuilt and plotted with generativity_replicate(). Memory
## is S * R * n_val * K * 8 bytes (~16.5 MB for S = 5, R = 20, n_val ~ 4100,
## K = 5); decoded QFs and transport plans are never kept.
##
## Cost note: for NHANES-sized splits (~4000x4000) each transport solve
## allocates a ~128 MB cost matrix and n_cores multiplies that footprint.
## Keep S, R modest for tuning runs.
evaluate_generativity <- function(
  pipeline, y_list,
  X                = NULL,
  fit_fun          = NULL,
  sample_fun       = NULL,
  ref_mask         = NULL,
  S, R,
  cost_type        = 'wasserstein',
  frac_train       = 0.5,
  pool_subsample_n = NULL,
  val_subsample_n  = NULL,
  J_aug            = 500,
  ridge            = 0,
  seed             = 12345,
  n_cores          = 1,
  keep_draws       = FALSE
) {
  ## Model axis
  aware <- !is.null(fit_fun) || !is.null(sample_fun)
  if (aware) {
    stopifnot(
      is.function(fit_fun), is.function(sample_fun),
      !is.null(X), NROW(X) == length(y_list)
    )
  } else {
    fit_fun    <- function(Z, X) fit_mean_cov(Z, ridge = ridge)
    sample_fun <- sample_mean_cov
    X          <- NULL   # covariate-blind: the model never sees X
  }

  ## Reference axis
  conditional <- !is.null(ref_mask)
  if (!conditional) ref_mask <- rep(TRUE, length(y_list))
  stopifnot(is.logical(ref_mask), length(ref_mask) == length(y_list),
            !anyNA(ref_mask))

  set.seed(seed)

  ## Build augmented grid + weights once
  p_grid     <- pipeline$training$cache$p_grid
  p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
  w_aug      <- get_quadrature_weights(p_grid_aug)

  split_seeds <- sample.int(.Machine$integer.max, S)

  splits <- vector('list', S)
  for (s in seq_len(S)) {
    set.seed(split_seeds[s])

    ## Optional pool-level subsample, then split -- unmasked, so the RNG draw
    ## sequence is independent of ref_mask.
    n_pool <- length(y_list)
    if (!is.null(pool_subsample_n)) {
      stopifnot(pool_subsample_n <= n_pool)
      idx_pool <- sample.int(n_pool, pool_subsample_n)
    } else {
      idx_pool <- seq_len(n_pool)
    }
    n_used   <- length(idx_pool)
    n_train  <- floor(frac_train * n_used)
    perm     <- sample(idx_pool)
    idx_tr   <- perm[seq_len(n_train)]
    idx_val  <- perm[(n_train + 1):n_used]

    ## Fit on the FULL training half (not ref_mask-restricted)
    Z_tr <- encode_y_to_latent(pipeline, y_list[idx_tr])
    X_tr <- if (is.null(X)) NULL else X[idx_tr, , drop = FALSE]
    fit  <- fit_fun(Z_tr, X_tr)

    ## Restrict to the reference set for scoring. Logical subsetting is not a
    ## random draw, so for an unconditional reference idx_tr_grp == idx_tr and
    ## idx_val_grp == idx_val exactly.
    idx_tr_grp  <- idx_tr[ref_mask[idx_tr]]
    idx_val_grp <- idx_val[ref_mask[idx_val]]

    if (!is.null(val_subsample_n)) {
      stopifnot(val_subsample_n <= length(idx_val_grp))
      idx_val_grp <- sample(idx_val_grp, val_subsample_n)
    }

    X_val_grp <- if (is.null(X)) NULL else X[idx_val_grp, , drop = FALSE]
    stopifnot(length(idx_tr_grp) >= 1, length(idx_val_grp) >= 1)

    Q_val_aug   <- encode_y_to_Qi_aug(pipeline, y_list[idx_val_grp], p_grid_aug)
    Q_train_aug <- encode_y_to_Qi_aug(pipeline, y_list[idx_tr_grp], p_grid_aug)

    T_tv <- compute_generativity_score(Q_train_aug, Q_val_aug, w_aug, cost_type)

    rep_seeds <- sample.int(.Machine$integer.max, R)

    gen_one_rep <- function(r) {
      # A fresh multisession worker has no py_pkg (src/cot.R:13 creates it
      # only in the launching session). stage_flow pipelines need a
      # worker-local py_pkg tied to that worker's own Python interpreter
      # (decode_fun_flow, src/cot.R). Guarded on existence so a reused
      # worker only pays the re-init cost once, not once per task.
      if (!exists("py_pkg", envir = .GlobalEnv, inherits = FALSE)) {
        setwd(.repo_root)
        source(file.path(.repo_root, "src", "cot.R"))
      }
      set.seed(rep_seeds[r])
      z_draws     <- sample_fun(length(idx_val_grp), fit, X_val_grp)
      Q_synth_aug <- decode_latent_to_Qi_aug(pipeline, z_draws, p_grid_aug, p_grid)
      T_sv_r      <- compute_generativity_score(Q_synth_aug, Q_val_aug, w_aug, cost_type)
      if (keep_draws) list(T_sv = T_sv_r, z = do.call(rbind, z_draws)) else T_sv_r
    }

    .repo_root <- getwd()

    if (n_cores > 1) {
      # Reuse a caller-established worker pool if one is already active
      # (a script can call future::plan(multisession, ...) once up front
      # to avoid repeated pool teardown/Python re-init across many calls);
      # otherwise self-manage a scoped pool for just this call.
      caller_has_pool <- !inherits(future::plan(), "sequential")
      if (!caller_has_pool) {
        old_plan <- future::plan(future::multisession, workers = n_cores)
        on.exit(future::plan(old_plan), add = TRUE)
      }
      # future.globals is explicit (not TRUE) so future's auto-detection
      # doesn't walk into decode_latent_to_Qi_aug -> decode -> decode_fun_flow
      # and try to export the parent's live py_pkg reticulate handle.
      # future.seed = NULL: each task already seeds itself deterministically
      # via rep_seeds[r], so future's own RNG-stream management is neither
      # needed nor safe to enable here -- future.seed = TRUE switches
      # workers to the L'Ecuyer-CMRG RNG kind, which makes
      # set.seed(rep_seeds[r]) draw a *different* stream than the plain
      # Mersenne-Twister default used by the n_cores <= 1 path below,
      # breaking numeric reproducibility across n_cores values. NULL (vs.
      # the FALSE default) also silences future.apply's "UNRELIABLE VALUE"
      # warning, which is a false positive given the manual per-task seed.
      rep_res <- future.apply::future_lapply(
        seq_len(R), gen_one_rep,
        future.seed    = NULL,
        future.globals = c("rep_seeds", "pipeline", "idx_val_grp", "fit",
                            "sample_fun", "X_val_grp", "Q_val_aug", "w_aug",
                            "p_grid_aug", "p_grid", ".repo_root", "keep_draws",
                            "cost_type")
      )
    } else {
      rep_res <- lapply(seq_len(R), gen_one_rep)
    }
    if (keep_draws) {
      T_sv    <- vapply(rep_res, `[[`, numeric(1), 'T_sv')
      z_draws <- lapply(rep_res, `[[`, 'z')
    } else {
      T_sv <- unlist(rep_res)
    }

    splits[[s]] <- list(
      T_tv      = T_tv,
      T_sv      = T_sv,
      log_norm  = log(T_sv / T_tv),
      n_tr_grp  = length(idx_tr_grp),
      n_val_grp = length(idx_val_grp)
    )
    if (keep_draws) {
      splits[[s]]$idx_tr_grp  <- idx_tr_grp
      splits[[s]]$idx_val_grp <- idx_val_grp
      splits[[s]]$z_draws     <- z_draws
    }
  }

  K_used <- ncol(encode_y_to_latent(pipeline, y_list[1]))
  list(
    splits = splits,
    meta = list(
      S = S, R = R,
      model            = if (aware) 'covariate-aware' else 'covariate-blind',
      reference        = if (conditional) 'conditional' else 'unconditional',
      n_ref            = sum(ref_mask),
      frac_train       = frac_train,
      pool_subsample_n = pool_subsample_n,
      val_subsample_n  = val_subsample_n,
      N_used           = if (is.null(pool_subsample_n)) length(y_list) else pool_subsample_n,
      K                = K_used,
      latent           = .latent_of(pipeline),
      cost_type        = cost_type,
      J_aug            = J_aug,
      keep_draws       = keep_draws
    )
  )
}


## Rebuild replicate r of split s from an evaluate_generativity() result run
## with keep_draws = TRUE: decodes that replicate's stored synthetic latents and
## re-encodes that split's validation QFs on the same augmented grid. Returns
## the same structure as generativity_example() (so plot_generativity_example()
## applies), with T_tv / T_sv / log_norm taken from the stored result -- the
## plotted dataset is exactly the one behind that score. w2_synth is each
## synthetic QF's W2 distance to the mean QF of that split's training half.
## pipeline and y_list must be the ones passed to evaluate_generativity().
generativity_replicate <- function(res, pipeline, y_list, s, r) {
  if (!isTRUE(res$meta$keep_draws)) {
    stop("Result has no stored draws; rerun evaluate_generativity with keep_draws = TRUE.")
  }
  sp <- res$splits[[s]]
  stopifnot(r >= 1, r <= length(sp$z_draws))

  p_grid     <- pipeline$training$cache$p_grid
  J_aug      <- res$meta$J_aug
  p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
  w_aug      <- get_quadrature_weights(p_grid_aug)

  z_mat       <- sp$z_draws[[r]]
  z_list      <- lapply(seq_len(nrow(z_mat)), function(i) z_mat[i, ])
  Q_synth_aug <- decode_latent_to_Qi_aug(pipeline, z_list, p_grid_aug, p_grid)
  Q_val_aug   <- encode_y_to_Qi_aug(pipeline, y_list[sp$idx_val_grp], p_grid_aug)
  Q_train_aug <- encode_y_to_Qi_aug(pipeline, y_list[sp$idx_tr_grp],  p_grid_aug)

  Q_train_mean <- colMeans(do.call(rbind, Q_train_aug))
  w2_synth <- sqrt(as.numeric(
    sweep(do.call(rbind, Q_synth_aug), 2, Q_train_mean)^2 %*% w_aug
  ))

  list(
    p_grid   = p_grid_aug,
    Q_val    = Q_val_aug,
    Q_synth  = Q_synth_aug,
    T_tv     = sp$T_tv,
    T_sv     = sp$T_sv[r],
    log_norm = sp$log_norm[r],
    idx_val  = sp$idx_val_grp,
    z_draws  = z_mat,
    w2_synth = w2_synth,
    split    = s,
    rep      = r
  )
}


## One worked example of evaluate_generativity's procedure, keeping the data:
## a single random train/val split (frac_train), latent model fit on the full
## training half, the reference set's validation QFs (ref_mask; NULL = all;
## optionally subsampled to val_subsample_n) vs one synthetic draw of the same
## size, both on the augmented p-grid. Same steps, scores and generativity type
## (fit_fun / sample_fun NULL = covariate-blind, ref_mask NULL = unconditional)
## as one (split, replicate) of evaluate_generativity; seed makes the example
## reproducible. Returns the grid,
## the real (val) and synthetic QF lists, T_tv, T_sv and
## log_norm = log(T_sv / T_tv), plus idx_val (indices into y_list of the
## validation QFs), z_draws (n x K matrix of the synthetic latents, in the
## space decode_latent_to_Qi_aug consumes) and w2_synth (each synthetic QF's
## W2 distance to the mean training-half QF, for flagging outlying draws).
generativity_example <- function(
  pipeline, y_list,
  X               = NULL,
  fit_fun         = NULL,
  sample_fun      = NULL,
  ref_mask        = NULL,
  cost_type       = 'wasserstein',
  frac_train      = 0.5,
  val_subsample_n = NULL,
  J_aug           = 500,
  ridge           = 0,
  seed            = 12345
) {
  ## Generativity type, as in evaluate_generativity
  if (!is.null(fit_fun) || !is.null(sample_fun)) {
    stopifnot(is.function(fit_fun), is.function(sample_fun),
              !is.null(X), NROW(X) == length(y_list))
  } else {
    fit_fun    <- function(Z, X) fit_mean_cov(Z, ridge = ridge)
    sample_fun <- sample_mean_cov
    X          <- NULL
  }
  if (is.null(ref_mask)) ref_mask <- rep(TRUE, length(y_list))
  stopifnot(is.logical(ref_mask), length(ref_mask) == length(y_list),
            !anyNA(ref_mask))
  set.seed(seed)

  p_grid     <- pipeline$training$cache$p_grid
  p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
  w_aug      <- get_quadrature_weights(p_grid_aug)

  ## Split, then fit on the full training half
  n_pool  <- length(y_list)
  n_train <- floor(frac_train * n_pool)
  perm    <- sample(seq_len(n_pool))
  idx_tr  <- perm[seq_len(n_train)]
  idx_val <- perm[(n_train + 1):n_pool]
  Z_tr <- encode_y_to_latent(pipeline, y_list[idx_tr])
  X_tr <- if (is.null(X)) NULL else X[idx_tr, , drop = FALSE]
  fit  <- fit_fun(Z_tr, X_tr)

  ## Restrict to the reference set; optional validation subsample
  idx_tr_grp  <- idx_tr[ref_mask[idx_tr]]
  idx_val_grp <- idx_val[ref_mask[idx_val]]
  if (!is.null(val_subsample_n)) {
    stopifnot(val_subsample_n <= length(idx_val_grp))
    idx_val_grp <- sample(idx_val_grp, val_subsample_n)
  }
  X_val_grp <- if (is.null(X)) NULL else X[idx_val_grp, , drop = FALSE]

  ## Real (val), held-out real (train) and synthetic QFs; scores
  Q_val_aug   <- encode_y_to_Qi_aug(pipeline, y_list[idx_val_grp], p_grid_aug)
  Q_train_aug <- encode_y_to_Qi_aug(pipeline, y_list[idx_tr_grp], p_grid_aug)
  T_tv        <- compute_generativity_score(Q_train_aug, Q_val_aug, w_aug, cost_type)
  z_draws     <- sample_fun(length(idx_val_grp), fit, X_val_grp)
  Q_synth_aug <- decode_latent_to_Qi_aug(pipeline, z_draws, p_grid_aug, p_grid)
  T_sv        <- compute_generativity_score(Q_synth_aug, Q_val_aug, w_aug, cost_type)

  ## W2 distance of each synthetic QF to the mean training-half QF
  Q_train_mean <- colMeans(do.call(rbind, Q_train_aug))
  w2_synth <- sqrt(as.numeric(
    sweep(do.call(rbind, Q_synth_aug), 2, Q_train_mean)^2 %*% w_aug
  ))

  list(
    p_grid   = p_grid_aug,
    Q_val    = Q_val_aug,
    Q_synth  = Q_synth_aug,
    T_tv     = T_tv,
    T_sv     = T_sv,
    log_norm = log(T_sv / T_tv),
    idx_val  = idx_val_grp,
    z_draws  = do.call(rbind, z_draws),
    w2_synth = w2_synth
  )
}

## Plot a generativity_example(): real (validation) QFs and synthetic QFs
## overlaid in random order (so neither set systematically hides the other),
## with a legend placed inside the axes clear of the curves (.legend_layout).
## Draws to the active device; main may be an expression (e.g. built with
## bquote for a tilde(G) title).
##
## outlier_w2: if non-NULL, synthetic QFs whose W2 distance to the mean
## training QF (ex$w2_synth) is >= outlier_w2 are drawn in col_outlier, opaque
## and on top of everything else, and the legend gains an entry with their
## count.
##
## show_legend = FALSE omits the legend (and the y headroom made for it), e.g.
## for small panels of a combined figure whose caption explains the colors.
plot_generativity_example <- function(ex, main = "",
                                      show_legend = TRUE,
                                      col_real  = "black",
                                      col_synth = "#1B9E77",
                                      alpha     = 0.25,
                                      lwd       = 0.25,
                                      lwd_leg   = 1.5,
                                      legend_cex = 1,
                                      legend_loc = "topleft",
                                      labels = c("real (val)", "synthetic"),
                                      outlier_w2    = NULL,
                                      col_outlier   = "#D95F02",
                                      lwd_outlier   = 0.75,
                                      label_outlier = "synthetic outlier") {
  x     <- ex$p_grid
  funs  <- c(ex$Q_val, ex$Q_synth)
  is_out <- c(rep(FALSE, length(ex$Q_val)),
              if (is.null(outlier_w2)) rep(FALSE, length(ex$Q_synth))
              else ex$w2_synth >= outlier_w2)
  cols  <- c(rep(adjustcolor(col_real,  alpha), length(ex$Q_val)),
             rep(adjustcolor(col_synth, alpha), length(ex$Q_synth)))
  ord   <- sample(length(funs))
  ord   <- ord[!is_out[ord]]
  xlim  <- range(x)
  ylim  <- range(unlist(funs), finite = TRUE)
  leg_args <- list(legend = labels, col = c(col_real, col_synth),
                   lwd = lwd_leg, lty = 1, cex = legend_cex, bty = "n")
  if (!is.null(outlier_w2)) {
    leg_args$legend <- c(leg_args$legend,
                         sprintf("%s (n = %d)", label_outlier, sum(is_out)))
    leg_args$col    <- c(leg_args$col, col_outlier)
  }
  objs <- lapply(funs, function(y) list(x = x, lo = y, hi = y))

  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "p", "Q(p)"))
  on.exit(par(old_mar), add = TRUE)
  if (show_legend) {
    lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc)
  } else {
    plot.new()
    lay <- list(ylim = ylim)
  }
  .draw_frame(xlim, lay$ylim, main = main, xlab = "p", ylab = "Q(p)",
              ylim_data = ylim)
  for (i in ord) lines(x, funs[[i]], col = cols[i], lwd = lwd)
  for (i in which(is_out)) lines(x, funs[[i]], col = col_outlier, lwd = lwd_outlier)
  if (show_legend) do.call(legend, c(list(lay$pos), leg_args))
  invisible(NULL)
}


## Generativity of one latent model (fit_fun / sample_fun with covariates X;
## all NULL = covariate-blind) against several reference sets: unconditional
## (all subjects) and conditional on each level of strata. strata is a vector
## or factor with one entry per element of y_list and any number of levels
## (factor levels set the order, otherwise sorted unique values). Every call
## uses the same seed, so all reference sets share splits and per-split model
## fits. Returns
##   gen:    evaluate_generativity results, named "All" and by level
##   n:      sample sizes in the data passed in, same names
##   levels: the levels in order, and is_factor (for plot labels)
## Draw with plot_generativity_strata().
evaluate_generativity_strata <- function(pipeline, y_list, strata,
                                         X = NULL, fit_fun = NULL,
                                         sample_fun = NULL, S, R,
                                         val_subsample_n = NULL,
                                         seed = 12345, n_cores = 1, ...) {
  stopifnot(length(strata) == length(y_list), !anyNA(strata))
  lv <- if (is.factor(strata)) levels(droplevels(strata)) else sort(unique(strata))
  lv_chr <- as.character(lv)

  run <- function(ref_mask, what) {
    start <- Sys.time()
    res <- evaluate_generativity(
      pipeline, y_list,
      X = X, fit_fun = fit_fun, sample_fun = sample_fun,
      ref_mask = ref_mask,
      S = S, R = R,
      val_subsample_n = val_subsample_n,
      seed = seed, n_cores = n_cores, ...
    )
    message(sprintf('  generativity (%s): %.1f min', what,
                    as.numeric(difftime(Sys.time(), start, units = 'mins'))))
    res
  }
  gen <- c(
    list(All = run(NULL, 'unconditional')),
    setNames(lapply(lv, function(l) run(strata == l, paste('stratum', l))),
             lv_chr)
  )
  n <- c(length(strata), vapply(lv, function(l) sum(strata == l), integer(1)))

  list(gen = gen, n = setNames(n, c("All", lv_chr)),
       levels = lv_chr, is_factor = is.factor(strata))
}

## Box plot of an evaluate_generativity_strata() result on the active device:
## "All" (unconditional), a dotted separator, then one box per stratum
## (conditional), each labeled with its sample size on a second line. labels
## maps each level (as character) to its display name. The subscript of a
## stratum's N is its value (e.g. N[0], N[1] for a binary covariate), or its
## level index for a factor. ... goes to plot_generativity_boxes.
plot_generativity_strata <- function(res, labels,
                                     ylab = "Relative generativity",
                                     main = "", ...) {
  lv <- res$levels
  stopifnot(all(lv %in% names(labels)))
  n <- res$n
  sub_code <- if (res$is_factor) seq_along(lv) else lv
  sublabels <- c(
    list(bquote("(" * N == .(n[[1]]) * ")")),
    lapply(seq_along(lv), function(j)
      bquote("(" * N[.(sub_code[j])] == .(n[[j + 1]]) * ")"))
  )
  plot_generativity_boxes(
    groups          = c("All", unname(labels[lv])),
    scores          = lapply(res$gen, function(g) lapply(g$splits, `[[`, "log_norm")),
    group_sublabels = sublabels,
    vsep            = 1.5,
    hline           = 0,
    ylab            = ylab,
    main            = main,
    collapse        = TRUE,
    ...
  )
}


## ---------- Plotting ---------- ##

empty_plot <- function() {
  plot(
    NULL, xlab = "", ylab = "",
    xaxt = "n", yaxt = "n",
    xlim = c(0, 10),
    ylim = c(0, 10),
    bty = "n"
  )
}

## Shade an uncertainty band between lo and hi over a sorted 1-D grid x. Draw it
## right after plot() so curves and reference lines sit on top. The fill is
## translucent so overlapping bands (e.g. two groups) remain distinguishable.
draw_band <- function(x, lo, hi, col = adjustcolor("gray50", alpha.f = 0.25)) {
  ok <- is.finite(x) & is.finite(lo) & is.finite(hi)
  polygon(c(x[ok], rev(x[ok])), c(lo[ok], rev(hi[ok])),
          col = col, border = NA)
}


## Sets text sizes and margins for enlarged titles / axis labels, growing the
## margins with the label sizes so text isn't clipped. cex (if non-NULL)
## overrides the base text size -- e.g. cex = 1 undoes mfrow's automatic
## shrink so the png pointsize is the true base size in multi-panel layouts.
## cex_axis (if non-NULL) sets the tick-label size.
## Returns the previous par values (for on.exit restore).
fig_par <- function(cex_main = 1.2, cex_lab = 1, cex = NULL, cex_axis = NULL) {
  lab_mar <- 1.5 + 1.5 * cex_lab
  args <- list(cex.main = cex_main, cex.lab = cex_lab,
               mar = c(lab_mar, lab_mar, 1 + cex_main, 1),
               mgp = c(0.6 + cex_lab, 0.5, 0))
  if (!is.null(cex)) args <- c(list(cex = cex), args)
  if (!is.null(cex_axis)) args$cex.axis <- cex_axis
  do.call(par, args)
}


## Horizontal color bar in the top-left corner of the current plot: the colors
## in pal span lims left to right, with tick labels at ticks below the bar and
## an optional title above it. Sizes and insets are in inches, so the bar looks
## the same regardless of the axis scales.
draw_colorbar <- function(pal, lims, ticks, title = NULL, cex = 1,
                          width_in = 1.2, height_in = 0.1,
                          inset_in = c(0.15, 0.12)) {
  usr      <- par("usr")
  x_per_in <- diff(usr[1:2]) / par("pin")[1]
  y_per_in <- diff(usr[3:4]) / par("pin")[2]
  title_in <- if (is.null(title)) 0 else
    strheight(title, units = "inches", cex = cex) + 0.05

  x0 <- usr[1] + inset_in[1] * x_per_in
  x1 <- x0 + width_in * x_per_in
  y1 <- usr[4] - (inset_in[2] + title_in) * y_per_in
  y0 <- y1 - height_in * y_per_in

  n  <- length(pal)
  xs <- seq(x0, x1, length.out = n + 1)
  rect(xs[-(n + 1)], y0, xs[-1], y1, col = pal, border = NA)
  rect(x0, y0, x1, y1, col = NA, border = "black", lwd = 0.5)

  tx <- x0 + (ticks - lims[1]) / diff(lims) * (x1 - x0)
  segments(tx, y0, tx, y0 - 0.04 * y_per_in, lwd = 0.5)
  text(tx, y0 - 0.06 * y_per_in, labels = ticks, adj = c(0.5, 1), cex = cex)
  if (!is.null(title)) {
    text(x0, y1 + 0.05 * y_per_in, labels = title, adj = c(0, 0), cex = cex)
  }
  invisible(NULL)
}


plot_funs <- function(
    fun_list,
    grid_list,
    ylim = NULL,
    xlab = 'p',
    ylab = 'Q(p)',
    cex.lab = 1.0,
    colors = NULL,
    widths = NULL,
    types = NULL,
    color_width_type_labels = NULL,
    legend_loc = 'topleft',
    main = '',
    path = NULL,
    fig_width = 800,
    fig_height = 600,
    bands = NULL,
    cex_main = NULL,
    cex = NULL,
    cex_axis = NULL,
    legend_cex = 1,
    after_plot = NULL
) {
  ## cex_main / cex / cex_axis: if any is set, text and margins follow fig_par().
  ## bands: optional list of list(x, lo, hi), shaded beneath the curves.
  ## after_plot: optional function of no arguments, called after the curves are
  ## drawn (in the plot's coordinates and par), e.g. to add a draw_colorbar().
  if (!is.null(path)) {
    png(path, width = fig_width, height = fig_height)
  }
  
  n_funs <- length(fun_list)
  
  ## ---- Defaults & validation ---- ##
  if (!all(lengths(grid_list) == lengths(fun_list))) {
    stop("Lengths of grid_list must match lengths of fun_list.")
  }
  
  if (is.null(colors)) {
    colors <- rep("black", n_funs)
  } else if (length(colors) != n_funs) {
    stop("Length of colors must match length of fun_list.")
  }
  
  if (is.null(widths)) {
    widths <- rep(1, n_funs)
  } else if (length(widths) != n_funs) {
    stop("Length of widths must match length of fun_list.")
  }
  
  if (is.null(types)) {
    types <- rep(1, n_funs)
  } else if (length(types) != n_funs) {
    stop("Length of types must match length of fun_list.")
  }
  
  ## ---- Plot ---- ##
  xlim <- c(
    min(unlist(grid_list)),
    max(unlist(grid_list))
  )
  if (is.null(ylim)) {
    band_vals <- unlist(lapply(bands, `[`, c("lo", "hi")))
    ylim <- range(c(unlist(fun_list), band_vals), finite = TRUE)
  }
  par(mfrow = c(1, 1))
  if (!is.null(cex_main) || !is.null(cex) || !is.null(cex_axis)) {
    old_par <- fig_par(cex_main %||% 1.2, cex.lab, cex, cex_axis)
    on.exit(par(old_par), add = TRUE)
    if (main == '') par(mar = replace(par("mar"), 3, 1))
  }
  plot(
    NULL,
    xlim = xlim,
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    cex.lab = cex.lab,
    main = main
  )
  for (b in bands) draw_band(b$x, b$lo, b$hi)

  for (i in seq_len(n_funs)) {
    lines(
      grid_list[[i]],
      fun_list[[i]],
      col = colors[i],
      lwd = widths[i],
      lty = types[i]
    )
  }

  if (!is.null(after_plot)) after_plot()

  ## ---- Legend ---- ##
  if (!is.null(color_width_type_labels)) {
    
    required_cols <- c("color", "width", "type", "label")
    if (!all(required_cols %in% names(color_width_type_labels))) {
      stop("color_width_type_labels must have columns: color, width, type, label.")
    }
    
    legend(
      legend_loc,
      legend = color_width_type_labels$label,
      col    = color_width_type_labels$color,
      lwd    = color_width_type_labels$width,
      lty    = color_width_type_labels$type,
      cex    = legend_cex,
      bty    = "n"
    )
  }
  
  if (!is.null(path)) dev.off()
}


plot_embeddings <- function(
    emb,
    colors = NULL,
    shapes = NULL,
    sizes = NULL,
    color_shape_size_labels = NULL,
    stats = TRUE,
    path = NULL,
    legend_cex = 1,
    legend_ncol = 1,
    cex_main = NULL,
    cex_lab = NULL,
    cex = NULL,
    cex_axis = NULL,
    ks = NULL
) {
  ## cex_main / cex_lab / cex / cex_axis: if any is set, text and margins
  ## follow fig_par().
  ## ks: component numbers of emb's columns, used in the panel titles and axis
  ## labels (e.g. c(1, 3, 5) when plotting a subset); default 1:ncol(emb).
  if (!is.null(path)) {
    png(path, width = 800, height = 600)
  }

  K <- ncol(emb)
  N <- nrow(emb)
  ks <- ks %||% seq_len(K)
  stopifnot(length(ks) == K)

  use_legend <- !is.null(color_shape_size_labels)
  
  ## ---- Color handling ---- ##
  if (is.null(colors)) {
    point_cols <- rep("black", N)
  } else {
    if (length(colors) != N) {
      stop("Length of colors must match number of rows in emb.")
    }
    point_cols <- colors
  }
  
  ## ---- Shape handling ---- ##
  if (is.null(shapes)) {
    point_shapes <- rep(19, N)
  } else {
    if (length(shapes) != N) {
      stop("Length of shapes must match number of rows in emb.")
    }
    point_shapes <- shapes
  }
  
  ## ---- Size handling ---- ##
  if (is.null(sizes)) {
    point_sizes <- rep(1, N)
  } else {
    if (length(sizes) != N) {
      stop("Length of sizes must match number of rows in emb.")
    }
    point_sizes <- sizes
  }
  
  ## ---- Legend validation ---- ##
  if (use_legend) {
    if (is.null(color_shape_size_labels)) {
      stop("color_shape_size_labels must be provided when colors is not NULL.")
    }
    
    required_cols <- c("color", "shape", "size", "label")
    if (!all(required_cols %in% colnames(color_shape_size_labels))) {
      stop("color_shape_size_labels must have columns: color, shape, size, label.")
    }
  }
  
  par(mfrow = c(K, K))
  if (!is.null(cex_main) || !is.null(cex_lab) || !is.null(cex) ||
      !is.null(cex_axis)) {
    old_par <- fig_par(cex_main %||% 1.2, cex_lab %||% 1, cex, cex_axis)
    on.exit(par(old_par), add = TRUE)
  }

  for (k1 in 1:K) {
    for (k2 in 1:K) {
      
      if (k1 == k2) {
        ## ---- QQ plot ---- ##
        if (stats) {
          pval <- round(shapiro.test(emb[, k1])$p.value, 3)
          main_txt <- paste("k =", ks[k1], "| p =", pval)
        } else {
          main_txt <- paste("k =", ks[k1])
        }
        
        qqnorm(
          emb[, k1],
          main = main_txt,
          col  = point_cols,
          pch  = point_shapes,
          cex  = point_sizes
        )
        qqline(emb[, k1])
        
      } else if (k1 > k2) {
        ## ---- Scatter plot ---- ##
        if (stats) {
          r <- round(cor(emb[, k1], emb[, k2]), 3)
          main_txt <- paste("r =", r)
        } else {
          main_txt <- ""
        }
        
        plot(
          emb[, k1], emb[, k2],
          main = main_txt,
          xlab = paste("k =", ks[k1]),
          ylab = paste("k =", ks[k2]),
          col  = point_cols,
          pch  = point_shapes,
          cex  = point_sizes
        )
        
      } else {
        empty_plot()
      }
      
      ## ---- Legend (draw once) ---- ##
      if (use_legend && k1 == 1 && k2 == 1) {
        legend(
          "topleft",
          legend = color_shape_size_labels$label,
          col    = color_shape_size_labels$color,
          pch    = color_shape_size_labels$shape,
          pt.cex = color_shape_size_labels$size,
          cex    = legend_cex,
          ncol   = legend_ncol,
          bty    = "n"
        )
      }
    }
  }
  if (!is.null(path)) dev.off()
}


## Reusable: produces an n_row x n_col grid of QF-reconstruction panels.
## Each cell (i, j) shows the subject whose reconstruction loss sits at a
## randomly sampled quantile p ~ U(lower_mat[i, j], upper_mat[i, j]) of the
## empirical loss distribution -- so the figure stratifies subjects across
## the loss CDF. Call set.seed() externally for reproducibility.
##
## Args:
##   Qi_orig_list   length-N list of original Qi vectors (e.g. Qi_ctx$payload)
##   Qi_recon_list  length-N list of reconstructed Qi vectors (e.g. Qi_ctx_$payload)
##   recon_losses   length-N numeric of per-subject reconstruction losses
##   lower_mat      n_row x n_col numeric matrix of interval lower bounds
##   upper_mat      n_row x n_col numeric matrix of interval upper bounds
##   log_x_plus_1   if TRUE, plot log(Q(p) + 1) instead of Q(p)
##   path           optional output png path; if NULL, draws to active device
##   res            png resolution (ppi); layout is fixed in inches, so higher
##                  res gives more pixels without changing relative sizes
##   cex_main       magnification of panel titles
##   cex_lab        magnification of x/y axis labels (margins scale to fit)
##   cex            base text size; NULL keeps mfrow's automatic shrink
##   cex_axis       magnification of tick labels; NULL keeps the default
##   share_ylim     TRUE: one y-range (incl. 0) across all panels, from the
##                  subjects shown; numeric length-2: use as ylim for every
##                  panel; FALSE: per-panel ylim
plot_qi_recon_grid <- function(Qi_orig_list, Qi_recon_list, recon_losses,
                                lower_mat, upper_mat,
                                log_x_plus_1 = FALSE,
                                path = NULL,
                                res = 72,
                                cex_main = 1,
                                cex_lab = 1,
                                cex = NULL,
                                cex_axis = NULL,
                                share_ylim = FALSE) {
  stopifnot(identical(dim(lower_mat), dim(upper_mat)))
  stopifnot(all(lower_mat <  upper_mat),
            all(lower_mat >= 0), all(upper_mat <= 1))
  n_row <- nrow(lower_mat)
  n_col <- ncol(lower_mat)
  N_    <- length(recon_losses)
  recon_losses_sort <- sort(recon_losses)

  if (!is.null(path)) png(path, width = 320 * n_col / 72, height = 280 * n_row / 72,
                          units = "in", res = res)
  old_par <- c(par(mfrow = c(n_row, n_col)), fig_par(cex_main, cex_lab, cex, cex_axis))
  on.exit(par(old_par), add = TRUE)

  ## First pass: draw each panel's loss quantile and subject (same RNG order
  ## as before), so a shared y-range can be computed over the shown subjects.
  disp <- if (log_x_plus_1) function(q) log(q + 1) else identity
  ylab <- if (log_x_plus_1) "log(Q(p) + 1)" else "Q(p)"
  p_mat   <- matrix(NA_real_,    n_row, n_col)
  idx_mat <- matrix(NA_integer_, n_row, n_col)
  for (i in seq_len(n_row)) {
    for (j in seq_len(n_col)) {
      p           <- runif(1, lower_mat[i, j], upper_mat[i, j])
      idx_sort    <- round(1 + (N_ - 1) * p)
      q_value     <- recon_losses_sort[idx_sort]
      p_mat[i, j]   <- p
      idx_mat[i, j] <- which(recon_losses == q_value)[1]   # break ties safely
    }
  }

  ylim_shared <- if (is.numeric(share_ylim)) share_ylim else if (isTRUE(share_ylim)) {
    range(c(0, unlist(lapply(idx_mat, function(k)
      c(disp(Qi_orig_list[[k]]), disp(Qi_recon_list[[k]]))))), finite = TRUE)
  } else NULL

  for (i in seq_len(n_row)) {
    for (j in seq_len(n_col)) {
      p           <- p_mat[i, j]
      idx_payload <- idx_mat[i, j]

      Qi_disp       <- disp(Qi_orig_list[[idx_payload]])
      Qi_recon_disp <- disp(Qi_recon_list[[idx_payload]])
      pg            <- pi_grid_fun(length(Qi_disp))

      ylim <- if (is.null(ylim_shared)) {
        range(c(0, Qi_disp, Qi_recon_disp), finite = TRUE)
      } else ylim_shared

      plot(pg, Qi_disp, type = "l",
           xlab = "p", ylab = ylab,
           ylim = ylim,
           main = bquote(tau == .(sprintf("%.4f", p))),   # loss quantile
           # main = str_glue("{lower_mat[i, j]*100}%-{upper_mat[i, j]*100}%tile, p={round(p,3)}"),
           cex.main = cex_main
      )
      lines(pg, Qi_recon_disp, col = rgb(0, 1, 0, alpha = 1.0))
    }
  }
  if (!is.null(path)) dev.off()
  invisible(NULL)
}


## Group-wise box plot of generativity scores with optional right-axis K line.
## Designed for two display modes:
##   - lambda-sweep:   groups = as.character(lambdas), Ks supplied, vline at lambda_star
##   - dataset compare: groups = c("NHANES", "CHOP"), Ks = NULL, vline_group = NULL
##   - taxonomy compare: groups = c("All", stratum names), vsep = 1.5 and
##                       group_sublabels with the sample sizes
plot_generativity_boxes <- function(groups,
                                     scores,
                                     Ks          = NULL,
                                     hline       = 0,
                                     vline_group = NULL,
                                     vsep        = NULL,
                                     group_sublabels = NULL,
                                     align_columns = FALSE,
                                     xlab        = "",
                                     ylab        = "relative generativity",
                                     main        = "",
                                     path        = NULL,
                                     width       = 3.25,
                                     height      = 2.15,
                                     pointsize   = 7,
                                     res         = 600,
                                     collapse    = FALSE,
                                     mar         = NULL,
                                     medlwd      = 1.5,
                                     whisklty    = 1,
                                     outcex      = 0.4,
                                     outlwd      = 0.5,
                                     highlight     = NULL,
                                     highlight_col = "#7570B3") {
  ## medlwd / whisklty: median line width and whisker line type.
  ## outcex / outlwd: outlier point size and line width (small so the many
  ## outliers of a large R don't dominate the boxes).
  ## highlight: optional data frame of individual scores to mark on the boxes,
  ## with columns group (name in groups, or index), value, and optionally label
  ## (drawn to the right of the box, nudged apart so labels don't overlap) and
  ## split (box within the group when collapse = FALSE).
  ## path = NULL draws to the caller's device (e.g. a fig_png() paper figure
  ## with fig_par() set). With a path, opens a png at the paper figure spec of
  ## demo_nhanes.R: width / height in inches at res dpi, pointsize the printed
  ## text size, and fig_par()'s baseline title / label / tick sizes.
  ## The default margins follow the current par: trimmed for an omitted title /
  ## axis label (.trimmed_mar), with room for the K axis when Ks is given. mar
  ## overrides.
  ## vsep: x positions of dotted gray separator lines between groups (e.g. 1.5
  ## to set the first box apart); unlike vline_group, not tied to a group.
  ## group_sublabels: optional second line of x tick labels under groups
  ## (character or expression vector / list of length L, e.g. sample sizes
  ## with plotmath subscripts); the bottom margin grows to fit it.
  ## align_columns: center group g at the center of the g-th of L equal-width
  ## columns of the whole figure (margins included), so panels 1 / L of the
  ## figure width placed below it in LaTeX line up with the boxes.
  L <- length(groups)
  stopifnot(length(scores) == L)
  if (!is.null(Ks)) stopifnot(length(Ks) == L)
  if (!is.null(group_sublabels)) stopifnot(length(group_sublabels) == L)

  ## Flatten per-group values to set y-limits and per-split box positions.
  S <- length(scores[[1]])
  if (!all(vapply(scores, length, integer(1)) == S)) {
    stop("Every group must have the same number of splits.")
  }

  ## Collapse pools each group's S splits into one vector, so every group
  ## renders as a single box instead of S side-by-side split-wise boxes.
  if (collapse) {
    scores <- lapply(scores, function(g) list(unlist(g, use.names = FALSE)))
    S <- 1
  }

  y_all  <- unlist(scores, use.names = FALSE)
  y_rng  <- range(c(y_all, hline), finite = TRUE)
  y_pad  <- 0.05 * diff(y_rng)
  y_lim  <- c(y_rng[1] - y_pad, y_rng[2] + y_pad)

  ## Within-group offsets so the S boxes per group sit side-by-side.
  half_span <- 0.25
  offsets   <- if (S == 1) 0 else seq(-half_span, half_span, length.out = S)
  box_w     <- if (S == 1) 0.55 else min(0.18, (2 * half_span) / (S - 1) * 0.8)

  if (!is.null(path)) {
    png(path, width = width, height = height, units = "in", res = res,
        pointsize = pointsize)
    fig_par(cex_main = 1.2, cex_lab = 1, cex = 1, cex_axis = 0.8)
  }
  if (is.null(mar)) {
    mar <- .trimmed_mar(par("mar"), main, xlab, ylab)
    ## Without an x label only the group names sit below the axis: fit them
    if (.is_blank(xlab)) mar[1] <- par("mgp")[2] + 1.2 * par("cex.axis") + 0.1
    if (!is.null(group_sublabels)) mar[1] <- mar[1] + 1.2 * par("cex.axis")
    if (!is.null(Ks)) mar[4] <- max(mar[4], par("mgp")[1] + 1)
  }
  old_par <- par(mar = mar)
  ## Restore only the caller's device; our own png is closed below
  if (is.null(path)) on.exit(par(old_par), add = TRUE)

  ## Column alignment: one x unit per figure column (fin / L inches), with
  ## x = 0.5 at the figure's left edge, so the plot region (inside the
  ## margins) spans the x range below exactly (xaxs = "i").
  if (align_columns) {
    in_per_x <- par("fin")[1] / L
    xlim <- c(0.5 + par("mai")[2] / in_per_x, L + 0.5 - par("mai")[4] / in_per_x)
    xaxs <- "i"
  } else {
    xlim <- c(0.5, L + 0.5)
    xaxs <- "r"
  }
  plot(NULL,
       xlim = xlim,
       ylim = y_lim,
       xaxt = "n", xaxs = xaxs,
       xlab = xlab, ylab = ylab, main = main)
  axis(1, at = seq_len(L), labels = groups)
  if (!is.null(group_sublabels)) {
    ## Second label line: an axis shifted down one tick-label height, with no
    ## line or ticks of its own
    axis(1, at = seq_len(L), labels = as.expression(group_sublabels),
         line = 1.2 * par("cex.axis"), tick = FALSE, lwd = 0)
  }

  abline(h = hline, col = "red", lty = 3, lwd = 1.2)
  if (!is.null(vsep)) abline(v = vsep, col = "gray40", lty = 3, lwd = 1)
  if (!is.null(vline_group)) {
    vx <- match(vline_group, groups)
    if (!is.na(vx)) abline(v = vx, col = "red", lty = 3, lwd = 1.2)
  }

  for (g in seq_len(L)) {
    for (s in seq_len(S)) {
      boxplot(scores[[g]][[s]],
              at      = g + offsets[s],
              boxwex  = box_w,
              add     = TRUE,
              axes    = FALSE,
              col     = "gray85",
              border  = "black",
              medlwd  = medlwd,
              whisklty = whisklty,
              outcex  = outcex,
              outlwd  = outlwd)
    }
  }

  if (!is.null(highlight)) {
    hl <- as.data.frame(highlight)
    gi <- if (is.character(hl$group)) match(hl$group, groups) else hl$group
    si <- if (!is.null(hl$split) && S > 1) hl$split else rep(1, nrow(hl))
    x  <- gi + offsets[si]
    if (!is.null(hl$label)) {
      x_lab   <- x + box_w / 2 + 0.05
      y_lab   <- hl$value
      min_gap <- 1.2 * strheight("0", cex = par("cex.axis"))
      for (xx in unique(x)) {
        o <- which(x == xx)
        o <- o[order(y_lab[o])]
        for (k in seq_along(o)[-1]) {
          y_lab[o[k]] <- max(y_lab[o[k]], y_lab[o[k - 1]] + min_gap)
        }
      }
      segments(x, hl$value, x_lab, y_lab, col = highlight_col, lwd = 0.5)
      text(x_lab, y_lab, hl$label, pos = 4, offset = 0.2,
           cex = par("cex.axis"), col = highlight_col, xpd = NA)
    }
    points(x, hl$value, pch = 21, bg = highlight_col, col = "black", cex = 1)
  }

  if (!is.null(Ks)) {
    K_rng  <- range(Ks)
    K_ylim <- if (diff(K_rng) == 0) K_rng + c(-1, 1) else K_rng + c(-0.5, 0.5)
    k_to_y <- function(k) {
      y_lim[1] + (k - K_ylim[1]) / diff(K_ylim) * diff(y_lim)
    }
    lines(seq_len(L),  k_to_y(Ks), col = "forestgreen", lwd = 2)
    points(seq_len(L), k_to_y(Ks), col = "forestgreen", pch = 19, cex = 1.2)
    k_ticks <- pretty(K_ylim)
    k_ticks <- k_ticks[k_ticks >= K_ylim[1] & k_ticks <= K_ylim[2]]
    axis(side = 4, at = k_to_y(k_ticks), labels = k_ticks,
         col = "forestgreen", col.axis = "forestgreen", las = 1)
    mtext("K", side = 4, line = par("mgp")[1], col = "forestgreen",
          cex = par("cex.lab"))
  }

  if (!is.null(path)) dev.off()
  invisible(NULL)
}


