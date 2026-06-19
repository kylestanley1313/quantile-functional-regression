library(reticulate)



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
  lqd_ae <- import("python.lqd_ae", convert = TRUE)
  lqd_hae <- import("python.lqd_hae", convert = TRUE)
  lqd_vae <- import("python.lqd_vae", convert = TRUE)
  importlib$reload(flow)
  importlib$reload(lqd_ae)
  importlib$reload(lqd_hae)
  importlib$reload(lqd_vae)
  importlib$reload(python)
  python
}


## ---------- P-Grid ---------- ##

pi_grid_fun <- function(J) {
  1:J / (1 + J)
}

p_grid_fun_2 <- function(breaks, interval_counts) {
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

p_grid_fun <- function(
    y_list, J,
    p_left = NULL, 
    p_right = NULL, 
    J_left = NULL, 
    J_right = NULL
) {
  Ji_max <- max(lengths(y_list))
  p_min <- 1 / (1 + Ji_max)
  p_max <- Ji_max / (1 + Ji_max)
  
  if (is.null(p_left) & is.null(p_right)) {  ## no tails
    p_grid <- seq(p_min, p_max, length.out = J)
  } else if (is.null(p_left)) {  ## right tail only
    if (is.null(J_right)) {
      stop("Must pass J_right to create right-tailed p-grid!")
    }
    p_grid <- c(
      seq(p_min, p_right, length.out = J - J_right),
      seq(p_right, p_max, length.out = J_right + 1)
    )
  } else if (is.null(p_right)) {  ## left tail only
    if (is.null(J_left)) {
      stop("Must pass J_left to create left-tailed p-grid!")
    }
    p_grid <- c(
      seq(p_min, p_left, length.out = J_left + 1),
      seq(p_left, p_max, length.out = J - J_left)
    )
  } else {  ## left and right tails
    if (is.null(J_left) || is.null(J_right)) {
      stop("Must pass J_left and J_right to create left- and right-tailed p-grid!")
    }
    p_grid <- c(
      seq(p_min, p_left, length.out = J_left + 1),
      seq(p_left, p_right, length.out = J - J_left - J_right),
      seq(p_right, p_max, length.out = J_right + 1)
    )
  }
  
  sort(unique(p_grid))
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

return_one <- function(...) {
  1
}

pairwise_distance <- function(
    Qi_list,
    loss_fun,
    pi_grid_list,
    p_grid_aug,
    supp_Y = NULL
) {
  N <- length(Qi_list)
  distances <- numeric(N * (N - 1) / 2)

  ## Place Qi on augmented p-grid
  for (i in 1:N) {
    Qi_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], pi_grid_list[[i]],
      p_grid_aug, supp_Y
    )
  }
  w_aug <- get_quadrature_weights(p_grid_aug)

  ## Compute pairwise distances
  idx <- 1
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      p_low  <- max(pi_grid_list[[i]][1], pi_grid_list[[j]][1])
      p_high <- min(tail(pi_grid_list[[i]], 1), tail(pi_grid_list[[j]], 1))
      mask   <- (p_grid_aug >= p_low) & (p_grid_aug <= p_high)
      w_ij   <- w_aug[mask] / sum(w_aug[mask])
      distances[idx] <- loss_fun(Qi_list[[i]][mask], Qi_list[[j]][mask], w_ij)
      idx <- idx + 1
    }
  }

  distances
}

quantile_pairwise_distance <- function(
    Qi_list,
    loss_fun,
    pi_grid_list,
    p_grid_aug,
    p_scale = 0.5,
    supp_Y = NULL
) {
  d <- pairwise_distance(Qi_list, loss_fun, pi_grid_list, p_grid_aug, supp_Y)
  unname(quantile(d, c(p_scale)))
}


loss_to_fun <- list(
  'one_minus_sqcor' = one_minus_sqcor,
  'one_minus_sqconc' = one_minus_sqconc,
  'wasserstein' = wasserstein
)


loss_scale_to_fun <- list(
  'none' = return_one,
  'quantile_pairwise_distance' = quantile_pairwise_distance
)


## ---------- Likelihood Computation ---------- ##

compute_likelihoods <- function(
    y,
    Q,
    p_grid,
    supp_Y = NULL,
    log = FALSE,
    min_lik = 1e-20
) {
  
  if(length(Q) != length(p_grid))
    stop("Q and p_grid must have same length")
  
  if(any(diff(p_grid) <= 0))
    stop("p_grid must be strictly increasing")
  
  y <- as.numeric(y)
  J <- length(Q)
  
  if(is.null(supp_Y)){
    
    slopes <- diff(Q) / diff(p_grid)
    idx <- findInterval(y, Q)
    
    valid <- (idx >= 1) & (idx <= J-1)
    
    lik <- numeric(length(y))
    lik[!valid] <- 0
    lik[valid] <- 1 / slopes[idx[valid]]
    
    lik[lik <= 0 | !is.finite(lik)] <- 0
    
  } else {
    
    supp_Y <- sort(unique(as.numeric(supp_Y)))
    
    # Only keep support inside Q range
    T_vals <- supp_Y[supp_Y > Q[1] & supp_Y <= Q[J]]
    
    if(length(T_vals) == 0){
      lik <- numeric(length(y))
    } else {
      
      # ----- Build inverse CDF via interpolation
      #   i.e. interpolate (Q_j , p_j)
      
      Q <- cummax(Q)
      p_at_T <- approx(
        x = Q,
        y = p_grid,
        xout = T_vals,
        method="linear",
        ties="ordered"
      )$y
      
      # Enforce monotone safety
      p_at_T[is.na(p_at_T)] <- 0
      
      # ----- Compute masses via finite differences
      
      mass <- numeric(length(T_vals))
      
      # Interior points
      mass[1:(length(T_vals)-1)] <- diff(p_at_T)
      
      # Right endpoint
      mass[length(T_vals)] <- p_grid[J] - p_at_T[length(T_vals)]
      
      # Left endpoint
      left_candidates <- supp_Y[supp_Y <= Q[1]]
      if(length(left_candidates)==0){
        T_min <- min(supp_Y)
      } else {
        T_min <- max(left_candidates)
        T_vals <- c(T_min, T_vals)
        mass <- c(p_at_T[1] - p_grid[1], mass)
      }
      
      # numerical guard
      mass[mass < 0] <- 0
      
      # normalize
      if (sum(mass) > 0) {
        mass <- mass / sum(mass)
      }
      
      # lookup
      lik <- mass[match(y, T_vals)]
      lik[is.na(lik)] <- 0
    }
  }
  
  ## Threshold
  lik[lik < min_lik] <- min_lik
  
  if(log){
    lik[lik <= 0] <- -Inf
    lik[lik > 0] <- log(lik[lik > 0])
  }
  
  as.numeric(lik)
}


## ---------- Encoding/Decoding ---------- ##

## Decode a list of latent z draws through the pipeline back to Qi (and the
## intermediate context payloads). Mirrors the encode chain in reverse.
decode_z_draws <- function(z_draws, pipeline, Ji = 1000) {

  draws <- list()

  ## Z
  z_draws_ctx <- new_context(
    payload = z_draws,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_draws)))
  )
  draws$z <- z_draws_ctx$payload

  ## C
  c_draws_ctx <- decode(pipeline, z_draws_ctx, from = 5, to = 4)
  draws$c <- c_draws_ctx$payload

  ## G
  G_Q_star_draws_ctx <- decode(pipeline, c_draws_ctx, from = 4, to = 3)
  draws$G_Q_star <- G_Q_star_draws_ctx$payload

  ## Q
  Q_draws_ctx <- decode(pipeline, G_Q_star_draws_ctx, from = 3, to = 2)
  draws$Q <- Q_draws_ctx$payload

  ## Qi
  Qi_draws_ctx <- decode(pipeline, Q_draws_ctx, from = 2, to = 1)
  draws$Qi <- Qi_draws_ctx$payload

  draws
}

decode_z_to_Qi <- function(z_list, Ji) {
  z_ctx <- new_context(
    payload = z_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_list)))
  )
  decode(pipeline, z_ctx, from = 5, to = 1)$payload
}

z_to_Qi_aug <- function(pipeline, z_list, p_grid_aug, p_grid) {
  N <- length(z_list)
  draws <- decode_z_draws(z_list, pipeline)

  Qi_list <- draws$Q
  supp_Y <- pipeline$training$cache$supp_Y
  for (i in seq_len(N)) {
    Qi_list[[i]] <- inv_eqf_cgrid(
      Qi_list[[i]], p_grid,
      p_grid_aug, supp_Y
    )
  }
  Qi_list
}

decode_z_rot_to_Qi <- function(z_rot_list, Ji) {
  z_rot_ctx <- new_context(
    payload = z_rot_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_rot_list)))
  )
  decode(pipeline, z_rot_ctx, from = 6, to = 1)$payload
}


## Encode y_list to the rotated latent space (z) and stack into an N x K matrix.
encode_to_Z <- function(pipeline, y_list) {
  y_ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  z_ctx <- encode(pipeline, y_ctx, from = 0, to = 5)
  do.call(rbind, z_ctx$payload)
}

## Encode y_list to Qi on the augmented p-grid.
y_to_Qi_aug <- function(pipeline, y_list, p_grid_aug) {
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


## ---------- Modeling ---------- ##

## Frequentist mean-covariance fit on the latent matrix Z.
fit_mean_cov <- function(Z, ridge = 0) {
  K <- ncol(Z)
  list(
    mu    = colMeans(Z),
    Sigma = cov(Z) + ridge * diag(K)
  )
}


## ---------- Generativity ---------- ##

## Compute the (per-real-observation) generativity costs and scalar score
## sqrt(sum(g)) by solving balanced OT between two empirical measures on a
## shared grid. Returns the cost vector, scalar score, and the raw flow plan.
compute_generativity <- function(
    Q_synth_list,
    Q_real_list,
    w
) {
  N_synth <- length(Q_synth_list)
  N <- length(Q_real_list)

  ## N_synth x N cost matrix of squared Wasserstein distances
  M1 <- do.call(rbind, Q_synth_list)
  M2 <- do.call(rbind, Q_real_list)
  M1w <- sweep(M1, 2, w, '*')
  sq1 <- rowSums(M1w * M1)
  M2w <- sweep(M2, 2, w, '*')
  sq2 <- rowSums(M2w * M2)
  cross <- M1w %*% t(M2)
  cost_mat <- outer(sq1, sq2, '+') - 2 * cross
  cost_mat <- pmax(cost_mat, 0)  # guard tiny negative roundoff

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
generativity_score <- function(Q_a, Q_b, w) {
  compute_generativity(Q_a, Q_b, w)$score
}


## Draw n iid samples from N(fit$mu, fit$Sigma), returned as a length-n list of
## K-vectors (matches the input shape decode_z_draws expects).
draw_mean_cov <- function(n, fit) {
  K <- length(fit$mu)
  draws <- MASS::mvrnorm(n, mu = fit$mu, Sigma = fit$Sigma)
  if (n == 1) draws <- matrix(draws, nrow = 1)
  asplit(draws, MARGIN = 1) |> lapply(as.numeric)
}

## Data-split normalized generativity assessment.
##
## For each of S splits: partition y_list into train/val (optionally
## subsampling to size subsample_n first), compute the baseline real-vs-val
## score T_tv from the held-out real training data, fit a mean-covariance
## latent model on the training half, then for each of R replicates draw
## synthetic z's of size matching the validation set, decode, and score
## synth-vs-val (T_sv). The reported quantity per replicate is
## log_norm = log(T_sv / T_tv).
##
## Cost note: for NHANES-sized splits (~4000x4000) each transport solve
## allocates a ~128 MB cost matrix and mc.cores multiplies that footprint.
## Keep S, R modest for tuning runs.
assess_generativity_split <- function(pipeline, y_list,
                                       S, R,
                                       frac_train  = 0.5,
                                       subsample_n = NULL,
                                       J_aug       = 500,
                                       ridge       = 0,
                                       seed        = 12345,
                                       n_cores     = 1) {
  set.seed(seed)

  ## Build augmented grid + weights once
  p_grid     <- pipeline$training$cache$p_grid
  p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
  w_aug      <- get_quadrature_weights(p_grid_aug)

  split_seeds <- sample.int(.Machine$integer.max, S)

  splits <- vector('list', S)
  for (s in seq_len(S)) {
    set.seed(split_seeds[s])

    ## Optional subsample, then split
    n_pool <- length(y_list)
    if (!is.null(subsample_n)) {
      stopifnot(subsample_n <= n_pool)
      idx_pool <- sample.int(n_pool, subsample_n)
    } else {
      idx_pool <- seq_len(n_pool)
    }
    n_used   <- length(idx_pool)
    n_train  <- floor(frac_train * n_used)
    perm     <- sample(idx_pool)
    idx_tr   <- perm[seq_len(n_train)]
    idx_val  <- perm[(n_train + 1):n_used]

    y_tr  <- y_list[idx_tr]
    y_val <- y_list[idx_val]

    Q_val_aug   <- y_to_Qi_aug(pipeline, y_val, p_grid_aug)
    Q_train_aug <- y_to_Qi_aug(pipeline, y_tr,  p_grid_aug)

    T_tv <- generativity_score(Q_train_aug, Q_val_aug, w_aug)

    Z_tr <- encode_to_Z(pipeline, y_tr)
    fit  <- fit_mean_cov(Z_tr, ridge = ridge)

    rep_seeds <- sample.int(.Machine$integer.max, R)
    T_sv <- unlist(parallel::mclapply(seq_len(R), function(r) {
      set.seed(rep_seeds[r])
      z_draws     <- draw_mean_cov(length(idx_val), fit)
      Q_synth_aug <- z_to_Qi_aug(pipeline, z_draws, p_grid_aug, p_grid)
      generativity_score(Q_synth_aug, Q_val_aug, w_aug)
    }, mc.cores = n_cores))

    splits[[s]] <- list(
      T_tv     = T_tv,
      T_sv     = T_sv,
      log_norm = log(T_sv / T_tv)
    )
  }

  K_used <- ncol(encode_to_Z(pipeline, y_list[1]))
  list(
    splits = splits,
    meta = list(
      S = S, R = R,
      frac_train  = frac_train,
      subsample_n = subsample_n,
      N_used      = if (is.null(subsample_n)) length(y_list) else subsample_n,
      K           = K_used
    )
  )
}

## Group-wise box plot of generativity scores with optional right-axis K line.
## Designed for two display modes:
##   - lambda-sweep:   groups = as.character(lambdas), Ks supplied, vline at lambda_star
##   - dataset compare: groups = c("NHANES", "CHOP"), Ks = NULL, vline_group = NULL
plot_generativity_boxes <- function(groups,
                                     scores,
                                     Ks          = NULL,
                                     hline       = 0,
                                     vline_group = NULL,
                                     xlab        = "",
                                     ylab        = "log normalized generativity",
                                     main        = "",
                                     path        = NULL,
                                     width       = 960,
                                     height      = 720,
                                     pointsize   = 14) {
  L <- length(groups)
  stopifnot(length(scores) == L)
  if (!is.null(Ks)) stopifnot(length(Ks) == L)

  ## Flatten per-group values to set y-limits and per-split box positions.
  S <- length(scores[[1]])
  if (!all(vapply(scores, length, integer(1)) == S)) {
    stop("Every group must have the same number of splits.")
  }
  y_all  <- unlist(scores, use.names = FALSE)
  y_rng  <- range(c(y_all, hline), finite = TRUE)
  y_pad  <- 0.05 * diff(y_rng)
  y_lim  <- c(y_rng[1] - y_pad, y_rng[2] + y_pad)

  ## Within-group offsets so the S boxes per group sit side-by-side.
  half_span <- 0.25
  offsets   <- if (S == 1) 0 else seq(-half_span, half_span, length.out = S)
  box_w     <- if (S == 1) 0.55 else min(0.18, (2 * half_span) / (S - 1) * 0.8)

  if (!is.null(path)) png(path, width = width, height = height, pointsize = pointsize)
  old_par <- par(mar = c(4.5, 4.5, 3, if (is.null(Ks)) 2 else 4.5))
  on.exit(par(old_par), add = TRUE)

  plot(NULL,
       xlim = c(0.5, L + 0.5),
       ylim = y_lim,
       xaxt = "n",
       xlab = xlab, ylab = ylab, main = main)
  axis(1, at = seq_len(L), labels = groups)

  abline(h = hline, col = "red", lty = 3, lwd = 1.2)
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
              border  = "black")
    }
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
    mtext("K", side = 4, line = 3, col = "forestgreen")
  }

  if (!is.null(path)) dev.off()
  invisible(NULL)
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


plot_funs <- function(
    fun_list,
    grid_list,
    ylim = NULL,
    ylab = 'Q(p)',
    colors = NULL,
    widths = NULL,
    types = NULL,
    color_width_type_labels = NULL,
    main = '',
    path = NULL
) {
  if (!is.null(path)) {
    png(path, width = 800, height = 600)
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
  if (is.null(ylim)) {
    ylim <- c(
      min(unlist(fun_list)),
      max(unlist(fun_list))
    )
  }
  par(mfrow = c(1, 1))
  plot(
    NULL,
    xlim = range(p_grid),
    ylim = ylim,
    xlab = "p",
    ylab = ylab,
    main = main
  )
  
  for (i in seq_len(n_funs)) {
    lines(
      grid_list[[i]],
      fun_list[[i]],
      col = colors[i],
      lwd = widths[i],
      lty = types[i]
    )
  }
  
  ## ---- Legend ---- ##
  if (!is.null(color_width_type_labels)) {
    
    required_cols <- c("color", "width", "type", "label")
    if (!all(required_cols %in% names(color_width_type_labels))) {
      stop("color_width_type_labels must have columns: color, width, type, label.")
    }
    
    legend(
      "topleft",
      legend = color_width_type_labels$label,
      col    = color_width_type_labels$color,
      lwd    = color_width_type_labels$width,
      lty    = color_width_type_labels$type,
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
    path = NULL
) {
  if (!is.null(path)) {
    png(path, width = 800, height = 600)
  }

  K <- ncol(emb)
  N <- nrow(emb)
  
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
  
  for (k1 in 1:K) {
    for (k2 in 1:K) {
      
      if (k1 == k2) {
        ## ---- QQ plot ---- ##
        if (stats) {
          pval <- round(shapiro.test(emb[, k1])$p.value, 3)
          main_txt <- paste("k =", k1, "| p =", pval)
        } else {
          main_txt <- paste("k =", k1)
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
          xlab = paste("k =", k1),
          ylab = paste("k =", k2),
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
          bty    = "n"
        )
      }
    }
  }
  if (!is.null(path)) dev.off()
}

plot_losslessness <- function(
    valid_losses_by_K,
    train_losses_by_K = NULL,
    jitter_width = 0.01,
    epsilon = 0.01,
    alpha = 0.05,
    plot_mean = TRUE,
    ylab = "Loss",
    xlab = "K",
    main = "Loss by K",
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
  
  Ks <- as.numeric(names(valid_losses_by_K))
  xlim_pad <- 2 * jitter_width
  
  y_vals <- unlist(valid_losses_by_K)
  if (!is.null(train_losses_by_K)) {
    y_vals <- c(y_vals, unlist(train_losses_by_K))
  }
  
  plot(
    NULL,
    xlim = c(min(Ks) - xlim_pad, max(Ks) + xlim_pad),
    ylim = c(0, max(y_vals)),
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
    points(xvals_v, v_losses, pch = 19, col = rgb(0,0,0,0.4))
    
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
    abline(v = K_star, lty = "dashed", col = 'black')
  }
  
  ## Close device only if we opened it
  if (!is.null(path)) dev.off()
}


## Group-wise jittered scatter of per-subject losses (one column per group),
## styled after plot_losslessness. Each entry in losses_list is a numeric
## vector; group_labels are the x-axis tick labels.
plot_losslessness_groups <- function(losses_list,
                                      group_labels,
                                      epsilon      = NULL,
                                      annotate_idx = NULL,
                                      jitter_width = 0.1,
                                      ylab         = "Loss",
                                      xlab         = "",
                                      main         = "",
                                      path         = NULL,
                                      width        = 960,
                                      height       = 720,
                                      pointsize    = 14) {
  L <- length(losses_list)
  stopifnot(length(group_labels) == L)

  if (!is.null(path)) png(path, width = width, height = height, pointsize = pointsize)
  old_par <- par(mar = c(4.5, 4.5, 3, 2))
  on.exit(par(old_par), add = TRUE)

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
    points(xvals, v, pch = 19, col = rgb(0, 0, 0, 0.4))
  }

  if (!is.null(epsilon)) {
    for (e in epsilon) abline(h = e, lty = "dashed", col = "red")
  }

  if (!is.null(annotate_idx)) {
    eps <- epsilon[1]
    txts <- vapply(annotate_idx, function(g) {
      pct <- 100 * mean(losses_list[[g]] < eps)
      sprintf("%.1f%% of %s < %.2f", pct, group_labels[g], eps)
    }, character(1))
    legend("topleft", legend = txts, bty = "n")
  }

  if (!is.null(path)) dev.off()
  invisible(NULL)
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
plot_qi_recon_grid <- function(Qi_orig_list, Qi_recon_list, recon_losses,
                                lower_mat, upper_mat,
                                log_x_plus_1 = FALSE,
                                path = NULL) {
  stopifnot(identical(dim(lower_mat), dim(upper_mat)))
  stopifnot(all(lower_mat <  upper_mat),
            all(lower_mat >= 0), all(upper_mat <= 1))
  n_row <- nrow(lower_mat)
  n_col <- ncol(lower_mat)
  N_    <- length(recon_losses)
  recon_losses_sort <- sort(recon_losses)

  if (!is.null(path)) png(path, width = 320 * n_col, height = 280 * n_row)
  old_par <- par(mfrow = c(n_row, n_col), mar = c(3, 3, 2, 1),
                 mgp = c(1.6, 0.5, 0))
  on.exit(par(old_par), add = TRUE)

  for (i in seq_len(n_row)) {
    for (j in seq_len(n_col)) {
      p           <- runif(1, lower_mat[i, j], upper_mat[i, j])
      idx_sort    <- round(1 + (N_ - 1) * p)
      q_value     <- recon_losses_sort[idx_sort]
      idx_payload <- which(recon_losses == q_value)[1]   # break ties safely

      Qi       <- Qi_orig_list[[idx_payload]]
      Qi_recon <- Qi_recon_list[[idx_payload]]
      pg       <- pi_grid_fun(length(Qi))

      if (log_x_plus_1) {
        Qi_disp       <- log(Qi + 1)
        Qi_recon_disp <- log(Qi_recon + 1)
        ylab          <- "log(Q(p) + 1)"
      } else {
        Qi_disp       <- Qi
        Qi_recon_disp <- Qi_recon
        ylab          <- "Q(p)"
      }

      plot(pg, Qi_disp, type = "l",
           xlab = "p", ylab = ylab,
           ylim = range(c(0, Qi_disp, Qi_recon_disp), finite = TRUE),
           main = sprintf("p ~ U(%.3g, %.3g)  p=%.4g", lower_mat[i, j], upper_mat[i, j], p),
           # main = str_glue("{lower_mat[i, j]*100}%-{upper_mat[i, j]*100}%tile, p={round(p,3)}"),
           cex.main = 2
      )
      lines(pg, Qi_recon_disp, col = rgb(0, 1, 0, alpha = 1.0))
    }
  }
  if (!is.null(path)) dev.off()
  invisible(NULL)
}



## ---------- Mass and Density Functions ---------- ##

y_to_pmf <- function(y) {
  if (length(y) == 0) {
    stop("y must be non-empty.")
  }
  
  if (any(is.na(y))) {
    stop("y must not contain NA values.")
  }
  
  tab <- table(y)
  
  pmf <- data.frame(
    value = as.numeric(names(tab)),
    prob  = as.integer(tab) / length(y),
    row.names = NULL
  )
  
  pmf[order(pmf$value), ]
}




