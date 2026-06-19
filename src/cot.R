library(glue)
library(irlba)
library(MASS)
library(minpack.lm)
library(pracma)
library(purrr)
library(stringr)
library(yaml)

source('src/utils.R')

init_python()
py_pkg <- reload_py_pkg()



## ========== Utilities ==========

new_context <- function(payload, cache = NULL, meta = NULL) {
  list(
    payload = payload,
    cache = cache,
    meta = meta
  )
}

differentiate <- function(Q, p_grid) {  ## via central differences
  stopifnot(length(Q) == length(p_grid))
  
  J <- length(Q)
  G <- numeric(J)
  
  # Interior
  G[2:(J - 1)] <-
    (Q[3:J] - Q[1:(J - 2)]) /
    (p_grid[3:J] - p_grid[1:(J - 2)])
  
  # Boundaries (one-sided)
  G[1] <- (Q[2] - Q[1]) /
    (p_grid[2] - p_grid[1])
  G[J] <- (Q[J] - Q[J - 1]) /
    (p_grid[J] - p_grid[J - 1])
  
  G
}

integrate <- function(G, p_grid, p_star, Q_star) {  ## via trapezoidal rule
  stopifnot(length(G) == length(p_grid))
  
  J <- length(G)
  
  p_star_idx <- which.min(abs(p_grid - p_star))
  
  Q <- numeric(J)
  Q[p_star_idx] <- Q_star
  
  # Forward integration (trapezoidal)
  for (j in (p_star_idx + 1):J) {
    dp <- p_grid[j] - p_grid[j - 1]
    Q[j] <- Q[j - 1] +
      0.5 * (G[j] + G[j - 1]) * dp
  }
  
  # Backward integration
  for (j in (p_star_idx - 1):1) {
    dp <- p_grid[j + 1] - p_grid[j]
    Q[j] <- Q[j + 1] -
      0.5 * (G[j + 1] + G[j]) * dp
  }
  
  Q
}

compute_quantlets <- function(
    G_list, K, sqrt_w,
    Q_list = NULL, p_grid = NULL, min_dQ = 1e-8,
    construction = c("pca", "q_mean_residual")
) {
  construction <- match.arg(construction)
  G_mat <- do.call(rbind, G_list)

  if (construction == "pca") {
    ## Legacy: K weighted PCs of G centered by colMeans(G).
    G_center <- colMeans(G_mat)
    G_mat_c  <- scale(G_mat, center = G_center, scale = FALSE)
    svd_res  <- irlba(sweep(G_mat_c, 2, sqrt_w, '*'), nv = K, nu = 0)
    E        <- svd_res$v / sqrt_w
    sd_k     <- svd_res$d[1:K] / sqrt(nrow(G_mat) - 1)
    return(list(E = E, G_center = G_center, sd_k = sd_k))
  }

  ## Q-mean-anchored: quantlet 1 = LQD(Q_mean), quantlets 2..K = first
  ## K-1 weighted PCs of (G_i - LQD(Q_mean)). Reconstruction model
  ##   G_i ~ E %*% c  with c[1] ~ 1 for typical subjects.
  ## G_center is the zero vector so the qg_pca solver consumes E directly.
  if (is.null(Q_list) || is.null(p_grid)) {
    stop("construction = 'q_mean_residual' requires Q_list and p_grid.")
  }
  Q_mat   <- do.call(rbind, Q_list)
  Q_mean  <- colMeans(Q_mat)
  dQ_mean <- differentiate(Q_mean, p_grid)
  dQ_mean[dQ_mean < min_dQ] <- min_dQ
  G_anchor <- log(dQ_mean)                       # length J

  G_resid <- sweep(G_mat, 2, G_anchor, '-')       # N x J

  if (K >= 2) {
    svd_res <- irlba(sweep(G_resid, 2, sqrt_w, '*'),
                     nv = K - 1, nu = 0)
    E_pc    <- svd_res$v / sqrt_w                 # J x (K-1)
    sd_pc   <- svd_res$d[1:(K - 1)] / sqrt(nrow(G_mat) - 1)
    E       <- cbind(G_anchor, E_pc)
  } else {
    ## K = 1: only the anchor.
    E     <- matrix(G_anchor, ncol = 1)
    sd_pc <- numeric(0)
  }

  ## sd_k[1]: stdev of the anchor coefficient c[1] across subjects, computed
  ## post-hoc by projecting each G_i onto E in the weighted inner product.
  ## (For typical subjects c[1] ~ 1 with small scatter; this lets downstream
  ## plotting / normalization code treat all K coefficients uniformly.)
  W       <- sqrt_w^2
  EtWE    <- crossprod(E, sweep(E, 1, W, '*'))
  EtWG    <- crossprod(E, sweep(t(G_mat), 1, W, '*'))   # K x N
  c_mat   <- t(solve(EtWE, EtWG))                       # N x K
  sd_k1   <- sd(c_mat[, 1])
  sd_k    <- c(sd_k1, sd_pc)

  list(E = E, G_center = numeric(ncol(G_mat)), sd_k = sd_k)
}

compute_q_pcs <- function(Q_list, K, sqrt_w) {
  Q_mat <- do.call(rbind, Q_list)
  Q_center <- colMeans(Q_mat)
  Q_mat <- scale(Q_mat, center = Q_center, scale = FALSE)
  svd_res <- irlba(sweep(Q_mat, 2, sqrt_w, '*'), nv = K, nu = 0)
  E <- svd_res$v / sqrt_w
  sd_k <- svd_res$d[1:K] / sqrt(nrow(Q_mat) - 1)
  list(E = E, Q_center = Q_center, sd_k = sd_k)
}


get_step_fun <- function(Q, supp_Y) {
  idx <- findInterval(Q, supp_Y, rightmost.closed = FALSE)
  idx <- pmax(idx, 1)
  idx <- pmin(idx, length(supp_Y))
  Q <- supp_Y[idx]
  Q
}


inv_eqf_cgrid <- function(Q, p_grid, pi_grid, supp_Y = NULL) {

  ## Linearly interpolate onto subject grid
  ## NOTE: `rule = 2` indicates that the value at the closest data extreme 
  ## is used for extrapolation
  Qi <- approx(x = p_grid, y = Q, xout = pi_grid, rule = 2)$y
  
  ## Get step function
  if (!is.null(supp_Y)) {
    Qi <- get_step_fun(Qi, supp_Y)
  }
  
  Qi
}

inv_lqd <- function(G, p_grid, p_star, Q_star) {
  dQ <- exp(G)
  Q <- integrate(dQ, p_grid, p_star, Q_star)
  Q
}

qg_pca <- function(
    Q_obs, E, G_center, p_grid, p_star, 
    Q_star = NULL,
    sqrt_w = NULL,
    c_init = NULL, 
    Q_star_init = NULL,
    lambda = NULL,
    control = nls.lm.control(maxiter = 200, ftol = 1e-10)
) {
  
  K <- ncol(E)
  J <- nrow(E)
  w <- sqrt_w^2
  lambda <- lambda %||% 0
  
  p_star_idx <- which.min(abs(p_grid - p_star))
  
  if (is.null(c_init)) c_init <- rep(0, K)
  
  if (is.null(Q_star)) {
    if (is.null(Q_star_init)) {
      G_0 <- G_center + as.vector(E %*% c_init)
      Q_hat_0 <- inv_lqd(G_0, p_grid, p_star = p_star, Q_star = 0)
      Q_star_init <- sum(w * (Q_obs - Q_hat_0)) / sum(w)
    }
  }
  
  ## ---- Residual function ----
  resid_fun <- function(par) {
    if (is.null(Q_star)) {
      c <- par[1:K]
      Q_star_ <- par[K + 1]
    } else {
      c <- par[1:K]
      Q_star_ <- Q_star
    }
    
    G_hat <- G_center + as.vector(E %*% c)
    Q_hat <- inv_lqd(G_hat, p_grid, p_star, Q_star_)
    
    r <- Q_obs - Q_hat
    if (!is.null(sqrt_w)) r <- sqrt_w * r
    
    ## L2 penalty as extra residuals
    if (lambda > 0) {
      r_pen <- sqrt(lambda) * c
      r <- c(r, r_pen)
    }
    
    r
  }
  
  ## ---- Jacobian function ----
  jac_fun <- function(par) {
    if (is.null(Q_star)) {
      c <- par[1:K]
      Pdim <- K + 1
    } else {
      c <- par[1:K]
      Pdim <- K
    }
    
    G_hat <- G_center + as.vector(E %*% c)
    dQ <- exp(G_hat)
    F  <- dQ * E
    dp <- diff(p_grid)
    
    trap <- 0.5 * (F[-1, , drop = FALSE] +
                     F[-J, , drop = FALSE]) * dp
    
    J_mat <- matrix(0, nrow = J, ncol = Pdim)
    
    ## Forward
    if (p_star_idx < J) {
      cs_fwd <- apply(
        trap[p_star_idx:(J - 1), , drop = FALSE],
        2, cumsum
      )
      J_mat[(p_star_idx + 1):J, 1:K] <- -cs_fwd
    }
    
    ## Backward
    if (p_star_idx > 1) {
      idx <- 1:(p_star_idx - 1)
      cs_bwd <- apply(
        trap[idx, , drop = FALSE][length(idx):1, , drop = FALSE],
        2, cumsum
      )
      J_mat[idx, 1:K] <- cs_bwd[length(idx):1, , drop = FALSE]
    }
    
    if (is.null(Q_star)) {
      J_mat[, K + 1] <- -1
    }
    
    if (!is.null(sqrt_w)) {
      J_mat <- diag(sqrt_w) %*% J_mat
    }
    
    ## Jacobian rows for L2 penalty
    if (lambda > 0) {
      J_pen <- matrix(0, nrow = K, ncol = Pdim)
      J_pen[, 1:K] <- sqrt(lambda) * diag(K)
      J_mat <- rbind(J_mat, J_pen)
    }
    
    J_mat
  }
  
  if (is.null(Q_star)) {
    par_init <- c(c_init, Q_star_init)
  } else {
    par_init <- c(c_init)
  }
  
  fit <- nls.lm(
    par = par_init,
    fn = resid_fun,
    jac = jac_fun,
    control = control
  )
  
  list(
    c = fit$par[1:K],
    Q_star = if (is.null(Q_star)) fit$par[K + 1] else Q_star,
    converged = (fit$info %in% c(1, 2)),
    Q_star_fixed = !is.null(Q_star)
  )
}

q_pca <- function(Q_obs, E, Q_center, sqrt_w) {
  Q_centered <- (Q_obs - Q_center) * sqrt_w
  c_i <- drop(t(E*sqrt_w) %*% Q_centered)
  c_i
}

g_pca <- function(G_obs, E, G_center, sqrt_w) {
  G_centered <- (G_obs - G_center) * sqrt_w
  c_i <- drop(t(E * sqrt_w) %*% G_centered)
  c_i
}

choose_near_lossless_K_qg_pca <- function(
    epsilon, alpha, V, K_max,
    loss_fun,
    G_list, Q_list, Qi_list,
    p_grid, Ji_vec, p_star,
    Q_star = NULL,
    sqrt_w = NULL,
    supp_Y = NULL,
    lambda = 0,
    min_dQ = 1e-8,
    construction = c("pca", "q_mean_residual"),
    seed = 12345
) {
  construction <- match.arg(construction)
  
  ## Set parameters
  N <- length(G_list)
  
  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))
  
  K <- 1
  while (K <= K_max) {
    message(str_glue("---------- K = {K} ----------"))
    
    losses <- vector(mode = "numeric", length = N)
    converged <- vector(length = N)
    for (v in 1:V) {
      
      ## Get train/valid indices
      idx_train <- which(folds != v)
      idx_valid <- which(folds == v)
      
      ## Compute quantlets E from training data
      out <- compute_quantlets(
        G_list[idx_train], K, sqrt_w,
        Q_list = Q_list[idx_train], p_grid = p_grid, min_dQ = min_dQ,
        construction = construction
      )
      E <- out$E
      G_center <- out$G_center
      
      for (i in idx_valid) {
        
        ## Compute c
        out <- qg_pca(
          Q_obs = Q_list[[i]], 
          E = E, 
          G_center = G_center, 
          p_grid = p_grid,
          p_star = p_star,
          Q_star  = Q_star,
          sqrt_w = sqrt_w,
          lambda = lambda
        )
        converged[i] <- out$converged
        
        ## Map c --> G --> Q
        G <- G_center + rowSums(E %*% out$c)
        Q <- inv_lqd(
          G = G, 
          p_grid = p_grid, 
          p_star = p_star, 
          Q_star = out$Q_star
        )
        Qi <- inv_eqf_cgrid(
          Q = Q,
          p_grid = p_grid,
          pi_grid = pi_grid_fun(Ji_vec[i]),
          supp_Y = supp_Y
        )
        
        ## Compute loss
        dp <- 1 / (Ji_vec[i] + 1)
        losses[i] <- loss_fun(Qi_list[[i]], Qi, dp)
        
      }
    }
    
    ## CV result reporting
    idx_outliers <- which(losses >= epsilon)
    prop_tol <- mean(losses < epsilon)
    msg <- glue(
      "converged = {sum(converged)} of {N}\n",
      "mean loss = {mean(losses)}\n",
      "max loss  = {max(losses)}\n",
      "prop tol  = {prop_tol}"
    )
    message(msg)
    
    ## Stopping
    if (prop_tol >= 1 - alpha) {
      message(str_glue("Qualifying dimension found. Setting K = {K}."))
      break
    }
    
    K <- K + 1
  }
  
  if (K == K_max + 1) {
    warning(glue(
      "Qualifying dimension not found. Candidates exhausted. ",
      "Setting K = {K_max} = K_max."
    ))
    K = K_max
  }
  return(list(K = K, idx_outliers = idx_outliers))
}


choose_near_lossless_K_q_pca <- function(
    epsilon, alpha, V, K_max,
    loss_fun,
    Q_list, Qi_list,
    p_grid, Ji_vec,
    sqrt_w = NULL,
    supp_Y = NULL,
    seed = 12345
) {
  
  ## Set parameters
  N <- length(Q_list)
  
  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))
  
  K <- 1
  while (K <= K_max) {
    message(str_glue("---------- K = {K} ----------"))
    
    losses <- vector(mode = "numeric", length = N)
    for (v in 1:V) {
      
      ## Get train/valid indices
      idx_train <- which(folds != v)
      idx_valid <- which(folds == v)
      
      ## Compute weighted PCA basis from training data
      out      <- compute_q_pcs(Q_list[idx_train], K, sqrt_w)
      E        <- out$E
      Q_center <- out$Q_center
      
      for (i in idx_valid) {
        
        ## Project onto basis: c = E^T W (Q - Q_center)
        c_i <- q_pca(Q_list[[i]], E, Q_center, sqrt_w)
        
        ## Reconstruct: Q_hat = Q_center + E c
        Q_hat <- Q_center + drop(E %*% c_i)
        
        ## Map Q_hat back to subject-specific grid
        Qi_hat <- inv_eqf_cgrid(
          Q      = Q_hat,
          p_grid = p_grid,
          pi_grid = pi_grid_fun(Ji_vec[i]),
          supp_Y = supp_Y
        )
        
        ## Compute loss
        dp        <- 1 / (Ji_vec[i] + 1)
        losses[i] <- loss_fun(Qi_list[[i]], Qi_hat, dp)
        
      }
    }
    
    ## CV result reporting
    idx_outliers <- which(losses >= epsilon)
    prop_tol     <- mean(losses < epsilon)
    msg <- glue(
      "mean loss = {mean(losses)}\n",
      "max loss  = {max(losses)}\n",
      "prop tol  = {prop_tol}"
    )
    message(msg)
    
    ## Stopping
    if (prop_tol >= 1 - alpha) {
      message(str_glue("Qualifying dimension found. Setting K = {K}."))
      break
    }
    
    K <- K + 1
  }
  
  if (K == K_max + 1) {
    warning(glue(
      "Qualifying dimension not found. Candidates exhausted. ",
      "Setting K = {K_max} = K_max."
    ))
    K = K_max
  }
  return(list(K = K, idx_outliers = idx_outliers))
}


choose_near_lossless_K_g_pca <- function(
    epsilon, alpha, V, K_max,
    loss_fun,
    G_list, Qi_list,
    p_grid, Ji_vec, p_star,
    Q_star_list,
    sqrt_w = NULL,
    supp_Y = NULL,
    seed = 12345
) {

  ## Set parameters
  N <- length(G_list)

  ## Assign folds
  set.seed(seed)
  folds <- sample(rep(1:V, length.out = N))

  K <- 1
  while (K <= K_max) {
    message(str_glue("---------- K = {K} ----------"))

    losses <- vector(mode = "numeric", length = N)
    for (v in 1:V) {

      ## Get train/valid indices
      idx_train <- which(folds != v)
      idx_valid <- which(folds == v)

      ## Compute weighted PCA basis in G-space from training data
      out      <- compute_quantlets(G_list[idx_train], K, sqrt_w)
      E        <- out$E
      G_center <- out$G_center

      for (i in idx_valid) {

        ## Project onto basis
        c_i <- g_pca(G_list[[i]], E, G_center, sqrt_w)

        ## Reconstruct: G_hat -> Q_hat -> Qi_hat
        G_hat <- G_center + drop(E %*% c_i)
        Q_hat <- inv_lqd(G_hat, p_grid, p_star, Q_star_list[[i]])
        Qi_hat <- inv_eqf_cgrid(
          Q      = Q_hat,
          p_grid = p_grid,
          pi_grid = pi_grid_fun(Ji_vec[i]),
          supp_Y = supp_Y
        )

        ## Compute loss
        dp        <- 1 / (Ji_vec[i] + 1)
        losses[i] <- loss_fun(Qi_list[[i]], Qi_hat, dp)

      }
    }

    ## CV result reporting
    idx_outliers <- which(losses >= epsilon)
    prop_tol     <- mean(losses < epsilon)
    msg <- glue(
      "mean loss = {mean(losses)}\n",
      "max loss  = {max(losses)}\n",
      "prop tol  = {prop_tol}"
    )
    message(msg)

    ## Stopping
    if (prop_tol >= 1 - alpha) {
      message(str_glue("Qualifying dimension found. Setting K = {K}."))
      break
    }

    K <- K + 1
  }

  if (K == K_max + 1) {
    warning(glue(
      "Qualifying dimension not found. Candidates exhausted. ",
      "Setting K = {K_max} = K_max."
    ))
    K = K_max
  }
  return(list(K = K, idx_outliers = idx_outliers))
}



## ========== Pipeline Architecture ==========

## ---------- Generic Functions

fit <- function(x, ...) {
  UseMethod("fit")
}

encode <- function(x, ...) {
  UseMethod("encode")
}

decode <- function(x, ...) {
  UseMethod("decode")
}


## ---------- Stage Object and Methods

new_stage <- function(
    name,
    input_space,
    output_space,
    requires_fit = TRUE,
    fit_fun = NULL,
    init_state = NULL,
    encode_fun,
    decode_fun
) {
  structure(
    list(
      name = name,
      input_space = input_space,
      output_space = output_space,
      requires_fit = requires_fit,
      fit_fun = fit_fun,
      encode_fun = encode_fun,
      decode_fun = decode_fun,
      fitted = FALSE,
      state = init_state
    ),
    class = "cot_stage"
  )
}

fit.cot_stage <- function(stage, context, ...) {
  if (!stage$requires_fit) return(stage)
  stage$state <- stage$fit_fun(context, stage$state, ...)
  stage$fitted <- TRUE
  stage
}

encode.cot_stage <- function(stage, context, ...) {
  if (stage$requires_fit && !stage$fitted)
    stop("Stage not fitted.")
  
  stage$encode_fun(context, stage$state, ...)
}

decode.cot_stage <- function(stage, context, ...) {
  stage$decode_fun(context, stage$state, ...)
}


## ---------- Pipeline Object and Methods

new_pipeline <- function(stages) {
  structure(
    list(
      stages = stages,
      n_stages = length(stages)
    ),
    class = "cot_pipeline"
  )
}

fit.cot_pipeline <- function(pipeline, data, ...) { 
  context <- list(
    payload = data,
    cache = pipeline$cache_init,
    meta = pipeline$meta_init
  )
  
  for (i in seq_len(pipeline$n_stages)) { 
    stage <- pipeline$stages[[i]] 
    if (stage$requires_fit && !stage$fitted) { 
      stage <- fit(stage, context, ...) 
      pipeline$stages[[i]] <- stage 
    } 
    context <- encode(stage, context, ...) 
  } 
  
  pipeline$training <- list(
    cache = context$cache,
    meta = context$meta
  )
  pipeline 
}

encode.cot_pipeline <- function(pipeline, context, from = 0, to = pipeline$n_stages, ...) {
  
  if (is.null(context$cache)) {
    context$cache <- pipeline$cache_init %||% list()
  }
  
  x <- context
  for (i in seq(from + 1, to)) {
    stage <- pipeline$stages[[i]]
    x <- encode(stage, x, ...)
  }
  x
}

decode.cot_pipeline <- function(pipeline, context, from = pipeline$n_stages, to = 0, ...) {
  x <- context
  for (i in seq(from, to + 1, by = -1)) {
    stage <- pipeline$stages[[i]]
    x <- decode(stage, x, ...)
  }
  x
}


## ========== Transforms: Fit/Encode/Decode Functions ==========

## ---------- Transform: EQF on Subject Grid

encode_fun_eqf_sgrid <- function(context, state) {
  y_list <- context$payload
  Qi_list <- lapply(y_list, function(y) sort(y))
  context$payload <- Qi_list
  context$meta$Qi_list <- Qi_list
  context$meta$Ji_vec <- lengths(Qi_list)
  context
}

decode_fun_eqf_sgrid <- function(context, state) {
  Qi_list <- context$payload
  y_list <- Qi_list  ## recovers input up to rearrangement
  context$payload <- y_list
  context
}

stage_eqf_sgrid <- function() {
  new_stage(
    name = "eqf_sgrid",
    input_space = "Y",
    output_space = "Qi",
    requires_fit = FALSE,
    encode_fun = encode_fun_eqf_sgrid,
    decode_fun = decode_fun_eqf_sgrid
  )
}


## ---------- Transform: Smooth EQF on Dense Common Grid

interpolate_onto_interior <- function(Qi, p_grid, Q_min = NULL) {
  Ji <- length(Qi)
  J <- length(p_grid)
  
  ## Get subject grid
  pi_grid <- pi_grid_fun(Ji)
  
  ## Get interior mask
  interior_mask <- rep(TRUE, J)
  interior_mask[p_grid < min(pi_grid)] <- FALSE
  interior_mask[p_grid > max(pi_grid)] <- FALSE
  
  ## Get anchor points
  anchors <- which(!duplicated(Qi))
  a <- pi_grid[anchors]
  v <- Qi[anchors]
  
  ## Update anchor points with theoretical min/max
  if (!is.null(Q_min)) {
    if (v[1] == Q_min) {  ## remove the leftmost anchor point
      a <- a[2:length(a)]
      v <- v[2:length(v)]
    } 
    a <- c(0, a)
    v <- c(Q_min, v)
    interior_mask[p_grid < min(pi_grid)] <- TRUE
  }
  
  ## Interpolate onto interior through anchors (leaving exterior NA)
  out <- rep(NA, J)
  out[interior_mask] <- spinterp(x = a, y = v, xp = p_grid[interior_mask])
  out
}


fit_fun_eqf_cgrid <- function(context, state, ...) {
  Qi_list <- context$payload
  Ji_vec <- context$meta$Ji_vec
  Q_min <- context$cache$Q_min
  ratio_trans <- context$cache$ratio_trans
  p_star <- context$cache$p_star
  p_grid <- state$p_grid
  
  ## Check p_grid
  J <- length(p_grid)
  Ji_max <- max(Ji_vec)
  min_Qi <- 1 / (1 + Ji_max)
  max_Qi <- Ji_max / (1 + Ji_max)
  if (p_grid[1] < min_Qi || p_grid[J] > max_Qi) {
    warning("Can't evaluate EQFs on tails of p_grid. Trimming as needed.")
    p_grid <- p_grid[p_grid >= min_Qi & p_grid <= max_Qi]
    p_grid <- unique(c(min_Qi, p_grid, max_Qi))
    J <- length(p_grid)
    state$p_grid <- p_grid
  }
  if (!(
    all(diff(p_grid) > 0) &&  ## strictly increasing
    J >= 3 &&
    p_grid[1] > 0 &&
    p_grid[J] < 1
  )) {
    stop("Invalid p-grid!")
  }
  if (p_grid[1] > min_Qi || p_grid[J] < max_Qi) {
    warning(glue(
      "Add points to the left and/or right tail of p_grid ",
      "to maximally preserve tail information."
    ))
  }

  ## Compute weights that will later be used to account for unbalanced grid
  w <- get_quadrature_weights(p_grid)
  # state$dp <- dp
  state$w <- w
  state$sqrt_w <- sqrt(w)
  
  ## Set p_star_idx
  p_star_idx <- which.min(abs(p_grid - p_star))
  state$p_star_idx <- p_star_idx
  
  ## Interpolate onto interior points
  N <- length(Qi_list)
  Q_mat <- matrix(nrow = N, ncol = J)
  for (i in 1:N) {
    Q_mat[i,] <- interpolate_onto_interior(Qi_list[[i]], p_grid, Q_min)
  }
  
  ## Get common grid points for which to compute ratios
  pi_max <- state$Ji_min / (1 + state$Ji_min)
  exterior_cols <- which(p_grid > pi_max)
  # if (!is.null(Q_min)) {
  if (is.null(Q_min)) {
    pi_min <- 1 / (1 + state$Ji_min)
    exterior_cols <- c(exterior_cols, which(p_grid < pi_min))
  }
  
  ## Get interior sets for those common grid points
  interior_sets <- lapply(
    exterior_cols,
    function(j) which(!is.na(Q_mat[,j]))
  )
  names(interior_sets) <- exterior_cols
  
  ## Handle NULL ratio_trans
  if (is.null(ratio_trans)) {
    ratio_trans <- function(x, ...) x
  }
  
  ## Compute ratios for tail extrapolation
  exterior_scales <- vector('list', length(exterior_cols))
  names(exterior_scales) <- exterior_cols
  p_mid <- which.min(abs(p_grid - 0.5))
  ext_cols_left <- sort(exterior_cols[exterior_cols < p_mid], decreasing = TRUE)
  ext_cols_right <- sort(exterior_cols[exterior_cols > p_mid])
  for (j in c(ext_cols_left, ext_cols_right)) {
    j_ <- ifelse(j < p_mid, j + 1, j - 1)
    I_j <- interior_sets[[as.character(j)]]  ## interior indices
    E_j <- which(is.na(Q_mat[,j]))           ## exterior indices
    exterior_scales[[as.character(j)]] <- ratio_trans(
      mean(ratio_trans(Q_mat[I_j,j] / Q_mat[I_j,j_])),
      inverse = TRUE
    )
  }
  # for (j in c(ext_cols_left, ext_cols_right)) {
  #   j_1 <- ifelse(j < p_mid, j + 1, j - 1)
  #   j_2 <- ifelse(j < p_mid, j + 2, j - 2)
  #   I_j <- interior_sets[[as.character(j)]]  ## interior indices
  #   E_j <- which(is.na(Q_mat[,j]))           ## exterior indices
  #   ratio_vec <- (Q_mat[I_j,j] - Q_mat[I_j,j_1]) / (Q_mat[I_j,j_1] - Q_mat[I_j,j_2])
  #   exterior_scales[[as.character(j)]] <- ratio_trans(mean(ratio_trans(ratio_vec)), inverse = TRUE)
  # } 
  
  state$exterior_scales <- exterior_scales
  state
}

encode_fun_eqf_cgrid <- function(context, state) {
  Qi_list <- context$payload
  p_grid <- state$p_grid
  Q_min <- context$cache$Q_min
  
  ## Interpolate onto interior points
  N <- length(Qi_list)
  J <- length(p_grid)
  Q_mat <- matrix(nrow = N, ncol = J)
  for (i in 1:N) {
    Q_mat[i,] <- interpolate_onto_interior(Qi_list[[i]], p_grid, Q_min)
  }
  
  ## Get interior sets
  exterior_cols <- which(colSums(is.na(Q_mat)) > 0)
  interior_sets <- lapply(
    exterior_cols,
    function(j) which(!is.na(Q_mat[, j]))
  )
  names(interior_sets) <- exterior_cols
  
  ## Use pre-computed exterior scales to extrapolate in left and right tails
  p_mid <- which.min(abs(p_grid - 0.5))
  ext_cols_left <- sort(exterior_cols[exterior_cols < p_mid], decreasing = TRUE)
  ext_cols_right <- sort(exterior_cols[exterior_cols > p_mid])
  for (j in c(ext_cols_left, ext_cols_right)) {
    j_ <- ifelse(j < p_mid, j + 1, j - 1)
    I_j <- interior_sets[[as.character(j)]]  ## interior indices
    E_j <- which(is.na(Q_mat[,j]))           ## exterior indices
    Q_mat[E_j,j] <- state$exterior_scales[[as.character(j)]] * Q_mat[E_j,j_]
  }
  # for (j in c(ext_cols_left, ext_cols_right)) {
  #   j_1 <- ifelse(j < p_mid, j + 1, j - 1)
  #   j_2 <- ifelse(j < p_mid, j + 2, j - 2)
  #   I_j <- interior_sets[[as.character(j)]]  ## interior indices
  #   E_j <- which(is.na(Q_mat[,j]))           ## exterior indices
  #   Q_mat[E_j,j] <- Q_mat[E_j,j_1] + 
  #     (Q_mat[E_j,j_1] - Q_mat[E_j,j_2]) * 
  #     state$exterior_scales[[as.character(j)]]
  # }
  
  Q_list <- asplit(Q_mat, MARGIN = 1)
  context$payload <- Q_list
  context$meta$Q_list <- Q_list
  context$cache$p_grid <- p_grid
  context$cache$dp <- state$dp
  context$cache$w <- state$w
  context$cache$sqrt_w <- state$sqrt_w
  context$cache$p_star_idx <- state$p_star_idx
  context
}


decode_fun_eqf_cgrid <- function(context, state) {
  Q_list <- context$payload
  Ji_vec <- context$meta$Ji_vec
  if (is.null(Ji_vec))
    stop("decode_fun_eqf_cgrid requires Ji_vec in context$meta")
  p_grid <- context$cache$p_grid
  J <- length(p_grid)
  supp_Y <- context$cache$supp_Y
  
  ## Linearly interpolate through p_grid onto pi_grid
  N <- length(Q_list)
  Qi_list <- vector(mode = "list", length = N)
  for (i in 1:N) {
    pi_grid <- pi_grid_fun(Ji_vec[i])
    Qi_list[[i]] <- inv_eqf_cgrid(Q_list[[i]], p_grid, pi_grid, supp_Y)
  }
  
  context$payload <- Qi_list
  context
}

stage_eqf_cgrid <- function(p_grid = seq(0.01, 0.99, by = 0.01), Ji_min = 100) {
  new_stage(
    name = "eqf_cgrid",
    input_space = "Qi",
    output_space = "Q",
    requires_fit = TRUE,
    init_state = list(p_grid = p_grid, Ji_min = Ji_min),
    fit_fun = fit_fun_eqf_cgrid,
    encode_fun = encode_fun_eqf_cgrid,
    decode_fun = decode_fun_eqf_cgrid
  )
}


## ---------- Transform: Q-PCA

fit_fun_q_pca <- function(context, state, ...) {
  Q_list <- context$meta$Q_list
  Qi_list <- context$meta$Qi_list
  p_grid <- context$cache$p_grid
  Ji_vec <- context$meta$Ji_vec
  supp_Y <- context$cache$supp_Y
  sqrt_w <- context$cache$sqrt_w
  loss_fun <- context$cache$loss_fun

  ## Choose qualifying dimension K
  if (is.null(state$K)) {
    out <- choose_near_lossless_K_q_pca(
      epsilon = state$epsilon, alpha = state$alpha,
      V = state$V, K_max = state$K_max,
      loss_fun = loss_fun,
      Q_list = Q_list, Qi_list = Qi_list, 
      p_grid = p_grid, Ji_vec = Ji_vec, 
      sqrt_w = sqrt_w,
      supp_Y = supp_Y,
      seed = state$seed
    )
    state$K <- out$K
    state$idx_outliers <- out$idx_outliers
  } 
  
  ## Compute E
  out <- compute_q_pcs(Q_list, state$K, sqrt_w)
  
  state$Q_center <- out$Q_center
  state$E <- out$E
  state$sd_k <- out$sd_k
  state
}

encode_fun_q_pca <- function(context, state) {
  Q_list <- context$meta$Q_list
  p_grid <- context$cache$p_grid
  sqrt_w <- context$cache$sqrt_w
  
  ## Set parameters
  N <- length(Q_list)
  converged <- vector(length = N)
  c_list <- vector('list', N)
  for (i in 1:N) {
    c_list[[i]] <- q_pca(Q_list[[i]], state$E, state$Q_center, sqrt_w)
  }
  
  context$meta$idx_outliers <- state$idx_outliers
  context$payload = c_list
  context
}

decode_fun_q_pca <- function(context, state) {
  c_list <- context$payload
  Q_list <- lapply(c_list, function(c) state$Q_center + rowSums(state$E %*% c))
  context$payload <- Q_list
  context
}

stage_q_pca <- function(
    K = NULL,
    K_max = 20,
    epsilon = 0.01,
    alpha = 0.05,
    V = 5,
    seed = gen_seed()
) {
  new_stage(
    name = "q_pca",
    input_space = "Q",
    output_space = "C",
    requires_fit = TRUE,
    init_state = list(
      K = K,
      K_max = K_max,
      epsilon = epsilon,
      alpha = alpha,
      V = V,
      seed = seed
    ),
    fit_fun = fit_fun_q_pca,
    encode_fun = encode_fun_q_pca,
    decode_fun = decode_fun_q_pca
  )
}


## ---------- Transform: G-PCA

fit_fun_g_pca <- function(context, state, ...) {
  G_list <- context$payload$G_list
  Q_star_list <- context$payload$Q_star_list
  Qi_list <- context$meta$Qi_list
  p_grid <- context$cache$p_grid
  Ji_vec <- context$meta$Ji_vec
  p_star <- context$cache$p_star
  supp_Y <- context$cache$supp_Y
  sqrt_w <- context$cache$sqrt_w
  loss_fun <- context$cache$loss_fun

  ## Choose qualifying dimension K
  if (is.null(state$K)) {
    out <- choose_near_lossless_K_g_pca(
      epsilon = state$epsilon, alpha = state$alpha,
      V = state$V, K_max = state$K_max,
      loss_fun = loss_fun,
      G_list = G_list, Qi_list = Qi_list,
      p_grid = p_grid, Ji_vec = Ji_vec,
      p_star = p_star, Q_star_list = Q_star_list,
      sqrt_w = sqrt_w,
      supp_Y = supp_Y,
      seed = state$seed
    )
    state$K <- out$K
    state$idx_outliers <- out$idx_outliers
  }

  ## Compute E
  out <- compute_quantlets(G_list, state$K, sqrt_w)

  state$G_center <- out$G_center
  state$E <- out$E
  state$sd_k <- out$sd_k
  state
}

encode_fun_g_pca <- function(context, state) {
  G_list <- context$payload$G_list
  Q_star_list <- context$payload$Q_star_list
  sqrt_w <- context$cache$sqrt_w

  ## Set parameters
  N <- length(G_list)
  c_list <- vector('list', N)
  for (i in 1:N) {
    c_list[[i]] <- g_pca(G_list[[i]], state$E, state$G_center, sqrt_w)
  }

  context$meta$idx_outliers <- state$idx_outliers
  context$payload <- list(c_list = c_list, Q_star_list = Q_star_list)
  context
}

decode_fun_g_pca <- function(context, state) {
  c_list <- context$payload$c_list
  Q_star_list <- context$payload$Q_star_list

  G_list <- lapply(c_list, function(c) state$G_center + rowSums(state$E %*% c))
  context$payload <- list(G_list = G_list, Q_star_list = Q_star_list)
  context
}

stage_g_pca <- function(
    K = NULL,
    K_max = 20,
    epsilon = 0.01,
    alpha = 0.05,
    V = 5,
    seed = gen_seed()
) {
  new_stage(
    name = "g_pca",
    input_space = "G",
    output_space = "C",
    requires_fit = TRUE,
    init_state = list(
      K = K,
      K_max = K_max,
      epsilon = epsilon,
      alpha = alpha,
      V = V,
      seed = seed
    ),
    fit_fun = fit_fun_g_pca,
    encode_fun = encode_fun_g_pca,
    decode_fun = decode_fun_g_pca
  )
}


## ---------- Transform: LQD

encode_fun_lqd <- function(context, state) {
  Q_list <- context$payload
  p_star_idx <- context$cache$p_star_idx
  Q_star <- context$cache$Q_star
  
  ## Set parameters
  N <- length(Q_list)
  J <- length(Q_list[[1]])
  p_grid <- context$cache$p_grid
  
  ## Get integration constants
  if (is.null(Q_star)) {
    Q_star_list <- lapply(Q_list, function(Q) Q[[p_star_idx]])
  } else {
    Q_star_list <- rep(list(Q_star), N)
  }
  
  ## LQD transform Q_list
  G_list <- vector("list", N)
  for (i in 1:N) {
    dQ <- differentiate(Q_list[[i]], p_grid)
    dQ[dQ < state$min_dQ] <- state$min_dQ
    G_list[[i]] <- log(dQ)
  }
  
  context$payload <- list(
    G_list = G_list, 
    Q_star_list = Q_star_list
  )
  context
}

decode_fun_lqd <- function(context, state) {
  G_list <- context$payload$G_list
  Q_star_list <- context$payload$Q_star_list
  p_grid <- context$cache$p_grid
  p_star <- context$cache$p_star
  
  ## Set parameters
  N <- length(G_list)
  J <- length(G_list[[1]])
  
  ## Inverse LQD transform G_list 
  Q_list <- vector("list", N)
  for (i in 1:N) {
    Q_list[[i]] <- inv_lqd(G_list[[i]], p_grid, p_star, Q_star_list[[i]])
  }
  
  context$payload <- Q_list
  context
}

stage_lqd <- function(min_dQ = 1e-8) {
  new_stage(
    name = "lqd",
    input_space = "Q",
    output_space = "G",
    requires_fit = FALSE,
    init_state = list(min_dQ = min_dQ),
    encode_fun = encode_fun_lqd,
    decode_fun = decode_fun_lqd
  )
}



## ---------- Transform: Q-G PCA

fit_fun_qg_pca <- function(context, state, ...) {
  G_list <- context$payload$G_list
  Q_list <- context$meta$Q_list
  Qi_list <- context$meta$Qi_list
  p_grid <- context$cache$p_grid
  Ji_vec <- context$meta$Ji_vec
  p_star <- context$cache$p_star
  Q_star <- context$cache$Q_star
  supp_Y <- context$cache$supp_Y
  sqrt_w <- context$cache$sqrt_w
  loss_fun <- context$cache$loss_fun

  ## Choose qualifying dimension K
  if (is.null(state$K)) {
    out <- choose_near_lossless_K_qg_pca(
      epsilon = state$epsilon, alpha = state$alpha,
      V = state$V, K_max = state$K_max,
      loss_fun = loss_fun,
      G_list = G_list, Q_list = Q_list, Qi_list = Qi_list,
      p_grid = p_grid, Ji_vec = Ji_vec,
      p_star = p_star, Q_star = Q_star,
      sqrt_w = sqrt_w,
      supp_Y = supp_Y,
      lambda = state$lambda,
      min_dQ = state$min_dQ,
      construction = state$quantlet_construction,
      seed = state$seed
    )
    state$K <- out$K
    state$idx_outliers <- out$idx_outliers
  }

  ## Compute E (Q-mean-anchored by default; see stage_qg_pca docs)
  out <- compute_quantlets(
    G_list, state$K, sqrt_w,
    Q_list = Q_list, p_grid = p_grid, min_dQ = state$min_dQ,
    construction = state$quantlet_construction
  )

  state$G_center <- out$G_center
  state$E <- out$E
  state$sd_k <- out$sd_k
  state
}

encode_fun_qg_pca <- function(context, state) {
  Q_list <- context$meta$Q_list
  p_grid <- context$cache$p_grid
  p_star <- context$cache$p_star
  Q_star <- context$cache$Q_star
  sqrt_w <- context$cache$sqrt_w
  
  ## Set parameters
  N <- length(Q_list)
  converged <- vector(length = N)
  c_list <- vector('list', N)
  Q_star_list <- vector('list', N)
  for (i in 1:N) {
    out <- qg_pca(
      Q_obs = Q_list[[i]], 
      E = state$E, 
      G_center = state$G_center, 
      p_grid = p_grid,
      p_star = p_star,
      Q_star = Q_star,
      sqrt_w = sqrt_w,
      lambda = state$lambda
    )
    converged[i] <- out$converged
    c_list[[i]] <- out$c
    Q_star_list[[i]] <- out$Q_star
  }
  message(str_glue("converged = {sum(converged)} of {N}"))
  
  context$meta$idx_outliers <- state$idx_outliers
  context$payload = list(
    c_list = c_list,
    Q_star_list = Q_star_list
  )
  context
}

decode_fun_qg_pca <- function(context, state) {
  c_list <- context$payload$c_list
  Q_star_list <- context$payload$Q_star_list
  
  G_list <- lapply(c_list, function(c) state$G_center + rowSums(state$E %*% c))
  context$payload <- list(G_list = G_list, Q_star_list = Q_star_list)
  context
}

stage_qg_pca <- function(
    K = NULL,
    K_max = 20,
    epsilon = 0.01,
    alpha = 0.05,
    V = 5,
    lambda = 0,
    min_dQ = 1e-8,
    quantlet_construction = c("q_mean_residual", "pca"),
    seed = gen_seed()
) {
  quantlet_construction <- match.arg(quantlet_construction)
  new_stage(
    name = "qg_pca",
    input_space = "G",
    output_space = "C",
    requires_fit = TRUE,
    init_state = list(
      K = K,
      K_max = K_max,
      epsilon = epsilon,
      alpha = alpha,
      V = V,
      lambda = lambda,
      min_dQ = min_dQ,
      quantlet_construction = quantlet_construction,
      seed = seed
    ),
    fit_fun = fit_fun_qg_pca,
    encode_fun = encode_fun_qg_pca,
    decode_fun = decode_fun_qg_pca
  )
}


## ---------- Transform: WAME (Wasserstein-Aware Monotone Embeddings)
##
## Composite stage that fuses LQD (Q -> G) and QG-PCA (G -> C) into a single
## stage. Internally holds a `stage_lqd` and a `stage_qg_pca` instance and
## chains their fit / encode / decode methods, so a pipeline of
##   Y-axis -> Smooth -> WAME -> Flow
## is numerically equivalent to
##   Y-axis -> Smooth -> LQD -> QG-PCA -> Flow
## with the same seeds and parameters.

fit_fun_wame <- function(context, state, ...) {
  ctx_lqd <- encode(state$child_lqd, context, ...)
  state$child_qg_pca <- fit(state$child_qg_pca, ctx_lqd, ...)
  state
}

encode_fun_wame <- function(context, state, ...) {
  ctx_lqd <- encode(state$child_lqd, context, ...)
  encode(state$child_qg_pca, ctx_lqd, ...)
}

decode_fun_wame <- function(context, state, ...) {
  ctx_qg <- decode(state$child_qg_pca, context, ...)
  decode(state$child_lqd, ctx_qg, ...)
}

stage_wame <- function(
    min_dQ = 1e-8,
    K = NULL,
    K_max = 20,
    epsilon = 0.01,
    alpha = 0.05,
    V = 5,
    lambda = 0,
    quantlet_construction = c("q_mean_residual", "pca"),
    seed = gen_seed()
) {
  quantlet_construction <- match.arg(quantlet_construction)
  child_lqd    <- stage_lqd(min_dQ = min_dQ)
  child_qg_pca <- stage_qg_pca(
    K = K, K_max = K_max,
    epsilon = epsilon, alpha = alpha, V = V,
    lambda = lambda,
    min_dQ = min_dQ,
    quantlet_construction = quantlet_construction,
    seed = seed
  )
  new_stage(
    name         = "wame",
    input_space  = "Q",
    output_space = "C",
    requires_fit = TRUE,
    init_state   = list(
      child_lqd    = child_lqd,
      child_qg_pca = child_qg_pca
    ),
    fit_fun      = fit_fun_wame,
    encode_fun   = encode_fun_wame,
    decode_fun   = decode_fun_wame
  )
}


## ---------- Transform: Normalizing Flow

fit_fun_flow <- function(context, state, ...) {
  c_list <- context$payload$c_list
  Q_star_list <- context$payload$Q_star_list
  
  ## Get c-embeddings
  c_mat <- do.call(rbind, c_list)
  Q_star <- context$cache$Q_star
  if (is.null(Q_star)) {  ## append Q_star to c-embeddings
    c_mat <- cbind(c_mat, unlist(Q_star_list))
  }
  
  ## Train flow (in Python via reticulate)
  flow <- py_pkg$train_flow(
    C = c_mat,
    n_layers = state$n_layers,
    max_epochs = state$max_epochs,
    lr = state$lr,
    print_epoch = 50
  )
  py_pkg$save_flow(flow, state$path)
  
  state
}

encode_fun_flow <- function(context, state) {
  c_list <- context$payload$c_list
  Q_star_list <- context$payload$Q_star_list
  
  ## Get c-embeddings
  c_mat <- do.call(rbind, c_list)
  Q_star <- context$cache$Q_star
  if (is.null(Q_star)) {  ## append Q_star to c-embeddings
    c_mat <- cbind(c_mat, unlist(Q_star_list))
  }
  
  ## Load flow
  flow <- py_pkg$load_flow(
    dim = ncol(c_mat),
    n_layers = state$n_layers,
    path = state$path
  )
  
  ## Pass C through flow
  z_mat <- py_pkg$flow_encode(c_mat, flow)
  z_list <- split(z_mat, seq_len(nrow(z_mat)))
  
  context$payload <- z_list
  context
}


decode_fun_flow <- function(context, state) {
  z_list <- context$payload
  
  ## Get z-embeddings
  z_mat <- do.call(rbind, z_list)
  Q_star <- context$cache$Q_star
  K <- ifelse(is.null(Q_star), ncol(z_mat) - 1, ncol(z_mat))
  
  ## Load flow
  flow <- py_pkg$load_flow(
    dim = ncol(z_mat),
    n_layers = state$n_layers,
    path = state$path
  )
  
  ## Pass z_mat through inverse flow
  c_mat <- py_pkg$flow_decode(z_mat, flow)
  if (K > 1) {
    if (nrow(c_mat) == 1) {
      c_list <- list(c_mat[1, 1:K])
    } else {
      c_list <- asplit(c_mat[, 1:K], MARGIN = 1)
    }
  } else {
    if (nrow(c_mat) == 1) {
      c_list <- list(c_mat[1, 1])
    } else {
      c_list <- as.list(c_mat[, 1])
    }
  }
  if (is.null(Q_star)) {
    if (nrow(c_mat) == 1) {
      Q_star_list <- list(c_mat[1, K+1])
    } else {
      Q_star_list <- as.list(c_mat[, K+1])
    }
  } else {
    Q_star_list <- rep(list(Q_star), length(c_list))
  }
  
  context$payload <- list(
    c_list = c_list,
    Q_star_list = Q_star_list
  )
  context
}

stage_flow <- function(
  n_layers = 16,
  max_epochs = 200,
  lr = 1e-3,
  path = "flow.pth",
  seed = gen_seed()
) {
  new_stage(
    name = "flow",
    input_space = "C",
    output_space = "Z",
    requires_fit = TRUE,
    init_state = list(
      n_layers = as.integer(n_layers),
      max_epochs = as.integer(max_epochs),
      lr = lr,
      path = path,
      seed = seed
    ),
    fit_fun = fit_fun_flow,
    encode_fun = encode_fun_flow,
    decode_fun = decode_fun_flow
  )
}


## ---------- Transform: PCA Rotation

fit_fun_pca_rotation <- function(context, state, ...) {
  z_list <- context$payload

  Z_mat <- do.call(rbind, z_list)
  state$z_center <- colMeans(Z_mat)
  Z_c <- scale(Z_mat, center = state$z_center, scale = FALSE)

  s <- svd(Z_c, nu = 0)
  state$R <- s$v
  state$sd_k <- s$d / sqrt(nrow(Z_mat) - 1)

  state
}

encode_fun_pca_rotation <- function(context, state) {
  z_list <- context$payload

  Z_mat <- do.call(rbind, z_list)
  Z_rot <- sweep(Z_mat, 2, state$z_center, '-') %*% state$R

  if (nrow(Z_rot) == 1) {
    z_rot_list <- list(Z_rot[1, ])
  } else {
    z_rot_list <- asplit(Z_rot, MARGIN = 1)
  }

  context$payload <- z_rot_list
  context
}

decode_fun_pca_rotation <- function(context, state) {
  z_rot_list <- context$payload

  Z_rot <- do.call(rbind, z_rot_list)
  Z_mat <- sweep(Z_rot %*% t(state$R), 2, state$z_center, '+')

  if (nrow(Z_mat) == 1) {
    z_list <- list(Z_mat[1, ])
  } else {
    z_list <- asplit(Z_mat, MARGIN = 1)
  }

  context$payload <- z_list
  context
}

stage_pca_rotation <- function(seed = gen_seed()) {
  new_stage(
    name = "pca_rotation",
    input_space = "Z",
    output_space = "Z",
    requires_fit = TRUE,
    init_state = list(seed = seed),
    fit_fun = fit_fun_pca_rotation,
    encode_fun = encode_fun_pca_rotation,
    decode_fun = decode_fun_pca_rotation
  )
}



## ========== Pipeline ==========

construct_pipeline <- function(
    stages,
    supp_Y = NULL,
    p_star = 0.5,
    Q_star = NULL,
    y_min = NULL,
    loss = "wasserstein", # TODO: Move to qg_pca state
    ratio_trans = NULL, # TODO: Move to eqf_cgrid state
    cache_init = NULL,
    meta_init = NULL,
    seed = 12345
) {
  
  ## ---------- Parameter Validation
  
  ## TODO: Validate stage sequence using input/output spaces
  
  ## ---------- Cache Initialization
  
  ## Initialize cache
  if (is.null(cache_init)) {
    cache_init <- list(
      supp_Y = supp_Y,
      p_star = p_star,
      Q_star = Q_star,
      Q_min = y_min,
      loss_fun = loss_to_fun[[loss]],
      ratio_trans = ratio_trans,
      seed = seed
    ) 
  }
  
  set.seed(seed)
  pipe <- new_pipeline(stages = stages)
  pipe$cache_init <- cache_init
  pipe$meta_init <- meta_init
  pipe
}
