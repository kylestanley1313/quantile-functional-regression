library(glue)
library(irlba)
library(minpack.lm)
library(pracma)
library(purrr)
library(stringr)
library(yaml)

## Prepare Python
library(reticulate)
config <- read_yaml(file.path("src", "config.yml"))
use_python(config$python$path, required = TRUE)
py_run_string(str_glue("
import sys
if '{config$python$module_path}' not in sys.path:
    sys.path.insert(0, 'config$python$module_path')
"))
py_flow <- import("flow", convert = TRUE)



## ========== Utilities ==========

new_context <- function(payload, cache = NULL, meta = NULL) {
  list(
    payload = payload,
    cache = cache,
    meta = meta
  )
}

gen_seed <- function() {
  sample.int(.Machine$integer.max, 1)
}

get_uniform_p_grid <- function(J) {
  1:J / (1 + J)
}

log_transform <- function(x, inverse = FALSE) {
  if (inverse) {
    exp(x) - 1
  } else {
    log(x + 1)
  }
}

one_minus_sqcor <- function(vec_obs, vec_est) {
  1 - cor(vec_obs, vec_est)^2
}

normalized_mse <- function(vec_obs, vec_est) {
  sum((vec_obs - vec_est)^2) / sum(vec_obs^2)
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

# compute_quantlets_old <- function(G_list, K) {
# compute_quantlets <- function(G_list, Q_list, p_grid, p_star, K) {
#   G_mat <- do.call(rbind, G_list)
#   G_center <- colMeans(G_mat)
#   G_mat <- scale(G_mat, center = G_center, scale = FALSE)
#   E <- irlba(G_mat, nv = K, nu = 0)$v
#   list(E = E, G_center = G_center)
# }

jacobian_inv_lqd <- function(G_center, p_grid, p_star) {

  J <- length(G_center)
  stopifnot(length(p_grid) == J)

  p_star_idx <- which.min(abs(p_grid - p_star))
  dQ <- exp(G_center)

  ## Allocate Jacobian
  Jmat <- matrix(0, nrow = J, ncol = J)

  ## Forward direction: j > p_star_idx
  if (p_star_idx < J) {
    for (j in (p_star_idx + 1):J) {
      for (k in p_star_idx:(j - 1)) {
        dp <- p_grid[k + 1] - p_grid[k]
        w <- if (k == p_star_idx || k == j - 1) 0.5 else 1
        Jmat[j, k] <- Jmat[j, k] + w * dp * dQ[k]
        Jmat[j, k + 1] <- Jmat[j, k + 1] + w * dp * dQ[k + 1]
      }
    }
  }

  ## Backward direction: j < p_star_idx
  if (p_star_idx > 1) {
    for (j in 1:(p_star_idx - 1)) {
      for (k in j:(p_star_idx - 1)) {
        dp <- p_grid[k + 1] - p_grid[k]
        w <- if (k == j || k == p_star_idx - 1) 0.5 else 1
        Jmat[j, k] <- Jmat[j, k] - w * dp * dQ[k]
        Jmat[j, k + 1] <- Jmat[j, k + 1] - w * dp * dQ[k + 1]
      }
    }
  }

  Jmat
}

compute_quantlets <- function(G_list, Q_list, p_grid, p_star, K) {

  ## Stack G and Q
  G_mat <- do.call(rbind, G_list)
  Q_mat <- do.call(rbind, Q_list)

  ## Centers
  G_center <- colMeans(G_mat)
  Q_center <- colMeans(Q_mat)

  ## Center G
  Gc <- scale(G_mat, center = G_center, scale = FALSE)

  ## --- 1. Jacobian of inverse LQD at mean ---
  J <- jacobian_inv_lqd(
    G_center = G_center,
    p_grid = p_grid,
    p_star = p_star
  )
  # J is (length(p_grid) x length(p_grid))

  ## --- 2. Pull back Q-metric to G-space ---
  # Equivalent to PCA under metric J^T J
  G_tilde <- Gc %*% t(J)

  ## --- 3. PCA in pulled-back space ---
  svd_out <- irlba(G_tilde, nv = K, nu = 0)

  ## --- 4. Map basis back to original G-space ---
  E <- svd_out$v
  # Columns of E are G-space basis vectors optimized for Q-loss

  list(
    E = E,
    G_center = G_center
  )
}

inv_eqf_cgrid <- function(Q, p_grid, pi_grid, supp_TY = NULL) {
  
  ## Linearly interpolate onto common grid
  ## NOTE: `rule = 2` indicates that the value at the closest data extreme 
  ## is used for extrapolation
  Qi <- approx(x = p_grid, y = Q, xout = pi_grid, rule = 2)$y
  
  ## Get step function
  if (!is.null(supp_TY)) {
    idx <- findInterval(Qi, supp_TY, rightmost.closed = FALSE)
    idx <- pmax(idx, 1)
    idx <- pmin(idx, length(supp_TY))
    Qi <- supp_TY[idx]
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
    sqrt_w = NULL,
    c_init = NULL, 
    Q_star_init = NULL,
    control = nls.lm.control(maxiter = 200, ftol = 1e-10)
) {
  
  ## Set parameters
  K <- ncol(E)
  J <- nrow(E)
  w <- sqrt_w^2
  
  ## Set p_star_idx
  p_star_idx <- which.min(abs(p_grid - p_star))
  
  ## Set initial values for c and Q_star
  if (is.null(c_init)) c_init <- rep(0, K)
  if (is.null(Q_star_init)) {
    ## Compute Q_hat for c_init to find Q_star that matches pth quantile
    G_0 <- G_center + as.vector(E %*% c_init)
    Q_hat_0 <- inv_lqd(G_0, p_grid, p_star = p_star, Q_star = 0)
    Q_star_init <- sum(w * (Q_obs - Q_hat_0)) / sum(w)
  }
  
  ## Residual function (return vector of length Ji)
  resid_fun <- function(par) {
    c <- par[1:K]
    Q_star <- par[K+1]
    G_hat <- G_center + as.vector(E %*% c)
    Q_hat <- inv_lqd(G_hat, p_grid, p_star, Q_star)
    r <- Q_obs - Q_hat
    if (!is.null(sqrt_w)) r <- sqrt_w * r
    r
  }
  
  ## Jacobian function
  jac_fun <- function(par) {
    
    ## Parameters
    c <- par[1:K]
    
    ## Recompute G and exp(G)
    G_hat <- G_center + as.vector(E %*% c)
    dQ <- exp(G_hat)
    
    ## Precompute f_jk = exp(G_j) * E_jk
    F  <- dQ * E                     # J x K
    dp <- diff(p_grid)               # length J-1
    
    ## Trapezoidal increments: interval (j-1, j)
    trap <- 0.5 * (F[-1, , drop = FALSE] +
                     F[-J, , drop = FALSE]) *
      dp                       # (J-1) x K
    
    ## Allocate Jacobian
    J_mat <- matrix(0, nrow = J, ncol = K + 1)
    
    ## ---- Forward part: j > p_star_idx ----
    if (p_star_idx < J) {
      cs_fwd <- apply(
        trap[p_star_idx:(J - 1), , drop = FALSE],
        2,
        cumsum
      )
      J_mat[(p_star_idx + 1):J, 1:K] <- -cs_fwd
    }
    
    ## ---- Backward part: j < p_star_idx ----
    if (p_star_idx > 1) {
      idx <- 1:(p_star_idx - 1)
      
      cs_bwd <- apply(
        trap[idx, , drop = FALSE][length(idx):1, , drop = FALSE],
        2,
        cumsum
      )
      
      J_mat[idx, 1:K] <- cs_bwd[length(idx):1, , drop = FALSE]
    }
    
    ## ---- Derivative w.r.t Q_star ----
    J_mat[, K + 1] <- -1
    
    if (!is.null(sqrt_w)) J_mat <- diag(sqrt_w) %*% J_mat
    J_mat
  }
  
  par_init <- c(c_init, Q_star_init)
  fit <- nls.lm(par = par_init, fn = resid_fun, jac = jac_fun, control = control)
  list(par = fit$par, fit = fit, converged = (fit$info %in% c(1,2)))
}

choose_near_lossless_K <- function(
    epsilon, alpha, V, K_max, loss_fun,
    G_list, Q_list, Qi_list, 
    p_grid, Ji_vec, p_star, 
    sqrt_w = NULL,
    supp_TY = NULL,
    seed = 12345
) {
  
  ## Set parameters
  N <- length(G_list)
  w <- sqrt_w^2
  
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
        G_list[idx_train], 
        Q_list[idx_train], 
        p_grid, 
        p_star, 
        K
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
          sqrt_w = sqrt_w
        )
        converged[i] <- out$converged
        c <- out$par
        
        ## Map c --> G --> Q
        G <- G_center + rowSums(E %*% c[1:K])
        Q <- inv_lqd(
          G = G, 
          p_grid = p_grid, 
          p_star = p_star, 
          Q_star = c[K+1]
        )
        Qi <- inv_eqf_cgrid(
          Q = Q,
          p_grid = p_grid,
          pi_grid = get_uniform_p_grid(Ji_vec[i]),
          supp_TY = supp_TY
        )
        
        ## Compute loss
        losses[i] <- loss_fun(Qi_list[[i]], Qi)
        
      }
    }
    
    ## CV result reporting
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
      return(K)
    }
    
    K <- K + 1
  }
  
  warning(glue(
    "Qualifying dimension not found. Candidates exhausted. ",
    "Setting K = {K_max} = K_max."
  ))
  return(K)
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
    meta = list()
  )
  
  for (i in seq_len(pipeline$n_stages)) { 
    stage <- pipeline$stages[[i]] 
    if (stage$requires_fit && !stage$fitted) { 
      stage <- fit(stage, context, ...) 
      pipeline$stages[[i]] <- stage 
    } 
    context <- encode(stage, context, ...) 
  } 
  
  pipeline$state <- pipeline$state
  pipeline$training <- context$meta
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

## ---------- Transform: Y-Axis

encode_fun_y_axis <- function(context, state) {
  y_list <- context$payload
  Ty_list <- lapply(y_list, context$cache$y_trans)
  context$payload <- Ty_list
  context
}

decode_fun_y_axis <- function(context, state) {
  Ty_list <- context$payload
  y_list <- lapply(Ty_list, function(Ty) context$cache$y_trans(Ty, inverse = TRUE))
  context$payload <- y_list
  context
}


## ---------- Transform: EQF on Subject Grid

encode_fun_eqf_sgrid <- function(context, state) {
  Ty_list <- context$payload
  Qi_list <- lapply(Ty_list, function(Ty) sort(Ty))
  context$payload <- Qi_list
  context$meta$Qi_list <- Qi_list
  context$meta$Ji_vec <- lengths(Qi_list)
  context
}

decode_fun_eqf_sgrid <- function(context, state) {
  Qi_list <- context$payload
  Ty_list <- Qi_list  ## recovers input up to rearrangement
  context$payload <- Ty_list
  context
}


## ---------- Transform: Smooth EQF on Dense Common Grid

encode_fun_eqf_cgrid <- function(context, state) {
  Qi_list <- context$payload
  Ji_vec <- context$meta$Ji_vec
  J <- length(context$cache$p_grid)
  p_grid <- context$cache$p_grid
  y_trans <- context$cache$y_trans
  ratio_trans <- context$cache$ratio_trans
  y_min <- context$cache$y_min
  
  ## Check p_grid
  max_Ji <- max(Ji_vec)
  min_Qi <- 1 / (1 + max_Ji)
  max_Qi <- max_Ji / (1 + max_Ji)
  if (p_grid[1] < min_Qi || p_grid[J] > max_Qi) {
    stop("Can't evaluate EQFs on tails of p_grid!")
  }
  if (p_grid[1] > min_Qi || p_grid[J] < max_Qi) {
    warning(glue(
      "Add points to the left and/or right tail of p_grid ",
      "to preserve tail information!"
    ))
  }
  
  ## Set parameters
  N <- length(Qi_list)
  Q_min <- NULL
  if (!is.null(y_min)) {
    Q_min <- y_trans(y_min)
  }
  Q_mat <- matrix(nrow = N, ncol = J)
  ratio_trans <- ratio_trans
  
  ## Interpolate onto interior points
  for (i in 1:N) {
    
    ## Get Qi for this subject
    Qi <- Qi_list[[i]]
    
    ## Get subject grid
    pi_grid <- get_uniform_p_grid(Ji_vec[i])
    
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
    }
    
    ## Interpolate onto interior through anchors
    Q_mat[i,interior_mask] <- spinterp(x = a, y = v, xp = p_grid[interior_mask])
    
  }
  
  ## Get interior sets
  exterior_cols <- which(colSums(is.na(Q_mat)) > 0)
  interior_sets <- lapply(
    exterior_cols,
    function(j) which(!is.na(Q_mat[, j]))
  )
  names(interior_sets) <- exterior_cols
  
  ## Ratio-based extrapolation in left and right tails
  p_mid <- J / 2
  ext_cols_left <- sort(exterior_cols[exterior_cols < p_mid], decreasing = TRUE)
  ext_cols_right <- sort(exterior_cols[exterior_cols > p_mid])
  for (j in c(ext_cols_left, ext_cols_right)) {
    j_ <- ifelse(j < p_mid, j + 1, j - 1)
    I_j <- interior_sets[[as.character(j)]]  ## interior indices
    E_j <- which(is.na(Q_mat[,j]))           ## exterior indices
    if (is.null(ratio_trans)) {
      ratio_trans <- function(x, ...) x
    }
    scale_j <- ratio_trans(
      mean(ratio_trans(Q_mat[I_j,j] / Q_mat[I_j,j_])), 
      inverse = TRUE
    )
    Q_mat[E_j,j] <- scale_j * Q_mat[E_j,j_]
  }
  
  Q_list <- asplit(Q_mat, MARGIN = 1)
  context$payload <- Q_list
  context$meta$Q_list <- Q_list
  context
}

decode_fun_eqf_cgrid <- function(context, state) {
  Q_list <- context$payload
  Ji_vec <- context$meta$Ji_vec
  if (is.null(Ji_vec))
    stop("decode_fun_eqf_cgrid requires Ji_vec in context$meta")
  p_grid <- context$cache$p_grid
  J <- length(p_grid)
  supp_TY <- context$cache$supp_TY
  
  ## Linearly interpolate through p_grid onto pi_grid
  N <- length(Q_list)
  Qi_list <- vector(mode = "list", length = N)
  for (i in 1:N) {
    pi_grid <- get_uniform_p_grid(Ji_vec[i])
    Qi_list[[i]] <- inv_eqf_cgrid(Q_list[[i]], p_grid, pi_grid, supp_TY)
  }
  
  context$payload <- Qi_list
  context
}


## ---------- Transform: LQD

encode_fun_lqd <- function(context, state) {
  Q_list <- context$payload
  p_star_idx <- context$cache$p_star_idx
  
  ## Set parameters
  N <- length(Q_list)
  J <- length(Q_list[[1]])
  p_grid <- context$cache$p_grid
  
  ## Get integration constants
  Q_star_list <- lapply(Q_list, function(Q) Q[[p_star_idx]])
  
  ## LQD transform Q_list
  G_list <- vector("list", N)
  for (i in 1:N) {
    dQ <- differentiate(Q_list[[i]], p_grid)
    dQ[dQ < state$min_dQ] <- state$min_dQ
    G_list[[i]] <- log(dQ)
  }
  
  context$payload <- list(G_list = G_list, Q_star_list = Q_star_list)
  context
}

decode_fun_lqd <- function(context, state) {
  G_Q_star_list <- context$payload
  p_grid <- context$cache$p_grid
  p_star <- context$cache$p_star
  
  ## Extract lists
  G_list <- G_Q_star_list$G_list
  Q_star_list <- G_Q_star_list$Q_star_list
  
  ## Set parameters
  N <- length(G_list)
  J <- length(G_list[[1]])
  
  ## Inverse LQD transform G_list 
  Q_list <- vector("list", N)
  for (i in 1:N) {
    Q_list[[i]] <- inv_lqd(G_list[[i]], p_grid,  p_star, Q_star_list[[i]])
  }
  
  context$payload <- Q_list
  context
}


## ---------- Transform: Q-G PCA

fit_fun_qg_pca <- function(context, state, ...) {
  G_list <- context$payload$G_list
  Q_list <- context$meta$Q_list
  Qi_list <- context$meta$Qi_list
  p_grid <- context$cache$p_grid
  Ji_vec <- context$meta$Ji_vec
  p_star <- context$cache$p_star
  supp_TY <- context$cache$supp_TY
  sqrt_w <- context$cache$sqrt_w
  loss_fun <- context$cache$loss_fun
  
  ## Choose qualifying dimension K
  if (is.null(state$K)) {
    print(state$epsilon)
    state$K <- choose_near_lossless_K(
      epsilon = state$epsilon, alpha = state$alpha, 
      V = state$V, K_max = state$K_max, loss_fun = loss_fun,
      G_list = G_list, Q_list = Q_list, Qi_list = Qi_list, 
      p_grid = p_grid, Ji_vec = Ji_vec, p_star = p_star,
      sqrt_w = sqrt_w,
      supp_TY = supp_TY,
      seed = state$seed
    )
  } 
  
  ## Compute E
  out <- compute_quantlets(G_list, Q_list, p_grid, p_star, state$K)
  
  state$G_center <- out$G_center
  state$E <- out$E
  state
}

encode_fun_qg_pca <- function(context, state) {
  Q_list <- context$meta$Q_list
  p_grid <- context$cache$p_grid
  p_star <- context$cache$p_star
  sqrt_w <- context$cache$sqrt_w
  
  ## Set parameters
  N <- length(Q_list)
  converged <- vector(length = N)
  c_list <- vector('list', N)
  for (i in 1:N) {
    out <- qg_pca(
      Q_obs = Q_list[[i]], 
      E = state$E, 
      G_center = state$G_center, 
      p_grid = p_grid,
      p_star = p_star,
      sqrt_w = sqrt_w
    )
    converged[i] <- out$converged
    c_list[[i]] <- out$par
  }
  message(str_glue("converged = {sum(converged)} of {N}"))
  
  context$payload <- c_list
  context
}

decode_fun_qg_pca <- function(context, state) {
  c_list <- context$payload
  G_list <- lapply(c_list, function(c) state$G_center + rowSums(state$E %*% c[1:state$K]))
  Q_star_list <- lapply(c_list, function(c) c[state$K + 1])
  context$payload <- list(G_list = G_list, Q_star_list = Q_star_list)
  context
}


## ---------- Transform: Normalizing Flow

fit_fun_flow <- function(context, state, ...) {
  C_list <- context$payload
  C_mat <- do.call(rbind, C_list)
  
  ## Train flow (in Python via reticulate)
  flow <- py_flow$train_flow(
    C = C_mat,
    n_layers = state$n_layers,
    epochs = state$epochs,
    lr = state$lr
  )
  py_flow$save_flow(flow, state$flow_path)
  
  state$C_dim <- ncol(C_mat)
  state
}

encode_fun_flow <- function(context, state) {
  C_list <- context$payload
  C_mat <- do.call(rbind, C_list)
  
  ## Load flow
  flow <- py_flow$load_flow(
    dim = state$C_dim,
    n_layers = state$n_layers,
    path = state$flow_path
  )
  
  ## Pass C through flow
  Z_mat <- py_flow$flow_encode(C_mat, flow)
  Z_list <- split(Z_mat, seq_len(nrow(Z_mat)))
  
  context$payload <- Z_list
  context
}

decode_fun_flow <- function(context, state) {
  Z_list <- context$payload
  Z_mat <- do.call(rbind, Z_list)
  
  ## Load flow
  flow <- py_flow$load_flow(
    dim = state$C_dim,
    n_layers = state$n_layers,
    path = state$flow_path
  )
  
  ## Pass Z_mat through inverse flow
  C_mat <- py_flow$flow_decode(Z_mat, flow)
  C_list <- split(C_mat, seq_len(nrow(C_mat)))
  
  context$payload <- C_list
  context
}


## ========== Pipeline ==========

construct_pipeline <- function(
    p_grid,
    supp_Y,   
    y_trans, 
    K,
    epsilon,
    alpha,
    K_max,
    loss_fun,
    V,
    p_star = 0.5,
    ratio_trans = NULL,
    y_min = NULL, 
    min_dQ = 1e-8, 
    flow_n_layers = 16,
    flow_epochs = 200,
    flow_lr = 1e-3,
    flow_path = "flow.pth",
    seed = 12345
) {
  
  ## Storage Locations
  ##    state   --> returned by fit functions
  ##            --> property of stages that tells you how to encode/decode
  ##            --> Ex: (Q-G PCA) K, epsilon, alpha, V, etc.
  ##    context --> returned by encode/decode functions
  ##            --> contains payload/cache/meta
  ##                  * payload --> transformed by encode/decode
  ##                      Ex: y_list, Ty_list, Qi_list, etc.
  ##                  * cache --> data-agnostic quantities used across stages
  ##                      Ex: p_grid, p_star, loss_fun, etc.
  ##                  * meta --> data-aware quantities used across stages
  ##                      Ex: Qi_list, Q_list, etc.
  
  
  ## ---------- Parameter Validation
  
  ## Validate user-specified p_grid
  J <- length(p_grid)
  if (!(
    all(diff(p_grid) > 0) &&  ## strictly increasing
    J >= 3 &&
    p_grid[1] > 0 &&
    p_grid[J] < 1
  )) {
    stop("Invalid p-grid!")
  }
  
  
  ## ---------- Cache Initialization
  
  ## Compute weights that will later be used to account for unbalanced grid
  dp <- diff(p_grid)
  w <- numeric(J)
  w[1] <- 0.5 * dp[1]
  w[J] <- 0.5 * dp[J - 1]
  w[2:(J-1)] <- 0.5 * (dp[1:(J-2)] + dp[2:(J-1)])
  
  ## Get default y_trans
  if (is.null(y_trans)) {
    y_trans <- function(x, inverse) x
  }
  
  ## Derive transformed support
  if (!is.null(supp_Y)) {
    supp_TY <- y_trans(supp_Y)
  } else {
    supp_TY <- NULL
  }
  
  ## Initialize cache
  cache_init <- list(
    p_grid = p_grid,
    dp = diff(p_grid),
    w = w,
    sqrt_w = sqrt(w),
    p_star = p_star,
    p_star_idx = which.min(abs(p_grid - p_star)),
    supp_TY = supp_TY,
    y_trans = y_trans,
    ratio_trans = ratio_trans,
    y_min = y_min,
    loss_fun = loss_fun
  )
  
  
  ## ---------- Stages
  set.seed(seed)
  
  y_axis_stage <- new_stage(
    name = "y_axis",
    input_space = "Y",
    output_space = "TY",
    requires_fit = FALSE,
    encode_fun = encode_fun_y_axis,
    decode_fun = decode_fun_y_axis
  )
  
  eqf_sgrid_stage <- new_stage(
    name = "eqf_sgrid",
    input_space = "TY",
    output_space = "Qi",
    requires_fit = FALSE,
    encode_fun = encode_fun_eqf_sgrid,
    decode_fun = decode_fun_eqf_sgrid
  )
  
  eqf_cgrid_stage <- new_stage(
    name = "eqf_cgrid",
    input_space = "Qi",
    output_space = "Q",
    requires_fit = FALSE,
    encode_fun = encode_fun_eqf_cgrid,
    decode_fun = decode_fun_eqf_cgrid
  )
  
  lqd_stage <- new_stage(
    name = "lqd",
    input_space = "Q",
    output_space = "G",
    requires_fit = FALSE,
    init_state = list(min_dQ = min_dQ),
    encode_fun = encode_fun_lqd,
    decode_fun = decode_fun_lqd
  )
  
  qg_pca_stage <- new_stage(
    name = "qg_pca",
    input_space = "G",
    output_space = "C",
    requires_fit = TRUE,
    init_state = list(
      K = K,
      epsilon = epsilon,
      alpha = alpha,
      K_max = K_max,
      V = V,
      seed = gen_seed()
    ),
    fit_fun = fit_fun_qg_pca,
    encode_fun = encode_fun_qg_pca,
    decode_fun = decode_fun_qg_pca
  )
  
  flow_stage <- new_stage(
    name = "flow",
    input_space = "C",
    output_space = "Z",
    requires_fit = TRUE,
    init_state = list(
      n_layers = as.integer(flow_n_layers),
      epochs = as.integer(flow_epochs),
      lr = flow_lr,
      flow_path = flow_path,
      seed = gen_seed()
    ),
    fit_fun = fit_fun_flow,
    encode_fun = encode_fun_flow,
    decode_fun = decode_fun_flow
  )
  
  pipeline <- new_pipeline(
    stages = list(
      y_axis_stage,
      eqf_sgrid_stage,
      eqf_cgrid_stage,
      lqd_stage,
      qg_pca_stage,
      flow_stage
    )
  )
  
  pipeline$cache_init <- cache_init
  pipeline
}
