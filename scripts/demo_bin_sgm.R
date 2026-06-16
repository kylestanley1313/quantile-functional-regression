library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## ---------- Colors ---------- ##

col_recon <- rgb(0, 1, 0, alpha = 0.5)
col_0 <- rgb(0, 0, 0, alpha = 0.5)
col_1 <- rgb(1, 0, 0, alpha = 0.5)
col_train <- 'black'
col_draw <- 'gray'
col_outlier <- rgb(1, 1, 0.75)
col_z_mean <- 'red'
col_c_mean <- 'green'
col_Q_mean <- 'blue'
col_draw_trans <- rgb(0.5, 0.5, 0.5, alpha = 0.5)
col_z_mean_trans <- rgb(1, 0, 0, alpha = 0.5)
col_c_mean_trans <- rgb(0, 1, 0, alpha = 0.5)
col_Q_mean_trans <- rgb(0, 0, 1, alpha = 0.5)



## ---------- Representation Learning ---------- ##

## Globals
dataset <- 'tean'
pipe_name <- str_glue('flow_{dataset}')
n_stage_plots <- 3
plot_recons <- TRUE

set.seed(12345)
if (pipe_name == 'flow_tean') {
  
  ## Load data
  path <- file.path('data', 'processed', 'tean_v2.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, J = 100,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'log', y_shift = 1),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 10,
        epsilon = 0.25,
        alpha = 0.01,
        V = 5,
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_tean.pth'
      )
    ),
    supp_Y = 0:(2*y_max),
    p_star = 0,
    y_star = y_star,
    y_min = 0,
    loss = 'wasserstein',
    loss_scale = 'median_pairwise_wasserstein',
    loss_scale_samp_rate = 0.1,
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_bicluster') {

  ## Load data
  path <- file.path('data', 'processed', 'bicluster_v4.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, J = 100
  )
  
  ## Construct pipeline
  y_star <- NULL
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'identity', y_shift = 0),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 20,
        epsilon = 0.15,
        alpha = 0.05,
        V = 5,
        lambda = 1e-6
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_bicluster.pth'
      )
    ),
    supp_Y = NULL,
    p_star = 0.5,
    y_star = y_star,
    y_min = NULL,
    loss = 'wasserstein',
    loss_scale = 'median_pairwise_wasserstein',
    loss_scale_samp_rate = 0.1,
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_nhanes') {
  
  ## Load data
  path <- file.path('data', 'processed', 'tean_v2.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, J = 100,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'log', y_shift = 1),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 10,
        epsilon = 0.15,
        alpha = 0.05,
        V = 5,
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_tean.pth'
      )
    ),
    supp_Y = 0:(2*y_max),
    p_star = 0,
    y_star = y_star,
    y_min = 0,
    loss = 'wasserstein',
    loss_scale = 'median_pairwise_wasserstein',
    loss_scale_samp_rate = 0.1,
    seed = gen_seed()
  )
  
} else {
  stop("Invalid pipe_name!")
}

## Fitting
pipeline <- fit(pipeline, y_list)

## Save pipeline
path <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'pipe.rds')
saveRDS(pipeline, path)

## New context
y_ctx <- new_context(
  payload = y_list,
  cache = pipeline$training$cache,
  meta = list()
)

## Encode/Decode
Ty_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Qi_ctx <- encode(pipeline, Ty_ctx, from = 1, to = 2)
Q_ctx <- encode(pipeline, Qi_ctx, from = 2, to = 3)
G_Q_star_ctx <- encode(pipeline, Q_ctx, from = 3, to = 4)
c_ctx <- encode(pipeline, G_Q_star_ctx, from = 4, to = 5)
z_ctx <- encode(pipeline, c_ctx, from = 5, to = 6)
c_ctx_ <- decode(pipeline, z_ctx, from = 6, to = 5)
G_Q_star_ctx_ <- decode(pipeline, c_ctx_, from = 5, to = 4)
Q_ctx_ <- decode(pipeline, G_Q_star_ctx_, from = 4, to = 3)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 3, to = 2)
Ty_ctx_ <- decode(pipeline, Qi_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Ty_ctx_, from = 1, to = 0)

## Plot stages
for (i in 1:2) {
  pi_grid <- pi_grid_fun(Ji_vec[[i]])
  y_max_i <- max(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
  y_min_i <- min(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
  Ty_max_i <- max(c(Ty_ctx$payload[[i]], Ty_ctx_$payload[[i]]))
  Ty_min_i <- min(c(Ty_ctx$payload[[i]], Ty_ctx_$payload[[i]]))
  breaks_y <- seq(y_min_i, y_max_i, length.out = 50)
  breaks_Ty <- seq(Ty_min_i, Ty_max_i, length.out = 50)
  
  par(mfrow=c(2,4))
  h <- hist(y_ctx$payload[[i]], breaks = breaks_y)
  hist(y_ctx_$payload[[i]], add = TRUE, col = col_recon, breaks = breaks_y)
  h <- hist(Ty_ctx$payload[[i]], breaks = breaks_Ty)
  hist(Ty_ctx_$payload[[i]], add = TRUE, col = col_recon, breaks = breaks_Ty)
  plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1))
  lines(pi_grid, Qi_ctx_$payload[[i]], type = 'l', col = col_recon)
  plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1), col = 'gray')
  lines(p_grid, Q_ctx$payload[[i]], type = 'l')
  lines(p_grid, Q_ctx_$payload[[i]], type = 'l', col = col_recon)
  plot(p_grid, G_Q_star_ctx$payload$G_list[[i]], type = 'l', xlim = c(0, 1))
  lines(p_grid, G_Q_star_ctx_$payload$G_list[[i]], type = 'l', col = col_recon)
  plot(c_ctx$payload$c_list[[i]])
  points(c_ctx_$payload$c_list[[i]], col = col_recon)
  plot(z_ctx$payload[[i]])
}


## Plot reconstructions
if (plot_recons) {
  par(mfrow = c(1,1))
  K_plot_min <- 3
  
  ## Plot z-embeddings
  Z_plot <- do.call(rbind, z_ctx$payload)
  Z_plot <- Z_plot[,1:min(ncol(Z_plot), K_plot_min)]
  plot_embeddings(Z_plot)
  
  ## Plot c-embeddings
  C_plot <- do.call(rbind, c_ctx$payload$c_list)
  C_plot <- C_plot[,1:min(ncol(C_plot), K_plot_min)]
  plot_embeddings(C_plot)
  
  ## Plot Smooth EQFs on common grid
  plot_funs(
    fun_list = Q_ctx$payload,
    grid_list = rep(list(p_grid), N)
  )
}


## ---------- EDA ---------- ##

if (dataset == 'tean') {
  
  ## Read covariates
  path <- file.path('data', 'processed', 'tean_v2_cov.rds')
  df_cov <- readRDS(path)
  
  ## Match y_list and df_cov indices
  df_cov <- df_cov[match(names(y_list), df_cov$id),]
  
  ## Create binary covariates
  df_cov$high_bmi <- as.numeric(df_cov$bmi_pct > 90)
  df_cov$old <- as.numeric(df_cov$age > 14)
  
  ## Set binary predictor for plotting
  bin_pred <- 'old'
  
  ## Plot: QF
  mask <- !is.na(df_cov[bin_pred])
  Q_to_plot <- Q_ctx$payload[mask]
  df_cov_to_plot <- df_cov[mask,]
  fun_list <- c(
    Q_to_plot
  )
  grid_list <- rep(list(p_grid), length(fun_list))
  colors = rep(col_0, length.out = length(fun_list))
  colors[df_cov_to_plot[bin_pred] == 1] <- col_1
  widths <- rep(1, length(fun_list))
  types <- rep(1, length(fun_list))
  color_width_type_labels <- data.frame(
    color = c(col_0, col_1),
    width = c(1, 1),
    type  = c(1, 1),
    label = c('0', '1')
  )
  plot_funs(
    fun_list = fun_list,
    grid_list = grid_list,
    colors = colors,
    widths = widths,
    types = types,
    color_width_type_labels = color_width_type_labels
  )
  
  ## Plot: Z
  mask <- !is.na(df_cov[bin_pred])
  Z_to_plot <- rbind(
    do.call(rbind, z_ctx$payload[mask])
  )[,1:3]
  df_cov_to_plot <- df_cov[mask,]
  colors = rep(col_0, length.out = length(fun_list))
  colors[df_cov_to_plot[bin_pred] == 1] <- col_1
  shapes <- c(
    rep(19, length.out = nrow(Z_to_plot))
  )
  sizes <- c(
    rep(1, length.out = nrow(Z_to_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(19, 19),
    size = c(1, 1),
    label = c('0', '1')
  )
  plot_embeddings(
    Z_to_plot, 
    colors = colors,
    shapes = shapes,
    sizes = sizes,
    color_shape_size_labels = color_shape_size_labels,
    stats = FALSE
  )
  
  ## Plot: C
  mask <- !is.na(df_cov[bin_pred])
  C_to_plot <- rbind(
    do.call(rbind, c_ctx$payload$c_list[mask])
  )[,1:4]
  df_cov_to_plot <- df_cov[mask,]
  colors = rep(col_0, length.out = nrow(Z_to_plot))
  colors[df_cov_to_plot[bin_pred] == 1] <- col_1
  shapes <- c(
    rep(19, length.out = nrow(C_to_plot))
  )
  sizes <- c(
    rep(1, length.out = nrow(C_to_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(19, 19),
    size = c(1, 1),
    label = c('0', '1')
  )
  plot_embeddings(
    C_to_plot, 
    colors = colors,
    shapes = shapes,
    sizes = sizes,
    color_shape_size_labels = color_shape_size_labels,
    stats = FALSE
  )
  
} else if (dataset == 'bicluster') {
  
  ## Read covariates
  path <- file.path('data', 'processed', 'bicluster_v4_groups.rds')
  x <- readRDS(path)
  idx <- sample(1:length(x))
  
  ## Plot: QF
  fun_list <- Q_ctx$payload[idx]
  grid_list <- rep(list(p_grid), length(fun_list))
  colors = rep(col_0, length.out = length(fun_list))
  colors[x[idx] == 1] <- col_1
  widths <- rep(1, length(fun_list))
  types <- rep(1, length(fun_list))
  color_width_type_labels <- data.frame(
    color = c(col_0, col_1),
    width = c(1, 1),
    type  = c(1, 1),
    label = c('0', '1')
  )
  plot_funs(
    fun_list = fun_list,
    grid_list = grid_list,
    colors = colors,
    widths = widths,
    types = types,
    color_width_type_labels = color_width_type_labels
  )
  
  ## Plot: Z
  Z_to_plot <- do.call(rbind, z_ctx$payload[idx])[,1:3]
  colors = rep(col_0, length.out = length(fun_list))
  colors[x[idx] == 1] <- col_1
  shapes <- c(
    rep(19, length.out = nrow(Z_to_plot))
  )
  sizes <- c(
    rep(1, length.out = nrow(Z_to_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(19, 19),
    size = c(1, 1),
    label = c('0', '1')
  )
  plot_embeddings(
    Z_to_plot, 
    colors = colors,
    shapes = shapes,
    sizes = sizes,
    color_shape_size_labels = color_shape_size_labels,
    stats = FALSE
  )
  
  ## Plot: C
  C_to_plot <- do.call(rbind, c_ctx$payload$c_list[idx])[,1:3]
  colors = rep(col_0, length.out = nrow(Z_to_plot))
  colors[x[idx] == 1] <- col_1
  shapes <- c(
    rep(19, length.out = nrow(C_to_plot))
  )
  sizes <- c(
    rep(1, length.out = nrow(C_to_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(19, 19),
    size = c(1, 1),
    label = c('0', '1')
  )
  plot_embeddings(
    C_to_plot, 
    colors = colors,
    shapes = shapes,
    sizes = sizes,
    color_shape_size_labels = color_shape_size_labels,
    stats = FALSE
  )
  
} else {
  stop("Dataset not supported!")
}



## ---------- Modeling ---------- ##

if (dataset == 'tean') {
  mask <- !is.na(df_cov[bin_pred])
  Z <- do.call(rbind, z_ctx$payload[mask])
  x <- df_cov[mask,bin_pred]
  samps <- fit_model_bin_pred(
    Z, 
    x,
    n_chains = 1, 
    burn_in = 2000, 
    thin_interval = 5, 
    n_samps = 5000,
    seed = 12345
  )
} else if (dataset == 'bicluster') {
  Z <- do.call(rbind, z_ctx$payload)
  samps <- fit_model_bin_pred(
    Z, 
    x,
    n_chains = 1, 
    burn_in = 2000, 
    thin_interval = 5, 
    n_samps = 5000,
    seed = 12345
  )
} else {
  stop("Dataset not supported!")
}




## ---------- Posterior Predictive Sampling ---------- ##

## Set parameters
K_ <- ncol(Z)
K <- ifelse(is.null(y_star), K_ - 1, K_)
S <- nrow(samps)
n_param <- ncol(samps)

## Posterior predictive samples
z_draws_0 <- vector('list', length = K_)
z_draws_1 <- vector('list', length = K_)
betas <- samps[,(n_param - 2*K_ + 1):(n_param - K_)]
mus <- samps[,(n_param - K_ + 1):n_param]
Sigmas <- samps[,1:(n_param - 2*K_)]
set.seed(12345)
for (s in 1:S) {
  
  ## Get parameters
  beta <- betas[s,]
  mu <- mus[s,]
  Sigma <- matrix(Sigmas[s,], nrow = K_)
  
  ## Simulate draws 
  z_draws_0[[s]] <- mvrnorm(1, mu = mu, Sigma = Sigma)
  z_draws_1[[s]] <- mvrnorm(1, mu = mu + beta, Sigma = Sigma)
  
}


## ----- Decode Draws

## Payload indices
idx_draws <- 1:S
idx_z_mean <- S + 1
idx_c_mean <- S + 2
idx_Q_mean <- S + 3

decode_draws <- function(z_draws) {

  draws <- list()
  
  ## Z
  z_draws_ctx <- new_context(
    payload = z_draws,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(1000, length.out = S + 3))
  )
  z_mean <- colMeans(do.call(rbind, z_draws_ctx$payload[idx_draws]))
  z_draws_ctx$payload[[idx_z_mean]] <- z_mean
  draws$z <- z_draws_ctx$payload
  
  ## C
  c_draws_ctx <- decode(pipeline, z_draws_ctx, from = 6, to = 5)
  c_mean <- colMeans(do.call(rbind, c_draws_ctx$payload$c_list[idx_draws]))
  if (is.null(y_star)) {
    Q_star_mean <- mean(unlist(c_draws_ctx$payload$Q_star_list[idx_draws]))
    c_draws_ctx$payload$c_list[[idx_c_mean]] <- c_mean
    c_draws_ctx$payload$Q_star_list[[idx_c_mean]] <- Q_star_mean
  } else {
    c_draws_ctx$payload$c_list[[idx_c_mean]] <- c_mean
    c_draws_ctx$payload$Q_star_list[[idx_c_mean]] <- y_star
  }
  draws$c <- c_draws_ctx$payload
  
  ## G
  G_Q_star_draws_ctx <- decode(pipeline, c_draws_ctx, from = 5, to = 4)
  draws$G_Q_star <- G_Q_star_draws_ctx$payload
  
  ## Q
  Q_draws_ctx <- decode(pipeline, G_Q_star_draws_ctx, from = 4, to = 3)
  Q_mean <- colMeans(do.call(rbind, Q_draws_ctx$payload[idx_draws]))
  Q_draws_ctx$payload[[idx_Q_mean]] <- Q_mean
  draws$Q <- Q_draws_ctx$payload
  
  ## Qi
  Qi_draws_ctx <- decode(pipeline, Q_draws_ctx, from = 3, to = 2)
  draws$Qi <- Qi_draws_ctx$payload
  
  ## TY
  Ty_draws_ctx <- decode(pipeline, Qi_draws_ctx, from = 2, to = 1)
  draws$Ty <- Ty_draws_ctx$payload
  
  ## Y
  y_draws_ctx <- decode(pipeline, Ty_draws_ctx, from = 1, to = 0)
  draws$y <- y_draws_ctx$payload
  
  draws
}

draws_0 <- decode_draws(z_draws_0)
draws_1 <- decode_draws(z_draws_1)

## Save draws
path_idx_draws <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'idx_draws.rds')
path_0 <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'draws_0.rds')
path_1 <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'draws_1.rds')
saveRDS(idx_draws, path_idx_draws)
saveRDS(draws_0, path_0)
saveRDS(draws_1, path_1)


## ---------- Plotting ---------- ##

K_plot <- min(K, 4)
K_shift <- 0

## Z
set.seed(12345)
Z_to_plot <- rbind(
  do.call(rbind, draws_0$z[idx_draws]),
  do.call(rbind, draws_1$z[idx_draws])
)[,(1+K_shift):(K_plot+K_shift)]
idx_shuff <- sample(1:nrow(Z_to_plot))
Z_to_plot <- Z_to_plot[idx_shuff,]
colors = c(
  rep(col_0, length.out = S), 
  rep(col_1, length.out = S)
)
colors <- colors[idx_shuff]
shapes <- c(
  rep(19, length.out = S + S)
)
sizes <- c(
  rep(1, length.out = S + S)
)
color_shape_size_labels = data.frame(
  color = c(col_0, col_1),
  shape = c(19, 19),
  size = c(1, 1),
  label = c('0', '1')
)
plot_embeddings(
  Z_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## C
set.seed(12345)
C_to_plot <- rbind(
  do.call(rbind, draws_0$c$c_list[idx_draws]),
  do.call(rbind, draws_1$c$c_list[idx_draws])
)[,(1+K_shift):(K_plot+K_shift)]
idx_shuff <- sample(1:nrow(C_to_plot))
C_to_plot <- C_to_plot[idx_shuff,]
colors = c(
  rep(col_0, length.out = S), 
  rep(col_1, length.out = S)
)
colors <- colors[idx_shuff]
shapes <- c(
  rep(19, length.out = S + S)
)
sizes <- c(
  rep(1, length.out = S + S)
)
color_shape_size_labels = data.frame(
  color = c(col_0, col_1),
  shape = c(19, 19),
  size = c(1, 1),
  label = c('0', '1')
)
plot_embeddings(
  C_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## Q
set.seed(12345)
fun_list <- c(
  draws_0$Q[idx_draws],
  draws_1$Q[idx_draws]
)
idx_shuff <- sample(1:nrow(C_to_plot))
fun_list <- fun_list[idx_shuff]
grid_list <- rep(list(p_grid), length(fun_list))
colors = c(
  rep(col_0, length.out = S), 
  rep(col_1, length.out = S)
)
colors <- colors[idx_shuff]
widths <- rep(1, length(fun_list))
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_0, col_1),
  width = c(1, 1),
  type  = c(1, 1),
  label = c('0', '1')
)
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)


################################# LIKELIHOOD ###################################

## Likelihood
supp_TY <- pipeline$training$cache$supp_TY
lik_mat_0 <- matrix(0, S, length(supp_TY))
lik_mat_1 <- matrix(0, S, length(supp_TY))
for (s in seq_len(S)) {
  lik_mat_0[s,] <- compute_likelihoods(
    supp_TY,
    draws_0$Q[[s]],
    p_grid,
    supp_Y = supp_TY,
    log = FALSE
  )
  lik_mat_1[s,] <- compute_likelihoods(
    supp_TY,
    draws_1$Q[[s]],
    p_grid,
    supp_Y = supp_TY,
    log = FALSE
  )
}
mean_lik_0 <- colMeans(lik_mat_0) 
mean_lik_1 <- colMeans(lik_mat_1)
start <- 1
stop <- 50
plot(supp_TY[start:stop], mean_lik_0[start:stop], type = 'l', col = col_0, ylim = c(0, 0.1))
lines(supp_TY[start:stop], mean_lik_1[start:stop], col = col_1)
sum(supp_TY * mean_lik_0)
sum(supp_TY * mean_lik_1)








