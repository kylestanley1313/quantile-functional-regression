library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## ---------- Colors ---------- ##

col_recon <- rgb(0, 1, 0, alpha = 0.5)
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
pipe_name <- 'flow_tean'
n_stage_plots <- 2
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
        K_max = 20,
        epsilon = 0.01,
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
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_mims') {
  
  ## Load data
  path <- file.path('data', 'processed', 'chop-mims_v1.rds')
  y_list <- readRDS(path)
  y_list <- y_list
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, 100,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'log', y_shift = 100),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 20,
        epsilon = 0.01,
        alpha = 0.05,
        V = 5,
        lambda = 1e-6
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_chop-mims.pth'
      )
    ),
    supp_Y = c(0, 0.6, seq(0.601, 4000, by = 0.001)),  ## TODO: Improve this?
    p_star = 0,
    y_star = y_star,
    y_min = -0.01,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_enmo') {
  
  ## Load data
  path <- file.path('data', 'processed', 'chop-enmo_v1.rds')
  y_list <- readRDS(path)
  y_list <- y_list
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, 200,
    p_left = 0.05, J_left = 50,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(
        y_trans = 'log',
        y_shift = 1e-3
      ),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 20,
        epsilon = 0.01,
        alpha = 0.05,
        V = 5,
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_chop-enmo.pth'
      )
    ),
    supp_Y = seq(0, 2*y_max, by = 0.0001),
    p_star = 0,
    y_star = 0,
    y_min = 0,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else if (startsWith(pipe_name, 'flow_multi')) {
  
  ## Load data
  if (pipe_name == 'flow_multi_250') {
    path <- file.path('data', 'processed', 'multinomial-250.rds')
  } else if (pipe_name == 'flow_multi_500') {
    path <- file.path('data', 'processed', 'multinomial-500.rds')
  } else if (pipe_name == 'flow_multi_1000') {
    path <- file.path('data', 'processed', 'multinomial-1000.rds')
  } else {
    stop("Invalid pipe_name!")
  }
  ylist <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define p_grid
  p_grid <- p_grid_fun(y_list, J = 100)
  
  ## Construct pipeline
  y_star <- 1
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'identity'),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 30,
        epsilon = 0.02,
        alpha = 0.05,
        V = 5,
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_multinomial.pth'
      )
    ),
    supp_Y = 1:10,
    p_star = 0,
    y_star = y_star,
    y_min = 1,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_bicluster') {
  
  ## Load data
  path <- file.path('data', 'processed', 'bicluster_v3.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define p_grid
  p_grid <- p_grid_fun(y_list, J = 100)
  
  ## Construct pipeline
  # y_star <- 0
  y_star <- NULL
  pipeline <- construct_pipeline(
    stages = list(
      # stage_y_axis(y_trans = 'log', y_shift = 1),
      stage_y_axis(y_trans = 'identity'),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K = 2,
        K_max = 30,
        epsilon = 0.01,
        alpha = 0.01,
        V = 5,
        # lambda = 0
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_bicluster.pth'
      )
    ),
    supp_Y = NULL,
    p_star = 0,
    y_star = y_star,
    # y_min = 0,
    y_min = NULL,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else {
  stop("Invalid pipe_name!")
}

## Fitting
pipeline <- fit(pipeline, y_list)

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
for (i in 1:n_stage_plots) {
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
  K_plot_min <- 5
  
  ## Plot z-embeddings
  Z_plot <- do.call(rbind, z_ctx$payload)
  Z_plot <- Z_plot[,1:min(ncol(Z_plot), K_plot_min)]
  plot_embeddings(Z_plot)
  
  ## Plot z-embeddings
  C_plot <- do.call(rbind, c_ctx$payload$c_list)
  C_plot <- C_plot[,1:min(ncol(C_plot), K_plot_min)]
  plot_embeddings(C_plot)
  
  ## Plot Smooth EQFs on common grid
  plot_funs(
    fun_list = Q_ctx$payload,
    grid_list = rep(list(p_grid), N)
  )
}



## ---------- Mean Modeling ---------- ##

Z <- do.call(rbind, z_ctx$payload)
samps <- fit_model_mean(
  Z, 
  n_chains = 1, 
  burn_in = 2000, 
  thin_interval = 5, 
  n_samps = 5000
)



## ---------- Posterior Predictive Sampling ---------- ##

## Set parameters
emb_dim <- ncol(Z)
K <- ifelse(is.null(y_star), emb_dim - 1, emb_dim)
S <- nrow(samps)
n_param <- ncol(samps)

## Posterior predictive samples
z_draws <- vector('list', length = emb_dim)
for (s in 1:S) {
  mu <- samps[s,(n_param-emb_dim+1):n_param]
  Sigma <- matrix(samps[s,1:(n_param-emb_dim)], nrow = emb_dim)
  z_draws[[s]] <- mvrnorm(1, mu = mu, Sigma = Sigma) 
}

## ----- Decode Draws

## Context setup
z_draws_ctx <- new_context(
  payload = z_draws,
  cache = pipeline$training$cache,
  meta = list(Ji_vec = rep(500, length.out = S + 3))
)

## Payload indices
idx_draws <- 1:S
idx_z_mean <- S + 1
idx_c_mean <- S + 2
idx_Q_mean <- S + 3

## Z --> C
z_mean <- colMeans(do.call(rbind, z_draws_ctx$payload[idx_draws]))
z_draws_ctx$payload[[idx_z_mean]] <- z_mean
c_draws_ctx <- decode(pipeline, z_draws_ctx, from = 6, to = 5)

## C --> G
c_mean <- colMeans(do.call(rbind, c_draws_ctx$payload[idx_draws]))
if (is.null(y_star)) {
  c_draws_ctx$payload$c_list[[idx_c_mean]] <- c_mean[1:(length(c_mean)-1)]
  c_draws_ctx$payload$Q_star_list[[idx_c_mean]] <- c_mean[length(c_mean)]
} else {
  c_draws_ctx$payload$c_list[[idx_c_mean]] <- c_mean
  c_draws_ctx$payload$Q_star_list[[idx_c_mean]] <- y_star
}
G_Q_star_draws_ctx <- decode(pipeline, c_draws_ctx, from = 5, to = 4)

## G --> Q
Q_draws_ctx <- decode(pipeline, G_Q_star_draws_ctx, from = 4, to = 3)

## Q --> Qi
Q_mean <- colMeans(do.call(rbind, Q_draws_ctx$payload[idx_draws]))
Q_draws_ctx$payload[[idx_Q_mean]] <- Q_mean
Qi_draws_ctx <- decode(pipeline, Q_draws_ctx, from = 3, to = 2)

## Qi --> TY
Ty_draws_ctx <- decode(pipeline, Qi_draws_ctx, from = 2, to = 1)

## TY --> Y
y_draws_ctx <- decode(pipeline, Ty_draws_ctx, from = 1, to = 0)


## ---------- Plotting ---------- ##

K_plot <- min(K, 4)
K_shift <- 0
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)

## z-draws vs. z-train plot
Z_to_plot <- rbind(
  do.call(rbind, z_draws_ctx$payload[idx_draws]),
  do.call(rbind, z_ctx$payload[-idx_outliers]),
  do.call(rbind, z_ctx$payload[idx_outliers]),
  z_draws_ctx$payload[[idx_z_mean]]
)[,(1+K_shift):(K_plot+K_shift)]
colors = c(
  rep(col_draw, length.out = S),
  rep(col_train, length.out = N - N_outlier),
  rep(col_outlier, length.out = N_outlier),
  col_z_mean
)
shapes <- c(
  rep(19, length.out = S + N + 1)
)
sizes <- c(
  rep(1, length.out = S + N + 1)
)
color_shape_size_labels = data.frame(
  color = c(col_train, col_draw, col_outlier, col_z_mean),
  shape = c(19, 19, 19, 19),
  size = c(1, 1, 1, 1),
  label = c('z-train', 'z-draws', 'outlier', 'z-mean')
)
plot_embeddings(
  Z_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## c-draws vs. c-train plot
C_to_plot <- rbind(
  do.call(rbind, c_draws_ctx$payload$c_list[idx_draws]),
  do.call(rbind, c_ctx$payload$c_list[-idx_outliers]),
  do.call(rbind, c_ctx$payload$c_list[idx_outliers]),
  c_draws_ctx$payload$c_list[[idx_z_mean]],
  c_draws_ctx$payload$c_list[[idx_c_mean]]
)[,(1+K_shift):(K_plot+K_shift)]
colors = c(
  rep(col_draw, length.out = S),
  rep(col_train, length.out = N - N_outlier),
  rep(col_outlier, length.out = N_outlier),
  col_z_mean,
  col_c_mean
)
shapes <- c(
  rep(19, length.out = S + N + 2)
)
sizes <- c(
  rep(1, length.out = S + N + 2)
)
color_shape_size_labels = data.frame(
  color = c(col_train, col_draw, col_outlier, col_z_mean, col_c_mean),
  shape = c(19, 19, 19, 19, 19),
  size = c(1, 1, 1, 1, 1),
  label = c('c-train', 'c-draws', 'outlier', 'z-mean', 'c-mean')
)
plot_embeddings(
  C_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## Q-draws vs. Q-train plot
fun_list <- c(
  Q_draws_ctx$payload[idx_draws],
  Q_ctx$payload[-idx_outliers],
  Q_ctx$payload[idx_outliers],
  list(Q_draws_ctx$payload[[idx_z_mean]]),
  list(Q_draws_ctx$payload[[idx_c_mean]]),
  list(Q_draws_ctx$payload[[idx_Q_mean]])
)
grid_list <- rep(list(p_grid), length(fun_list))
colors = c(
  rep(col_draw, length.out = S),
  rep(col_train, length.out = N - N_outlier),
  rep(col_outlier, length.out = N_outlier),
  col_z_mean,
  col_c_mean,
  col_Q_mean
)
widths <- rep(1, length(fun_list))
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_draw, col_train, col_outlier, col_z_mean, col_c_mean, col_Q_mean),
  width = c(1, 1, 1, 1, 1, 1),
  type  = c(1, 1, 1, 1, 1, 1),
  label = c('Q-draw', 'Q-train', 'outlier', 'z-mean', 'c-mean', 'Q-mean')
)
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  # ylim = c(4.5, 6),
  color_width_type_labels = color_width_type_labels
)


## Y-draws vs. Y-train plot
y_plot_train <- unlist(Ty_ctx$payload)
y_plot_draws <- unlist(Ty_draws_ctx$payload[idx_draws])
y_plot_z_mean <- Ty_draws_ctx$payload[[idx_z_mean]]
y_plot_c_mean <- Ty_draws_ctx$payload[[idx_c_mean]]
y_plot_Q_mean <- Ty_draws_ctx$payload[[idx_Q_mean]]
y_plot_all <- c(y_plot_train, y_plot_draws, y_plot_z_mean, y_plot_c_mean, y_plot_Q_mean)
y_max_i <- max(y_plot_all)
y_min_i <- min(y_plot_all)
breaks_y <- seq(y_min_i, y_max_i, length.out = 50)
y_max <- 0.4
## ... z-mean
h <- hist(y_plot_train, breaks = breaks_y, probability = TRUE, col = col_train, ylim = c(0, y_max))
hist(y_plot_z_mean, breaks = breaks_y, probability = TRUE, add = TRUE, col = col_z_mean_trans)
## ... c-mean
h <- hist(y_plot_train, breaks = breaks_y, probability = TRUE, col = col_train, ylim = c(0, y_max))
hist(y_plot_c_mean, breaks = breaks_y, probability = TRUE, add = TRUE, col = col_c_mean_trans)
## ... Q-mean
h <- hist(y_plot_train, breaks = breaks_y, probability = TRUE, col = col_train, ylim = c(0, y_max))
hist(y_plot_Q_mean, breaks = breaks_y, probability = TRUE, add = TRUE, col = col_Q_mean_trans)
## ... Y-draws
h <- hist(y_plot_train, breaks = breaks_y, probability = TRUE, col = col_train, ylim = c(0, y_max))
hist(y_plot_draws, breaks = breaks_y, probability = TRUE, add = TRUE, col = col_draw_trans)
