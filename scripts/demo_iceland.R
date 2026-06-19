source('src/cot.R')
source('src/utils.R')


## ========== Data Processing ========== ##

## Load raw data
path <- file.path('data', 'raw', 'iceland.csv')
df <- read.csv(path, header = TRUE)
path <- file.path('data', 'raw', 'iceland_idsp.csv')
df_ids <- read.csv(path, header = TRUE)

## Get split lists
y_list <- apply(df, 2, na.omit, simplify = FALSE)
y_train_list <- y_list[df_ids[,3] == 1]
y_test_time_list <- y_list[df_ids[,3] == 0]
y_test_spacetime_list <- y_list[df_ids[,3] == 2]

## Save processed data
path <- file.path('data', 'processed', 'iceland_train.rds')
saveRDS(y_train_list, path)
path <- file.path('data', 'processed', 'iceland_test_time.rds')
saveRDS(y_test_time_list, path)
path <- file.path('data', 'processed', 'iceland_test_spacetime.rds')
saveRDS(y_test_spacetime_list, path)


## NOTES
##  - Maximum of J = 91 observations per site-year
##  - About 4% of site-year times are NA
##  - Each site-year is code 1 (train), 0 (time test), 2 (space-time test)

## PLAN 
##  - Common grid on 1/92, ..., 91/92
##  - Embeddings for only training split


## ========== Representation Learning ========== ##

## Load data
path <- file.path('data', 'processed', 'iceland_train.rds')
y_list <- readRDS(path)
# y_list <- y_list[1:500]
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
y_max <- max(unlist(y_list))


## Define grid
p_grid <- p_grid_fun_2(
  breaks = c(1/(Ji_max + 1), Ji_max/(Ji_max + 1)),
  interval_counts = c(Ji_max)
)

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = Ji_min),
    stage_wame(
      K_max = 20,
      epsilon = 0.5,
      alpha = 0.01,
      V = 5,
      quantlet_construction = 'pca',
      lambda = 0
    ),
    stage_flow(
      n_layers = 16,
      max_epochs = 1000,
      lr = 1e-3,
      path = 'artifacts/demo_iceland/flow_iceland.pth'
    )
  ),
  supp_Y = seq(-30, 20, by = 0.0001),
  p_star = 0.5,
  Q_star = NULL,
  y_min = NULL,
  loss = 'wasserstein',
  seed = 12345
)

## Fitting
pipeline <- fit(pipeline, y_list)
path <- 'artifacts/demo_iceland/pipe.rds'
saveRDS(pipeline, path)

## New context
y_ctx <- new_context(
  payload = y_list,
  cache = pipeline$training$cache,
  meta = list()
)

## Encode/Decode
Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
c_ctx_ <- decode(pipeline, z_ctx, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)



## --- Pairwise distances

## Augmented p-grid
J_aug <- 500
p_grid_aug <- c(p_grid, pi_grid_fun(J_aug - length(p_grid)))
p_grid_aug <- sort(unique(p_grid_aug))

## Pairwise distances
set.seed(12345)
Qi_list <- Qi_ctx$payload[sample(1:N, size = 1000)]
distances <- pairwise_distance(
  Qi_list = Qi_list,
  loss_fun = wasserstein,
  pi_grid_list = lapply(lengths(Qi_list), pi_grid_fun),
  p_grid_aug = p_grid_aug,
  supp_Y = pipeline$training$cache$supp_Y
)
quantile(distances, c(0.5, 0.1, 0.01, 0.005, 0.001))

## NOTE: We have set the tolerance level to eps = 0.5. This is
## roughly equivalent to the 0.5th percentile of pairwise Wasserstein
## distances in the training sample.

## --- Visualize reconstructions

recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_ctx$payload[[i]]
  Q2 <- Qi_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

set.seed(12345)

lower_mat_3x5 <- matrix(
  c(0.00, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90,
    0.95, 0.96, 0.97, 0.98, 0.99),
  nrow = 3, byrow = TRUE
)
upper_mat_3x5 <- matrix(
  c(0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95,
    0.96, 0.97, 0.98, 0.99, 1.00),
  nrow = 3, byrow = TRUE
)

log_scale <- FALSE   # toggle: TRUE => plot on log(Q(p) + 1) scale

plot_qi_recon_grid( ## See demo_iceland.R
  Qi_orig_list  = Qi_ctx$payload,
  Qi_recon_list = Qi_ctx_$payload,
  recon_losses  = recon_losses,
  lower_mat     = lower_mat_3x5,
  upper_mat     = upper_mat_3x5,
  log_x_plus_1  = log_scale,
  path          = file.path('artifacts', 'demo_iceland', 'plots',
                            sprintf('Qi-recon_grid_3x5%s.png',
                                    if (log_scale) '_log' else ''))
)





## ----- Visualize data in different spaces

## Plotting globals
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- rgb(1, 0.8, 0.5, alpha = 0.25)

## Plot z-embeddings
path_plot <- file.path(
  'artifacts', 'demo_iceland', 'plots',
  str_glue('data_z.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Z_plot <- rbind(
  do.call(rbind, z_ctx$payload)[-idx_outliers,1:3],
  do.call(rbind, z_ctx$payload)[idx_outliers,1:3]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(19, length.out = nrow(Z_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(Z_plot))
)
color_shape_size_labels = data.frame(
  color = c(col_train, col_outlier),
  shape = c(19, 19),
  size = c(0.5, 0.5),
  label = c('non-outlier', 'outlier')
)
plot_embeddings(
  Z_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)
dev.off()

## Plot c-embeddings
path_plot <- file.path(
  'artifacts', 'demo_iceland', 'plots',
  str_glue('data_c.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
C_plot <- rbind(
  do.call(rbind, c_ctx$payload$c_list)[-idx_outliers,1:3],
  do.call(rbind, c_ctx$payload$c_list)[idx_outliers,1:3]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(19, length.out = nrow(C_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(C_plot))
)
color_shape_size_labels = data.frame(
  color = c(col_train, col_outlier),
  shape = c(19, 19),
  size = c(0.5, 0.5),
  label = c('non-outlier', 'outlier')
)
plot_embeddings(
  C_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)
dev.off()


## Plot Q
path_plot <- file.path(
  'artifacts', 'demo_iceland', 'plots',
  str_glue('data_q.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
fun_list <- c(
  Q_ctx$payload[-idx_outliers],
  Q_ctx$payload[idx_outliers]
)
grid_list <- rep(list(p_grid), length(fun_list))
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
widths <- c(
  rep(0.25, length.out = length(fun_list))
)
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_train, col_outlier),
  width = c(2.5, 2.5),
  type  = c(1, 1),
  label = c('non-outlier', 'outlier')
)
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)
dev.off()


## ---------- Generativity

## Get Z-draws from fitted mean-covariance model
set.seed(12345)
Z <- do.call(rbind, z_ctx$payload)
fit <- fit_mean_cov(Z)
z_draws <- draw_mean_cov(N, fit)

## Decode Z-draws
z_draws_ctx <- new_context(
  payload = z_draws,
  cache = pipeline$training$cache,
  meta = list(Ji_vec = rep(Ji_max, length.out = length(z_draws)))
)
Qi_draws <- decode(pipeline, z_draws_ctx, from = 4, to = 1)$payload

## Plot Q
set.seed(12345)
idx_shuff <- sample(1:(2*N))
col_data <- rgb(0, 0, 0, alpha = 0.25)
col_draws <- rgb(1, 0, 0, alpha = 0.25)
path_plot <- file.path(
  'artifacts', 'demo_iceland', 'plots',
  str_glue('draws_q.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
fun_list <- c(
  Qi_ctx$payload,
  Qi_draws
)
grid_list <- c(
  lapply(Ji_vec, pi_grid_fun),
  lapply(rep(Ji_max, N), pi_grid_fun)
)
colors <- c(
  rep(col_data, N),
  rep(col_draws, N)
)
fun_list <- fun_list[idx_shuff]
grid_list <- grid_list[idx_shuff]
colors <- colors[idx_shuff]
widths <- c(
  rep(0.25, length.out = length(fun_list))
)
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_data, col_draws),
  width = c(2.5, 2.5),
  type  = c(1, 1),
  label = c('data', 'draws')
)
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)
dev.off()


