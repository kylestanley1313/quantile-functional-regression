library(dplyr)
library(MASS)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')



## ========== Helpers ========== ##

decode_z_to_Qi <- function(z_list, Ji) {
  z_ctx <- new_context(
    payload = z_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_list)))
  )
  decode(pipeline, z_ctx, from = 5, to = 1)$payload
}


## ========== Representation Learning ========== ##

## Load data
path <- file.path('data', 'processed', 'bicluster_v6.rds')
y_list <- readRDS(path)
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
y_max <- max(unlist(y_list))
path_plot_tmp = "scratch/plots/tmp_plot.png"

## Define grid
p_grid <- p_grid_fun_2(
  breaks = c(1/(1+Ji_min), Ji_min/(1+Ji_min)),
  interval_counts = c(Ji_min)
)

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_lqd(),
    stage_qg_pca(
      K_max = 20,
      epsilon = 1, # 0.25,
      alpha = 0.05,
      V = 5,
      lambda = 0.1
    ),
    stage_flow(
      n_layers = 16,
      max_epochs = 1000,
      lr = 1e-3,
      path = 'artifacts/demo_bicluster/flow_bicluster.pth'
    )
  ),
  supp_Y = NULL,
  p_star = 0.5,
  Q_star = NULL,
  y_min = NULL,
  loss = 'wasserstein',
  seed = gen_seed()
)

## Fitting
pipeline <- fit(pipeline, y_list)
path <- 'artifacts/demo_bicluster/pipe_bicluster.rds'
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
G_Q_star_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
c_ctx <- encode(pipeline, G_Q_star_ctx, from = 3, to = 4)
z_ctx <- encode(pipeline, c_ctx, from = 4, to = 5)
c_ctx_ <- decode(pipeline, z_ctx, from = 5, to = 4)
G_Q_star_ctx_ <- decode(pipeline, c_ctx_, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, G_Q_star_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)


## ----- Visualize reconstructions

## Compute reconstruction losses
recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_ctx$payload[[i]]
  Q2 <- Qi_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

## Plot select reconstructions
p_vec <- c(0.5, 0.95, 1)
for (p in p_vec) {

  ## Get payload index
  recon_losses_sort <- sort(recon_losses)
  idx_float <- 1 + (length(recon_losses) - 1) * p
  q_value <- recon_losses_sort[round(idx_float)]
  idx_payload <-  which(recon_losses == q_value)
  
  ## Plot reconstruction
  path_plot <- file.path(
    'artifacts', 'demo_bicluster', 'plots', 
    str_glue('Qi-recon_p-{100*p}.png')
  )
  Qi <- Qi_ctx$payload[[idx_payload]]
  Qi_recon <- Qi_ctx_$payload[[idx_payload]]
  pi_grid <- pi_grid_fun(length(Qi))
  png(path_plot, width = 480, height = 480)
  plot(
    pi_grid, Qi, 
    type = 'l',
    xlab = 'p', 
    ylab = 'Q(p)',
    ylim = c(min(c(Qi, Qi_recon)), max(c(Qi, Qi_recon))),
    main = str_glue("{100*p}th Percentile Validation Loss")
  )
  lines(pi_grid, Qi_recon, col = rgb(0, 1, 0, alpha = 1.0))
  dev.off()
}


## ----- Visualize data in different spaces

## Plotting globals
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- rgb(1, 0.8, 0.5, alpha = 0.25)


## Plot z-embeddings
path_plot <- file.path(
  'artifacts', 'demo_bicluster', 'plots',
  str_glue('data_z.png')
)
png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
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
  'artifacts', 'demo_bicluster', 'plots',
  str_glue('data_c.png')
)
png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
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
  'artifacts', 'demo_bicluster', 'plots',
  str_glue('data_q.png')
)
png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
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

## Q: Can you characterize outlyingness? 
##  - Hard to characterize in Q-space. Try visualizig log(Q).
##  - Can characterize in C-space, but not interpretable.




## ========== Frechet Means ========== ##

## Q-mean
mean_q <- colMeans(do.call(rbind, Qi_ctx$payload))

## Z-mean
Ji <- length(Qi_ctx$payload[[1]])
mean_z <- colMeans(do.call(rbind, z_ctx$payload))
mean_z <- decode_z_to_Qi(list(mean_z), Ji = Ji)[[1]]

## F-mean (L2 in CDF space)
pi_grid <- pi_grid_fun(Ji)
y_range <- range(unlist(Qi_ctx$payload))
y_grid  <- seq(y_range[1], y_range[2], length.out = 10000)
F_bar   <- numeric(length(y_grid))
for (Qi in Qi_ctx$payload) {
  F_bar <- F_bar + findInterval(y_grid, sort(Qi)) / length(Qi)
}
F_bar  <- F_bar / length(Qi_ctx$payload)
mean_f <- approx(
  x = F_bar, y = y_grid,
  xout = pi_grid,
  rule = 2, ties = "ordered"
)$y

## Plotting
path_plot <- file.path(
  'artifacts', 'demo_bicluster', 'plots',
  str_glue('frechet_bicluster.png')
)
png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
col_q <- 'red'
col_f <- 'green'
col_z <- 'blue'
fun_list <- c(
  Qi_ctx$payload,
  list(mean_q),
  list(mean_f),
  list(mean_z)
)
grid_list <- rep(list(pi_grid), length(fun_list))
colors <- c(
  rep(col_train, length(fun_list) - 3),
  col_q,
  col_f,
  col_z
)
widths <- c(
  rep(0.25, length.out = length(fun_list) - 3),
  c(2.5, 2.5, 2.5)
)
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_train, col_q, col_f, col_z),
  width = c(2.5, 2.5, 2.5, 2.5),
  type  = c(1, 1, 1, 1),
  label = c('data', 'Q-mean', 'F-mean', 'Z-mean')
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
