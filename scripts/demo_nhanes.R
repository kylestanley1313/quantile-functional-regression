library(biotools)
library(car)
library(dplyr)
library(kableExtra)
library(knitr)
library(MASS)
library(mgcv)
library(moments)
library(MVN)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')




## ========== Representation Learning ========== ##

## Globals
dir_art <- 'demo_nhanes_v6'

## Latent representation used by the Group Comparison and AMM sections:
##   'z_rot' -> output of the PCA layer (stage 5)
##   'c'     -> output of the WAME layer (stage 3)
## Analysis output directories are suffixed for anything other than z_rot, so
## the two runs sit side by side under the same dir_art (and therefore share a
## single fitted pipeline -- see the pipeline load guard below).
latent <- 'z_rot'
suffix <- if (latent == 'z_rot') '' else str_glue('-{latent}')


## Load data
path <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path)
# y_list <- y_list[1:500]
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
y_max <- max(unlist(y_list))
path_plot_tmp <- "scratch/plots/tmp_plot.png"

## Define grid
p_grid <- p_grid_fun(
  breaks = c(1/(7350 + 1), 0.95, 7350/(7350 + 1)),
  interval_counts = c(51, 50)
)

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100),
    stage_wame(
      K = 5,
      K_max = 20,
      epsilon = 1.45,
      alpha = 0.05,
      V = 5,
      lambda = 0,
      order_by = 'importance'
    ),
    stage_flow(
      n_layers = 16, # 4 (good for K = 5),
      hidden = 64, # 32 (good for K = 5),
      max_epochs = 1000,
      lr = 1e-3,
      path = str_glue('artifacts/{dir_art}/flow_nhanes.pth')
    ),
    stage_pca_rotation()
  ),
  supp_Y = c(0, seq(0.006, 500, by = 0.001)),
  p_star = 0,
  y_star = 0,
  y_min = 0,
  seed = 12345
)

## Fitting
## NOTE: stage seeds come from gen_seed(), which is evaluated before
## construct_pipeline() calls set.seed(), so re-fitting produces a DIFFERENT
## flow each session. Reuse the saved pipeline so that the z_rot and c runs are
## two views of one fit and the comparison isn't confounded by retraining.
path <- str_glue('artifacts/{dir_art}/pipe_nhanes.rds')
if (file.exists(path)) {
  pipeline <- readRDS(path)
} else {
  pipeline <- fit(pipeline, y_list)
  saveRDS(pipeline, path)
}
pipeline$latent <- latent
pipeline <- set_rotation_signs(pipeline, flip = c(1, 3))
pipeline <- set_c_signs(pipeline, flip = c(1))

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
z_rot_ctx <- encode(pipeline, z_ctx, from = 4, to = 5)
z_ctx_ <- decode(pipeline, z_rot_ctx, from = 5, to = 4)
c_ctx_ <- decode(pipeline, z_ctx_, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)

## N x K matrix of the selected latent, consumed by the Group Comparison and
## AMM sections below. The c branch goes through c_ctx_to_matrix() so that it
## carries the pipeline's c sign convention, matching decode_latent_to_Qi().
Z_latent <- switch(
  latent,
  z_rot = do.call(rbind, z_rot_ctx$payload),
  c     = c_ctx_to_matrix(pipeline, c_ctx),
  stop(str_glue("Unknown latent: '{latent}'"))
)


## ----- Visualize reconstructions

## Compute reconstruction losses
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

plot_qi_recon_grid(
  Qi_orig_list  = Qi_ctx$payload,
  Qi_recon_list = Qi_ctx_$payload,
  recon_losses  = recon_losses,
  lower_mat     = lower_mat_3x5,
  upper_mat     = upper_mat_3x5,
  log_x_plus_1  = log_scale,
  path          = file.path('artifacts', dir_art, 'plots', 'representation',
                            sprintf('Qi-recon_grid_3x5%s.png',
                                    if (log_scale) '_log' else ''))
)



## ----- Visualize data in different spaces

## Plotting globals
mark_outliers <- FALSE
include_legend <- FALSE
k_from <- 1
k_to <- 5
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.25), col_train)

## Plot z-rot-embeddings
path_plot <- file.path(
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_z-rot.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Z_plot <- rbind(
  do.call(rbind, z_rot_ctx$payload)[idx_train,k_from:k_to],
  do.call(rbind, z_rot_ctx$payload)[idx_outliers,k_from:k_to]
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
if (include_legend) {
  color_shape_size_labels <- data.frame(
    color = c(col_train, col_outlier),
    shape = c(19, 19),
    size = c(0.5, 0.5),
    label = c('non-outlier', 'outlier')
  )
} else {
  color_shape_size_labels <- NULL
}
plot_embeddings(
  Z_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)
dev.off()

## Plot z-embeddings
path_plot <- file.path(
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_z.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Z_plot <- rbind(
  do.call(rbind, z_ctx$payload)[idx_train,k_from:k_to],
  do.call(rbind, z_ctx$payload)[idx_outliers,k_from:k_to]
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
if (include_legend) {
  color_shape_size_labels = data.frame(
    color = c(col_train, col_outlier),
    shape = c(19, 19),
    size = c(0.5, 0.5),
    label = c('non-outlier', 'outlier')
  )
} else {
  color_shape_size_labels <- NULL
}
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
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_c.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
C_plot <- rbind(
  c_ctx_to_matrix(pipeline, c_ctx)[idx_train,k_from:k_to],
  c_ctx_to_matrix(pipeline, c_ctx)[idx_outliers,k_from:k_to]
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
if (include_legend) {
  color_shape_size_labels = data.frame(
    color = c(col_train, col_outlier),
    shape = c(19, 19),
    size = c(0.5, 0.5),
    label = c('non-outlier', 'outlier')
  )
} else {
  color_shape_size_labels <- NULL
}
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
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_q.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
fun_list <- c(
  Q_ctx$payload[idx_train],
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
if (include_legend) {
  color_width_type_labels <- data.frame(
    color = c(col_train, col_outlier),
    width = c(2.5, 2.5),
    type  = c(1, 1),
    label = c('non-outlier', 'outlier')
  )
} else {
  color_width_type_labels <- NULL
}
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)
dev.off()


## Plot subsample of Q
## NOTE: For Jeff's slides
N_samp <- 500
if (!is.null(N_samp)) {
  set.seed(12345)

  ## Compute overall mean
  Q_mean <- colMeans(do.call(rbind, Q_ctx$payload))

  ## Subsample
  idx_samp <- sample(N, size = N_samp)
  Q_samp <- Q_ctx$payload[idx_samp]

  ## Plot
  path_plot <- file.path(
    'artifacts', dir_art, 'plots', 'representation',
    str_glue('data_q-subsamp.png')
  )
  png(path_plot, width = 960, height = 960, pointsize = 18)
  fun_list <- c(
    Q_samp,
    list(Q_mean)
  )
  grid_list <- rep(list(p_grid), length(fun_list))
  colors <- c(
    rep('black', N_samp),
    rep('red', 1)
  )
  widths <- c(
    rep(0.25, length.out = N_samp),
    rep(4, 1)
  )
  types <- rep(1, length(fun_list))
  if (FALSE) {
    color_width_type_labels <- data.frame(
      color = c(col_train, col_outlier),
      width = c(2.5, 2.5),
      type  = c(1, 1),
      label = c('non-outlier', 'outlier')
    )
  } else {
    color_width_type_labels <- NULL
  }
  plot_funs(
    fun_list = fun_list,
    grid_list = grid_list,
    colors = colors,
    widths = widths,
    types = types,
    color_width_type_labels = color_width_type_labels
  )
  dev.off()
}


## Plot Qi
path_plot <- file.path(
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_qi.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
fun_list <- c(
  Qi_ctx$payload[idx_train],
  Qi_ctx$payload[idx_outliers]
)
grid_list <- c(
  lapply(Ji_vec[idx_train], pi_grid_fun), 
  lapply(Ji_vec[idx_outliers], pi_grid_fun)
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
widths <- c(
  rep(0.25, length.out = length(fun_list))
)
types <- rep(1, length(fun_list))
if (include_legend) {
  color_width_type_labels <- data.frame(
    color = c(col_train, col_outlier),
    width = c(2.5, 2.5),
    type  = c(1, 1),
    label = c('non-outlier', 'outlier')
  )
} else {
  color_width_type_labels <- NULL
}
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)
dev.off()


## Plot Qi vs. Q (one subject)
i <- 1
path_plot <- file.path(
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('data_qi-v-q_i-{i}.png')
)
pi_grid <- pi_grid_fun(Ji_vec[i])
Qi <- Qi_ctx$payload[[i]]
Q <- Q_ctx$payload[[i]]
png(path_plot, width = 960, height = 960, pointsize = 18)
plot(
  pi_grid, Qi,
  type = 'l',
  lwd = 4,
  xlim = c(0, 1),
  xlab = "p",
  ylab = "Q(p)",
  main = str_glue("Subject {i}")
)
lines(
  p_grid, Q,
  lwd = 4,
  col = rgb(0.9, 0, 0)
)
points(
  x = p_grid,
  y = rep(0, length(p_grid)),
  pch = 16,
  col = rgb(0, 0.9, 0)
)
dev.off()


## ----- Visualize synthetic

## Globals
K  <- pipeline$stages[[3]]$state$child_qg_pca$state$K
Ji <- 7350

## Train/val split (same convention as evaluate_pipeline_generativity): fit the
## latent mean-cov model on the train half, then draw |val| synthetic z's
## and compare against the held-out real Qi.
set.seed(12345)
frac_train <- 0.5
n_train    <- floor(frac_train * N)
perm       <- sample(N)
idx_tr     <- perm[seq_len(n_train)]
idx_val    <- perm[(n_train + 1):N]

fit     <- fit_mean_cov(encode_y_to_z(pipeline, y_list[idx_tr]), ridge = 0)
z_draws <- sample_mean_cov(length(idx_val), fit)
Qi_draws <- decode_z_to_Qi(pipeline, z_draws, Ji = Ji)

## Plot
col_real  <- rgb(0, 0, 0, alpha = 0.25)
col_synth <- rgb(1, 0, 0, alpha = 0.25)
path_plot <- file.path(
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('synth_qi.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Qi_val    <- Qi_ctx$payload[idx_val]
n_real    <- length(Qi_val)
n_synth   <- length(Qi_draws)
fun_list  <- c(Qi_val, Qi_draws)
grid_list <- c(
  lapply(Ji_vec[idx_val], pi_grid_fun),
  lapply(rep(Ji, n_synth), pi_grid_fun)
)
colors <- c(rep(col_real, n_real), rep(col_synth, n_synth))
widths <- rep(0.25, length(fun_list))
types  <- rep(1,    length(fun_list))
shuff     <- sample(length(fun_list))
fun_list  <- fun_list[shuff]
grid_list <- grid_list[shuff]
colors    <- colors[shuff]
color_width_type_labels <- data.frame(
  color = c(col_real, col_synth),
  width = c(2.5, 2.5),
  type  = c(1, 1),
  label = c('real (val)', 'synth')
)
plot_funs(
  fun_list  = fun_list,
  grid_list = grid_list,
  colors    = colors,
  widths    = widths,
  types     = types,
  color_width_type_labels = color_width_type_labels
)
dev.off()




## ========== Frechet Means ========== ##

## Q-mean
mean_q <- colMeans(do.call(rbind, Qi_ctx$payload))

## Z-mean
Ji <- length(Qi_ctx$payload[[1]])
mean_z <- colMeans(do.call(rbind, z_rot_ctx$payload))
mean_z <- decode_z_to_Qi(pipeline, list(mean_z), Ji = Ji)[[1]]

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
  'artifacts', dir_art, 'plots', 'representation',
  str_glue('frechet_nhanes.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
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



## ========== EDA ========== ##

## Create directory
dir_eda <- file.path('artifacts', dir_art, 'plots', 'eda')
if (!dir.exists(dir_eda)) dir.create(dir_eda)

## Load covariates
# path <- file.path('data', 'processed', 'nhanes_cov.rds')
path <- file.path('data', 'processed', 'nhanes_cov_cdc.rds')
df_cov <- readRDS(path)
df_cov <- df_cov %>%
  filter(sub_id %in% names(y_list)) %>%
  arrange(match(sub_id, names(y_list)))

## Create binary covariates
df_cov$low_income <- as.numeric(df_cov$pir <= 1)
df_cov$old <- as.numeric(df_cov$age >= 65)
df_cov$male <- as.numeric(df_cov$gender == 1)

## Age- and Sex- adjusted covariates
df_cov$bmd_ll_adj <- residuals(
  gam(bmd_ll ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$bmd_rl_adj <- residuals(
  gam(bmd_l_min ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$bmd_adj <- residuals(
  gam(bmd_l_min ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$hdl_adj <- residuals(
  gam(hdl ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$ghb_adj <- residuals(
  gam(ghb ~ male + s(age), data = df_cov, na.action = na.exclude)
)


get_plot_name_for_cov <- function(cov, other_cov = FALSE) {
  if (other_cov) {
    name_map <- list(
      'old' = '< 65',
      'male' = 'Female',
      'low_income' = 'Above Poverty Line'
    )
  } else {
    name_map <- list(
      'old' = '>= 65',
      'male' = 'Male',
      'low_income' = 'Below Poverty Line'
    )
  }
  name_map[[cov]]
}

## Plot QFs color-coded by covariate level
for (bin_pred in c('old', 'male', 'low_income')) {

  k_from <- 1
  k_to <- 2

  ## Plot: QF
  set.seed(12345)
  path_plot <- file.path(
    dir_eda,
    str_glue('eda_Qi_{bin_pred}.png')
  )
  png(path_plot, width = 960, height = 960, pointsize = 18)
  col_0 <- rgb(0, 0, 0, alpha = 0.25)
  col_1 <- rgb(1, 0, 0, alpha = 0.25)
  mask <- !is.na(df_cov[bin_pred])
  idx <- sample(1:sum(mask))
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
    width = c(2.5, 2.5),
    type  = c(1, 1),
    label = c(
      get_plot_name_for_cov(bin_pred, TRUE), 
      get_plot_name_for_cov(bin_pred, FALSE)
    )
  )
  plot_funs(
    fun_list = fun_list[idx],
    grid_list = grid_list[idx],
    colors = colors[idx],
    widths = widths[idx],
    types = types[idx],
    color_width_type_labels = color_width_type_labels
  )
  dev.off()

  ## Plot: z-rot-embeddings
  set.seed(12345)
  path_plot <- file.path(
    dir_eda,
    str_glue('eda_z-rot_{bin_pred}.png')
  )
  png(path_plot, width = 960, height = 960, pointsize = 18)
  col_0 <- rgb(0, 0, 0, alpha = 0.25)
  col_1 <- rgb(1, 0, 0, alpha = 0.25)
  mask <- !is.na(df_cov[bin_pred])
  idx <- sample(1:sum(mask))
  Z_plot <- do.call(rbind, z_rot_ctx$payload)[mask,k_from:k_to]
  df_cov_to_plot <- df_cov[mask,]
  colors = rep(col_0, length.out = nrow(Z_plot))
  colors[df_cov_to_plot[bin_pred] == 1] <- col_1
  shapes <- c(
    rep(19, length.out = nrow(Z_plot))
  )
  sizes <- c(
    rep(0.5, length.out = nrow(Z_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(19, 19),
    size = c(0.5, 0.5),
    label = c(
      get_plot_name_for_cov(bin_pred, TRUE), 
      get_plot_name_for_cov(bin_pred, FALSE)
    )
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

}



## ========== Group Comparison ========== ##

## ---------- Latent Traversal Plots ---------- ##

## Sweep one latent coordinate across its empirical quantiles, holding the
## others at their mean, and decode. Characterizes what each direction of the
## latent space means; these are not covariate effects.

## Create directory
dir_trav_plots <- file.path('artifacts', dir_art, 'plots',
                            str_glue('traversal-plots{suffix}'))
if (!dir.exists(dir_trav_plots)) dir.create(dir_trav_plots, recursive = TRUE)

K <- pipeline$stages[[3]]$state$child_qg_pca$state$K
k_star_quantiles <- seq(0.1, 0.9, by = 0.1)
for (k_star in 1:K) {

  ## Set latent embeddings to decode
  Z <- Z_latent
  Z_mean <- colMeans(Z)
  Z_plot <- matrix(
    Z_mean, 
    nrow = length(k_star_quantiles), 
    ncol = length(Z_mean), 
    byrow = TRUE
  )
  Z_plot[,k_star] <- quantile(Z[,k_star], k_star_quantiles)

  ## Decode
  Ji <- length(Qi_ctx$payload[[1]])
  Qi_plot <- decode_latent_to_Qi(
    pipeline,
    latent_list = asplit(Z_plot, MARGIN = 1),
    Ji          = Ji
  )

  ## Center each QF by subtracting the q = 0.5 QF
  idx_med <- which(k_star_quantiles == 0.5)
  if (length(idx_med) != 1) {
    stop("k_star_quantiles must contain exactly one entry equal to 0.5.")
  }
  Qi_med  <- Qi_plot[[idx_med]]
  Qi_plot <- lapply(Qi_plot, function(q) q - Qi_med)

  ## Plot
  pi_grid <- pi_grid_fun(Ji)
  path_plot <- file.path(
    dir_trav_plots,
    str_glue('traversal{suffix}_k-{k_star}.png')
  )
  pal <- colorRampPalette(c("blue", "gray", "red"))(101)
  line_cols <- pal[round(k_star_quantiles * 100) + 1]
  y_lim <- range(unlist(Qi_plot))
  png(path_plot, width = 720, height = 720, pointsize = 14)
  plot(
    NULL,
    xlim = c(0, 1),
    ylim = y_lim,
    xlab = "p",
    ylab = "Q(p) - Q_0.5(p)",
    main = str_glue("Traversal of component k = {k_star}")
  )
  for (i in seq_along(Qi_plot)) {
    lines(pi_grid, Qi_plot[[i]], col = line_cols[i], lwd = 2)
  }
  legend(
    "topleft",
    legend = sprintf("%.1f", k_star_quantiles),
    col    = line_cols,
    lwd    = 2,
    title  = "quantile level",
    bty    = "n"
  )
  dev.off()

}



## ---------- Frequentist Group Comparison (via conditional Z-means) ---------- ##

## Set binary predictor
bin_pred <- 'low_income'
run_latent_diagnostics <- FALSE
run_generativity_diagnostics <- FALSE

## Covariate names for plotting
gp1_name <- get_plot_name_for_cov(bin_pred, FALSE)
gp0_name <- get_plot_name_for_cov(bin_pred, TRUE)

## Create directory
dir_gc <- file.path('artifacts', dir_art, 'plots',
                    str_glue('group-comp{suffix}'), bin_pred)
if (!dir.exists(dir_gc)) dir.create(dir_gc, recursive = TRUE)


## ----- Fit linear regression

## Fit linear model
mask <- !is.na(df_cov[bin_pred])
Z <- Z_latent[mask, , drop = FALSE]
N <- nrow(Z)
K <- ncol(Z)
x <- df_cov[mask,bin_pred]
model <- fit_binary(Z, x)


## ----- Model diagnostics

## Latent diagnostics
if (run_latent_diagnostics) {
  binary_diagnostics(model, dir_gc, col_train)
}


## Generativity diagnostics
if (run_generativity_diagnostics) {
  S <- 5
  R <- 20
  X <- matrix(x, ncol = 1)
  gen_scores <- list()
  for (v in c(0, 1)) {
    y_mask <- x == v
    start <- Sys.time()
    gen_scores[[v + 1]] <- evaluate_generativity(
      pipeline = pipeline,
      y_list = y_list[!is.na(df_cov[bin_pred])],
      X = X,
      y_mask = y_mask,
      fit_fun = fit_binary,
      sample_fun = sample_binary,
      S = S,
      R = R,
      val_subsample_n = 500,
      n_cores = 1
    )
    stop <- Sys.time()
    print(stop - start)
  }
  plot_generativity_boxes(
    groups      = c(gp0_name, gp1_name),
    scores      = list(
      lapply(gen_scores[[1]]$splits, `[[`, "log_norm"),
      lapply(gen_scores[[2]]$splits, `[[`, "log_norm")
    ),
    Ks          = NULL,
    vline_group = NULL,
    hline       = 0,
    ylab        = "Log Normalized Conditional Generativity Scores",
    main        = str_glue("Conditional Generativity"),
    path        = file.path(dir_gc, str_glue('gen_{bin_pred}.png')),
    collapse    = TRUE
  )
}


## ----- Hotelling's T-Squared Test and Roy-Bose Intervals

res_hotel <- binary_test(model)
print(res_hotel)

res_rb <- binary_intervals(model)
print(res_rb)

## Save results
path <- file.path(
  dir_gc,
  str_glue('mod_{bin_pred}.txt')
)
sink(path)
print("========== Hotelling ==========")
res_hotel
print("========== Roy-Bose ==========")
res_rb
sink()

## Plot Roy-Bose Intervals
path_plot <- file.path(
  dir_gc,
  str_glue('roy-bose_{bin_pred}.png')
)
xlim_rb <- if (latent == 'z_rot') c(-1.2, 1.2) else NULL
png(path_plot, width = 600, height = 800, pointsize = 14)
binary_plot_intervals(
  res_rb,
  xlim = xlim_rb,
  main = str_glue("Joint Bands for Latent Regression Coefficients ({gp1_name})")
)
dev.off()

## MIMS activity-intensity thresholds (per-minute units) for CDF plots.
mims_thresholds <- list(
  list(
    label = "light",
    value = 15.05,
    col = 'darkgreen',
    eval_type = 'cdf',
    main = "Time in sedentary"
  ),
  list(
    label = "mvpa",
    value = 19.61,
    col = 'darkblue',
    eval_type = 'survival',
    main = "Time above MVPA threshold"
  ),
  list(
    label = "strenuous",
    value = 50.00,
    col = 'darkred',
    eval_type = 'survival',
    main = "Time above strenuous threshold"
  )
)

## Default quantile levels for QF plots
quantile_level_labels_fun <- function(q_levels) {
  sprintf("p = %g", q_levels)
}
quantile_levels <- list(
  list(
    label = quantile_level_labels_fun(0.5),
    value = 0.5,
    col = 'darkgreen'
  ),
  list(
    label = quantile_level_labels_fun(0.75),
    value = 0.75,
    col = 'darkblue'
  ),
  list(
    label = quantile_level_labels_fun(0.99),
    value = 0.99,
    col = 'darkred'
  )
)

## ----- Conditional QF Plots

## --- Parametric bootstrap of conditional latent means

alpha     <- 0.05
boot_res  <- binary_boot_interpret(model, R = 1000, seed = 12345)
z_hat_0   <- boot_res$z_hat_0
z_hat_1   <- boot_res$z_hat_1
z_boot_0  <- boot_res$z_boot_0
z_boot_1  <- boot_res$z_boot_1

## Decode original and bootstrap conditional means to Qi-space
Qi_hat_0  <- decode_latent_to_Qi(pipeline, list(z_hat_0), Ji)[[1]]
Qi_hat_1  <- decode_latent_to_Qi(pipeline, list(z_hat_1), Ji)[[1]]
Qi_boot_0 <- decode_latent_to_Qi(pipeline, z_boot_0, Ji)
Qi_boot_1 <- decode_latent_to_Qi(pipeline, z_boot_1, Ji)


## --- Plot conditional QFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_qfs.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qfs(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = "Decoded Conditional Means",
  group_labels = c(gp0_name, gp1_name)
)
dev.off()


## --- Plot difference of conditional QFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_qfs-contrast.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qf_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = "Decoded Conditional Mean Contrast",
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  legend_loc = 'bottomleft'
)
dev.off()


## --- Shared y-grid for CDF plots
y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1)


## --- Plot conditional CDFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_cdf.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdfs(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha      = alpha,
  main       = "Decoded Conditional CDFs",
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  group_labels = c(gp0_name, gp1_name)
)
dev.off()


## --- Plot CDF difference with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_cdf-contrast.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdf_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha      = alpha,
  main       = "Decoded Conditional CDF Contrast",
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  legend_loc = 'topright'
)
dev.off()


## --- Plot conditional survival functions (1-F(x)) with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_survival.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_survivals(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha      = alpha,
  main       = "Decoded Conditional Survival Curves",
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  group_labels = c(gp0_name, gp1_name),
  band_type  = 'pointwise'
)
dev.off()


## --- Plot survival function difference with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_survival-contrast.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_survival_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha      = alpha,
  main       = "Decoded Conditional Survival Contrast",
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  band_type  = 'pointwise',
  legend_loc = 'bottomright'
)
dev.off()


## --- Plot thresholds with bands (2 panels, one per threshold)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_threshold.png')
)
png(path_plot, width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_thresholds(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  thresholds   = mims_thresholds,
  eval_type    = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains  = sapply(mims_thresholds, `[[`, "main"),
  y_grid       = y_grid,
  alpha        = alpha,
  main         = "Decoded Conditional Threshold Probabilities",
  group_labels = c(gp0_name, gp1_name)
)
dev.off()


## --- Plot threshold constrasts with bands (2 panels, one per threshold)

path_plot <- file.path(
  dir_gc,
str_glue('gc_{bin_pred}_threshold-contrast.png')
)
png(path_plot, width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_threshold_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  thresholds  = mims_thresholds,
  eval_type   = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains = sapply(mims_thresholds, `[[`, "main"),
  y_grid      = y_grid,
  alpha       = alpha,
  main        = "Decoded Conditional Threshold Probability Contrast",
  diff_label  = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()


## --- Plot quantile levels with bands (one panel per quantile level)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_quantile-level.png')
)
png(path_plot, width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_quantile_levels(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  quantile_levels = quantile_levels,
  alpha           = alpha,
  main            = "Decoded Conditional Quantile Levels",
  group_labels    = c(gp0_name, gp1_name)
)
dev.off()


## --- Plot quantile-level contrasts with bands (one panel per quantile level)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_quantile-level-contrast.png')
)
png(path_plot, width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_quantile_level_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  quantile_levels = quantile_levels,
  alpha           = alpha,
  main            = "Decoded Conditional Quantile Level Contrast",
  diff_label      = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()


## --- Plot moments with bands (4 panels, one per moment)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_moment.png')
)
png(path_plot, width = 1280, height = 240, pointsize = 14)
plot_decoded_conditional_moments(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = "Decoded Conditional Moments",
  group_labels = c(gp0_name, gp1_name)
)
dev.off()


## --- Contrast in conditional moments (group 1 - group 0)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_moment-contrast.png')
)
png(path_plot, width = 1280, height = 240, pointsize = 14)
plot_decoded_conditional_moment_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = "Decoded Conditional Moment Contrasts",
  diff_label = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()




## ========== Additive Mixed Model ========== ##

## Create directory
dir_amm <- file.path('artifacts', dir_art, 'plots', str_glue('amm{suffix}'))
if (!dir.exists(dir_amm)) dir.create(dir_amm, recursive = TRUE)

## ---------- Fit ----------

n_knots_age <- 20
n_knots_pir <- 20
Z_amm <- Z_latent
covariates_amm <- data.frame(
  male = df_cov$male,
  age  = df_cov$age,
  pir  = df_cov$pir
)
start <- Sys.time()
amm <- fit_amm(
  Z_amm, covariates_amm, 
  n_knots_age = n_knots_age, 
  n_knots_pir = n_knots_pir
)  
stop <- Sys.time()
print(stop - start)

## NOTE: For more nuanced knot selection, see the following references.
##  - "Selecting the number of knots for penlalized splines", Ruppert (2002).
##  - "Semiparametric Regression", Ruppert, Wand, and Carroll (2003).


## ---------- Diagnostics ----------

run_latent_diagnostics <- FALSE
run_generativity_diagnostics <- FALSE

## Create directory
dir_amm_diags <- file.path(dir_amm, 'diagnostics')
if (!dir.exists(dir_amm_diags)) dir.create(dir_amm_diags)

## ----- Latent Diagnostics

if (run_latent_diagnostics) {

  K_amm <- ncol(Z_amm)

  ## (1) Per-column summary table
  summary_tbl <- amm_summary(amm)
  print(summary_tbl)
  write.csv(
    summary_tbl,
    file.path(dir_amm_diags, 'amm_summary.csv'),
    row.names = FALSE
  )

  ## (2) Smoothing-parameter plot
  amm_plot_lambda(amm, file.path(dir_amm_diags, 'amm_lambda.png'))

  ## (3) Residual Q-Q panels
  amm_plot_resid_qq(amm, file.path(dir_amm_diags, 'amm_resid_qq.png'),
                    col_train)

  ## (4) Residual covariance heatmap (Sigma_E)
  amm_plot_sigma_e(amm, file.path(dir_amm_diags, 'amm_sigma_e.png'))

  ## (5) Random-effect covariance heatmaps
  amm_plot_re_cov(
    amm$U_age_hat, K_amm,
    expression(hat(Sigma)[U]^"(age)"),
    file.path(dir_amm_diags, 'amm_sigma_u_age.png')
  )
  amm_plot_re_cov(
    amm$U_pir_hat, K_amm,
    expression(hat(Sigma)[U]^"(pir)"),
    file.path(dir_amm_diags, 'amm_sigma_u_pir.png')
  )

}


## ----- Generativity Diagnostics 

if (run_generativity_diagnostics) {

  ## --- Replicate group comparison diagnostics
  ## NOTE: Two differences from group comparison setting
  ##  - AMM fitting and sampling is aware of all covariates
  ##  - AMM sample size < group comparison sample size

  ## fit_fun for evaluate_generativity()
  fit_amm_nhanes <- function(Z, X) {
    fit_amm(Z, X, n_knots_age = n_knots_age, n_knots_pir = n_knots_pir)
  }

  ## Restrict to complete-case sample (male, age, pir all observed)
  mask_gen     <- complete.cases(covariates_amm)
  y_list_gen   <- y_list[mask_gen]
  X_gen        <- covariates_amm[mask_gen, , drop = FALSE]

  S <- 5
  R <- 20
  for (bin_pred in c('male', 'low_income', 'old')) {

    gp1_name <- get_plot_name_for_cov(bin_pred, FALSE)
    gp0_name <- get_plot_name_for_cov(bin_pred, TRUE)
    x <- df_cov[mask_gen, bin_pred]

    gen_scores <- list()
    for (v in c(0, 1)) {
      y_mask <- x == v
      start <- Sys.time()
      gen_scores[[v + 1]] <- evaluate_generativity(
        pipeline = pipeline,
        y_list = y_list_gen,
        X = X_gen,
        y_mask = y_mask,
        fit_fun = fit_amm_nhanes,
        sample_fun = sample_amm,
        S = S,
        R = R,
        val_subsample_n = 500,
        n_cores = 1
      )
      stop <- Sys.time()
      print(stop - start)
    }
    plot_generativity_boxes(
      groups      = c(gp0_name, gp1_name),
      scores      = list(
        lapply(gen_scores[[1]]$splits, `[[`, "log_norm"),
        lapply(gen_scores[[2]]$splits, `[[`, "log_norm")
      ),
      Ks          = NULL,
      vline_group = NULL,
      hline       = 0,
      ylab        = "Log Normalized Conditional Generativity Scores",
      main        = str_glue("Conditional Generativity"),
      path        = file.path(dir_amm_diags, str_glue('amm_gen_{bin_pred}.png')),
      collapse    = TRUE
    )
  }

}



## ========== AMM Interpretation ========== ##

## Create directories for continuous variable interpretation
dir_amm_interp <- file.path(dir_amm, 'interpretation')
if (!dir.exists(dir_amm_interp)) dir.create(dir_amm_interp)


## ---------- Driver ----------

Ji_interp <- length(Qi_ctx$payload[[1]])
B_interp  <- 20  ## TODO: Increase this!

amm_boots <- amm_boot_interpret(
  amm,
  covariates_kept = covariates_amm[amm$keep, , drop = FALSE],
  B               = B_interp,
  n_cores         = 5,
  seed            = 12345
)

## Interpret continuous covariates
cov_to_plot <- c('age', 'pir') # c('age', 'pir')
fam_to_plot <- c('latent', 'qf', 'cdf', 'survival', 'moment', 'threshold', 'quantile')
for (cov in cov_to_plot) {

  ## Create covariate directory
  dir_amm_cov <- file.path(dir_amm_interp, cov)
  if (!dir.exists(dir_amm_cov)) dir.create(dir_amm_cov)

  ## Reference QF / CDF (decoded at male = 0, age = 40, pir = 1)
  cv_ref_decode <- list(male = 0, age = 40, pir = 2)
  Qi_ref_decode <- .decode_z_vec(pipeline,
    amm_compute_conditional_mean(amm, cv_ref_decode), Ji_interp)
  pi_grid_ref <- pi_grid_fun(Ji_interp)

  ## Latent effect plots
  if ('latent' %in% fam_to_plot) {
    path_plot <- file.path(dir_amm_cov, str_glue("amm_{cov}_latent-effect.png"))
    png(path_plot, width = 1280, height = 640, pointsize = 14)
    amm_plot_latent_effects(amm, cov, band = TRUE,
                            boot_reps = amm_boots,
                            covariates_df = covariates_amm)
    dev.off()
  }

  ## QF Reference
  if ('qf' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf_reference.png")),
        width = 960, height = 960, pointsize = 24)
    plot_funs(
      fun_list  = list(Qi_ref_decode),
      grid_list = list(pi_grid_ref),
      colors    = "gray60",
      widths    = 4,
      types     = 1,
      ylab      = "Q(p)",
      main = "Reference QF"
    )
    dev.off()
  }

  ## CDF Reference
  if ('cdf' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf_reference.png")),
        width = 960, height = 960, pointsize = 24)
    y_grid_ref <- .amm_default_y_grid(list(Qi_ref_decode))
    F_ref_decode <- qi_to_cdf(Qi_ref_decode, y_grid_ref)
    plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
        xlab = "y", ylab = "F(y)", main = "Reference CDF")
    lines(y_grid_ref, F_ref_decode, col = "gray60", lwd = 4)
    dev.off()
  }

  ## Survival Reference
  if ('survival' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival_reference.png")),
        width = 960, height = 960, pointsize = 24)
    S_ref_decode <- .amm_qi_to_survival(Qi_ref_decode, y_grid_ref)
    plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
        xlab = "y", ylab = "1 - F(y)", main = "Reference Survival Function")
    lines(y_grid_ref, S_ref_decode, col = "gray60", lwd = 4)
    dev.off()
  }

  ## QF family
  if ('qf' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_conditional_qf(amm, cov, pipeline, Ji = Ji_interp,
                            quantile_lines = quantile_levels)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_qf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                    boot_reps = amm_boots,
                                    quantile_lines = quantile_levels)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf-contrast.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_contrast_qf(amm, cov, pipeline, Ji = Ji_interp,
                        quantile_lines = quantile_levels)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf-contrast-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_contrast_qf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                boot_reps = amm_boots,
                                quantile_lines = quantile_levels)
    dev.off()
  }

  ## CDF family
  if ('cdf' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_conditional_cdf(amm, cov, pipeline, Ji = Ji_interp,
                            threshold_lines = mims_thresholds)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_cdf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                    threshold_lines = mims_thresholds,
                                    boot_reps = amm_boots)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf-contrast.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_contrast_cdf(amm, cov, pipeline, Ji = Ji_interp,
                          threshold_lines = mims_thresholds)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf-contrast-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_contrast_cdf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                  threshold_lines = mims_thresholds,
                                  boot_reps = amm_boots)
    dev.off()
  }

  ## Survival function family
  if ('survival' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_conditional_survival(amm, cov, pipeline, Ji = Ji_interp,
                                  threshold_lines = mims_thresholds)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_survival_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                          threshold_lines = mims_thresholds,
                                          boot_reps = amm_boots)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-contrast.png")),
        width = 960, height = 720, pointsize = 14)
    amm_plot_contrast_survival(amm, cov, pipeline, Ji = Ji_interp,
                              threshold_lines = mims_thresholds)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-contrast-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_contrast_survival_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                      threshold_lines = mims_thresholds,
                                      boot_reps = amm_boots)
    dev.off()
  }

  ## Moments family (joint bands over the covariate)
  if ('moment' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_moments.png")),
        width = 1280, height = 240, pointsize = 14)
    amm_plot_conditional_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                         boot_reps = amm_boots)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_moments-contrast.png")),
        width = 1280, height = 240, pointsize = 14)
    amm_plot_contrast_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                      boot_reps = amm_boots)
    dev.off()
  }

  ## Threshold-crossing (one panel per MIMS threshold; absolute + contrast)
  if ('threshold' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_threshold.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_threshold_crossing_panels(
      amm, cov, mims_thresholds, pipeline,
      main      = str_glue("AMM Decoded Threshold Probabilities by {cov}"),
      Ji        = Ji_interp,
      boot_reps = amm_boots
    )
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_threshold-contrast.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_threshold_crossing_panels(
      amm, cov, mims_thresholds, pipeline,
      contrast  = TRUE,
      main      = str_glue("AMM Decoded Threshold Probability Contrasts by {cov}"),
      Ji        = Ji_interp,
      boot_reps = amm_boots
    )
    dev.off()
  }

  ## Quantile-crossing (one panel per quantile level; absolute + contrast)
  if ('quantile' %in% fam_to_plot) {
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_quantile.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_quantile_crossing_panels(
      amm, cov, quantile_levels, pipeline,
      main      = str_glue("AMM Decoded Quantile Levels by {cov}"),
      Ji        = Ji_interp,
      boot_reps = amm_boots
    )
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_quantile-contrast.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_quantile_crossing_panels(
      amm, cov, quantile_levels, pipeline,
      contrast  = TRUE,
      main      = str_glue("AMM Decoded Quantile Level Contrasts by {cov}"),
      Ji        = Ji_interp,
      boot_reps = amm_boots
    )
    dev.off()
  }
}


## --- Plots for binary covariates

## Create covariate directory
dir_amm_cov <- file.path(dir_amm_interp, 'male')
if (!dir.exists(dir_amm_cov)) dir.create(dir_amm_cov)

## Covariate names for plotting
gp1_name <- get_plot_name_for_cov('male', FALSE)
gp0_name <- get_plot_name_for_cov('male', TRUE)

## Set reference values
amm_male_age_ref <- 40
amm_male_pir_ref <- 1
amm_male_alpha   <- 0.05
cv_male_0 <- list(male = 0, age = amm_male_age_ref, pir = amm_male_pir_ref)
cv_male_1 <- list(male = 1, age = amm_male_age_ref, pir = amm_male_pir_ref)

## Latent point estimates and bootstrap replicate matrices (B x K)
z_hat_male_0  <- as.numeric(amm_compute_conditional_mean(amm, cv_male_0))
z_hat_male_1  <- as.numeric(amm_compute_conditional_mean(amm, cv_male_1))
z_boot_male_0 <- amm_replicate_z_at(amm_boots, cv_male_0)
z_boot_male_1 <- amm_replicate_z_at(amm_boots, cv_male_1)

## Decode to Qi (point + B x Ji bootstrap matrices)
Qi_hat_male_0  <- .decode_z_vec(pipeline, z_hat_male_0, Ji_interp)
Qi_hat_male_1  <- .decode_z_vec(pipeline, z_hat_male_1, Ji_interp)
Qi_boot_male_0 <- t(apply(z_boot_male_0, 1,
                          function(z) .decode_z_vec(pipeline, z, Ji_interp)))
Qi_boot_male_1 <- t(apply(z_boot_male_1, 1,
                          function(z) .decode_z_vec(pipeline, z, Ji_interp)))
main_suffix <- str_glue("male @ age={amm_male_age_ref}, pir={amm_male_pir_ref}")

## QF and QF contrast
png(file.path(dir_amm_cov, "amm_male_qf.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Means ({main_suffix})"),
  group_labels = c(gp0_name, gp1_name)
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_qf-contrast.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Mean Contrast ({main_suffix})"),
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  legend_loc = 'bottomleft'
)
dev.off()

## CDF and CDF contrast
y_grid_male <- default_y_grid_two_group(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1
)
png(file.path(dir_amm_cov, "amm_male_cdf.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional CDFs ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  group_labels = c(gp0_name, gp1_name)
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_cdf-contrast.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional CDF Contrast ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  legend_loc = 'topright'
)
dev.off()

## Survival function and survival function contrast
png(file.path(dir_amm_cov, "amm_male_survival.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_survivals(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional Survival Curves ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  group_labels = c(gp0_name, gp1_name),
  band_type  = 'pointwise'
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_survival-contrast.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_survival_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional Survival Contrast ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  diff_label = str_glue("{gp1_name} - {gp0_name}"),
  band_type  = 'pointwise',
  legend_loc = 'bottomright'
)
dev.off()

## Moments and moment contrast
png(file.path(dir_amm_cov, "amm_male_moment.png"),
    width = 1280, height = 240, pointsize = 14)
plot_decoded_conditional_moments(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Moments ({main_suffix})"),
  group_labels = c(gp0_name, gp1_name)
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_moment-contrast.png"),
    width = 1280, height = 240, pointsize = 14)
plot_decoded_conditional_moment_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Moment Contrasts ({main_suffix})"),
  diff_label = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()

## Threshold probabilities and threshold contrast (one panel per threshold)
png(file.path(dir_amm_cov, "amm_male_threshold.png"),
    width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_thresholds(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  thresholds  = mims_thresholds,
  eval_type   = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains = sapply(mims_thresholds, `[[`, "main"),
  y_grid      = y_grid_male,
  alpha       = amm_male_alpha,
  main        = str_glue("AMM Decoded Conditional Threshold Probabilities ({main_suffix})"),
  group_labels = c(gp0_name, gp1_name)
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_threshold-contrast.png"),
    width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_threshold_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  thresholds  = mims_thresholds,
  eval_type   = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains = sapply(mims_thresholds, `[[`, "main"),
  y_grid      = y_grid_male,
  alpha       = amm_male_alpha,
  main        = str_glue("AMM Decoded Conditional Threshold Probability Contrast ({main_suffix})"),
  diff_label = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()

## Quantile levels and quantile-level contrast (one panel per quantile level)
png(file.path(dir_amm_cov, "amm_male_quantile-level.png"),
    width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_quantile_levels(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  quantile_levels = quantile_levels,
  alpha           = amm_male_alpha,
  main            = str_glue("AMM Decoded Conditional Quantile Levels ({main_suffix})"),
  group_labels    = c(gp0_name, gp1_name)
)
dev.off()
png(file.path(dir_amm_cov, "amm_male_quantile-level-contrast.png"),
    width = 960, height = 240, pointsize = 12)
plot_decoded_conditional_quantile_level_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  quantile_levels = quantile_levels,
  alpha           = amm_male_alpha,
  main            = str_glue("AMM Decoded Conditional Quantile Level Contrast ({main_suffix})"),
  diff_label      = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()



## ========== Hierarchical Clustering ========== ##

## Ward clustering of the latent embeddings, characterized by decoding
## embedding-space group means back to QF space, with a QF-space clustering as
## a baseline. Outputs land in plots/hca/<latent> so that the c and z_rot runs
## sit side by side rather than overwriting each other.

## ---------- Helpers ---------- ##

## Place every Qi on the shared augmented p-grid, as a J x N matrix. Subjects
## have different Ji, so this is what makes them comparable pointwise.
hca_qf_aug <- function(Qi_list, p_grid_aug, supp_Y) {
  vapply(
    Qi_list,
    function(Qi) inv_eqf_cgrid(Qi, pi_grid_fun(length(Qi)), p_grid_aug, supp_Y),
    numeric(length(p_grid_aug))
  )
}

## Pairwise Wasserstein distances between QFs, as a `dist`, from the augmented
## matrix above. The metric is a w-weighted L2, so scaling coordinates by
## sqrt(w) reduces it to plain Euclidean distance and dist() does the O(N^2)
## work in C -- the R-level loop in compute_pwds() is hopeless at N ~ 8k.
hca_qf_dist <- function(Q_aug, p_grid_aug) {
  w_aug <- get_quadrature_weights(p_grid_aug)
  dist(sweep(t(Q_aug), 2, sqrt(w_aug), `*`))
}

## Wasserstein barycenter of the columns in `idx`. In one dimension the W2
## barycenter has a closed form -- it is the pointwise mean of the quantile
## functions -- so this is exact, not an iterative solve.
hca_qf_barycenter <- function(Q_aug, idx) {
  rowMeans(Q_aug[, idx, drop = FALSE])
}

## Relabel clusters 1..k manually: cutree() cluster i becomes cluster
## `new_ids[i]`, so `new_ids` must be a permutation of the cutree() ids.
hca_relabel_manual <- function(cl, new_ids) {
  stopifnot(setequal(new_ids, cl), !anyDuplicated(new_ids))
  new_ids[cl]
}

## Overlaid per-cluster densities of a covariate.
hca_plot_density <- function(v, cl, cols, xlab, main, path) {
  ok    <- !is.na(v)
  dens  <- lapply(seq_along(cols), function(g) density(v[ok & cl == g]))
  png(path, width = 960, height = 720, pointsize = 18)
  plot(
    NULL,
    xlim = range(sapply(dens, function(d) range(d$x))),
    ylim = c(0, max(sapply(dens, function(d) max(d$y)))),
    xlab = xlab, ylab = 'density', main = main
  )
  for (g in seq_along(dens)) lines(dens[[g]], col = cols[g], lwd = 3)
  legend(
    'topright',
    legend = str_glue('cluster {seq_along(cols)}'),
    col = cols, lwd = 3, bty = 'n'
  )
  dev.off()
}

## Cluster-wise horizontal boxplots stacked vertically, with the pooled median
## marked, drawn into the current device. Groups are passed in reverse because
## boxplot() draws its first group at the bottom, and cluster 1 reads better at
## the top.
##
## Args:
##   show_names      label the cluster axis; turn off for all but the first of
##                   several panels sharing that axis
##   faint_outliers  draw outliers as small translucent dots, so that the
##                   boxes stay readable when N is large
##   show_n          print each cluster's non-missing count in the right margin
##   hide_outliers   draw no outliers at all, which also limits the value
##                   axis to the span of the whiskers so that a few extreme
##                   points do not compress the boxes. Overrides
##                   faint_outliers.
##   mean_ci         overlay each cluster's mean (white diamond) and its 95%
##                   t-interval (black bar). The reference line then marks the
##                   pooled mean rather than the pooled median, so that it is
##                   the quantity the intervals are compared against.
##   labels          group names for the cluster axis, indexed by group id;
##                   defaults to the ids themselves
hca_draw_boxes <- function(v, cl, cols, xlab, main, ylab = 'cluster',
                           show_names = TRUE, faint_outliers = FALSE,
                           show_n = FALSE, hide_outliers = FALSE,
                           mean_ci = FALSE, labels = seq_along(cols)) {
  idx    <- rev(seq_along(cols))
  groups <- lapply(idx, function(g) v[cl == g & !is.na(v)])
  out_par <- if (faint_outliers) {
    list(outpch = 16, outcex = 0.4, outcol = adjustcolor('black', alpha.f = 0.25))
  }
  do.call(boxplot, c(
    list(
      groups,
      horizontal = TRUE,
      names      = if (show_names) labels[idx] else rep('', length(idx)),
      col        = cols[idx],
      las        = 1,
      xlab       = xlab,
      ylab       = ylab,
      main       = main,
      outline    = !hide_outliers
    ),
    out_par
  ))
  if (mean_ci) {
    abline(v = mean(v, na.rm = TRUE), lty = 3, lwd = 2)
    m    <- vapply(groups, mean, numeric(1))
    ## NA for a cluster of fewer than two, which then draws no bar
    half <- vapply(
      groups,
      function(x) qt(0.975, max(length(x) - 1, 1)) * sd(x) / sqrt(length(x)),
      numeric(1)
    )
    at <- seq_along(groups)
    ## White casing keeps the bar visible on top of the box and median line
    segments(m - half, at, m + half, at, lwd = 5, col = 'white', lend = 'butt')
    segments(m - half, at, m + half, at, lwd = 2.5, lend = 'butt')
    points(m, at, pch = 23, bg = 'white', cex = 0.9)
  } else {
    abline(v = median(v, na.rm = TRUE), lty = 3, lwd = 2)
  }
  if (show_n) hca_margin_n(lengths(groups))
}

## Counts in the right margin of a cluster-axis panel, one per row.
hca_margin_n <- function(n) {
  mtext(n, side = 4, at = seq_along(n), las = 1, line = 0.3,
        cex = 0.7 * par('cex'), col = 'grey40')
}

## hca_draw_boxes() as a standalone png. `width` and `height` are in pixels at
## 72 dpi; the file written is res / 72 times larger in each direction.
hca_plot_boxes <- function(v, cl, cols, xlab, main, path,
                           width = 960, height = 720, pointsize = 18,
                           res = 72, mar = c(5, 5, 4, 2), ...) {
  ## An individual empty cluster is fine (boxplot just leaves a gap), but an
  ## all-missing covariate has no finite xlim and would error.
  if (all(is.na(v))) {
    warning(str_glue("hca_plot_boxes: '{xlab}' is entirely missing; skipping."))
    return(invisible(NULL))
  }
  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit(dev.off(), add = TRUE)
  par(mar = mar)
  hca_draw_boxes(v, cl, cols, xlab = xlab, main = main, ...)
}

## One figure summarizing every covariate by cluster: a row of panels sharing
## the cluster axis (cluster 1 at the top). A binary covariate gets a bar per
## cluster for its proportion; a continuous one gets cluster-wise boxplots
## without outliers, overlaid with each cluster's mean and 95% CI. The dotted line is the pooled proportion / mean, and the grey numbers
## in each panel's right margin are the non-missing counts behind each row,
## since missingness differs by covariate and by cluster.
##
## `cl` may hold any grouping coded 1..length(cols), not just clusters: pass
## the group names as `labels` and the axis title as `group_lab`, and the left
## margin widens to fit the names.
hca_plot_cov_summary <- function(covs, cl, cols, path,
                                 width = 1200, height = 520, pointsize = 16,
                                 res = 288, labels = NULL,
                                 group_lab = 'cluster') {
  covs <- as.data.frame(covs)
  k    <- length(cols)
  idx  <- rev(seq_len(k))
  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit(dev.off(), add = TRUE)
  ## Cluster labels live in the outer margin so that all panels are equally
  ## wide. mfrow shrinks cex for a row this long, so set it back explicitly.
  par(mfrow = c(1, ncol(covs)), mar = c(4, 0.8, 2.5, 2.6),
      mgp = c(2.4, 0.7, 0), cex = 0.8)
  if (is.null(labels)) {
    labels   <- seq_len(k)
    oma_left <- 3.5
  } else {
    ## Widest name, in margin lines, plus room for the tick and axis title
    oma_left <- max(strwidth(labels, units = 'inches')) /
      (par('mai')[1] / par('mar')[1]) + 2.5
  }
  par(oma = c(0, oma_left, 0, 0))
  for (j in seq_len(ncol(covs))) {
    v  <- covs[[j]]
    nm <- names(covs)[j]
    if (all(is.na(v))) {
      warning(str_glue("hca_plot_cov_summary: '{nm}' is entirely missing."))
      plot.new()
      title(main = nm)
      next
    }
    if (all(v[!is.na(v)] %in% c(0, 1))) {
      prop <- vapply(idx, function(g) mean(v[cl == g], na.rm = TRUE), numeric(1))
      ## Same row positions and bar width as boxplot(), so rows line up
      plot(NULL, xlim = c(0, 1), ylim = c(0.5, k + 0.5), yaxt = 'n',
           xlab = 'proportion', ylab = '', main = nm)
      rect(0, seq_len(k) - 0.4, prop, seq_len(k) + 0.4, col = cols[idx])
      axis(2, at = seq_len(k), labels = if (j == 1) labels[idx] else FALSE,
           las = 1)
      abline(v = mean(v, na.rm = TRUE), lty = 3, lwd = 2)
      hca_margin_n(vapply(idx, function(g) sum(cl == g & !is.na(v)), numeric(1)))
    } else {
      hca_draw_boxes(v, cl, cols, xlab = '', main = nm, ylab = '',
                     show_names = j == 1, show_n = TRUE,
                     hide_outliers = TRUE, mean_ci = TRUE, labels = labels)
    }
  }
  mtext(group_lab, side = 2, outer = TRUE, line = oma_left - 1.3, cex = 0.8)
}

## Move labels at heights `y` apart until neighbours are at least `gap` apart,
## keeping each run of colliding labels centred on its original positions.
hca_spread_labels <- function(y, gap) {
  o   <- order(y)
  ys  <- y[o]
  grp <- seq_along(ys)
  repeat {
    pos <- ys
    for (g in unique(grp)) {
      i <- which(grp == g)
      pos[i] <- mean(ys[i]) + (seq_along(i) - (length(i) + 1) / 2) * gap
    }
    clash <- which(diff(pos) < gap - 1e-9 & diff(grp) != 0)
    if (length(clash) == 0) break
    grp[grp == grp[clash[1] + 1]] <- grp[clash[1]]
  }
  y[o] <- pos
  y
}

## --- Clustered-heatmap colour scales

## Missing covariate values get their own colour rather than falling back on
## white: in a binary column white is already 0, so NA would otherwise be
## indistinguishable from it (low_income, say, inherits every pir NA). Orange
## sits outside both covariate ramps and outside the green cluster rules.
hca_na_col <- 'orange'

## Black/white for a binary covariate.
hca_cols_bw <- function(v, na_col = hca_na_col) {
  out <- rep('white', length(v))
  out[!is.na(v) & v == 1] <- 'black'
  out[is.na(v)] <- na_col
  out
}

## Greys for a continuous covariate.
hca_grey_pal <- function(n = 256) gray.colors(n, start = 0.92, end = 0)
hca_cols_grey <- function(v, n = 256, na_col = hca_na_col) {
  pal <- hca_grey_pal(n)
  rng <- range(v, na.rm = TRUE)
  out <- rep(na_col, length(v))
  ok  <- !is.na(v)
  out[ok] <- if (diff(rng) > 0) {
    pal[round((v[ok] - rng[1]) / diff(rng) * (n - 1)) + 1]
  } else {
    pal[1]
  }
  out
}

## Blue (negative) - white (zero) - red (positive) over [-lim, lim].
hca_div_pal <- function(n = 255) colorRampPalette(c('blue', 'white', 'red'))(n)
hca_cols_div <- function(v, lim, n = 255) {
  if (!is.finite(lim) || lim <= 0) lim <- 1
  vc <- pmax(pmin(v, lim), -lim)
  hca_div_pal(n)[round((vc + lim) / (2 * lim) * (n - 1)) + 1]
}

## One labelled colour ramp in the key panel.
hca_legend_ramp <- function(pal, y0, y1, title, labels) {
  rasterImage(as.raster(matrix(rev(pal), ncol = 1)), 0.05, y0, 0.30, y1)
  text(0.05, y1 + 0.035, title, adj = 0, font = 2)
  text(0.34, seq(y1, y0, length.out = length(labels)), labels, adj = 0)
}

## Color each dendrogram branch that lies wholly inside one cluster with that
## cluster's color; branches above the cut keep the default. `cl` is indexed by
## observation, as returned by cutree(), and `cols` by cluster id.
hca_color_dendrogram <- function(dend, cl, cols) {
  rec <- function(node) {
    if (is.leaf(node)) {
      g <- cl[as.integer(node)]
    } else {
      gs <- integer(length(node))
      for (i in seq_along(node)) {
        child     <- rec(node[[i]])
        node[[i]] <- child$node
        gs[i]     <- child$g
      }
      g <- if (!anyNA(gs) && all(gs == gs[1])) gs[1] else NA_integer_
    }
    if (!is.na(g)) attr(node, 'edgePar') <- list(col = cols[g])
    list(node = node, g = g)
  }
  rec(dend)$node
}

## Clustered heatmap: an N x (A + K) raster of covariate and latent values,
## rows in dendrogram order with the tree drawn alongside (one leaf per row).
## Columns are never re-ordered, so the latent block stays in component order.
##
## Args:
##   hc         hclust whose leaves index the rows of `covs` and `Z`
##   Z          N x K latent matrix. Each column gets its own symmetric scale,
##              +/- max|Z[, k]|, so zero is white everywhere and every
##              component is legible regardless of its variance.
##   covs       data.frame of covariates, one per heatmap column, drawn left of
##              the latent block in the order given
##   cov_types  optional 'binary'/'continuous' per covariate; the default
##              infers 'binary' when every non-NA value of a column is 0 or 1
##   n_clusts   positive integer -> cutree(hc, n_clusts) to partition the rows;
##              NULL (default) leaves the rows unmarked, the dendrogram
##              uncolored, and makes the two flags below no-ops
##   cl         optional cluster label per row, used in place of the raw
##              cutree() ids so relabelled clusters print consistently with
##              the other figures. Must be a relabelling of that same cut.
##   cl_cols    color per cluster id, for the dendrogram branches; defaults
##              to the 'Dark 3' palette used by the other figures
##   cluster_lines   draw rules (in `line_col`) between the row blocks
##   label_clusters  print each cluster's number in the right margin, centred
##                   on its block
##   na_col     colour for missing covariate values, in both column types
##   z_labs     column labels for the Z block; defaults to 'k = 1..K'. Pass a
##              mostly-'' vector when Z is wide (Qi on the augmented p-grid)
##              so the axis stays readable.
##   cov_frac   share of the plot width given to the covariate block; the
##              default A/(A + K) makes every column equally wide, which is
##              right for a K-column latent but squeezes the covariates to a
##              sliver when K is in the hundreds
##   path       output png
##   width, height   figure size in pixels at 72 dpi; the file written is
##                   res / 72 times larger in each direction
##   res        resolution in dpi. Text, lines and layout scale with it, so
##              raising it sharpens the figure without changing its look.
hca_plot_heatmap <- function(hc, Z, covs, path,
                            cov_types = NULL, n_clusts = NULL, cl = NULL,
                            cl_cols = NULL,
                            cluster_lines = TRUE, label_clusters = TRUE,
                            line_col = 'green3',
                            na_col = hca_na_col, z_labs = NULL,
                            cov_frac = NULL,
                            width = 1000, height = 1200, pointsize = 16,
                            res = 288) {
  covs <- as.data.frame(covs)
  nr   <- nrow(Z)
  A    <- ncol(covs)
  K    <- ncol(Z)
  stopifnot(nrow(covs) == nr, length(hc$order) == nr, A >= 1)

  if (is.null(cov_types)) {
    cov_types <- vapply(
      covs,
      function(v) if (all(v[!is.na(v)] %in% c(0, 1))) 'binary' else 'continuous',
      character(1)
    )
  }
  stopifnot(length(cov_types) == A,
            cov_types %in% c('binary', 'continuous'))

  ## vapply (not sapply/mapply) so a single covariate still yields a matrix
  cols_cov <- vapply(
    seq_len(A),
    function(j) {
      v <- covs[[j]]
      if (cov_types[j] == 'binary') {
        hca_cols_bw(v, na_col = na_col)
      } else {
        hca_cols_grey(v, na_col = na_col)
      }
    },
    character(nr)
  )
  ## One symmetric scale per component: +/- max|Z[, k]|, so each column uses
  ## the full blue-to-red range regardless of how much variance it carries.
  lims <- apply(abs(Z), 2, max, na.rm = TRUE)
  cols_lat <- vapply(
    seq_len(K),
    function(k) hca_cols_div(Z[, k], lims[k]),
    character(nr)
  )

  ord      <- hc$order
  cols_cov <- cols_cov[ord, , drop = FALSE]
  cols_lat <- cols_lat[ord, , drop = FALSE]
  if (is.null(z_labs)) z_labs <- paste('k =', seq_len(K))
  stopifnot(length(z_labs) == K)
  ## The two blocks are drawn as separate rasters on a [0, 1] axis so their
  ## widths are independent of their column counts -- otherwise a wide Z (Qi
  ## on the augmented p-grid, K ~ 500) squeezes the covariates to a sliver.
  if (is.null(cov_frac)) cov_frac <- A / (A + K)
  stopifnot(cov_frac > 0, cov_frac < 1)

  dend <- as.dendrogram(hc)
  if (!is.null(n_clusts)) {
    stopifnot(length(n_clusts) == 1, n_clusts == round(n_clusts), n_clusts >= 1)
    if (is.null(cl)) cl <- cutree(hc, k = n_clusts)
    if (is.null(cl_cols)) cl_cols <- hcl.colors(n_clusts, palette = 'Dark 3')
    stopifnot(length(cl) == nr, length(cl_cols) == n_clusts)
    dend <- hca_color_dendrogram(dend, cl, cl_cols)
  }

  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit({dev.off(); layout(1)}, add = TRUE)
  layout(matrix(1:3, nrow = 1), widths = c(1.1, 2.4, 0.9))

  ## Dendrogram: yaxs = 'i' makes leaf i sit at y = i, matching the raster
  par(mar = c(5, 2, 6, 0))
  plot(dend, horiz = TRUE, axes = FALSE, yaxs = 'i', leaflab = 'none')

  ## Heatmap. as.raster() draws its first row at the top, so flip to put
  ## dendrogram-order row 1 at the bottom where the first leaf is.
  ## Right margin holds the cluster numbers, between heatmap and key
  par(mar = c(5, 0, 6, if (label_clusters) 2.5 else 1))
  plot(NULL, xlim = c(0, 1), ylim = 0.5 + c(0, nr),
      xaxs = 'i', yaxs = 'i', axes = FALSE, xlab = '', ylab = '')
  rasterImage(as.raster(cols_cov[nr:1, , drop = FALSE]),
              xleft = 0, ybottom = 0.5, xright = cov_frac, ytop = nr + 0.5,
              interpolate = FALSE)
  rasterImage(as.raster(cols_lat[nr:1, , drop = FALSE]),
              xleft = cov_frac, ybottom = 0.5, xright = 1, ytop = nr + 0.5,
              interpolate = FALSE)
  ## Column centres in axis units, per block
  at_cov <- (seq_len(A) - 0.5) / A * cov_frac
  at_z   <- cov_frac + (seq_len(K) - 0.5) / K * (1 - cov_frac)
  ## Two axis() calls over only the labelled positions: one call spanning both
  ## blocks lets axis() thin out labels it judges too dense, which silently
  ## drops the sparse p-marks used for a wide Z.
  axis(3, at = at_cov, labels = names(covs), las = 2, tick = FALSE, line = -0.5)
  keep <- nzchar(z_labs)
  if (any(keep)) {
    axis(3, at = at_z[keep], labels = z_labs[keep], las = 2,
         tick = FALSE, line = -0.5)
  }
  abline(v = cov_frac, lwd = 2)
  if (!is.null(n_clusts)) {
    ## Clusters are contiguous in dendrogram order, so run-lengths give both
    ## the cut points and each block's extent in one pass
    runs <- rle(unname(cl[ord]))
    stopifnot(length(runs$values) == n_clusts)
    ends <- cumsum(runs$lengths)
    if (cluster_lines) {
      abline(h = head(ends, -1) + 0.5, col = line_col, lwd = 2)
    }
    if (label_clusters) {
      ## mtext rather than axis() so no label is dropped when a block is thin
      mids <- ends - (runs$lengths - 1) / 2
      for (i in seq_along(mids)) {
        mtext(runs$values[i], side = 4, at = mids[i], line = 0.6, las = 1)
      }
    }
  }
  box()

  ## Key
  par(mar = c(5, 1, 6, 1))
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = '', ylab = '')
  rect(0.05, c(0.90, 0.83), 0.30, c(0.96, 0.89),
      col = c('white', 'black'), border = 'grey50')
  text(0.05, 0.995, 'binary', adj = 0, font = 2)
  text(0.34, c(0.93, 0.86), c('0', '1'), adj = 0)
  hca_legend_ramp(hca_grey_pal(), 0.52, 0.74, 'continuous', c('max', 'min'))
  ## Scale is per column, so the key shows sign rather than a shared bound
  hca_legend_ramp(hca_div_pal(), 0.14, 0.36, 'latent', c('+', '0', '-'))
  ## NA applies to the covariate columns only; latent values are never missing
  rect(0.05, 0.035, 0.30, 0.085, col = na_col, border = 'grey50')
  text(0.05, 0.115, 'missing', adj = 0, font = 2)
  text(0.34, 0.06, 'NA', adj = 0)
}


## ---------- Settings ---------- ##

# hca_latent   <- 'z_rot'      # 'c' or 'z_rot'
# hca_k        <- 20        # set after eyeballing hca_dendrogram.png, then re-run

hca_J_aug <- 500      # augmented p-grid size for the Qi representation

## Shared by the four figures that make up the activity panel in LaTeX
## (cluster-qfs-contrast.png over the three box-q*.png), so text sizes match.
hca_pointsize <- 14
hca_res       <- 288

## Cluster Relabels, by representation and then k. Entry i is the new label
## for cutree() cluster i, so 1:k keeps the default labeling. A representation
## or k with no entry also keeps the default.
cluster_relabels <- list(
  'c' = list(
    '11' = c(2, 3, 1, 8, 9, 7, 10, 4, 6, 5, 11)
  )
)

## Phenotypes defined manually post-hoc
phenotypes_to_clusters <- list(
  'c' = list(
    '11' = list(
      'very active' = c(1, 2),
      'moderately active' = c(3, 4, 5),
      'weekend warrior' = c(6),
      'frequent mover' = c(7),
      'inactive' = c(8, 9, 10),
      'very inactive' = c(11)
    )
  )
)

## Subject-level activity quantiles, boxplotted per cluster alongside the
## covariates. These are a property of the data, not of the clustering, so
## they are evaluated once rather than per representation or per k.
hca_q_levels <- c(0.5, 0.75, 0.99)
Q_lvl_hca <- t(hca_qf_aug(Qi_ctx$payload, hca_q_levels,
                          pipeline$training$cache$supp_Y))

## 'c' and 'z_rot' are latent embeddings; 'qi' clusters the quantile functions
## themselves, on the augmented p-grid under the Wasserstein metric. Qi is not
## a latent representation, so it skips the pairwise-embedding scatter plot --
## every other figure is produced for it exactly as for the embeddings.
for (hca_latent in c('c')) {

  hca_is_qi     <- hca_latent == 'qi'
  hca_rep_label <- switch(
    hca_latent,
    z_rot = 'z_rot-embeddings',
    c     = 'c-embeddings',
    qi    = 'Qi',
    stop(str_glue("Unknown hca_latent: '{hca_latent}'"))
  )

  dir_hca <- file.path('artifacts', dir_art, 'plots', 'hca', hca_latent)
  if (!dir.exists(dir_hca)) dir.create(dir_hca, recursive = TRUE)

  ## NOTE: N, K, Ji, mask, x, y_grid, ... are all clobbered by earlier
  ## sections, so every name introduced below carries an _hca suffix.
  Ji_hca         <- length(Qi_ctx$payload[[1]])
  pi_grid_hca    <- pi_grid_fun(Ji_hca)
  p_grid_aug_hca <- sort(unique(c(
    p_grid, pi_grid_fun(hca_J_aug - length(p_grid))
  )))

  ## Representation, metric, and cluster representative. The tree does not
  ## depend on k, so it is built once per representation rather than once per
  ## k -- the Qi distance matrix alone is ~8k x 8k.
  start <- Sys.time()
  if (hca_is_qi) {
    ## Qi on the shared augmented grid; rows are subjects, as for a latent
    Q_aug_hca <- hca_qf_aug(Qi_ctx$payload, p_grid_aug_hca,
                            pipeline$training$cache$supp_Y)
    Z_hca     <- t(Q_aug_hca)
    d_hca     <- hca_qf_dist(Q_aug_hca, p_grid_aug_hca)
    grid_hca  <- p_grid_aug_hca
    ## In 1-D the W2 barycenter is the pointwise mean of the QFs, so it is the
    ## exact analogue of decoding the mean of a cluster's embeddings.
    hca_rep_mean <- function(idx) hca_qf_barycenter(Q_aug_hca, idx)
  } else {
    Z_hca <- switch(
      hca_latent,
      z_rot = do.call(rbind, z_rot_ctx$payload),
      c     = c_ctx_to_matrix(pipeline, c_ctx)
    )
    d_hca    <- dist(Z_hca)
    grid_hca <- pi_grid_hca
    pipe_hca <- pipeline
    pipe_hca$latent <- hca_latent
    hca_rep_mean <- function(idx) {
      z_bar <- colMeans(Z_hca[idx, , drop = FALSE])
      decode_latent_to_Qi(pipe_hca, list(z_bar), Ji_hca)[[1]]
    }
  }
  N_hca <- nrow(Z_hca)
  K_hca <- ncol(Z_hca)
  stopifnot(nrow(df_cov) == N_hca)

  hc_hca <- hclust(d_hca, method = 'ward.D2')
  ## dist objects are ~N^2/2 doubles; drop once the tree is built
  rm(d_hca)
  gc()
  print(Sys.time() - start)

  png(file.path(dir_hca, 'dendrogram.png'),
      width = 1280, height = 720, pointsize = 14)
  plot(hc_hca, labels = FALSE, hang = -1, xlab = '', sub = '',
       main = str_glue("Ward linkage on {hca_rep_label}"))
  dev.off()

  ## Heatmap Z block: for Qi the natural analogue of the latent columns is the
  ## QF itself, centred on the overall barycenter so that the diverging scale
  ## has a meaningful zero. 500 columns, so label only a few p levels.
  if (hca_is_qi) {
    Z_heat_hca <- sweep(Z_hca, 2, hca_qf_barycenter(Q_aug_hca, seq_len(N_hca)))
    z_labs_hca <- rep('', K_hca)
    p_marks    <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    z_labs_hca[sapply(p_marks, function(p) which.min(abs(p_grid_aug_hca - p)))] <-
      sprintf('p = %g', p_marks)
    ## ~500 Z columns vs a handful of covariates, so fix the split by width
    cov_frac_hca <- 0.3
  } else {
    Z_heat_hca   <- Z_hca
    z_labs_hca   <- NULL
    cov_frac_hca <- NULL   # equal-width columns, as before
  }

  for (hca_k in c(11)) {

    ## One directory per k: the figure names below carry no k suffix,
    ## so the path alone identifies a plot.
    dir_hca_k <- file.path(dir_hca, str_glue('k-{hca_k}'))
    if (!dir.exists(dir_hca_k)) dir.create(dir_hca_k, recursive = TRUE)

    ## Number clusters manually via cluster_relabels, so that ids, and hence
    ## colors, are interpretable.
    relabel_hca <- cluster_relabels[[hca_latent]][[as.character(hca_k)]]
    if (is.null(relabel_hca)) relabel_hca <- seq_len(hca_k)
    cl_hca   <- hca_relabel_manual(cutree(hc_hca, k = hca_k), relabel_hca)
    cols_hca <- hcl.colors(hca_k, palette = 'Dark 3')

    ## rect.hclust() fills `border` left-to-right along hc$order, not by cluster id,
    ## so the palette has to be permuted to keep box colors consistent with every
    ## other plot in this section.
    hca_rect_cols <- function(hc, cl, cols) cols[unique(cl[hc$order])]

    png(file.path(dir_hca_k, 'dendrogram.png'),
        width = 1280, height = 720, pointsize = 14)
    plot(hc_hca, labels = FALSE, hang = -1, xlab = '', sub = '',
        main = str_glue("Ward linkage on {hca_rep_label} (k = {hca_k})"))
    rect.hclust(hc_hca, k = hca_k,
                border = hca_rect_cols(hc_hca, cl_hca, cols_hca))
    dev.off()

    print(table(cluster = cl_hca))


    ## ---------- Clustered Heatmap ---------- ##

    hca_plot_heatmap(
      hc             = hc_hca,
      Z              = Z_heat_hca,
      covs           = df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
      n_clusts       = hca_k,
      cl             = cl_hca,
      cl_cols        = cols_hca,
      cluster_lines  = TRUE,
      label_clusters = TRUE,
      z_labs         = z_labs_hca,
      cov_frac       = cov_frac_hca,
      path           = file.path(dir_hca_k, 'heatmap.png')
    )


    ## ---------- Embedding space ---------- ##

    ## Qi has no embedding coordinates to scatter, so this is the one figure
    ## the Qi representation skips.
    if (!hca_is_qi) {
      k_from <- 1
      k_to <- 3
      png(file.path(dir_hca_k, 'embeddings.png'),
          width = 960, height = 960, pointsize = 18)
      plot_embeddings(
        Z_hca[,k_from:k_to],
        colors = adjustcolor(cols_hca[cl_hca], alpha.f = 0.3),
        shapes = rep(19, N_hca),
        sizes  = rep(0.5, N_hca),
        color_shape_size_labels = data.frame(
          color = cols_hca,
          shape = 19,
          size  = 1,
          label = str_glue('cluster {seq_len(hca_k)}')
        ),
        stats = FALSE,
        ## Small, and in two columns for a large k, so the legend stays in
        ## the corner of the first Q-Q panel clear of its points
        legend_cex  = 0.65,
        legend_ncol = if (hca_k > 12) 2 else 1
      )
      dev.off()
    }


    ## ---------- Distribution space ---------- ##

    ## Cluster QF := decode(mean of the cluster's embeddings) for a latent,
    ## or the cluster's Wasserstein barycenter for Qi.
    Qi_cluster <- lapply(
      seq_len(hca_k),
      function(g) hca_rep_mean(which(cl_hca == g))
    )

    png(file.path(dir_hca_k, 'cluster-qfs.png'),
        width = 960, height = 960, pointsize = 18)
    plot_funs(
      fun_list  = Qi_cluster,
      grid_list = rep(list(grid_hca), hca_k),
      colors    = cols_hca,
      widths    = rep(3, hca_k),
      types     = rep(1, hca_k),
      color_width_type_labels = data.frame(
        color = cols_hca,
        width = 3,
        type  = 1,
        label = str_glue('cluster {seq_len(hca_k)}')
      ),
      main = str_glue("Cluster QFs ({hca_rep_label}, k = {hca_k})")
    )
    dev.off()

    ## Same curves against the root QF -- the representative of the whole
    ## sample -- so each cluster reads as a departure from the overall
    ## distribution.
    Qi_root <- hca_rep_mean(seq_len(N_hca))

    ## Sized as the top row of a LaTeX panel: 2:1 at full width, above the
    ## three box-q*.png figures (2:3 each, a third of the width). All four
    ## share pointsize and res so their text matches once placed.
    Qi_contrast <- lapply(Qi_cluster, function(q) q - Qi_root)

    png(file.path(dir_hca_k, 'cluster-qfs-contrast.png'),
        width = 960 * hca_res / 72, height = 480 * hca_res / 72,
        pointsize = hca_pointsize, res = hca_res)
    par(mar = c(4, 4, 2.5, 3), mgp = c(2.4, 0.7, 0))
    plot_funs(
      fun_list  = Qi_contrast,
      grid_list = rep(list(grid_hca), hca_k),
      colors    = cols_hca,
      widths    = rep(2, hca_k),
      types     = rep(1, hca_k),
      ylab = 'Q(p) Contrast',
      main = str_glue("Cluster QF Contrasts")
    )
    abline(h = 0, lty = 2, col = 'gray50')
    ## The quantile levels boxplotted in box-q*.png
    abline(v = hca_q_levels, lty = 3, col = 'gray30')
    mtext(hca_q_levels, side = 3, at = hca_q_levels, line = 0.1, cex = 0.7,
          col = 'gray30')
    ## Cluster numbers in the right margin in place of a legend, at the height
    ## where each curve ends, spread apart where curves end close together.
    usr   <- par('usr')
    y_end <- vapply(Qi_contrast, function(q) q[length(q)], numeric(1))
    y_lab <- hca_spread_labels(y_end, gap = 1.15 * strheight('0', cex = 0.8))
    x_lab <- usr[2] + 0.02 * diff(usr[1:2])
    segments(max(grid_hca), y_end, x_lab, y_lab, col = cols_hca, xpd = NA)
    text(x_lab, y_lab, seq_len(hca_k), col = cols_hca, pos = 4, offset = 0.2,
         cex = 0.8, font = 2, xpd = NA)
    dev.off()




    ## ---------- Covariate summaries ---------- ##

    tbl_hca_male <- data.frame(cluster = cl_hca, male = df_cov$male) %>%
      group_by(cluster) %>%
      summarise(prop_male = mean(male, na.rm = TRUE)) %>%
      as.data.frame()
    print(tbl_hca_male)
    write.csv(
      tbl_hca_male,
      file.path(dir_hca_k, 'cov-male.csv'),
      row.names = FALSE
    )

    ## Each continuous covariate gets a density overlay and a stack of
    ## cluster-wise boxplots. File stems match the previous explicit calls.
    hca_cov_plots <- c('age', 'pir', 'bmd_adj', 'hdl_adj', 'ghb_adj')
    for (cv in hca_cov_plots) {
      slug <- gsub('_', '-', cv)
      hca_plot_density(
        df_cov[[cv]], cl_hca, cols_hca,
        xlab = cv, main = str_glue("{cv} by cluster (k = {hca_k})"),
        path = file.path(dir_hca_k, str_glue('cov-{slug}.png'))
      )
      hca_plot_boxes(
        df_cov[[cv]], cl_hca, cols_hca,
        xlab = cv, main = str_glue("{cv} by cluster (k = {hca_k})"),
        path = file.path(dir_hca_k, str_glue('box-{slug}.png'))
      )
    }

    ## All covariates in one figure
    hca_plot_cov_summary(
      df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
      cl_hca, cols_hca,
      path = file.path(dir_hca_k, 'cov-summary.png')
    )

    ## Same figure grouped on the manual phenotypes, where this
    ## representation and k have any. Phenotypes keep their listed order and
    ## are not color-coded, since the palette belongs to the clusters.
    pheno_hca <- phenotypes_to_clusters[[hca_latent]][[as.character(hca_k)]]
    if (!is.null(pheno_hca)) {
      ## Every cluster must belong to exactly one phenotype
      stopifnot(setequal(unlist(pheno_hca), seq_len(hca_k)),
                !anyDuplicated(unlist(pheno_hca)))
      ph_hca <- rep(seq_along(pheno_hca), lengths(pheno_hca))[
        match(cl_hca, unlist(pheno_hca))
      ]
      print(table(phenotype = factor(ph_hca, labels = names(pheno_hca))))
      hca_plot_cov_summary(
        df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
        ph_hca, rep('grey80', length(pheno_hca)),
        path      = file.path(dir_hca_k, 'cov-summary-phenotype.png'),
        labels    = names(pheno_hca),
        group_lab = 'phenotype'
      )
    }

    ## Activity quantiles on the original scale. Sized as the bottom row of
    ## the LaTeX panel under cluster-qfs-contrast.png: 2:3, a third as wide.
    for (j in seq_along(hca_q_levels)) {
      q_pct <- round(100 * hca_q_levels[j])
      hca_plot_boxes(
        Q_lvl_hca[, j], cl_hca, cols_hca,
        xlab = 'MIMS',
        main = str_glue("Q({hca_q_levels[j]})"),
        path = file.path(dir_hca_k, str_glue('box-q{q_pct}.png')),
        width = 320, height = 480, pointsize = hca_pointsize, res = hca_res,
        mar = c(4, 4, 2.5, 0.8),
        faint_outliers = TRUE
      )
    }


  }
}

