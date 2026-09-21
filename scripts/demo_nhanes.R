library(biotools)
library(car)
library(dplyr)
library(kableExtra)
library(knitr)
library(MASS)
library(moments)
library(MVN)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')




## ========== Representation Learning ========== ##

## Globals
dir_art <- 'demo_nhanes_v6'

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
pipeline <- fit(pipeline, y_list)
path <- str_glue('artifacts/{dir_art}/pipe_nhanes.rds')
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
z_rot_ctx <- encode(pipeline, z_ctx, from = 4, to = 5)
z_ctx_ <- decode(pipeline, z_rot_ctx, from = 5, to = 4)
c_ctx_ <- decode(pipeline, z_ctx_, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
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
  path          = file.path('artifacts', dir_art, 'plots',
                            sprintf('Qi-recon_grid_3x5%s.png',
                                    if (log_scale) '_log' else ''))
)



## ----- Visualize data in different spaces

## Plotting globals
mark_outliers <- FALSE
include_legend <- FALSE
k_from <- 1
k_to <- 3
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.25), col_train)

## Plot z-rot-embeddings
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
  str_glue('data_c.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
C_plot <- rbind(
  do.call(rbind, c_ctx$payload$c_list)[idx_train,k_from:k_to],
  do.call(rbind, c_ctx$payload$c_list)[idx_outliers,k_from:k_to]
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
  'artifacts', dir_art, 'plots',
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
    'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
path <- file.path('data', 'processed', 'nhanes_cov.rds')
df_cov <- readRDS(path)
df_cov <- df_cov %>%
  filter(sub_id %in% names(y_list)) %>%
  arrange(match(sub_id, names(y_list)))

## Create binary covariates
df_cov$low_income <- as.numeric(df_cov$PIR <= 1)
df_cov$old <- as.numeric(df_cov$age >= 65)
df_cov$male <- sapply(df_cov$gender, function(g) ifelse(g == 'Male', 1, 0))

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

## ---------- Effect Plots ---------- ##

## Create directory
dir_eff_plots <- file.path('artifacts', dir_art, 'plots', 'effect-plots')
if (!dir.exists(dir_eff_plots)) dir.create(dir_eff_plots)

K <- pipeline$stages[[3]]$state$child_qg_pca$state$K
k_star_quantiles <- seq(0.1, 0.9, by = 0.1)
for (k_star in 1:K) {

  ## Set z-embeddings to decode
  Z <- do.call(rbind, z_rot_ctx$payload)
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
  Qi_plot <- decode_z_rot_to_Qi(
    pipeline,
    z_rot_list = asplit(Z_plot, MARGIN = 1),
    Ji         = Ji
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
    dir_eff_plots,
    str_glue('effect_k-{k_star}.png')
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
    main = str_glue("Effect of component k = {k_star}")
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
run_generativity_diagnostics <- TRUE

## Covariate names for plotting
gp1_name <- get_plot_name_for_cov(bin_pred, FALSE)
gp0_name <- get_plot_name_for_cov(bin_pred, TRUE)

## Create directory
dir_gc <- file.path('artifacts', dir_art, 'plots', 'group-comparison', bin_pred)
if (!dir.exists(dir_gc)) dir.create(dir_gc)


## ----- Fit linear regression

## Fit linear model
mask <- !is.na(df_cov[bin_pred])
Z <- do.call(rbind, z_rot_ctx$payload[mask])
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
    path        = file.path(dir_gc, str_glue('gen_{bin_pred}.png'))
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
png(path_plot, width = 600, height = 800, pointsize = 14)
binary_plot_intervals(
  res_rb, 
  xlim = c(-1.2, 1.2), 
  main = str_glue("Joint Bands for Latent Regression Coefficients ({gp1_name})")
)
dev.off()

## MIMS activity-intensity thresholds (per-minute units) for CDF plots.
mims_thresholds <- list(
  list(
    label = "light",
    value = 15.05,
    col = 'darkblue'
  ),
  list(
    label = "mvpa",
    value = 19.61,
    col = 'darkorange'
  ),
  list(
    label = "extremely vigorous",
    value = 40.00,
    col = 'darkgreen'
  )
)

## Default quantile levels for QF plots (parallel to mims_thresholds for CDF plots).
quantile_level_labels_fun <- function(q_levels) {
  sprintf("p = %g", q_levels)
}
quantile_levels <- list(
  list(
    label = quantile_level_labels_fun(0.75),
    value = 0.75,
    col = 'darkblue'
  ),
  list(
    label = quantile_level_labels_fun(0.99),
    value = 0.99,
    col = 'darkorange'
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
Qi_hat_0  <- decode_z_rot_to_Qi(pipeline, list(z_hat_0), Ji)[[1]]
Qi_hat_1  <- decode_z_rot_to_Qi(pipeline, list(z_hat_1), Ji)[[1]]
Qi_boot_0 <- decode_z_rot_to_Qi(pipeline, z_boot_0, Ji)
Qi_boot_1 <- decode_z_rot_to_Qi(pipeline, z_boot_1, Ji)


## --- Plot conditional QFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_cond-qfs_{bin_pred}.png')
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
  str_glue('gc_cond-qfs-diff_{bin_pred}.png')
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
  str_glue('gc_cond-cdfs_{bin_pred}.png')
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
  str_glue('gc_cond-cdfs-diff_{bin_pred}.png')
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
  str_glue('gc_cond-survivals_{bin_pred}.png')
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
  str_glue('gc_cond-survivals-diff_{bin_pred}.png')
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
  str_glue('gc_cond-thresholds_{bin_pred}.png')
)
png(path_plot, width = 960, height = 360, pointsize = 12)
plot_decoded_conditional_thresholds(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  thresholds   = mims_thresholds,
  eval_type    = c('cdf', 'survival', 'survival'),
  panel_mains  = c('Time in sedentary', 'Time above MVPA threshold', 'Time above very vigorous threshold'),
  panel_ylabs  = c(
    str_glue("F(y = {mims_thresholds[[1]]$value})"),
    str_glue("1 - F(y = {mims_thresholds[[2]]$value})"),
    str_glue("1 - F(y = {mims_thresholds[[3]]$value})")
  ),
  alpha        = alpha,
  main         = "Decoded Conditional Threshold Probabilities",
  group_labels = c(gp0_name, gp1_name)
)
dev.off()


## --- Plot threshold constrasts with bands (2 panels, one per threshold)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_cond-thresholds-diff_{bin_pred}.png')
)
png(path_plot, width = 960, height = 360, pointsize = 12)
plot_decoded_conditional_threshold_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  thresholds  = mims_thresholds,
  eval_type   = c('cdf', 'survival', 'survival'),
  panel_mains = c('Time in sedentary', 'Time above MVPA threshold', 'Time above very vigorous threshold'),
  panel_ylabs = c(
    str_glue("F(y = {mims_thresholds[[1]]$value}) Contrast"),
    str_glue("1 - F(y = {mims_thresholds[[2]]$value}) Contrast"),
    str_glue("1 - F(y = {mims_thresholds[[3]]$value}) Contrast")
  ),
  alpha       = alpha,
  main        = "Decoded Conditional Threshold Probability Contrast",
  diff_label  = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()




## --- Plot moments with bands (4 panels, one per moment)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_cond-moments_{bin_pred}.png')
)
png(path_plot, width = 1280, height = 360, pointsize = 14)
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
  str_glue('gc_cond-moments-diff_{bin_pred}.png')
)
png(path_plot, width = 1280, height = 360, pointsize = 14)
plot_decoded_conditional_moment_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = "Decoded Conditional Moment Contrasts",
  diff_label = str_glue("{gp1_name} - {gp0_name}")
)
dev.off()




## ========== Additive Mixed Model ========== ##


## ---------- Fit ----------

Z_amm <- do.call(rbind, z_rot_ctx$payload)
covariates_amm <- data.frame(
  male = df_cov$male,
  age  = df_cov$age,
  pir  = df_cov$PIR
)
start <- Sys.time()
amm <- fit_amm(Z_amm, covariates_amm, n_knots_age = 10, n_knots_pir = 10)
stop <- Sys.time()
print(stop - start)


## ---------- Diagnostics ----------

K_amm <- ncol(Z_amm)

## (1) Per-column summary table
summary_tbl <- amm_summary(amm)
print(summary_tbl)
write.csv(
  summary_tbl,
  file.path('artifacts', dir_art, 'amm_summary.csv'),
  row.names = FALSE
)

## (2) Smoothing-parameter plot
amm_plot_lambda(amm, file.path('artifacts', dir_art, 'plots', 'amm_lambda.png'))

## (3) Residual Q-Q panels
amm_plot_resid_qq(amm, file.path('artifacts', dir_art, 'plots', 'amm_resid_qq.pdf'),
                  col_train)

## (4) Residual covariance heatmap (Sigma_E)
amm_plot_sigma_e(amm, file.path('artifacts', dir_art, 'plots', 'amm_sigma_e.png'))

## (5) Random-effect covariance heatmaps
amm_plot_re_cov(
  amm$U_age_hat, K_amm,
  expression(hat(Sigma)[U]^"(age)"),
  file.path('artifacts', dir_art, 'plots', 'amm_sigma_u_age.png')
)
amm_plot_re_cov(
  amm$U_pir_hat, K_amm,
  expression(hat(Sigma)[U]^"(pir)"),
  file.path('artifacts', dir_art, 'plots', 'amm_sigma_u_pir.png')
)


## ========== AMM Inference ========== ##

## ---------- Driver ----------

## Reduced fits for CR and bootstrap smooth tests
amm_red_age <- fit_amm_reduced(Z_amm, covariates_amm, drop = "age",
                                n_knots_age = 10, n_knots_pir = 10)
amm_red_pir <- fit_amm_reduced(Z_amm, covariates_amm, drop = "pir",
                                n_knots_age = 10, n_knots_pir = 10)

## Hotelling test for linear effects
hot_male <- amm_test_linear(amm, "male")
hot_age  <- amm_test_linear(amm, "age")
hot_pir  <- amm_test_linear(amm, "pir")

## Crainiceanu-Ruppert exact RLRT for smooth effects
cr_age <- amm_test_smooth(amm, amm_red_age,
                           covariates_amm, "age",
                           B = 10000, seed_base = 12345)
cr_pir <- amm_test_smooth(amm, amm_red_pir,
                           covariates_amm, "pir",
                           B = 10000, seed_base = 12345)

## Roy-Bose simultaneous CIs for linear effects
ci_male <- amm_intervals_linear(amm, "male", alpha = 0.05)
ci_age  <- amm_intervals_linear(amm, "age",  alpha = 0.05)
ci_pir  <- amm_intervals_linear(amm, "pir",  alpha = 0.05)

## Parametric bootstrap (ISS and LRT; VC-only null; for validation)
grid_age <- seq(min(covariates_amm$age, na.rm = TRUE),
                max(covariates_amm$age, na.rm = TRUE), length.out = 200)
grid_pir <- seq(min(covariates_amm$pir, na.rm = TRUE),
                max(covariates_amm$pir, na.rm = TRUE), length.out = 200)
B_amm <- 20  # development; raise for final
set.seed(12345)
boot_age_iss <- amm_boot_test(amm, amm_red_age, covariates_amm, "age",
                               grid = grid_age, statistic = "iss",
                               null_type = "vc_only", B = B_amm, n_cores = 5)
boot_age_lrt <- amm_boot_test(amm, amm_red_age, covariates_amm, "age",
                               statistic = "lrt", null_type = "vc_only", B = B_amm, n_cores = 5)
boot_pir_iss <- amm_boot_test(amm, amm_red_pir, covariates_amm, "pir",
                               grid = grid_pir, statistic = "iss",
                               null_type = "vc_only", B = B_amm, n_cores = 5)
boot_pir_lrt <- amm_boot_test(amm, amm_red_pir, covariates_amm, "pir",
                               statistic = "lrt", null_type = "vc_only", B = B_amm, n_cores = 5)
saveRDS(
  list(
    boot_age_iss = boot_age_iss, boot_age_lrt = boot_age_lrt,
    boot_pir_iss = boot_pir_iss, boot_pir_lrt = boot_pir_lrt
  ),
  file.path('artifacts', dir_art, 'amm_bootstraps.rds')
)

## Read Bootstrap
tmp <- readRDS(file.path('artifacts', dir_art, 'amm_bootstraps.rds'))
boot_age_iss <- tmp[['boot_age_iss']]
boot_age_lrt <- tmp[['boot_age_lrt']]
boot_pir_iss <- tmp[['boot_pir_iss']]
boot_pir_lrt <- tmp[['boot_pir_lrt']]


## ---------- Inference summary ----------

inference_tbl <- data.frame(
  covariate = c(
    "male", "age (linear)", "pir (linear)",
    "age (smooth)", "age (smooth)", "age (smooth)",
    "pir (smooth)", "pir (smooth)", "pir (smooth)"
  ),
  test      = c(
    "Hotelling", "Hotelling", "Hotelling",
    "Crainiceanu-Ruppert (Bonferroni)",
    "Parametric boot (ISS)", "Parametric boot (LRT)",
    "Crainiceanu-Ruppert (Bonferroni)",
    "Parametric boot (ISS)", "Parametric boot (LRT)"
  ),
  statistic = c(
    hot_male$F_stat, hot_age$F_stat, hot_pir$F_stat,
    min(cr_age$p_per_k) * cr_age$K,
    boot_age_iss$T_obs, boot_age_lrt$T_obs,
    min(cr_pir$p_per_k) * cr_pir$K,
    boot_pir_iss$T_obs, boot_pir_lrt$T_obs
  ),
  p_value   = c(
    hot_male$p_value, hot_age$p_value, hot_pir$p_value,
    cr_age$p_Bonf,
    boot_age_iss$p_value, boot_age_lrt$p_value,
    cr_pir$p_Bonf,
    boot_pir_iss$p_value, boot_pir_lrt$p_value
  )
)
print(inference_tbl)
write.csv(
  inference_tbl,
  file.path('artifacts', dir_art, 'amm_inference_summary.csv'),
  row.names = FALSE
)

## LaTeX Table
p_val_to_text <- function(p_val) {
  if (p_val < 0.001) {
    return("<.001")
  } else {
    return(sub("^0", "", format(round(p_val, 3), nsmall = 2)))
  }
}
inf_tbl_latex <- data.frame(
  "Effect Type" = c("Linear", "Smooth"),
  "Sex" = c(p_val_to_text(hot_male$p_value), "{---}"),
  "Age" = c(p_val_to_text(hot_age$p_value), p_val_to_text(cr_age$p_Bonf)),
  "Income" = c(p_val_to_text(hot_pir$p_value), p_val_to_text(cr_pir$p_Bonf))
)
path <- file.path(
  'artifacts', dir_art, 'plots', 
  str_glue('amm_inf_latex.txt')
)
sink(path)
kbl(inf_tbl_latex, 
    format = "latex", 
    booktabs = TRUE, 
    escape = FALSE,
    align = c("l", "S", "S", "S"),
    caption = "Inference for Semiparametric Regression",
    col.names = c("", "{Sex}", "{Age}", "{Income}")) %>%
  kable_styling(latex_options = c("HOLD_position"))
sink()


## ---------- Diagnostic plots ----------

amm_plot_boot_hist(boot_age_iss, "Bootstrap (age, ISS, VC-only)",
                   file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_age_iss.png'))
amm_plot_boot_hist(boot_age_lrt, "Bootstrap (age, LRT, VC-only)",
                   file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_age_lrt.png'))
amm_plot_boot_hist(boot_pir_iss, "Bootstrap (pir, ISS, VC-only)",
                   file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_pir_iss.png'))
amm_plot_boot_hist(boot_pir_lrt, "Bootstrap (pir, LRT, VC-only)",
                   file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_pir_lrt.png'))

amm_plot_cr_per_k(cr_age, "Crainiceanu-Ruppert per-column null (age)",
                  file.path('artifacts', dir_art, 'plots', 'amm_cr_per_k_age.png'))
amm_plot_cr_per_k(cr_pir, "Crainiceanu-Ruppert per-column null (pir)",
                  file.path('artifacts', dir_art, 'plots', 'amm_cr_per_k_pir.png'))
amm_plot_cr_joint(cr_age, "Crainiceanu-Ruppert joint null (age)",
                  file.path('artifacts', dir_art, 'plots', 'amm_cr_joint_age.png'))
amm_plot_cr_joint(cr_pir, "Crainiceanu-Ruppert joint null (pir)",
                  file.path('artifacts', dir_art, 'plots', 'amm_cr_joint_pir.png'))



## ========== AMM Interpretation ========== ##

## ---------- Driver ----------

Ji_interp <- length(Qi_ctx$payload[[1]])
B_interp  <- 20

amm_boots <- amm_boot_interpret(
  amm,
  covariates_kept = covariates_amm[amm$keep, , drop = FALSE],
  B               = B_interp,
  n_cores         = 5,
  seed            = 12345
)

interp_dir <- file.path('artifacts', dir_art, 'plots')

## Per-covariate quantile levels for QF crossings / markings.
quantile_levels_by_cov <- list(
  age = list(
    list(label = quantile_level_labels_fun(0.75), value = 0.75,  col = 'darkblue'),
    list(label = quantile_level_labels_fun(0.99), value = 0.99,  col = 'darkorange')
  ),
  pir = list(
    list(label = quantile_level_labels_fun(0.75),  value = 0.75,  col = 'darkblue'),
    list(label = quantile_level_labels_fun(0.999), value = 0.999, col = 'darkorange')
  )
)

## --- Latent-space effect curves
for (cov in c("age", "pir")) {
  path_plot <- file.path(interp_dir, str_glue("amm_latent_effect_{cov}.png"))
  png(path_plot, width = 1280, height = 960, pointsize = 14)
  amm_plot_latent_effects(amm, cov, band = TRUE,
                          boot_reps = amm_boots,
                          covariates_df = covariates_amm)
  dev.off()
}

## --- Decoded plots
for (cov in c("age", "pir")) {
  q_list <- quantile_levels_by_cov[[cov]]

  ## Reference QF / CDF (decoded at male = 0, age = 40, pir = 1)
  cv_ref_decode <- list(male = 0, age = 40, pir = 1)
  Qi_ref_decode <- .decode_z_vec(pipeline,
    amm_compute_conditional_mean(amm, cv_ref_decode), Ji_interp)
  pi_grid_ref <- pi_grid_fun(Ji_interp)

  ## QF Reference
  png(file.path(interp_dir, str_glue("amm_qf_reference_{cov}.png")),
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

  ## CDF Reference
  png(file.path(interp_dir, str_glue("amm_cdf_reference_{cov}.png")),
      width = 960, height = 960, pointsize = 24)
  y_grid_ref <- .amm_default_y_grid(list(Qi_ref_decode))
  F_ref_decode <- qi_to_cdf(Qi_ref_decode, y_grid_ref)
  plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)", main = "Reference CDF")
  lines(y_grid_ref, F_ref_decode, col = "gray60", lwd = 4)
  dev.off()

  ## QF family
  png(file.path(interp_dir, str_glue("amm_qf_conditional_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  amm_plot_conditional_qf(amm, cov, pipeline, Ji = Ji_interp,
                          quantile_lines = q_list)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_conditional_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  amm_plot_conditional_qf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                  boot_reps = amm_boots,
                                  quantile_lines = q_list)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_contrast_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  amm_plot_contrast_qf(amm, cov, pipeline, Ji = Ji_interp,
                       quantile_lines = q_list)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_contrast_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  amm_plot_contrast_qf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                               boot_reps = amm_boots,
                               quantile_lines = q_list)
  dev.off()

  ## CDF family
  png(file.path(interp_dir, str_glue("amm_cdf_conditional_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  amm_plot_conditional_cdf(amm, cov, pipeline, Ji = Ji_interp,
                           threshold_lines = mims_thresholds)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_conditional_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  amm_plot_conditional_cdf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                   threshold_lines = mims_thresholds,
                                   boot_reps = amm_boots)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_contrast_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  amm_plot_contrast_cdf(amm, cov, pipeline, Ji = Ji_interp,
                        threshold_lines = mims_thresholds)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_contrast_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  amm_plot_contrast_cdf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                threshold_lines = mims_thresholds,
                                boot_reps = amm_boots)
  dev.off()

  ## Moments family
  png(file.path(interp_dir, str_glue("amm_moments_conditional_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  amm_plot_conditional_moments(amm, cov, pipeline, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_conditional_faceted_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  amm_plot_conditional_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                       boot_reps = amm_boots)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_contrast_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  amm_plot_contrast_moments(amm, cov, pipeline, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_contrast_faceted_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  amm_plot_contrast_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                    boot_reps = amm_boots)
  dev.off()

  ## Threshold-crossing (one per MIMS threshold; absolute + contrast)
  for (thr in mims_thresholds) {
    thr_slug <- gsub(" / ", "_", thr$label)
    thr_y    <- thr$value
    png(file.path(interp_dir,
                  str_glue("amm_threshold_{thr_slug}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    amm_plot_threshold_crossing(amm, cov, thr_y, pipeline, Ji = Ji_interp,
                                boot_reps = amm_boots)
    dev.off()
    png(file.path(interp_dir,
                  str_glue("amm_threshold_contrast_{thr_slug}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    amm_plot_contrast_threshold_crossing(amm, cov, thr_y, pipeline,
                                         Ji = Ji_interp,
                                         boot_reps = amm_boots)
    dev.off()
  }

  ## Quantile-crossing (one per quantile level; absolute + contrast)
  for (ql in 1:length(q_list)) {
    p_star <- q_list[[ql]]$value
    png(file.path(interp_dir,
                  str_glue("amm_quantile_q{ql}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    amm_plot_quantile_crossing(amm, cov, p_star, pipeline, Ji = Ji_interp,
                               boot_reps = amm_boots)
    dev.off()
    png(file.path(interp_dir,
                  str_glue("amm_quantile_contrast_q{ql}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    amm_plot_contrast_quantile_crossing(amm, cov, p_star, pipeline,
                                        Ji = Ji_interp,
                                        boot_reps = amm_boots)
    dev.off()
  }
}


## --- Male (binary) effect: same six-panel two-group interpretation as the
## group-comparison section, but conditioning on AMM-fitted ref values for
## age and pir and using the AMM parametric-bootstrap replicates for bands.

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

## QF and QF difference
png(file.path(interp_dir, "amm_male_cond_qfs.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Means ({main_suffix})")
)
dev.off()

png(file.path(interp_dir, "amm_male_cond_qfs_diff.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Mean Difference ({main_suffix})")
)
dev.off()

## Shared y-grid for the CDF pair
y_grid_male <- default_y_grid_two_group(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1
)

## CDF and CDF difference
png(file.path(interp_dir, "amm_male_cond_cdfs.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional CDFs ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds
)
dev.off()

png(file.path(interp_dir, "amm_male_cond_cdfs_diff.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  main       = str_glue("AMM Decoded Conditional CDF Difference ({main_suffix})"),
  y_grid     = y_grid_male,
  thresholds = mims_thresholds
)
dev.off()

## Moments and moment contrast
png(file.path(interp_dir, "amm_male_cond_moments.png"),
    width = 1280, height = 360, pointsize = 14)
plot_decoded_conditional_moments(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Moments ({main_suffix})")
)
dev.off()

png(file.path(interp_dir, "amm_male_cond_moments_diff.png"),
    width = 1280, height = 360, pointsize = 14)
plot_decoded_conditional_moment_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha = amm_male_alpha,
  main  = str_glue("AMM Decoded Conditional Moment Contrasts ({main_suffix})")
)
dev.off()


