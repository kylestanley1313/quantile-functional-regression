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



## ========== Helpers ========== ##

decode_z_to_Qi <- function(z_list, Ji) {
  z_ctx <- new_context(
    payload = z_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_list)))
  )
  decode(pipeline, z_ctx, from = 6, to = 2)$payload
}

decode_z_rot_to_Qi <- function(z_rot_list, Ji) {
  z_rot_ctx <- new_context(
    payload = z_rot_list,
    cache = pipeline$training$cache,
    meta = list(Ji_vec = rep(Ji, length.out = length(z_rot_list)))
  )
  decode(pipeline, z_rot_ctx, from = 7, to = 2)$payload
}


## ========== Representation Learning ========== ##

## Globals
dir_art <- 'demo_nhanes'

## Load data
path <- file.path('data', 'processed', 'nhanes_v1_nofilter.rds')
y_list <- readRDS(path)
# y_list <- y_list[1:500]
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
y_max <- max(unlist(y_list))
path_plot_tmp = "scratch/plots/tmp_plot.png"

## Define grid
p_grid <- p_grid_fun_2(
  breaks = c(1/(Ji_min + 1), 0.95, Ji_min/(Ji_min + 1)),
  interval_counts = c(51, 50)
)

## NOTE: Unfiltered NHANES data contains 10080 observations for each subject. This
## means extrapolation isn't necessary. 

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_y_axis(y_trans = 'identity', y_shift = 0),
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100), 
    stage_lqd(),
    stage_qg_pca(
      K_max = 20,
      epsilon = 1.25, # 0.25,
      alpha = 0.05,
      V = 5,
      quantlet_construction = 'pca',
      lambda = 0.1
    ),
    stage_flow(
      n_layers = 16,
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
  loss = 'wasserstein',
  loss_scale = 'none',
  loss_scale_samp_rate = 1.0,
  p_scale = NULL, # 0.025
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
Ty_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Qi_ctx <- encode(pipeline, Ty_ctx, from = 1, to = 2)
Q_ctx <- encode(pipeline, Qi_ctx, from = 2, to = 3)
G_Q_star_ctx <- encode(pipeline, Q_ctx, from = 3, to = 4)
c_ctx <- encode(pipeline, G_Q_star_ctx, from = 4, to = 5)
z_ctx <- encode(pipeline, c_ctx, from = 5, to = 6)
z_rot_ctx <- encode(pipeline, z_ctx, from = 6, to = 7)
z_ctx_ <- decode(pipeline, z_rot_ctx, from = 7, to = 6)
c_ctx_ <- decode(pipeline, z_ctx_, from = 6, to = 5)
G_Q_star_ctx_ <- decode(pipeline, c_ctx_, from = 5, to = 4)
Q_ctx_ <- decode(pipeline, G_Q_star_ctx_, from = 4, to = 3)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 3, to = 2)
Ty_ctx_ <- decode(pipeline, Qi_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Ty_ctx_, from = 1, to = 0)


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
    'artifacts', dir_art, 'plots', 
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
    ylim = c(0, max(c(Qi, Qi_recon))),
    main = str_glue("{100*p}th Percentile Validation Loss")
  )
  lines(pi_grid, Qi_recon, col = rgb(0, 1, 0, alpha = 1.0))
  dev.off()
}


## ----- Grid of QF reconstructions stratified by loss quantile

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
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- rgb(1, 0.8, 0.5, alpha = 0.25)


## Plot z-rot-embeddings
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('data_z-rot.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Z_plot <- rbind(
  do.call(rbind, z_rot_ctx$payload)[-idx_outliers,1:3],
  do.call(rbind, z_rot_ctx$payload)[idx_outliers,1:3]
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

## Plot z-embeddings
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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
  'artifacts', dir_art, 'plots',
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


## Plot Qi
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('data_qi.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
fun_list <- c(
  Qi_ctx$payload[-idx_outliers],
  Qi_ctx$payload[idx_outliers]
)
grid_list <- c(
  lapply(Ji_vec[-idx_outliers], pi_grid_fun), 
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


## ----- Pairwise-distance scatter plots: Wasserstein (Qi) vs LQD-L2 (G) and Z-Euclidean

N_samp <- 200
set.seed(12345)
idx_samp <- sample(N, N_samp)

## Augmented p-grid + quadrature weights on the native p-grid
p_grid     <- pipeline$training$cache$p_grid
J_aug      <- 500
p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
w_p_grid   <- get_quadrature_weights(p_grid)
supp_TY    <- pipeline$training$cache$supp_TY

## Pairwise Wasserstein distances on Qi (existing utility handles the
## per-subject pi-grid -> p_grid_aug placement and weighted L2 integration).
Qi_samp      <- Qi_ctx$payload[idx_samp]
pi_grid_list <- lapply(Qi_samp, function(Qi) pi_grid_fun(length(Qi)))
d_wass <- pairwise_distance(
  Qi_list      = Qi_samp,
  loss_fun     = wasserstein,
  pi_grid_list = pi_grid_list,
  p_grid_aug   = p_grid_aug,
  supp_TY      = supp_TY
)

## Pairwise L2 in LQD space and Euclidean in Z-space, looped in the SAME
## (i, j>i) order pairwise_distance uses so d_wass[k] / d_lqd[k] / d_z[k]
## refer to the same pair. wasserstein() is reused here as the
## quadrature-weighted L2 norm on the native p_grid.
G_samp <- G_Q_star_ctx$payload$G_list[idx_samp]
C_samp <- do.call(rbind, c_ctx$payload$c_list[idx_samp])
Z_samp <- do.call(rbind, z_ctx$payload[idx_samp])
n_pair <- N_samp * (N_samp - 1) / 2
d_lqd  <- numeric(n_pair)
d_c    <- numeric(n_pair)
d_z    <- numeric(n_pair)
k <- 1
for (i in 1:(N_samp - 1)) {
  for (j in (i + 1):N_samp) {
    d_lqd[k] <- wasserstein(G_samp[[i]], G_samp[[j]], w_p_grid)
    d_c[k]   <- sqrt(sum((C_samp[i, ] - C_samp[j, ])^2))
    d_z[k]   <- sqrt(sum((Z_samp[i, ] - Z_samp[j, ])^2))
    k <- k + 1
  }
}

fmt_cors <- function(x, y) {
  c(
    sprintf("Pearson:  %.3f", cor(x, y, method = "pearson")),
    sprintf("Spearman: %.3f", cor(x, y, method = "spearman")),
    sprintf("Kendall:  %.3f", cor(x, y, method = "kendall"))
  )
}

png(file.path('artifacts', dir_art, 'plots',
              'pairwise_wass_vs_lqd.png'),
    width = 720, height = 720, pointsize = 14)
plot(d_wass, d_lqd,
     xlab = "pairwise Wasserstein (Qi)",
     ylab = "pairwise L2 (G, LQD space)",
     pch  = 19, col = rgb(0, 0, 0, 0.25),
     main = str_glue("Pairwise distances: Wass vs LQD (N_samp = {N_samp})"))
legend("topleft", legend = fmt_cors(d_wass, d_lqd), bty = "n")
dev.off()

png(file.path('artifacts', dir_art, 'plots',
              'pairwise_wass_vs_c.png'),
    width = 720, height = 720, pointsize = 14)
plot(d_wass, d_c,
     xlab = "pairwise Wasserstein (Qi)",
     ylab = "pairwise Euclidean (C)",
     pch  = 19, col = rgb(0, 0, 0, 0.25),
     main = str_glue("Pairwise distances: Wass vs C (N_samp = {N_samp})"))
legend("topleft", legend = fmt_cors(d_wass, d_c), bty = "n")
dev.off()

png(file.path('artifacts', dir_art, 'plots',
              'pairwise_wass_vs_z.png'),
    width = 720, height = 720, pointsize = 14)
plot(d_wass, d_z,
     xlab = "pairwise Wasserstein (Qi)",
     ylab = "pairwise Euclidean (Z)",
     pch  = 19, col = rgb(0, 0, 0, 0.25),
     main = str_glue("Pairwise distances: Wass vs Z (N_samp = {N_samp})"))
legend("topleft", legend = fmt_cors(d_wass, d_z), bty = "n")
dev.off()


## ----- Visualize synthetic

## Globals
K  <- pipeline$stages[[5]]$state$K
Ji <- 10080

## Train/val split (same convention as assess_generativity_split): fit the
## latent mean-cov model on the train half, then draw |val| synthetic z's
## and compare against the held-out real Qi.
set.seed(12345)
frac_train <- 0.5
n_train    <- floor(frac_train * N)
perm       <- sample(N)
idx_tr     <- perm[seq_len(n_train)]
idx_val    <- perm[(n_train + 1):N]

fit     <- fit_mean_cov(encode_to_Z(pipeline, y_list[idx_tr]), ridge = 0)
z_draws <- draw_mean_cov(length(idx_val), fit)
Qi_draws <- decode_z_draws(z_draws, pipeline, Ji = Ji)$Qi

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

## Set binary predictor for plotting
bin_pred <- 'old'

## Plot: QF
set.seed(12345)
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('eda_Qi_{bin_pred}.png')
)
png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
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
  width = c(1, 1),
  type  = c(1, 1),
  label = c('0', '1')
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
  'artifacts', dir_art, 'plots',
  str_glue('eda_z-rot_{bin_pred}.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
col_0 <- rgb(0, 0, 0, alpha = 0.25)
col_1 <- rgb(1, 0, 0, alpha = 0.25)
mask <- !is.na(df_cov[bin_pred])
idx <- sample(1:sum(mask))
Z_plot <- do.call(rbind, z_rot_ctx$payload)[mask,1:3]
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
  label = c('0', '1')
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




## ========== Group Comparison ========== ##


## ---------- Helpers ---------- ##

get_pointwise_bands <- function(Qi_list, level = 0.99) {
  Qi_mat <- do.call(rbind, Qi_list)
  alpha  <- 1 - level
  list(
    mean  = colMeans(Qi_mat),
    lower = apply(Qi_mat, 2, quantile, probs = alpha / 2),
    upper = apply(Qi_mat, 2, quantile, probs = 1 - alpha / 2)
  )
}

get_joint_credible_bands <- function(betas, alpha = 0.05) {
  
  S <- nrow(betas)
  K <- ncol(betas)
  
  # Posterior mean and standard deviations
  beta_hat <- colMeans(betas)
  beta_sd  <- apply(betas, 2, sd)
  
  # Standardized deviations from posterior mean: S-by-K matrix
  a <- sweep(betas, 2, beta_hat, "-")
  a <- sweep(a,     2, beta_sd,  "/")
  
  # Row-wise maximum absolute standardized deviation
  a_max <- apply(abs(a), 1, max)
  
  # (1 - alpha) empirical quantile of a_max
  a_crit <- quantile(a_max, probs = 1 - alpha)
  
  # Joint credible intervals
  lower <- beta_hat - a_crit * beta_sd
  upper <- beta_hat + a_crit * beta_sd
  
  list(
    beta_hat = beta_hat,
    beta_sd  = beta_sd,
    a_crit   = a_crit,
    lower    = lower,
    upper    = upper
  )
}

simbas <- function(betas) {
  S <- nrow(betas)
  beta_hat <- colMeans(betas)
  beta_sd  <- apply(betas, 2, sd)
  a <- sweep(betas, 2, beta_hat, "-")
  a <- sweep(a,     2, beta_sd,  "/")
  a_max <- apply(abs(a), 1, max)
  
  # For each k, find smallest alpha such that |beta_hat_k / beta_sd_k| > a_crit(alpha)
  # Equivalently: proportion of a_max exceeding |beta_hat_k / beta_sd_k|
  t_obs <- abs(beta_hat / beta_sd)
  sapply(t_obs, function(t) mean(a_max > t))
}



## ---------- Effect Plots ---------- ##

K <- pipeline$stages[[5]]$state$K
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
    'artifacts', dir_art, 'plots',
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
bin_pred <- 'old'
run_diagnostics <- FALSE


## ----- Fit linear regression

## Fit linear model
mask <- !is.na(df_cov[bin_pred])
Z <- do.call(rbind, z_rot_ctx$payload[mask])
N <- nrow(Z)
K <- ncol(Z)
x <- df_cov[mask,bin_pred]
model <- lm(Z ~ x)


## ----- Model diagnostics

if (run_diagnostics) {

  ## Setup
  resid <- residuals(lm(Z ~ x))

  ## Check for marginal normality via Q-Q plots
  nc <- ceiling(sqrt(K))
  nr <- ceiling(K / nc)
  path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-marg-qq.png')
  png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k in seq_len(K)) {
    qqnorm(resid[, k], main = paste0("k = ", k), pch = 19, cex = 0.25, col = col_train)
    qqline(resid[, k], col = "red")
  }
  dev.off()
  par(mfrow = c(1, 1))
  ## NOTE: Looks pretty good

  ## Check for marginal normality via histograms
  nc <- ceiling(sqrt(K))
  nr <- ceiling(K / nc)
  path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-marg-hist.png')
  png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k in seq_len(K)) {
    hist(resid[,k], main = paste0("k = ", k))
  }
  dev.off()
  par(mfrow = c(1, 1))
  ## NOTE: Definitely looks good

  ## Check for multivariate normality via pairwise scatterplots
  n_plots <- choose(K, 2)
  nc <- ceiling(sqrt(n_plots))
  nr <- ceiling(n_plots / nc)
  path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-mv-pw-plots.png')
  png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k1 in 1:(K-1)) {
    for (k2 in (k1+1):K) {
      plot(
        resid[,k1], resid[,k2], 
        main = str_glue("k = {k1} vs. k = {k2}"),
        pch = 19, cex = 0.25, col = col_train)
    }
  }
  dev.off()
  par(mfrow = c(1, 1))
  ## NOTE: Looks good.

  ## Check for both univariate and multivariate normality via formal checks
  result <- mvn(resid, mvn_test = "mardia", univariate_test = "AD")
  print(result$multivariate_normality)
  print(result$univariate_normality)
  ## NOTES: Formal tests reveal departure from MVN. Large sample hypersensitivity.

  ## Check for multivariate normality via squared Mahalanobis distances
  # cov_resid  <- cov(resid)
  # mean_resid <- colMeans(resid)
  # d2 <- mahalanobis(resid, center = mean_resid, cov = cov_resid)
  # chi2_q <- qchisq(ppoints(N), df = K)
  # par(mar = c(4, 4, 2, 1))
  # plot(
  #   chi2_q, sort(d2),
  #   xlab = expression(chi[K]^2 ~ "quantiles"),
  #   ylab = "Sorted squared Mahalanobis distances",
  #   main = "Multivariate Normality: Chi-squared Q-Q Plot",
  #   pch = 19, cex = 0.5, 
  #   col = col_train
  # )
  # abline(0, 1, col = "red")
  ## NOTE: Mahalanobis plot suggests substantial deviation from MVN.

  ## Check for homescedasticity via covariance heatmaps
  cov0 <- cov(Z[x == 0, ])
  cov1 <- cov(Z[x == 1, ])
  K_dim    <- nrow(cov0)
  zlim     <- range(c(cov0, cov1))
  zmax     <- max(abs(zlim))
  breaks   <- seq(-zmax, zmax, length.out = 101)
  col_pal  <- colorRampPalette(c("#2166ac", "white", "#b2182b"))(100)

  plot_cov_heatmap <- function(M, title) {
    image(
      1:K_dim, 1:K_dim, t(M)[, K_dim:1],
      zlim = c(-zmax, zmax), col = col_pal, breaks = breaks,
      axes = FALSE, xlab = "", ylab = "", main = title
    )
    axis(1, at = 1:K_dim, labels = 1:K_dim, las = 1, tick = FALSE)
    axis(2, at = 1:K_dim, labels = K_dim:1, las = 1, tick = FALSE)
    for (i in 1:K_dim) for (j in 1:K_dim) if (i >= j) {
      text(j, K_dim - i + 1, sprintf("%.2f", M[i, j]), cex = 0.7)
    }
  }

  png(path_plot_tmp, width = 1200, height = 540, pointsize = 14)
  layout(matrix(c(1, 2, 3), nrow = 1), widths = c(4, 4, 1))
  par(mar = c(4, 4, 3, 1))
  plot_cov_heatmap(cov0, "Group 0 covariance")
  plot_cov_heatmap(cov1, "Group 1 covariance")
  par(mar = c(4, 0.5, 3, 4))
  legend_levels <- seq(-zmax, zmax, length.out = 100)
  image(
    1, legend_levels, t(matrix(legend_levels, ncol = 1)),
    col = col_pal, breaks = breaks, axes = FALSE, xlab = "", ylab = ""
  )
  axis(4, las = 1)
  dev.off()
  ## NOTE: Looks good

  ## Check for homoscedasticity via Box's M
  boxm_result <- boxM(Z, x)
  print(boxm_result)
  ## NOTE: Failed. Large sample hypersensitivity.

}



## ----- Hotelling's T-Squared Test and Roy-Bose Intervals

## Hotelling's T
res_manova <- manova(model)
res_hotel <- summary(res_manova, test = "Hotelling-Lawley")
print(res_hotel)

## Roy-Bose Intervals
roy_bose_intervals <- function(model, alpha = 0.05) {
  
  beta_hat <- coef(model)["x", ]
  N <- nrow(model$model)
  K <- length(beta_hat)
  
  n0    <- sum(model$model[, "x"] == 0)
  n1    <- sum(model$model[, "x"] == 1)
  n_eff <- n0 * n1 / (n0 + n1)
  
  S_p <- crossprod(residuals(model)) / (N - 2)
  se  <- sqrt(diag(S_p) / n_eff)
  
  # Component-wise t-statistics
  t_stat <- abs(beta_hat) / se
  
  # Roy-Bose critical value
  F_crit  <- qf(1 - alpha, df1 = K, df2 = N - K - 1)
  rb_crit <- sqrt((N - 2) * K / (N - K - 1) * F_crit)
  
  # Simultaneous intervals
  lower <- beta_hat - rb_crit * se
  upper <- beta_hat + rb_crit * se
  
  # Multiplicity-adjusted p-values via inversion of Roy-Bose critical value
  # p_k = P(F_{K, N-K-1} > (N-K-1)/(K*(N-2)) * t_k^2)
  F_stat  <- (N - K - 1) / (K * (N - 2)) * t_stat^2
  p_adj   <- pf(F_stat, df1 = K, df2 = N - K - 1, lower.tail = FALSE)
  
  data.frame(
    component   = seq_len(K),
    estimate    = beta_hat,
    se          = se,
    lower       = lower,
    upper       = upper,
    t_stat      = t_stat,
    p_adj       = p_adj,
    significant = (lower > 0) | (upper < 0)
  )
}
res_rb <- roy_bose_intervals(model)
print(res_rb)

## Save results
path <- file.path(
  'artifacts', dir_art, 'plots', 
  str_glue('mod_{bin_pred}.txt')
)
sink(path)
print("========== Hotelling ==========")
res_hotel
print("========== Roy-Bose ==========")
res_rb
sink()

## Plot Roy-Bose Intervals

## Visualize the Roy-Bose simultaneous confidence intervals as a horizontal
## forest plot: one row per component k, with the point estimate, CI bar with
## end caps, and a small p_adj-colored box just inside the y-axis. A vertical
## colorbar legend in the top right encodes the multiplicity-adjusted p-value
## with three equal-height segments (0-0.001, 0.001-0.05, 0.05-1).
plot_roy_bose <- function(rb_df, xlim = NULL, main = "") {
  K       <- nrow(rb_df)
  bright_red  <- "#FF0000" # "#FF3333"
  medium_red  <- "#C54E57" # "#B22222"
  dark_red    <- "#847777" # "#4B0000"
  dark_blue   <- "#5E819D" # "#08306B"
  bright_blue <- "#0000FF" # "#4292C6"

  p_to_color <- function(p) {
    sapply(p, function(pp) {
      if (pp <= 0.001) {
        t <- pp / 0.001
        v <- colorRamp(c(bright_red, medium_red))(t)
      } else if (pp <= 0.05) {
        t <- (pp - 0.001) / (0.05 - 0.001)
        v <- colorRamp(c(medium_red, dark_red))(t)
      } else {
        t <- (pp - 0.05) / (1 - 0.05)
        v <- colorRamp(c(dark_blue, bright_blue))(t)
      }
      rgb(v[1], v[2], v[3], maxColorValue = 255)
    })
  }

  if (is.null(xlim)) {
    rng <- range(c(rb_df$lower, rb_df$upper, 0))
    pad <- 0.08 * diff(rng)
    xlim <- c(rng[1] - pad, rng[2] + pad)
  }

  y_pos <- K:1   # k = 1 at top
  old_par <- par(mar = c(4.5, 4, 3, 2))
  on.exit(par(old_par), add = TRUE)

  plot(NULL, xlim = xlim, ylim = c(0.5, K + 0.5),
       xlab = expression(hat(beta)[k]),
       ylab = "k",
       yaxt = "n", main = main)
  axis(2, at = y_pos, labels = seq_len(K), las = 1)

  abline(v = 0, col = "gray80", lwd = 1)

  cap_h <- 0.18
  for (i in seq_len(K)) {
    yi <- y_pos[i]
    segments(rb_df$lower[i], yi, rb_df$upper[i], yi, col = "black", lwd = 1.5)
    segments(rb_df$lower[i], yi - cap_h, rb_df$lower[i], yi + cap_h,
             col = "black", lwd = 1.5)
    segments(rb_df$upper[i], yi - cap_h, rb_df$upper[i], yi + cap_h,
             col = "black", lwd = 1.5)
    points(rb_df$estimate[i], yi, pch = 19, cex = 1.0)
  }

  ## p_adj color boxes just inside the y-axis
  usr <- par("usr")
  xr  <- diff(usr[1:2])
  yr  <- diff(usr[3:4])
  box_w <- 0.022 * xr
  box_h <- 0.35
  box_x0 <- usr[1] + 0.006 * xr
  box_x1 <- box_x0 + box_w
  cols_p <- p_to_color(rb_df$p_adj)
  for (i in seq_len(K)) {
    yi <- y_pos[i]
    rect(box_x0, yi - box_h / 2, box_x1, yi + box_h / 2,
         col = cols_p[i], border = "black", lwd = 0.5)
  }

  ## Vertical colorbar legend (top right of plot area)
  lx0 <- usr[1] + 0.89 * xr
  lx1 <- usr[1] + 0.93 * xr
  ly0 <- usr[3] + 0.75 * yr # 0.55
  ly1 <- usr[3] + 0.92 * yr
  third <- (ly1 - ly0) / 3

  draw_seg <- function(yb, yt, col_lo, col_hi, n = 60) {
    ramp <- colorRamp(c(col_lo, col_hi))
    ys <- seq(yb, yt, length.out = n + 1)
    for (s in seq_len(n)) {
      v <- ramp((s - 0.5) / n)
      rect(lx0, ys[s], lx1, ys[s + 1],
           col = rgb(v[1], v[2], v[3], maxColorValue = 255), border = NA)
    }
  }
  draw_seg(ly0,             ly0 + third,     bright_red,  medium_red)
  draw_seg(ly0 + third,     ly0 + 2 * third, medium_red,  dark_red)
  draw_seg(ly0 + 2 * third, ly1,             dark_blue,   bright_blue)
  rect(lx0, ly0, lx1, ly1, col = NA, border = "black")

  lab_x <- lx1 + 0.008 * xr
  text(lab_x, ly0,             labels = "0",     adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + third,     labels = "0.001", adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + 2 * third, labels = "0.05",  adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly1,             labels = "1",     adj = c(0, 0.5), cex = 0.75)
  text((lx0 + lx1) / 2, ly1 + 0.025 * yr,
       labels = expression(p[adj]), adj = c(0.5, 0))

  invisible(NULL)
}

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('roy-bose_{bin_pred}.png')
)
png(path_plot, width = 600, height = 800, pointsize = 14)
plot_roy_bose(res_rb, xlim = c(-1, 1), main = str_glue("Roy-Bose Intervals ({bin_pred})"))
dev.off()



## ---------- Decoded conditional plot helpers ----------
##
## Reusable plotting primitives for two-group decoded-space interpretation,
## driven by a bootstrap distribution of conditional latent means. Each
## function takes Qi point estimates and Qi bootstrap replicate matrices
## (or lists) for the two groups, computes its own pointwise quantile bands
## internally, and draws to the active device. Caller manages png/dev.off.
##
## Consumed by both the group-comparison section (binary `x` predictor) and
## the AMM interpretation section (binary `male` predictor evaluated at
## fixed reference values of age / pir).

qi_to_cdf <- function(Qi, y_grid) {
  Ji        <- length(Qi)
  Qi_sorted <- sort(Qi)
  pi_local  <- (1:Ji) / (Ji + 1)
  approx(
    x = Qi_sorted, y = pi_local,
    xout = y_grid, rule = 2, ties = "ordered"
  )$y
}

qi_at_p <- function(Qi, p_star) {
  pi_local <- seq_along(Qi) / (length(Qi) + 1)
  approx(
    x = pi_local, y = Qi,
    xout = p_star, rule = 2, ties = "ordered"
  )$y
}

## A Qi is a length-Ji vector at p-grid (1:Ji)/(Ji+1) ~ uniform on (0,1),
## so the k-th moment of the distribution is well approximated by mean(Qi^k).
moments_from_Qi <- function(Qi) {
  mu <- mean(Qi)
  s2 <- mean((Qi - mu)^2)
  sd <- sqrt(s2)
  c(
    mean     = mu,
    variance = s2,
    skewness = mean((Qi - mu)^3) / sd^3,
    kurtosis = mean((Qi - mu)^4) / sd^4
  )
}

## MIMS activity-intensity thresholds (per-minute units) for CDF plots.
mims_thresholds       <- c(sed_light = 15.05, light_mvpa = 19.61)
mims_threshold_cols   <- c(sed_light = "darkblue", light_mvpa = "darkorange")
mims_threshold_labels <- c(
  sed_light  = "sed / light (15.05)",
  light_mvpa = "light / mvpa (19.61)"
)

## Default quantile levels for QF plots (parallel to mims_thresholds for CDF plots).
quantile_levels       <- c(q1 = 0.75, q2 = 0.99)
quantile_level_cols   <- c(q1 = "darkblue", q2 = "darkorange")
quantile_level_labels_fun <- function(q_levels) {
  sprintf("p = %g", q_levels)
}
quantile_level_labels <- quantile_level_labels_fun(quantile_levels)


## Coerce bootstrap input to an R x Ji matrix (accepts list of length-Ji
## vectors or an R x Ji matrix already).
.as_qi_boot_matrix <- function(Qi_boot) {
  if (is.list(Qi_boot)) do.call(rbind, Qi_boot) else Qi_boot
}

## Build a y_grid that spans both groups' point estimates and bootstrap
## replicates. Useful when callers want consistent y-axes across the
## "CDFs" and "CDF difference" plots.
default_y_grid_two_group <- function(Qi_hat_0, Qi_hat_1,
                                      Qi_boot_0, Qi_boot_1) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  y_pool <- c(Qi_hat_0, Qi_hat_1,
              as.vector(Qi_boot_0), as.vector(Qi_boot_1))
  sort(unique(c(min(y_pool), Qi_hat_0, Qi_hat_1, max(y_pool))))
}

## Two-group conditional QFs with pointwise bands on the (0,1) p-grid.
plot_decoded_conditional_qfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  band_lo_0 <- apply(Qi_boot_0, 2, quantile, probs = alpha)
  band_hi_0 <- apply(Qi_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1 <- apply(Qi_boot_1, 2, quantile, probs = alpha)
  band_hi_1 <- apply(Qi_boot_1, 2, quantile, probs = 1 - alpha)

  fun_list  <- list(Qi_hat_0, band_lo_0, band_hi_0,
                    Qi_hat_1, band_lo_1, band_hi_1)
  grid_list <- rep(list(pi_grid), length(fun_list))
  colors    <- rep(group_colors, each = 3)
  widths    <- rep(c(1, 0.5, 0.5), 2)
  types     <- rep(c(1, 3, 3), 2)
  cwt_lbl   <- data.frame(
    color = group_colors, width = c(4, 4),
    type  = c(1, 1), label = group_labels
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = colors, widths = widths, types = types,
    ylab      = "Q(p)",
    color_width_type_labels = cwt_lbl, main = main
  )
  invisible(NULL)
}

## Group-difference QF (group 1 - group 0) with pointwise bands. Bands use
## within-replicate subtraction so they pinch at any common point.
plot_decoded_conditional_qf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  Qi_hat_diff   <- Qi_hat_1 - Qi_hat_0
  mat_boot_diff <- Qi_boot_1 - Qi_boot_0
  band_lo_diff  <- apply(mat_boot_diff, 2, quantile, probs = alpha)
  band_hi_diff  <- apply(mat_boot_diff, 2, quantile, probs = 1 - alpha)

  zero_line <- rep(0, length(pi_grid))
  fun_list  <- list(Qi_hat_diff, band_lo_diff, band_hi_diff, zero_line)
  grid_list <- rep(list(pi_grid), length(fun_list))
  colors    <- c("black", "black", "black", "red")
  widths    <- c(4, 1, 1, 1)
  types     <- c(1, 3, 3, 2)
  cwt_lbl   <- data.frame(
    color = "black", width = 4, type = 1, label = diff_label
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = colors, widths = widths, types = types,
    ylab      = "Q(p) difference",
    color_width_type_labels = cwt_lbl, main = main
  )
  invisible(NULL)
}

## Two-group conditional CDFs on a shared y-grid with pointwise bands and
## optional vertical threshold lines.
plot_decoded_conditional_cdfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL,
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    thresholds = NULL, threshold_cols = NULL, threshold_labels = NULL
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  F_hat_0  <- qi_to_cdf(Qi_hat_0, y_grid)
  F_hat_1  <- qi_to_cdf(Qi_hat_1, y_grid)
  F_boot_0 <- t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  F_boot_1 <- t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  band_lo_0 <- apply(F_boot_0, 2, quantile, probs = alpha)
  band_hi_0 <- apply(F_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1 <- apply(F_boot_1, 2, quantile, probs = alpha)
  band_hi_1 <- apply(F_boot_1, 2, quantile, probs = 1 - alpha)

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)", main = main)
  if (!is.null(thresholds)) {
    abline(v = thresholds, col = threshold_cols, lty = 4, lwd = 1.5)
  }
  lines(y_grid, F_hat_0,   col = group_colors[1], lwd = 4)
  lines(y_grid, band_lo_0, col = group_colors[1], lwd = 0.5, lty = 3)
  lines(y_grid, band_hi_0, col = group_colors[1], lwd = 0.5, lty = 3)
  lines(y_grid, F_hat_1,   col = group_colors[2], lwd = 4)
  lines(y_grid, band_lo_1, col = group_colors[2], lwd = 0.5, lty = 3)
  lines(y_grid, band_hi_1, col = group_colors[2], lwd = 0.5, lty = 3)

  legend("topleft",
    legend = c(group_labels, threshold_labels),
    col    = c(group_colors, threshold_cols),
    lwd    = c(4, 4, rep(1.5, length(threshold_labels))),
    lty    = c(1, 1, rep(4, length(threshold_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Group-difference CDF (group 1 - group 0) on a shared y-grid with
## pointwise bands and optional threshold lines.
plot_decoded_conditional_cdf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL, diff_label = "1 - 0",
    thresholds = NULL, threshold_cols = NULL, threshold_labels = NULL
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  F_hat_0  <- qi_to_cdf(Qi_hat_0, y_grid)
  F_hat_1  <- qi_to_cdf(Qi_hat_1, y_grid)
  F_boot_0 <- t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  F_boot_1 <- t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  F_hat_diff      <- F_hat_1 - F_hat_0
  F_boot_diff_mat <- F_boot_1 - F_boot_0
  band_lo_diff    <- apply(F_boot_diff_mat, 2, quantile, probs = alpha)
  band_hi_diff    <- apply(F_boot_diff_mat, 2, quantile, probs = 1 - alpha)

  y_lim <- range(c(F_hat_diff, band_lo_diff, band_hi_diff, 0))
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "F(y) difference", main = main)
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(thresholds)) {
    abline(v = thresholds, col = threshold_cols, lty = 4, lwd = 1.5)
  }
  lines(y_grid, F_hat_diff,   col = "black", lwd = 4)
  lines(y_grid, band_lo_diff, col = "black", lwd = 1, lty = 3)
  lines(y_grid, band_hi_diff, col = "black", lwd = 1, lty = 3)

  legend("topleft",
    legend = c(diff_label, "y = 0", threshold_labels),
    col    = c("black", "red", threshold_cols),
    lwd    = c(4, 1, rep(1.5, length(threshold_labels))),
    lty    = c(1, 2, rep(4, length(threshold_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Two-group moments (4 panels: mean, variance, skewness, kurtosis) with
## interval bars from the bootstrap. Layout via par(mfrow = c(1, 4)).
plot_decoded_conditional_moments <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  mom_hat_0  <- moments_from_Qi(Qi_hat_0)
  mom_hat_1  <- moments_from_Qi(Qi_hat_1)
  mom_boot_0 <- t(apply(Qi_boot_0, 1, moments_from_Qi))
  mom_boot_1 <- t(apply(Qi_boot_1, 1, moments_from_Qi))
  band_lo_0  <- apply(mom_boot_0, 2, quantile, probs = alpha)
  band_hi_0  <- apply(mom_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1  <- apply(mom_boot_1, 2, quantile, probs = alpha)
  band_hi_1  <- apply(mom_boot_1, 2, quantile, probs = 1 - alpha)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_0[k], band_hi_0[k], mom_hat_0[k],
                band_lo_1[k], band_hi_1[k], mom_hat_1[k])
    y_pad <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "group", ylab = moment_names[k],
         main = moment_names[k])
    axis(1, at = c(1, 2), labels = group_labels)
    arrows(1, band_lo_0[k], 1, band_hi_0[k],
           angle = 90, code = 3, length = 0.08,
           col = group_colors[1], lwd = 1.5)
    points(1, mom_hat_0[k], col = group_colors[1], pch = 19, cex = 1.6)
    arrows(2, band_lo_1[k], 2, band_hi_1[k],
           angle = 90, code = 3, length = 0.08,
           col = group_colors[2], lwd = 1.5)
    points(2, mom_hat_1[k], col = group_colors[2], pch = 19, cex = 1.6)
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}

## Group-difference moments (4 panels) with interval bars from the
## within-replicate bootstrap contrast.
plot_decoded_conditional_moment_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  mom_hat_0     <- moments_from_Qi(Qi_hat_0)
  mom_hat_1     <- moments_from_Qi(Qi_hat_1)
  mom_boot_0    <- t(apply(Qi_boot_0, 1, moments_from_Qi))
  mom_boot_1    <- t(apply(Qi_boot_1, 1, moments_from_Qi))
  mom_hat_diff  <- mom_hat_1 - mom_hat_0
  mom_boot_diff <- mom_boot_1 - mom_boot_0
  band_lo_diff  <- apply(mom_boot_diff, 2, quantile, probs = alpha)
  band_hi_diff  <- apply(mom_boot_diff, 2, quantile, probs = 1 - alpha)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_diff[k], band_hi_diff[k], mom_hat_diff[k], 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = str_glue("{moment_names[k]} difference"),
         main = moment_names[k])
    axis(1, at = 1, labels = diff_label)
    abline(h = 0, col = "red", lty = 2)
    arrows(1, band_lo_diff[k], 1, band_hi_diff[k],
           angle = 90, code = 3, length = 0.08,
           col = "black", lwd = 1.5)
    points(1, mom_hat_diff[k], col = "black", pch = 19, cex = 1.6)
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}


## ----- Conditional QF Plots

## --- Parametric bootstrap of conditional latent means

## Extract fitted ingredients
X         <- cbind(1, x)
B_hat     <- coef(model)
E_hat     <- residuals(model)
Sigma_hat <- crossprod(E_hat) / (N - 2)
N_fit     <- nrow(E_hat)
mu_hat    <- X %*% B_hat

## Original conditional latent means
z_hat_0 <- as.numeric(B_hat["(Intercept)", ])
z_hat_1 <- as.numeric(B_hat["(Intercept)", ] + B_hat["x", ])

## Bootstrap conditional latent means
R     <- 1000
alpha <- 0.05
z_boot_0 <- vector('list', R)
z_boot_1 <- vector('list', R)
set.seed(12345)
for (r in 1:R) {
  E_star  <- MASS::mvrnorm(N_fit, mu = rep(0, ncol(B_hat)), Sigma = Sigma_hat)
  Z_star  <- mu_hat + E_star
  model_r <- lm(Z_star ~ x)
  B_r     <- coef(model_r)
  z_boot_0[[r]] <- as.numeric(B_r["(Intercept)", ])
  z_boot_1[[r]] <- as.numeric(B_r["(Intercept)", ] + B_r["x", ])
}

## Decode original and bootstrap conditional means to Qi-space
Qi_hat_0  <- decode_z_rot_to_Qi(list(z_hat_0), Ji)[[1]]
Qi_hat_1  <- decode_z_rot_to_Qi(list(z_hat_1), Ji)[[1]]
Qi_boot_0 <- decode_z_rot_to_Qi(z_boot_0, Ji)
Qi_boot_1 <- decode_z_rot_to_Qi(z_boot_1, Ji)


## --- Plot conditional QFs with bands

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-qfs_{bin_pred}.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qfs(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = str_glue("Decoded Conditional Means ({bin_pred})")
)
dev.off()


## --- Plot difference of conditional QFs with bands

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-qfs-diff_{bin_pred}.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_qf_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = str_glue("Decoded Conditional Mean Difference ({bin_pred})")
)
dev.off()


## --- Shared y-grid for CDF plots
y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1)


## --- Plot conditional CDFs with bands

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-cdfs_{bin_pred}.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdfs(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha            = alpha,
  main             = str_glue("Decoded Conditional CDFs ({bin_pred})"),
  y_grid           = y_grid,
  thresholds       = mims_thresholds,
  threshold_cols   = mims_threshold_cols,
  threshold_labels = mims_threshold_labels
)
dev.off()


## --- Plot CDF difference with bands

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-cdfs-diff_{bin_pred}.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdf_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha            = alpha,
  main             = str_glue("Decoded Conditional CDF Difference ({bin_pred})"),
  y_grid           = y_grid,
  thresholds       = mims_thresholds,
  threshold_cols   = mims_threshold_cols,
  threshold_labels = mims_threshold_labels
)
dev.off()


## --- Plot moments with bands (4 panels, one per moment)

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-moments_{bin_pred}.png')
)
png(path_plot, width = 1280, height = 360, pointsize = 14)
plot_decoded_conditional_moments(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = str_glue("Decoded Conditional Moments ({bin_pred})")
)
dev.off()


## --- Contrast in conditional moments (group 1 - group 0)

path_plot <- file.path(
  'artifacts', dir_art, 'plots',
  str_glue('gc_cond-moments-diff_{bin_pred}.png')
)
png(path_plot, width = 1280, height = 360, pointsize = 14)
plot_decoded_conditional_moment_diff(
  Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
  alpha = alpha,
  main  = str_glue("Decoded Conditional Moment Contrasts ({bin_pred})")
)
dev.off()




## ========== Additive Mixed Model ========== ##

library(future)
library(future.apply)
library(mgcv)
library(nlme)
library(RLRsim)
library(splines)


## ---------- Helper: O'Sullivan (Demmler-Reinsch) spline design ----------

build_DR_design <- function(x, n_knots = 5, range_x = NULL) {
  if (is.null(range_x)) range_x <- range(x)
  a <- range_x[1]; b <- range_x[2]

  ## Interior knots at equally-spaced quantiles of x
  knots_interior <- as.numeric(quantile(
    unique(x),
    probs = seq(0, 1, length.out = n_knots + 2)[-c(1, n_knots + 2)]
  ))

  ## Cubic B-spline knot sequence (boundary knots with multiplicity 4)
  all_knots <- c(rep(a, 4), knots_interior, rep(b, 4))

  ## B-spline design at x
  B <- splineDesign(knots = all_knots, x = x, ord = 4, outer.ok = TRUE)

  ## O'Sullivan penalty Omega via Simpson's rule on a fine grid
  n_grid  <- 401
  grid    <- seq(a, b, length.out = n_grid)
  B2      <- splineDesign(knots = all_knots, x = grid, ord = 4, derivs = 2, outer.ok = TRUE)
  h       <- (b - a) / (n_grid - 1)
  w_simp  <- rep(0, n_grid)
  w_simp[1]                              <- h / 3
  w_simp[n_grid]                         <- h / 3
  w_simp[seq(2, n_grid - 1, by = 2)]     <- 4 * h / 3
  w_simp[seq(3, n_grid - 1, by = 2)]     <- 2 * h / 3
  Omega <- t(B2) %*% (w_simp * B2)

  ## Spectral decomposition; null space (constant + linear) sorted first
  eig <- eigen(Omega, symmetric = TRUE)
  d   <- eig$values
  P   <- eig$vectors
  ord <- order(d)
  d   <- d[ord]; P <- P[, ord]

  Z_Omega  <- P[, -(1:2)]
  d_nz     <- d[-(1:2)]
  Z_scaled <- Z_Omega %*% diag(1 / sqrt(d_nz))

  W <- B %*% Z_scaled
  attr(W, "knots_interior") <- knots_interior
  attr(W, "all_knots")      <- all_knots
  attr(W, "Z_scaled")       <- Z_scaled
  attr(W, "range")          <- range_x
  W
}


## ---------- Helper: Fit per-K LMMs ----------

fit_latent_amm <- function(Z, covariates,
                            n_knots_age = 10, n_knots_pir = 10,
                            linear_covariates = c("male", "age", "pir"),
                            smooth_covariates = c("age", "pir")) {

  ## Drop rows with any missing covariate
  keep <- complete.cases(covariates[, c("male", "age", "pir")])
  Z    <- Z[keep, , drop = FALSE]
  cov_ <- covariates[keep, , drop = FALSE]
  N    <- nrow(Z)
  K    <- ncol(Z)

  has_age_lin <- "age" %in% linear_covariates
  has_pir_lin <- "pir" %in% linear_covariates
  has_male    <- "male" %in% linear_covariates
  has_age_sm  <- "age" %in% smooth_covariates
  has_pir_sm  <- "pir" %in% smooth_covariates

  ## Fixed-effect design
  X_cols <- list(`(Intercept)` = rep(1, N))
  if (has_male)    X_cols[["male"]] <- cov_$male
  if (has_age_lin) X_cols[["age"]]  <- cov_$age
  if (has_pir_lin) X_cols[["pir"]]  <- cov_$pir
  X_design <- do.call(cbind, X_cols)
  colnames(X_design) <- names(X_cols)
  A1 <- ncol(X_design)
  beta_names <- colnames(X_design)
  fixed_rhs <- if (length(beta_names) > 1) {
    paste(setdiff(beta_names, "(Intercept)"), collapse = " + ")
  } else {
    "1"
  }
  fixed_form <- as.formula(paste("z ~", fixed_rhs))

  ## Demmler-Reinsch spline designs (only for included smooths)
  W_age <- if (has_age_sm) build_DR_design(cov_$age, n_knots = n_knots_age) else NULL
  W_pir <- if (has_pir_sm) build_DR_design(cov_$pir, n_knots = n_knots_pir) else NULL
  M_a   <- if (!is.null(W_age)) ncol(W_age) else 0
  M_p   <- if (!is.null(W_pir)) ncol(W_pir) else 0

  ## Eigenvalues of W'W (used for edf via Demmler-Reinsch formula)
  phi_age <- if (M_a > 0) eigen(crossprod(W_age), symmetric = TRUE, only.values = TRUE)$values else numeric(0)
  phi_pir <- if (M_p > 0) eigen(crossprod(W_pir), symmetric = TRUE, only.values = TRUE)$values else numeric(0)

  ## Storage
  beta_hat       <- matrix(NA_real_, nrow = A1, ncol = K,
                           dimnames = list(beta_names, NULL))
  U_age_hat      <- if (M_a > 0) matrix(NA_real_, nrow = M_a, ncol = K) else NULL
  U_pir_hat      <- if (M_p > 0) matrix(NA_real_, nrow = M_p, ncol = K) else NULL
  sigma2_U_age   <- if (M_a > 0) numeric(K) else NULL
  sigma2_U_pir   <- if (M_p > 0) numeric(K) else NULL
  sigma2_E_diag  <- numeric(K)
  edf_age        <- if (M_a > 0) numeric(K) else NULL
  edf_pir        <- if (M_p > 0) numeric(K) else NULL
  edf            <- numeric(K)
  fits           <- vector("list", K)
  ## Per-K diagnostic: mgcv refit summary at each lme failure.
  ## Each element is NULL on success, or a list(gam_sp, gam_edf, gam_logLik)
  ## on lme failure. Boundary collapse <=> sp at upper bound and edf ~= 1 for
  ## the offending smooth term.
  boundary_diag  <- vector("list", K)

  ## Named-column data frame so pdIdent formulas resolve cleanly
  W_cols_age <- if (M_a > 0) paste0("Wa", seq_len(M_a)) else character(0)
  W_cols_pir <- if (M_p > 0) paste0("Wp", seq_len(M_p)) else character(0)
  form_age   <- if (M_a > 0) as.formula(paste0("~ -1 + ", paste(W_cols_age, collapse = " + "))) else NULL
  form_pir   <- if (M_p > 0) as.formula(paste0("~ -1 + ", paste(W_cols_pir, collapse = " + "))) else NULL
  group      <- factor(rep(1, N))

  df_base <- data.frame(
    male = cov_$male,
    age  = cov_$age,
    pir  = cov_$pir,
    grp  = group
  )
  for (j in seq_len(M_a)) df_base[[W_cols_age[j]]] <- W_age[, j]
  for (j in seq_len(M_p)) df_base[[W_cols_pir[j]]] <- W_pir[, j]

  ## Random-effects structure
  random_struct <- NULL
  if (M_a > 0 && M_p > 0) {
    random_struct <- list(grp = pdBlocked(list(pdIdent(form_age), pdIdent(form_pir))))
  } else if (M_a > 0) {
    random_struct <- list(grp = pdIdent(form_age))
  } else if (M_p > 0) {
    random_struct <- list(grp = pdIdent(form_pir))
  }

  ## Per-K model
  for (k in seq_len(K)) {
    message(str_glue("---- AMM fit: k = {k} of {K} ----"))
    df_k    <- df_base
    df_k$z  <- Z[, k]

    if (is.null(random_struct)) {
      ## No smooths => plain OLS. No zero-fill recovery here: lm failures
      ## are not boundary-collapse (no variance components to collapse), so
      ## the "this column contributes nothing" interpretation does not apply.
      ## Treat lm failures the same way the original code did: skip.
      ## do.call materializes `fixed_form` into the captured fit_k$call so
      ## update() works after fit_latent_amm() returns.
      fit_k <- tryCatch(
        do.call(lm, list(formula = fixed_form, data = df_k)),
        error = function(e) {
          message(str_glue("  lm failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      if (is.null(fit_k)) next
      fits[[k]] <- fit_k

      beta_hat[, k] <- coef(fit_k)[beta_names]
      sigma2_E_diag[k] <- summary(fit_k)$sigma^2
      edf[k] <- A1
    } else {
      ## do.call materializes `fixed_form` and `random_struct` into the
      ## captured fit_k$call, so RLRsim::exactRLRT() (and any other caller
      ## that runs update.lme on the fit) can reconstruct the call after
      ## this function returns and its locals fall out of scope.
      fit_k <- tryCatch(
        do.call(lme, list(
          fixed   = fixed_form,
          random  = random_struct,
          data    = df_k,
          method  = "REML",
          control = lmeControl(opt = "optim")
        )),
        error = function(e) {
          message(str_glue("  lme failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      ## NOTE: lme's foreign-call errors are tentatively interpreted as
      ## variance-component boundary collapse (failure mode whenever the
      ## true VC for this column is ~0). The zero-fill encodes "this column
      ## contributes nothing" so downstream consumers see finite values.
      ## To verify the boundary interpretation, refit the failing column
      ## with mgcv::gam (which handles the boundary cleanly) and record
      ## the smoothing parameters and edf. Boundary collapse <=> gam_sp at
      ## upper bound AND gam_edf ~= 1 for the offending smooth term.
      if (is.null(fit_k)) {
        beta_hat[, k] <- 0
        if (M_a > 0) U_age_hat[, k] <- 0
        if (M_p > 0) U_pir_hat[, k] <- 0
        gam_k <- tryCatch(
          mgcv::gam(
            z ~ male + s(age, bs = "ps") + s(pir, bs = "ps"),
            data = df_k, method = "REML"
          ),
          error = function(e) NULL
        )
        if (!is.null(gam_k)) {
          s_tbl <- summary(gam_k)$s.table
          boundary_diag[[k]] <- list(
            gam_sp     = gam_k$sp,
            gam_edf    = setNames(s_tbl[, "edf"], rownames(s_tbl)),
            gam_logLik = as.numeric(logLik(gam_k))
          )
          message(str_glue(
            "    -> mgcv refit succeeded: sp = ({paste(signif(gam_k$sp, 3), collapse=', ')}), ",
            "edf = ({paste(signif(s_tbl[, 'edf'], 3), collapse=', ')})"
          ))
        }
        next
      }
      fits[[k]] <- fit_k

      beta_hat[, k] <- fixef(fit_k)[beta_names]

      re_vec <- as.numeric(ranef(fit_k)[1, ])
      if (M_a > 0) U_age_hat[, k] <- re_vec[seq_len(M_a)]
      if (M_p > 0) U_pir_hat[, k] <- re_vec[(M_a + 1):(M_a + M_p)]

      sigma2_E_diag[k] <- fit_k$sigma^2
      vc <- VarCorr(fit_k)
      ## VarCorr rows for pdBlocked(pdIdent, pdIdent): M_a rows then M_p rows then Residual.
      ## All rows within a pdIdent block share one variance.
      if (M_a > 0) sigma2_U_age[k] <- as.numeric(vc[1, "Variance"])
      if (M_p > 0) sigma2_U_pir[k] <- as.numeric(vc[max(M_a, 0) + 1, "Variance"])

      edf_k_total <- A1
      if (M_a > 0) {
        lambda_a <- sigma2_E_diag[k] / sigma2_U_age[k]
        edf_age[k] <- sum(phi_age / (phi_age + lambda_a))
        edf_k_total <- edf_k_total + edf_age[k]
      }
      if (M_p > 0) {
        lambda_p <- sigma2_E_diag[k] / sigma2_U_pir[k]
        edf_pir[k] <- sum(phi_pir / (phi_pir + lambda_p))
        edf_k_total <- edf_k_total + edf_pir[k]
      }
      edf[k] <- edf_k_total
    }
  }

  ## Residual matrix
  R_mat <- Z - X_design %*% beta_hat
  if (M_a > 0) R_mat <- R_mat - W_age %*% U_age_hat
  if (M_p > 0) R_mat <- R_mat - W_pir %*% U_pir_hat

  edf_mean <- mean(edf)
  Sigma_E  <- crossprod(R_mat) / (N - edf_mean)

  list(
    beta_hat          = beta_hat,
    U_age_hat         = U_age_hat,
    U_pir_hat         = U_pir_hat,
    sigma2_U_age      = sigma2_U_age,
    sigma2_U_pir      = sigma2_U_pir,
    sigma2_E_diag     = sigma2_E_diag,
    Sigma_E           = Sigma_E,
    edf               = edf,
    edf_age           = edf_age,
    edf_pir           = edf_pir,
    lambda_age        = if (M_a > 0) sigma2_E_diag / sigma2_U_age else NULL,
    lambda_pir        = if (M_p > 0) sigma2_E_diag / sigma2_U_pir else NULL,
    X_design          = X_design,
    W_age             = W_age,
    W_pir             = W_pir,
    knots_age         = if (M_a > 0) attr(W_age, "knots_interior") else NULL,
    knots_pir         = if (M_p > 0) attr(W_pir, "knots_interior") else NULL,
    linear_covariates = linear_covariates,
    smooth_covariates = smooth_covariates,
    keep              = keep,
    fits              = fits,
    R_mat             = R_mat,
    boundary_diag     = boundary_diag
  )
}


## ---------- Reduced-fit wrappers ----------

fit_latent_amm_drop_smooth <- function(Z, covariates, drop = NULL, ...) {
  fit_latent_amm(Z, covariates,
                 smooth_covariates = setdiff(c("age", "pir"), drop), ...)
}

fit_latent_amm_drop_covariate <- function(Z, covariates, drop = NULL, ...) {
  fit_latent_amm(Z, covariates,
                 linear_covariates = setdiff(c("male", "age", "pir"), drop),
                 smooth_covariates = setdiff(c("age", "pir"), drop), ...)
}


## ---------- Fit ----------

Z_amm <- do.call(rbind, z_rot_ctx$payload)
covariates_amm <- data.frame(
  male = df_cov$male,
  age  = df_cov$age,
  pir  = df_cov$PIR
)
start <- Sys.time()
amm <- fit_latent_amm(Z_amm, covariates_amm, n_knots_age = 10, n_knots_pir = 10)
stop <- Sys.time()
print(stop - start)


## ---------- Diagnostics ----------

K_amm <- ncol(Z_amm)

## (1) Per-column summary table
summary_tbl <- data.frame(
  k            = seq_len(K_amm),
  sigma2_U_age = amm$sigma2_U_age,
  sigma2_U_pir = amm$sigma2_U_pir,
  sigma2_E     = amm$sigma2_E_diag,
  lambda_age   = amm$lambda_age,
  lambda_pir   = amm$lambda_pir,
  edf          = amm$edf
)
print(summary_tbl)
write.csv(
  summary_tbl,
  file.path('artifacts', dir_art, 'amm_summary.csv'),
  row.names = FALSE
)


## (2) Smoothing-parameter plot (log10 lambda by k)
path_plot <- file.path('artifacts', dir_art, 'plots', 'amm_lambda.png')
png(path_plot, width = 960, height = 540, pointsize = 14)
par(mfrow = c(1, 1), mar = c(4, 5, 3, 1))
lambda_log <- log10(c(amm$lambda_age, amm$lambda_pir))
plot(
  NULL,
  xlim = c(1, K_amm),
  ylim = range(lambda_log, finite = TRUE),
  xlab = "k", ylab = expression(log[10](lambda)),
  main = "AMM smoothing parameters by latent dimension"
)
lines(1:K_amm, log10(amm$lambda_age), col = "darkblue",   lwd = 2, type = "b", pch = 19)
lines(1:K_amm, log10(amm$lambda_pir), col = "darkorange", lwd = 2, type = "b", pch = 17)
legend("topright",
  legend = c("age", "pir"),
  col    = c("darkblue", "darkorange"),
  lwd    = 2, pch = c(19, 17), bty = "n"
)
dev.off()


## (3) Residual Q-Q panels
nc <- ceiling(sqrt(K_amm))
nr <- ceiling(K_amm / nc)
path_plot <- file.path('artifacts', dir_art, 'plots', 'amm_resid_qq.pdf')
pdf(path_plot, width = 3 * nc, height = 3 * nr)
par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
for (k in seq_len(K_amm)) {
  qqnorm(amm$R_mat[, k], main = paste0("k = ", k),
         pch = 19, cex = 0.25, col = col_train)
  qqline(amm$R_mat[, k], col = "red")
}
dev.off()
par(mfrow = c(1, 1))


## (4) Residual covariance heatmap (Sigma_E)
path_plot <- file.path('artifacts', dir_art, 'plots', 'amm_sigma_e.png')
zmax    <- max(abs(amm$Sigma_E))
breaks  <- seq(-zmax, zmax, length.out = 101)
col_pal <- colorRampPalette(c("#0000FF", "white", "#FF000000"))(100)

png(path_plot, width = 800, height = 720, pointsize = 14)
par(mar = c(4, 4, 3, 5))
image(
  1:K_amm, 1:K_amm,
  t(amm$Sigma_E)[, K_amm:1],
  zlim = c(-zmax, zmax),
  col  = col_pal, breaks = breaks,
  axes = FALSE, xlab = "", ylab = "",
  main = expression(hat(Sigma)[E])
)
axis(1, at = 1:K_amm, labels = 1:K_amm, las = 1, tick = FALSE)
axis(2, at = 1:K_amm, labels = K_amm:1, las = 1, tick = FALSE)
for (i in 1:K_amm) for (j in 1:K_amm) if (i >= j) {
  text(j, K_amm - i + 1, sprintf("%.2f", amm$Sigma_E[i, j]), cex = 0.6)
}
dev.off()


## (5) Random-effect covariance heatmaps: diagnostic on diagonal assumption.
##
## The model fits per-K independently with pdIdent shape, so it ASSUMES the
## joint covariance of U_a across (basis dim, latent component) is block-
## diagonal with sigma2_U_a[k] * I on each block — equivalently, the
## cross-K random-effect covariance is exactly diagonal with sigma2_U_a[k]
## on the diagonal and 0 off-diagonal.
##
## This plot empirically checks that diagonal assumption. Treat each basis
## dimension m = 1..M_a as one observation of a K-dim BLUP vector and
## compute cov(U_a_hat) (M_a x K, rows = basis dims, cols = latent
## components). Under the model the off-diagonals should be 0. Substantial
## off-diagonals suggest the per-K fits are not independent — typically
## because Sigma_E has off-diagonals that induce correlated smooth
## estimates across latent components.
##
## Caveat: M_a = 7 (or M_p = 7) is small, so the empirical covariance is
## rank-deficient (rank <= M-1) and individual off-diagonal entries are
## noisy. Read this as a qualitative pattern check, not a precise estimate.

plot_RE_cov_diag <- function(U_hat, K_, title_expr, path) {
  Sigma_U <- cov(U_hat)
  zmax    <- max(abs(Sigma_U))
  breaks  <- seq(-zmax, zmax, length.out = 101)
  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF0000"))(100)

  png(path, width = 800, height = 720, pointsize = 14)
  par(mar = c(4, 4, 3, 5))
  image(
    1:K_, 1:K_,
    t(Sigma_U)[, K_:1],
    zlim = c(-zmax, zmax),
    col  = col_pal, breaks = breaks,
    axes = FALSE, xlab = "", ylab = "",
    main = title_expr
  )
  axis(1, at = 1:K_, labels = 1:K_, las = 1, tick = FALSE)
  axis(2, at = 1:K_, labels = K_:1, las = 1, tick = FALSE)
  for (i in 1:K_) for (j in 1:K_) if (i >= j) {
    text(j, K_ - i + 1, sprintf("%.2f", Sigma_U[i, j]), cex = 0.6)
  }
  dev.off()
}

plot_RE_cov_diag <- function(U_hat, K_, title_expr, path) {
  Sigma_U <- cov(U_hat)
  d       <- sqrt(diag(Sigma_U))
  Corr_U  <- Sigma_U / tcrossprod(d)           # correlation matrix

  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF0000"))(100)
  breaks  <- seq(-1, 1, length.out = 101)

  png(path, width = 1100, height = 720, pointsize = 14)
  layout(matrix(c(1, 2), nrow = 1), widths = c(2, 1))

  par(mar = c(4, 4, 3, 5))
  image(1:K_, 1:K_, t(Corr_U)[, K_:1],
        zlim = c(-1, 1), col = col_pal, breaks = breaks,
        axes = FALSE, xlab = "", ylab = "", main = title_expr)
  axis(1, at = 1:K_, labels = 1:K_,    las = 1, tick = FALSE)
  axis(2, at = 1:K_, labels = K_:1,    las = 1, tick = FALSE)
  for (i in 1:K_) for (j in 1:K_) if (i >= j)
    text(j, K_ - i + 1, sprintf("%.2f", Corr_U[i, j]), cex = 0.6)

  par(mar = c(4, 5, 3, 2))
  barplot(log10(diag(Sigma_U)), names.arg = 1:K_, horiz = FALSE,
          xlab = "k", ylab = expression(log[10]~hat(sigma)[U[k]]^2),
          main = "Component variances")
  dev.off()
}

plot_RE_cov_diag(
  amm$U_age_hat, K_amm,
  expression(hat(Sigma)[U]^"(age)"),
  file.path('artifacts', dir_art, 'plots', 'amm_sigma_u_age.png')
)
plot_RE_cov_diag(
  amm$U_pir_hat, K_amm,
  expression(hat(Sigma)[U]^"(pir)"),
  file.path('artifacts', dir_art, 'plots', 'amm_sigma_u_pir.png')
)


## ========== AMM Inference ========== ##

## ---------- Inference helpers ----------

p_mixture_chisq <- function(q, dfs, weights) {
  stopifnot(length(dfs) == length(weights))
  contrib <- numeric(length(dfs))
  for (j in seq_along(dfs)) {
    if (dfs[j] == 0) {
      contrib[j] <- ifelse(q > 0, 0, 1)
    } else {
      contrib[j] <- pchisq(q, df = dfs[j], lower.tail = FALSE)
    }
  }
  sum(weights * contrib)
}

evaluate_smooth <- function(fit_result, covariate_name, grid, include_linear = TRUE) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  U_hat <- fit_result[[paste0("U_", covariate_name, "_hat")]]
  if (is.null(W_obj) || is.null(U_hat)) {
    stop(str_glue("Smooth for '{covariate_name}' not present in fit_result."))
  }
  all_knots <- attr(W_obj, "all_knots")
  Z_scaled  <- attr(W_obj, "Z_scaled")
  range_x   <- attr(W_obj, "range")

  grid_clipped <- pmax(range_x[1], pmin(range_x[2], grid))
  B_new <- splineDesign(knots = all_knots, x = grid_clipped, ord = 4, outer.ok = TRUE)
  W_new <- B_new %*% Z_scaled

  out <- W_new %*% U_hat  # J x K
  if (include_linear && covariate_name %in% rownames(fit_result$beta_hat)) {
    beta_slope <- fit_result$beta_hat[covariate_name, ]
    out <- out + outer(grid, beta_slope)
  }
  out
}

compute_lrt <- function(fits_full, fits_reduced) {
  K_ <- length(fits_full)
  stopifnot(length(fits_reduced) == K_)
  ## NOTE: A NULL fit at column k indicates a boundary collapse during
  ## fit_latent_amm (see note inside the per-K loop there). Treat that
  ## column's log-likelihood contribution as 0, matching the "zero spline
  ## contribution" interpretation used to fill the BLUPs/coefficients.
  -2 * sum(sapply(seq_len(K_), function(k) {
    if (is.null(fits_full[[k]]) || is.null(fits_reduced[[k]])) return(0)
    as.numeric(logLik(fits_reduced[[k]])) - as.numeric(logLik(fits_full[[k]]))
  }))
}

compute_fitted_reduced <- function(fit_result_reduced) {
  ## lme$fitted() returns marginal mean X*beta + W*u; lm has the same convention.
  Z_hat <- sapply(fit_result_reduced$fits, fitted)
  if (!is.matrix(Z_hat)) Z_hat <- matrix(Z_hat, ncol = length(fit_result_reduced$fits))
  Z_hat
}


## ---------- Test 1: Hotelling-type test for linear effect ----------

test_linear <- function(fit_result, covariate_name) {
  a <- which(rownames(fit_result$beta_hat) == covariate_name)
  if (length(a) != 1) stop(str_glue("Covariate '{covariate_name}' not found in beta_hat."))

  b      <- fit_result$beta_hat[a, ]
  K_     <- length(b)
  N_kept <- sum(fit_result$keep)
  nu     <- nrow(fit_result$beta_hat) +
            (if (!is.null(fit_result$edf_age)) mean(fit_result$edf_age) else 0) +
            (if (!is.null(fit_result$edf_pir)) mean(fit_result$edf_pir) else 0)

  c_aa <- mean(sapply(fit_result$fits, function(f) {
    if (inherits(f, "lme")) {
      vcov(f)[covariate_name, covariate_name] / f$sigma^2
    } else {
      vcov(f)[covariate_name, covariate_name] / summary(f)$sigma^2
    }
  }))

  T2     <- as.numeric(t(b) %*% solve(fit_result$Sigma_E, b) / c_aa)
  df1    <- K_
  df2    <- N_kept - nu - K_
  F_stat <- (N_kept - nu - K_) / ((N_kept - nu - 1) * K_) * T2
  p_value <- pf(F_stat, df1, df2, lower.tail = FALSE)

  list(T2 = T2, F_stat = F_stat, df1 = df1, df2 = df2, p_value = p_value)
}


## ---------- Test 1 companion: Roy-Bose simultaneous intervals ----------

## Component-wise simultaneous CIs for beta_k, k = 1..K, derived from the
## same Hotelling/F machinery as test_linear. The Roy-Bose union-intersection
## bound gives joint coverage >= 1 - alpha across ALL linear combinations
## a^T beta; specializing a = e_k yields the per-k intervals
##   beta_hat_k +/- sqrt(T2_crit) * sqrt(c_aa * Sigma_E[k, k]),
## where T2_crit = (N_kept - nu - 1) * K / (N_kept - nu - K) * F_{K, df2, 1-alpha}
## matches the F-transform inside test_linear.
intervals_linear <- function(fit_result, covariate_name, alpha = 0.05) {
  a <- which(rownames(fit_result$beta_hat) == covariate_name)
  if (length(a) != 1) stop(str_glue("Covariate '{covariate_name}' not found in beta_hat."))

  b      <- fit_result$beta_hat[a, ]
  K_     <- length(b)
  N_kept <- sum(fit_result$keep)
  nu     <- nrow(fit_result$beta_hat) +
            (if (!is.null(fit_result$edf_age)) mean(fit_result$edf_age) else 0) +
            (if (!is.null(fit_result$edf_pir)) mean(fit_result$edf_pir) else 0)

  c_aa <- mean(sapply(fit_result$fits, function(f) {
    if (inherits(f, "lme")) {
      vcov(f)[covariate_name, covariate_name] / f$sigma^2
    } else {
      vcov(f)[covariate_name, covariate_name] / summary(f)$sigma^2
    }
  }))

  df1     <- K_
  df2     <- N_kept - nu - K_
  F_crit  <- qf(1 - alpha, df1, df2)
  T2_crit <- (N_kept - nu - 1) * K_ / (N_kept - nu - K_) * F_crit

  se_k <- sqrt(c_aa * diag(fit_result$Sigma_E))
  half <- sqrt(T2_crit) * se_k

  list(
    covariate_name = covariate_name,
    alpha          = alpha,
    beta_hat       = b,
    se             = se_k,                # per-k Wald SE: sqrt(c_aa * Sigma_E[k,k])
    T2_crit        = T2_crit,
    df1            = df1,
    df2            = df2,
    half_width     = half,
    lower          = b - half,
    upper          = b + half,
    method         = "Roy-Bose simultaneous (T^2 union-intersection)"
  )
}


## ---------- Test 2: Wood (2013) conditional F-test (mgcv route) ----------

## NOTE: Based on different GAM parameterization!
test_smooth_wood <- function(Z, covariates_df, covariate_name) {
  K_ <- ncol(Z)
  per_k <- data.frame(k = seq_len(K_), edf = NA_real_, F = NA_real_, p = NA_real_)
  smooth_row <- paste0("s(", covariate_name, ")")
  for (k in seq_len(K_)) {
    df_k <- data.frame(
      z    = Z[, k],
      male = covariates_df$male,
      age  = covariates_df$age,
      pir  = covariates_df$pir
    )
    fit_k <- tryCatch(
      mgcv::gam(z ~ male + s(age, bs = "ps") + s(pir, bs = "ps"), data = df_k),
      error = function(e) NULL
    )
    if (is.null(fit_k)) next
    s_tbl <- summary(fit_k)$s.table
    per_k$edf[k] <- s_tbl[smooth_row, "edf"]
    per_k$F[k]   <- s_tbl[smooth_row, "F"]
    per_k$p[k]   <- s_tbl[smooth_row, "p-value"]
  }
  p_clean      <- pmin(pmax(per_k$p, .Machine$double.xmin), 1)
  p_fisher     <- pchisq(-2 * sum(log(p_clean)), df = 2 * K_, lower.tail = FALSE)
  p_bonferroni <- min(K_ * min(p_clean), 1)
  list(per_k = per_k, p_fisher = p_fisher, p_bonferroni = p_bonferroni)
}


## ---------- Test 3: Self & Liang LRT ----------

test_smooth_self_liang <- function(fit_full, fit_reduced) {
  Lambda  <- compute_lrt(fit_full$fits, fit_reduced$fits)
  K_      <- ncol(fit_full$beta_hat)
  dfs     <- 0:K_
  weights <- choose(K_, dfs) / 2^K_
  p_value <- p_mixture_chisq(Lambda, dfs, weights)
  list(Lambda = Lambda, K = K_, p_value = p_value,
       mixture_dfs = dfs, mixture_weights = weights)
}


## ---------- Test 4: Crainiceanu-Ruppert exact RLRT ----------

## Per-column exact RLRT via RLRsim::exactRLRT, aggregated across K via
## independent-summation convolution + Fisher's method + Bonferroni.
##
## Per-column null sampling: exactRLRT short-circuits and returns an empty
## sample when the observed RLRT is 0 (p-value trivially 1). For the joint
## convolution test we need an honest per-column null sample regardless of
## the observed statistic, so for short-circuited columns we fall back to
## RLRsim::RLRTSim() with the original (X, W_tested) design. This fallback
## IGNORES the nuisance variance component (e.g., the pir smooth when
## testing age), so the resulting null sample is a conservative
## approximation (slightly inflated null tail => larger p-values).
## Returned `method_per_k` flags which path produced each column's sample.
##
## Validity of convolution and Fisher: requires cross-column independence
## of per-column RLRT statistics under H_0. Justified by approximate
## diagonality of Sigma_E. Bonferroni holds under arbitrary dependence.
test_smooth_crainiceanu_ruppert <- function(fit_full, fit_reduced,
                                             covariates,
                                             covariate_name,
                                             B = 10000,
                                             seed_base = 1234) {
  K_           <- ncol(fit_full$beta_hat)
  Lambda_per_k <- numeric(K_)
  null_samples <- matrix(NA_real_, nrow = B, ncol = K_)
  p_per_k      <- numeric(K_)
  method_per_k <- character(K_)

  ## Reconstruct the kept-row fixed-effect and tested-smooth designs for the
  ## RLRTSim fallback. Order matches the original fit_latent_amm construction.
  cov_kept   <- covariates[fit_full$keep, , drop = FALSE]
  X_kept     <- cbind(
    `(Intercept)` = 1,
    male = cov_kept$male,
    age  = cov_kept$age,
    pir  = cov_kept$pir
  )
  W_tested <- fit_full[[paste0("W_", covariate_name)]]

  for (k in seq_len(K_)) {
    fit_k_full <- fit_full$fits[[k]]
    fit_k_red  <- fit_reduced$fits[[k]]
    if (is.null(fit_k_full) || is.null(fit_k_red)) {
      ## Per-K fit collapsed inside fit_latent_amm: no observed statistic,
      ## no null sample (set all to 0; same convention as compute_lrt).
      Lambda_per_k[k]   <- 0
      null_samples[, k] <- 0
      p_per_k[k]        <- 1
      method_per_k[k]   <- "collapsed"
      next
    }
    ## Always compute the observed RLRT manually so it's available even
    ## when exactRLRT short-circuits.
    Lambda_per_k[k] <- max(
      0,
      -2 * (as.numeric(logLik(fit_k_red)) - as.numeric(logLik(fit_k_full)))
    )

    ## Primary path: exactRLRT (handles nuisance VC correctly).
    rlrt_k <- tryCatch(
      RLRsim::exactRLRT(m = fit_k_full, mA = fit_k_full, m0 = fit_k_red,
                        nsim = B, seed = seed_base + k),
      error = function(e) {
        message(str_glue("  exactRLRT failed at k = {k}: {conditionMessage(e)}"))
        NULL
      }
    )

    sample_k <- if (!is.null(rlrt_k)) as.numeric(rlrt_k$sample) else numeric(0)

    if (length(sample_k) >= B) {
      null_samples[, k] <- sample_k[1:B]
      method_per_k[k]   <- "exact"
    } else {
      ## Short-circuit (observed RLRT = 0): exactRLRT skipped simulation.
      ## Fall back to RLRTSim with the original design, ignoring nuisance.
      sim_k <- tryCatch(
        RLRsim::RLRTSim(
          X = X_kept, Z = W_tested,
          ## pdIdent shape: random-effect covariance is sigma^2 * I, so the
          ## shape factor (the part that isn't the scalar variance being
          ## tested) is the identity.
          sqrt.Sigma = diag(ncol(W_tested)),
          nsim = B, seed = seed_base + k + 100000L
        ),
        error = function(e) {
          message(str_glue("  RLRTSim fallback failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      if (is.null(sim_k)) {
        null_samples[, k] <- 0
        method_per_k[k]   <- "fallback_failed"
      } else {
        null_samples[, k] <- as.numeric(sim_k)[1:B]
        method_per_k[k]   <- "approx_no_nuisance"
      }
    }

    p_per_k[k] <- (1 + sum(null_samples[, k] >= Lambda_per_k[k])) / (1 + B)
  }

  ## Joint statistic and convolution reference
  Lambda_joint     <- sum(Lambda_per_k)
  joint_null_sample <- rowSums(null_samples)
  p_conv <- (1 + sum(joint_null_sample >= Lambda_joint)) / (1 + B)

  ## Fisher's method (chi^2_{2K} under independence)
  p_clean  <- pmin(pmax(p_per_k, .Machine$double.xmin), 1)
  X_Fisher <- -2 * sum(log(p_clean))
  p_Fisher <- pchisq(X_Fisher, df = 2 * K_, lower.tail = FALSE)

  ## Bonferroni (arbitrary dependence; conservative)
  p_Bonf <- min(K_ * min(p_per_k), 1)

  n_approx <- sum(method_per_k == "approx_no_nuisance")
  if (n_approx > 0) {
    message(str_glue(
      "  Note: {n_approx} of {K_} columns used the RLRTSim fallback ",
      "(observed RLRT = 0). Their null samples ignore the nuisance ",
      "variance component (approximate, conservative)."
    ))
  }

  ## Bonferroni-adjusted per-k p-values for simultaneous component-wise
  ## inference (joint FWER <= alpha across the K components).
  p_per_k_Bonf <- pmin(K_ * p_per_k, 1)

  list(
    covariate_name    = covariate_name,
    Lambda_per_k      = Lambda_per_k,
    Lambda_joint      = Lambda_joint,
    p_per_k           = p_per_k,
    p_per_k_Bonf      = p_per_k_Bonf,
    p_conv            = p_conv,
    p_Fisher          = p_Fisher,
    p_Bonf            = p_Bonf,
    X_Fisher          = X_Fisher,
    null_samples      = null_samples,
    joint_null_sample = joint_null_sample,
    method_per_k      = method_per_k,
    B                 = B,
    K                 = K_
  )
}


## ---------- Test 5: Parametric bootstrap ----------

boot_test_smooth <- function(fit_full, fit_reduced, covariates, covariate_name,
                              grid = NULL,
                              statistic = c("iss", "lrt"),
                              null_type = c("vc_only", "full"),
                              B = 500, n_cores = 4) {
  statistic <- match.arg(statistic)
  null_type <- match.arg(null_type)

  if (statistic == "lrt" && null_type != "vc_only") {
    stop("statistic='lrt' requires null_type='vc_only' (REML log-likelihoods need identical fixed-effect structure).")
  }
  if (statistic == "iss" && is.null(grid)) {
    stop("statistic='iss' requires a non-NULL grid.")
  }

  ## Observed statistic
  T_obs <- if (statistic == "iss") {
    sm <- evaluate_smooth(fit_full, covariate_name, grid,
                          include_linear = (null_type == "full"))
    sum(sm^2)
  } else {
    compute_lrt(fit_full$fits, fit_reduced$fits)
  }

  ## Bootstrap setup
  Z_hat_red <- compute_fitted_reduced(fit_reduced)
  L         <- chol(fit_reduced$Sigma_E)
  cov_kept  <- covariates[fit_reduced$keep, , drop = FALSE]
  N_kept    <- nrow(Z_hat_red)
  K_        <- ncol(Z_hat_red)

  boot_one <- function(b) {
    G_b <- matrix(rnorm(N_kept * K_), N_kept, K_)
    Z_b <- Z_hat_red + G_b %*% L
    fit_b <- tryCatch(fit_latent_amm(Z_b, cov_kept), error = function(e) NULL)
    ## Boundary failure => spline at zero => T_b = 0 (principled, not discarded)
    if (is.null(fit_b)) return(0)  # <-- was NA_real_
    if (statistic == "iss") {
      sm_b <- evaluate_smooth(fit_b, covariate_name, grid,
                              include_linear = (null_type == "full"))
      sum(sm_b^2)
    } else {
      fit_b_red <- tryCatch(
        fit_latent_amm_drop_smooth(Z_b, cov_kept, drop = covariate_name),
        error = function(e) NULL
      )
      if (is.null(fit_b_red)) return(0)  # <-- was NA_real_
      compute_lrt(fit_b$fits, fit_b_red$fits)
    }
  }

  ## Parallelization: PSOCK multisession (via future.apply) instead of fork.
  ## Fork (mclapply) is unsafe on macOS in this session because reticulate's
  ## init_python() ran at script startup; forking that into a child crashes
  ## the worker silently. PSOCK starts fresh R processes, side-stepping the
  ## issue. future.apply auto-detects and exports globals.
  T_boot <- if (n_cores > 1) {
    old_plan <- future::plan(future::multisession, workers = n_cores)
    on.exit(future::plan(old_plan), add = TRUE)
    unlist(future.apply::future_lapply(
      seq_len(B), boot_one,
      future.seed = TRUE
    ))
  } else {
    sapply(seq_len(B), boot_one)
  }
  T_boot_ok <- T_boot[!is.na(T_boot)]
  p_value <- (1 + sum(T_boot_ok >= T_obs)) / (1 + length(T_boot_ok))

  list(T_obs = T_obs, T_boot = T_boot, p_value = p_value,
       B = B, B_effective = length(T_boot_ok),
       statistic = statistic, null_type = null_type)
}


## ---------- Driver ----------

## Reduced fits used by Tests 3 and 4
amm_drop_smooth_age <- fit_latent_amm_drop_smooth(Z_amm, covariates_amm, drop = "age",
                                                   n_knots_age = 10, n_knots_pir = 10)
amm_drop_smooth_pir <- fit_latent_amm_drop_smooth(Z_amm, covariates_amm, drop = "pir",
                                                   n_knots_age = 10, n_knots_pir = 10)

## Test 1: Hotelling for linear effects
hot_male <- test_linear(amm, "male")
hot_age  <- test_linear(amm, "age")
hot_pir  <- test_linear(amm, "pir")

## Test 2: Wood F via mgcv
wood_age <- test_smooth_wood(Z_amm[amm$keep, ],
                             covariates_amm[amm$keep, ], "age")
wood_pir <- test_smooth_wood(Z_amm[amm$keep, ],
                             covariates_amm[amm$keep, ], "pir")

## Test 3: Self & Liang
sl_age <- test_smooth_self_liang(amm, amm_drop_smooth_age)
sl_pir <- test_smooth_self_liang(amm, amm_drop_smooth_pir)

## Test 4: Crainiceanu-Ruppert exact RLRT (per-column via RLRsim, joint via
## convolution/Fisher/Bonferroni). Modern update to Self & Liang's
## asymptotic mixture; same per-column null hypothesis (VC-only).
cr_age <- test_smooth_crainiceanu_ruppert(amm, amm_drop_smooth_age,
                                           covariates_amm, "age",
                                           B = 10000, seed_base = 12345)
cr_pir <- test_smooth_crainiceanu_ruppert(amm, amm_drop_smooth_pir,
                                           covariates_amm, "pir",
                                           B = 10000, seed_base = 12345)


## Roy-Bose simultaneous CIs for linear effects (companion to test_linear)
ci_male <- intervals_linear(amm, "male", alpha = 0.05)
ci_age  <- intervals_linear(amm, "age",  alpha = 0.05)
ci_pir  <- intervals_linear(amm, "pir",  alpha = 0.05)


## Test 5: Parametric bootstrap (ISS and LRT; both with VC-only null)
# grid_age <- seq(min(covariates_amm$age, na.rm = TRUE),
#                 max(covariates_amm$age, na.rm = TRUE), length.out = 200)
# grid_pir <- seq(min(covariates_amm$pir, na.rm = TRUE),
#                 max(covariates_amm$pir, na.rm = TRUE), length.out = 200)
# B_amm <- 100  # development; raise for final
# set.seed(12345)
# boot_age_iss <- boot_test_smooth(amm, amm_drop_smooth_age, covariates_amm, "age",
#                                   grid = grid_age, statistic = "iss",
#                                   null_type = "vc_only", B = B_amm, n_cores = 5)
# boot_age_lrt <- boot_test_smooth(amm, amm_drop_smooth_age, covariates_amm, "age",
#                                   statistic = "lrt", null_type = "vc_only", B = B_amm, n_cores = 5)
# boot_pir_iss <- boot_test_smooth(amm, amm_drop_smooth_pir, covariates_amm, "pir",
#                                   grid = grid_pir, statistic = "iss",
#                                   null_type = "vc_only", B = B_amm, n_cores = 5)
# boot_pir_lrt <- boot_test_smooth(amm, amm_drop_smooth_pir, covariates_amm, "pir",
#                                   statistic = "lrt", null_type = "vc_only", B = B_amm, n_cores = 5)
# saveRDS(
#   list(
#     boot_age_iss = boot_age_iss, boot_age_lrt = boot_age_lrt,
#     boot_pir_iss = boot_pir_iss, boot_pir_lrt = boot_pir_lrt
#   ),
#   file.path('artifacts', dir_art, 'amm_bootstraps.rds')
# )

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
    ## age (smooth) block
    "age (smooth)", "age (smooth)", "age (smooth)",
    "age (smooth)", "age (smooth)", "age (smooth)",
    "age (smooth)", "age (smooth)",
    ## pir (smooth) block
    "pir (smooth)", "pir (smooth)", "pir (smooth)",
    "pir (smooth)", "pir (smooth)", "pir (smooth)",
    "pir (smooth)", "pir (smooth)"
  ),
  test      = c(
    "Hotelling", "Hotelling", "Hotelling",
    "Wood F (Fisher)", "Wood F (Bonferroni)", "Self & Liang",
    "Crainiceanu-Ruppert (convolution)",
    "Crainiceanu-Ruppert (Fisher)",
    "Crainiceanu-Ruppert (Bonferroni)",
    "Parametric boot (ISS)", "Parametric boot (LRT)",
    "Wood F (Fisher)", "Wood F (Bonferroni)", "Self & Liang",
    "Crainiceanu-Ruppert (convolution)",
    "Crainiceanu-Ruppert (Fisher)",
    "Crainiceanu-Ruppert (Bonferroni)",
    "Parametric boot (ISS)", "Parametric boot (LRT)"
  ),
  statistic = c(
    hot_male$F_stat, hot_age$F_stat, hot_pir$F_stat,
    NA, NA, sl_age$Lambda,
    cr_age$Lambda_joint, cr_age$X_Fisher, min(cr_age$p_per_k) * cr_age$K,
    boot_age_iss$T_obs, boot_age_lrt$T_obs,
    NA, NA, sl_pir$Lambda,
    cr_pir$Lambda_joint, cr_pir$X_Fisher, min(cr_pir$p_per_k) * cr_pir$K,
    boot_pir_iss$T_obs, boot_pir_lrt$T_obs
  ),
  p_value   = c(
    hot_male$p_value, hot_age$p_value, hot_pir$p_value,
    wood_age$p_fisher, wood_age$p_bonferroni, sl_age$p_value,
    cr_age$p_conv, cr_age$p_Fisher, cr_age$p_Bonf,
    boot_age_iss$p_value, boot_age_lrt$p_value,
    wood_pir$p_fisher, wood_pir$p_bonferroni, sl_pir$p_value,
    cr_pir$p_conv, cr_pir$p_Fisher, cr_pir$p_Bonf,
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

## Bootstrap histograms
plot_boot_hist <- function(boot_res, title, path) {
  png(path, width = 720, height = 480, pointsize = 14)
  par(mar = c(4, 4, 3, 1))
  hist(boot_res$T_boot, breaks = 40, col = "gray85", border = "white",
       xlab = "T", main = title,
       xlim = range(c(boot_res$T_boot, boot_res$T_obs), na.rm = TRUE))
  abline(v = boot_res$T_obs, col = "red", lwd = 2)
  mtext(sprintf("T_obs = %.3g    p = %.3g    B_eff = %d",
                boot_res$T_obs, boot_res$p_value, boot_res$B_effective),
        side = 3, line = 0.2, cex = 0.9)
  dev.off()
}
plot_boot_hist(boot_age_iss, "Bootstrap (age, ISS, VC-only)",
               file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_age_iss.png'))
plot_boot_hist(boot_age_lrt, "Bootstrap (age, LRT, VC-only)",
               file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_age_lrt.png'))
plot_boot_hist(boot_pir_iss, "Bootstrap (pir, ISS, VC-only)",
               file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_pir_iss.png'))
plot_boot_hist(boot_pir_lrt, "Bootstrap (pir, LRT, VC-only)",
               file.path('artifacts', dir_art, 'plots', 'amm_boot_hist_pir_lrt.png'))

## Self & Liang mixture-weights plots
plot_sl_mixture <- function(sl_res, title, path) {
  png(path, width = 720, height = 480, pointsize = 14)
  par(mar = c(4, 4, 3, 1))
  barplot(sl_res$mixture_weights,
          names.arg = sl_res$mixture_dfs,
          xlab = "chi^2 degrees of freedom", ylab = "weight",
          col = "steelblue", border = "white",
          main = title)
  mtext(sprintf("Lambda = %.3g    p = %.3g    K = %d",
                sl_res$Lambda, sl_res$p_value, sl_res$K),
        side = 3, line = 0.2, cex = 0.9)
  dev.off()
}
plot_sl_mixture(sl_age, "Self & Liang mixture (age)",
                file.path('artifacts', dir_art, 'plots', 'amm_sl_mixture_age.png'))
plot_sl_mixture(sl_pir, "Self & Liang mixture (pir)",
                file.path('artifacts', dir_art, 'plots', 'amm_sl_mixture_pir.png'))


## Crainiceanu-Ruppert: per-column null distributions (faceted) and joint
## null distribution histograms, matching the SL-mixture aesthetic.

plot_cr_per_k <- function(cr_res, title, path) {
  K_ <- cr_res$K
  nc <- ceiling(sqrt(K_))
  nr <- ceiling(K_ / nc)
  png(path, width = 200 * nc, height = 200 * nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(1.8, 0.6, 0))
  for (k in seq_len(K_)) {
    xlim_k <- range(c(cr_res$null_samples[, k], cr_res$Lambda_per_k[k]),
                    finite = TRUE)
    hist(cr_res$null_samples[, k],
         breaks = 40, col = "gray85", border = "white",
         xlab = expression(Lambda[k]), main = paste0("k = ", k),
         xlim = xlim_k)
    abline(v = cr_res$Lambda_per_k[k], col = "red", lwd = 2)
    mtext(sprintf("p = %.3g", cr_res$p_per_k[k]),
          side = 3, line = 0.2, cex = 0.8)
  }
  mtext(title, outer = TRUE, cex = 1.1, font = 2)
  dev.off()
}

plot_cr_joint <- function(cr_res, title, path) {
  png(path, width = 720, height = 480, pointsize = 14)
  par(mar = c(4, 4, 3, 1))
  hist(cr_res$joint_null_sample, breaks = 40,
       col = "gray85", border = "white",
       xlab = expression(Lambda), main = title,
       xlim = range(c(cr_res$joint_null_sample, cr_res$Lambda_joint),
                    finite = TRUE))
  abline(v = cr_res$Lambda_joint, col = "red", lwd = 2)
  mtext(sprintf("Lambda = %.3g    p (conv) = %.3g    p (Fisher) = %.3g    p (Bonf) = %.3g",
                cr_res$Lambda_joint, cr_res$p_conv,
                cr_res$p_Fisher, cr_res$p_Bonf),
        side = 3, line = 0.2, cex = 0.85)
  dev.off()
}

plot_cr_per_k(cr_age, "Crainiceanu-Ruppert per-column null (age)",
              file.path('artifacts', dir_art, 'plots', 'amm_cr_per_k_age.png'))
plot_cr_per_k(cr_pir, "Crainiceanu-Ruppert per-column null (pir)",
              file.path('artifacts', dir_art, 'plots', 'amm_cr_per_k_pir.png'))
plot_cr_joint(cr_age, "Crainiceanu-Ruppert joint null (age)",
              file.path('artifacts', dir_art, 'plots', 'amm_cr_joint_age.png'))
plot_cr_joint(cr_pir, "Crainiceanu-Ruppert joint null (pir)",
              file.path('artifacts', dir_art, 'plots', 'amm_cr_joint_pir.png'))



## ========== AMM Interpretation ========== ##

## ---------- Interpretation helpers ----------

compute_conditional_mean <- function(fit_result, covariate_values) {
  beta <- fit_result$beta_hat                  # (A+1) x K
  K_   <- ncol(beta)
  out  <- beta["(Intercept)", , drop = TRUE]   # length K

  if ("male" %in% rownames(beta) && !is.null(covariate_values$male)) {
    out <- out + covariate_values$male * beta["male", , drop = TRUE]
  }
  if ("age" %in% fit_result$smooth_covariates && !is.null(covariate_values$age)) {
    out <- out + as.numeric(evaluate_smooth(fit_result, "age",
                                            covariate_values$age,
                                            include_linear = TRUE))
  } else if ("age" %in% rownames(beta) && !is.null(covariate_values$age)) {
    out <- out + covariate_values$age * beta["age", , drop = TRUE]
  }
  if ("pir" %in% fit_result$smooth_covariates && !is.null(covariate_values$pir)) {
    out <- out + as.numeric(evaluate_smooth(fit_result, "pir",
                                            covariate_values$pir,
                                            include_linear = TRUE))
  } else if ("pir" %in% rownames(beta) && !is.null(covariate_values$pir)) {
    out <- out + covariate_values$pir * beta["pir", , drop = TRUE]
  }
  out
}

compute_smooth_edf <- function(fit_result, covariate_name) {
  fit_result[[paste0("edf_", covariate_name)]]
}

## ---------- AMM parametric bootstrap (interpretation) ----------

## Conditional parametric bootstrap from the fitted AMM: simulate full
## synthetic datasets Z* = mu_hat + E* (mu_hat = X*beta + W_age*U_age +
## W_pir*U_pir, the fitted mean; E* ~ N(0, Sigma_E) row-wise), refit
## fit_latent_amm on each, and expose lightweight evaluators that produce
## the same B x K / J x K x B shapes the prior mgcv helpers did.
##
## Mirrors the group-comparison parametric bootstrap pattern at
## scripts/demo_nhanes.R:826-873: fitted mean + resampled residuals + refit,
## extended to the AMM with fit_latent_amm as the per-replicate primitive.
##
## Parallelism uses future::multisession + future.apply::future_lapply,
## matching boot_test_smooth's pattern (PSOCK workers; avoids the
## reticulate-fork crash on macOS).

## Wrap a replicate's coefficient triple into a fit-like object so the
## existing AMM-native helpers (evaluate_smooth, compute_conditional_mean)
## work on it unchanged. Design matrices and basis attributes are shared
## by reference from the original fit.
make_replicate <- function(amm_fit, beta_hat_r, U_age_hat_r, U_pir_hat_r) {
  list(
    beta_hat          = beta_hat_r,
    U_age_hat         = U_age_hat_r,
    U_pir_hat         = U_pir_hat_r,
    W_age             = amm_fit$W_age,
    W_pir             = amm_fit$W_pir,
    smooth_covariates = amm_fit$smooth_covariates,
    linear_covariates = amm_fit$linear_covariates
  )
}

## Simulate one synthetic Z from the fitted AMM (Option A: BLUPs held as
## parameters, residuals resampled from chol(Sigma_E)).
amm_simulate_one <- function(amm_fit) {
  N_kept <- sum(amm_fit$keep)
  K_     <- ncol(amm_fit$beta_hat)
  mu     <- amm_fit$X_design %*% amm_fit$beta_hat
  if (!is.null(amm_fit$U_age_hat)) mu <- mu + amm_fit$W_age %*% amm_fit$U_age_hat
  if (!is.null(amm_fit$U_pir_hat)) mu <- mu + amm_fit$W_pir %*% amm_fit$U_pir_hat
  L  <- chol(amm_fit$Sigma_E)
  G  <- matrix(rnorm(N_kept * K_), N_kept, K_)
  mu + G %*% L
}

## B parametric-bootstrap replicates of the fitted AMM. Returns a length-B
## list of fit-like objects (made via make_replicate). On full-refit failure
## the replicate is zero-coefficient (principled "this replicate contributes
## nothing", matching boot_test_smooth's convention).
amm_boot <- function(amm_fit, covariates_kept, B,
                     n_cores = 1, seed = 12345) {
  set.seed(seed)
  boot_one <- function(b) {
    Z_b   <- amm_simulate_one(amm_fit)
    fit_b <- tryCatch(fit_latent_amm(Z_b, covariates_kept),
                      error = function(e) NULL)
    if (is.null(fit_b)) return(NULL)
    list(
      beta_hat  = fit_b$beta_hat,
      U_age_hat = fit_b$U_age_hat,
      U_pir_hat = fit_b$U_pir_hat
    )
  }

  reps <- if (n_cores > 1) {
    old_plan <- future::plan(future::multisession, workers = n_cores)
    on.exit(future::plan(old_plan), add = TRUE)
    future.apply::future_lapply(seq_len(B), boot_one, future.seed = TRUE)
  } else {
    lapply(seq_len(B), boot_one)
  }

  zero_beta  <- matrix(0, nrow = nrow(amm_fit$beta_hat),
                       ncol = ncol(amm_fit$beta_hat),
                       dimnames = dimnames(amm_fit$beta_hat))
  zero_U_age <- if (!is.null(amm_fit$U_age_hat))
                  matrix(0, nrow = nrow(amm_fit$U_age_hat),
                            ncol = ncol(amm_fit$U_age_hat)) else NULL
  zero_U_pir <- if (!is.null(amm_fit$U_pir_hat))
                  matrix(0, nrow = nrow(amm_fit$U_pir_hat),
                            ncol = ncol(amm_fit$U_pir_hat)) else NULL

  n_failed <- sum(vapply(reps, is.null, logical(1)))
  if (n_failed > 0) {
    message(str_glue("  amm_boot: {n_failed} of {B} replicates failed to refit; ",
                     "filled with zero coefficients."))
  }

  lapply(reps, function(r) {
    if (is.null(r)) {
      make_replicate(amm_fit, zero_beta, zero_U_age, zero_U_pir)
    } else {
      make_replicate(amm_fit, r$beta_hat, r$U_age_hat, r$U_pir_hat)
    }
  })
}

## B x K matrix of bootstrap latent conditional means at covariate_values
## (replaces sample_z_boot).
replicate_z_at <- function(boot_reps, covariate_values) {
  do.call(rbind, lapply(boot_reps, function(rep_) {
    as.numeric(compute_conditional_mean(rep_, covariate_values))
  }))
}

## J x K x B array of bootstrap smooth-curve evaluations on x_grid
## (replaces sample_smooth_boot).
replicate_smooth_at <- function(boot_reps, covariate_name, x_grid,
                                include_linear = TRUE) {
  B_ <- length(boot_reps)
  J_ <- length(x_grid)
  K_ <- ncol(boot_reps[[1]]$beta_hat)
  out <- array(NA_real_, dim = c(J_, K_, B_))
  for (b in seq_len(B_)) {
    out[, , b] <- evaluate_smooth(boot_reps[[b]], covariate_name,
                                  x_grid, include_linear = include_linear)
  }
  out
}


## ---------- Latent-space effect curves ----------

plot_latent_effect_curves <- function(fit_result, covariate_name,
                                       x_grid = NULL, n_grid = 100,
                                       band = FALSE, alpha = 0.05,
                                       boot_reps = NULL,
                                       covariates_df = NULL,
                                       ncol = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  if (is.null(W_obj)) stop(str_glue("No smooth for '{covariate_name}' in fit_result."))
  range_x <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(range_x[1], range_x[2], length.out = n_grid)

  K_    <- ncol(fit_result$beta_hat)
  f_mat <- evaluate_smooth(fit_result, covariate_name, x_grid, include_linear = TRUE)
  edf_k <- compute_smooth_edf(fit_result, covariate_name)
  beta_lin <- fit_result$beta_hat[covariate_name, ]

  ## Optional bootstrap band
  band_lo <- band_hi <- NULL
  if (isTRUE(band) && !is.null(boot_reps)) {
    sm_arr <- replicate_smooth_at(boot_reps, covariate_name, x_grid,
                                  include_linear = TRUE)
    band_lo <- apply(sm_arr, c(1, 2), quantile, probs = alpha / 2)       # J x K
    band_hi <- apply(sm_arr, c(1, 2), quantile, probs = 1 - alpha / 2)
  }

  nc <- ncol %||% ceiling(sqrt(K_))
  nr <- ceiling(K_ / nc)
  old_par <- par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1), mgp = c(1.8, 0.6, 0))
  on.exit(par(old_par), add = TRUE)

  rug_x <- if (!is.null(covariates_df)) covariates_df[[covariate_name]][fit_result$keep]
           else NULL

  for (k in seq_len(K_)) {
    y_vals <- c(f_mat[, k], beta_lin[k] * x_grid, band_lo[, k], band_hi[, k])
    y_lim  <- range(y_vals, finite = TRUE)
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = covariate_name, ylab = "f(x)",
         main = sprintf("k = %d (edf = %.2f)", k, edf_k[k]))
    if (!is.null(band_lo)) {
      lines(x_grid, band_lo[, k], col = "gray60", lwd = 0.5, lty = 3)
      lines(x_grid, band_hi[, k], col = "gray60", lwd = 0.5, lty = 3)
    }
    abline(h = 0, col = "gray85", lty = 1)
    lines(x_grid, beta_lin[k] * x_grid, col = "gray40", lwd = 1.5, lty = 2)
    lines(x_grid, f_mat[, k], col = "black", lwd = 2)
    if (!is.null(rug_x)) rug(rug_x, col = rgb(0, 0, 0, 0.2))
  }
  invisible(NULL)
}


## ---------- Decoded-space helpers ----------

decode_z_to_Qi_vec <- function(z, Ji) {
  decode_z_rot_to_Qi(list(as.numeric(z)), Ji)[[1]]
}

## Compute decoded Qi-band at one covariate combination by drawing the
## replicate latent z's, decoding each, and taking pointwise quantiles.
## Returns J x 3 matrix with columns (mean_hat, lo, hi) on pi_grid.
compute_band_decoded_qf <- function(fit_result, boot_reps, covariate_values,
                                     alpha, Ji) {
  z_hat <- compute_conditional_mean(fit_result, covariate_values)
  Qi_hat <- decode_z_to_Qi_vec(z_hat, Ji)

  Z_boot <- replicate_z_at(boot_reps, covariate_values)
  B_     <- nrow(Z_boot)
  Qi_boot <- vapply(seq_len(B_),
                    function(b) decode_z_to_Qi_vec(Z_boot[b, ], Ji),
                    numeric(Ji))                                            # Ji x B

  band_lo <- apply(Qi_boot, 1, quantile, probs = alpha / 2)
  band_hi <- apply(Qi_boot, 1, quantile, probs = 1 - alpha / 2)
  cbind(mean_hat = Qi_hat, lo = band_lo, hi = band_hi)
}

## Same but for the contrast Qi(x) - Qi(x_ref). Band uses within-replicate
## subtraction so it pinches at the reference.
compute_band_decoded_qf_contrast <- function(fit_result, boot_reps,
                                              covariate_name, x_value,
                                              alpha, Ji,
                                              age_ref = 40, pir_ref = 1,
                                              male_ref = 0) {
  cov_x   <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cov_ref <- cov_x
  cov_x[[covariate_name]] <- x_value
  cov_ref[[covariate_name]] <- switch(covariate_name, age = age_ref, pir = pir_ref)

  z_x   <- compute_conditional_mean(fit_result, cov_x)
  z_ref <- compute_conditional_mean(fit_result, cov_ref)
  Qi_x   <- decode_z_to_Qi_vec(z_x,   Ji)
  Qi_ref <- decode_z_to_Qi_vec(z_ref, Ji)
  Qi_hat_diff <- Qi_x - Qi_ref

  Zb_x   <- replicate_z_at(boot_reps, cov_x)
  Zb_ref <- replicate_z_at(boot_reps, cov_ref)
  B_     <- nrow(Zb_x)
  Qi_diff_boot <- vapply(seq_len(B_), function(b) {
    decode_z_to_Qi_vec(Zb_x[b, ], Ji) - decode_z_to_Qi_vec(Zb_ref[b, ], Ji)
  }, numeric(Ji))

  band_lo <- apply(Qi_diff_boot, 1, quantile, probs = alpha / 2)
  band_hi <- apply(Qi_diff_boot, 1, quantile, probs = 1 - alpha / 2)
  cbind(mean_hat = Qi_hat_diff, lo = band_lo, hi = band_hi)
}

## Sequential blue-gray-red palette indexed by an ordered set of x values.
.x_palette <- function(x_values, ref = NULL) {
  ramp <- colorRampPalette(c("blue", "gray60", "red"))(101)
  if (is.null(ref)) {
    ## Normalize x to (0, 1) by rank
    r <- rank(x_values, ties.method = "min")
    idx <- round((r - 1) / max(r - 1, 1) * 100) + 1
  } else {
    rng <- max(abs(x_values - ref))
    if (rng == 0) rng <- 1
    norm <- (x_values - ref) / (2 * rng) + 0.5
    idx  <- pmin(pmax(round(norm * 100) + 1, 1), 101)
  }
  ramp[idx]
}


## ---------- QF family ----------

.default_x_values <- function(covariate_name, faceted = FALSE) {
  if (covariate_name == "age") {
    if (faceted) c(10, 30, 50, 70) else c(10, 20, 30, 40, 50, 60, 70, 80)
  } else if (covariate_name == "pir") {
    if (faceted) c(0.25, 0.5, 2, 4) else c(0.25, 0.5, 1, 2, 4, 5)
  } else stop(str_glue("No default x_values for '{covariate_name}'."))
}

plot_conditional_qf <- function(fit_result, covariate_name, x_values = NULL,
                                 age_ref = 40, pir_ref = 1, male_ref = 0,
                                 Ji = NULL,
                                 quantile_lines = NULL,
                                 quantile_line_cols = quantile_level_cols,
                                 quantile_line_labels = quantile_level_labels) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  pi_grid <- pi_grid_fun(Ji)

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
  })
  colors <- .x_palette(x_values)

  grid_list <- rep(list(pi_grid), length(Qi_list))
  cwt_lbl   <- data.frame(
    color = c(colors,
              if (!is.null(quantile_lines)) quantile_line_cols else NULL),
    width = c(rep(4, length(x_values)),
              if (!is.null(quantile_lines)) rep(1.5, length(quantile_lines)) else NULL),
    type  = c(rep(1, length(x_values)),
              if (!is.null(quantile_lines)) rep(4, length(quantile_lines)) else NULL),
    label = c(sprintf("%s = %g", covariate_name, x_values),
              if (!is.null(quantile_lines)) quantile_line_labels else NULL)
  )
  plot_funs(
    fun_list  = Qi_list,
    grid_list = grid_list,
    colors    = colors,
    widths    = rep(4, length(x_values)),
    types     = rep(1, length(x_values)),
    ylab      = "Q(p)",
    color_width_type_labels = cwt_lbl,
    main      = sprintf("Conditional QFs by %s", covariate_name)
  )
  if (!is.null(quantile_lines)) {
    abline(v = quantile_lines, col = quantile_line_cols, lty = 4, lwd = 1.5)
  }
  invisible(NULL)
}

plot_conditional_qf_faceted <- function(fit_result, covariate_name, x_values = NULL,
                                         age_ref = 40, pir_ref = 1, male_ref = 0,
                                         Ji = NULL, alpha = 0.05,
                                         boot_reps = NULL,
                                         quantile_lines = NULL,
                                         quantile_line_cols = quantile_level_cols) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  pi_grid <- pi_grid_fun(Ji)

  bands <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    compute_band_decoded_qf(fit_result, boot_reps, cv, alpha, Ji)
  })
  y_lim <- range(sapply(bands, function(b) range(b)))

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p", ylab = "Q(p)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    if (!is.null(quantile_lines)) {
      abline(v = quantile_lines, col = quantile_line_cols, lty = 4, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "lo"], col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "hi"], col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}

plot_contrast_qf <- function(fit_result, covariate_name, x_values = NULL,
                              age_ref = 40, pir_ref = 1, male_ref = 0,
                              Ji = NULL,
                              quantile_lines = NULL,
                              quantile_line_cols = quantile_level_cols,
                              quantile_line_labels = quantile_level_labels) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  pi_grid <- pi_grid_fun(Ji)
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  Qi_ref <- {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- ref_val
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
  }
  contrasts <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji) - Qi_ref
  })
  colors <- .x_palette(x_values, ref = ref_val)

  y_lim <- range(unlist(contrasts), 0)
  plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p", ylab = "Q(p) contrast",
       main = sprintf("QF contrasts vs %s = %g", covariate_name, ref_val))
  if (!is.null(quantile_lines)) {
    abline(v = quantile_lines, col = quantile_line_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(pi_grid, contrasts[[i]], col = colors[i], lwd = 2)
  }
  legend("topleft",
         legend = c(sprintf("%s = %g", covariate_name, x_values),
                    if (!is.null(quantile_lines)) quantile_line_labels else NULL),
         col    = c(colors,
                    if (!is.null(quantile_lines)) quantile_line_cols else NULL),
         lwd    = c(rep(2, length(colors)),
                    if (!is.null(quantile_lines)) rep(1.5, length(quantile_lines)) else NULL),
         lty    = c(rep(1, length(colors)),
                    if (!is.null(quantile_lines)) rep(4, length(quantile_lines)) else NULL),
         bty = "n")
  invisible(NULL)
}

plot_contrast_qf_faceted <- function(fit_result, covariate_name, x_values = NULL,
                                      age_ref = 40, pir_ref = 1, male_ref = 0,
                                      Ji = NULL, alpha = 0.05,
                                      boot_reps = NULL,
                                      quantile_lines = NULL,
                                      quantile_line_cols = quantile_level_cols) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  pi_grid <- pi_grid_fun(Ji)

  bands <- lapply(x_values, function(x) {
    compute_band_decoded_qf_contrast(fit_result, boot_reps, covariate_name, x,
                                     alpha, Ji,
                                     age_ref = age_ref, pir_ref = pir_ref,
                                     male_ref = male_ref)
  })
  y_lim <- range(sapply(bands, function(b) range(b)), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p",
         ylab = "Q(p) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    abline(h = 0, col = "red", lty = 2)
    if (!is.null(quantile_lines)) {
      abline(v = quantile_lines, col = quantile_line_cols, lty = 4, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "lo"], col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "hi"], col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}


## ---------- CDF family ----------

.default_y_grid <- function(Qi_list, n = 1000) {
  rng <- range(unlist(Qi_list))
  sort(unique(c(rng[1], unlist(Qi_list), rng[2])))
}

plot_conditional_cdf <- function(fit_result, covariate_name, x_values = NULL,
                                  y_grid = NULL,
                                  age_ref = 40, pir_ref = 1, male_ref = 0,
                                  Ji = NULL,
                                  threshold_lines = mims_thresholds) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .default_y_grid(Qi_list)
  F_list <- lapply(Qi_list, qi_to_cdf, y_grid = y_grid)
  colors <- .x_palette(x_values)

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)",
       main = sprintf("Conditional CDFs by %s", covariate_name))
  if (!is.null(threshold_lines)) {
    abline(v = threshold_lines, col = mims_threshold_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(y_grid, F_list[[i]], col = colors[i], lwd = 2)
  }
  legend("topleft",
         legend = c(sprintf("%s = %g", covariate_name, x_values),
                    if (!is.null(threshold_lines)) mims_threshold_labels else NULL),
         col    = c(colors,
                    if (!is.null(threshold_lines)) mims_threshold_cols else NULL),
         lwd    = c(rep(2, length(colors)),
                    if (!is.null(threshold_lines)) rep(1.5, length(threshold_lines)) else NULL),
         lty    = c(rep(1, length(colors)),
                    if (!is.null(threshold_lines)) rep(4,   length(threshold_lines)) else NULL),
         bty = "n")
  invisible(NULL)
}

plot_conditional_cdf_faceted <- function(fit_result, covariate_name, x_values = NULL,
                                          y_grid = NULL,
                                          age_ref = 40, pir_ref = 1, male_ref = 0,
                                          Ji = NULL,
                                          threshold_lines = mims_thresholds,
                                          alpha = 0.05,
                                          boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")

  ## Decoded Qi bands per x
  Qi_bands <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    compute_band_decoded_qf(fit_result, boot_reps, cv, alpha, Ji)
  })
  if (is.null(y_grid)) y_grid <- .default_y_grid(lapply(Qi_bands, function(b) b[, "mean_hat"]))

  F_means <- lapply(Qi_bands, function(b) qi_to_cdf(b[, "mean_hat"], y_grid))
  ## Bands via per-replicate decode (recompute to get per-y bands)
  cv_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    cv
  })
  Z_boots <- lapply(cv_list, function(cv) replicate_z_at(boot_reps, cv))
  F_bands <- lapply(seq_along(x_values), function(i) {
    Zb <- Z_boots[[i]]
    B_ <- nrow(Zb)
    Fb <- vapply(seq_len(B_),
                 function(b) qi_to_cdf(decode_z_to_Qi_vec(Zb[b, ], Ji), y_grid),
                 numeric(length(y_grid)))
    list(lo = apply(Fb, 1, quantile, probs = alpha / 2),
         hi = apply(Fb, 1, quantile, probs = 1 - alpha / 2))
  })

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
         xlab = "y", ylab = "F(y)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    if (!is.null(threshold_lines)) {
      abline(v = threshold_lines, col = mims_threshold_cols, lty = 4, lwd = 1.5)
    }
    lines(y_grid, F_bands[[i]]$lo, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_bands[[i]]$hi, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_means[[i]],    col = "black", lwd = 4)
  }
  invisible(NULL)
}

plot_contrast_cdf <- function(fit_result, covariate_name, x_values = NULL,
                               y_grid = NULL,
                               age_ref = 40, pir_ref = 1, male_ref = 0,
                               Ji = NULL,
                               threshold_lines = mims_thresholds) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref <- decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv_ref), Ji)

  Qi_list <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .default_y_grid(c(list(Qi_ref), Qi_list))
  F_ref <- qi_to_cdf(Qi_ref, y_grid)
  F_diff <- lapply(Qi_list, function(Qi) qi_to_cdf(Qi, y_grid) - F_ref)
  colors <- .x_palette(x_values, ref = ref_val)

  y_lim <- range(unlist(F_diff), 0)
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "F(y) contrast",
       main = sprintf("CDF contrasts vs %s = %g", covariate_name, ref_val))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(threshold_lines)) {
    abline(v = threshold_lines, col = mims_threshold_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(y_grid, F_diff[[i]], col = colors[i], lwd = 2)
  }
  legend("topright", legend = sprintf("%s = %g", covariate_name, x_values),
         col = colors, lwd = 2, bty = "n")
  invisible(NULL)
}

plot_contrast_cdf_faceted <- function(fit_result, covariate_name, x_values = NULL,
                                       y_grid = NULL,
                                       age_ref = 40, pir_ref = 1, male_ref = 0,
                                       Ji = NULL,
                                       threshold_lines = mims_thresholds,
                                       alpha = 0.05,
                                       boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref_pt <- decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv_ref), Ji)

  ## Determine y_grid from all point-estimate Qi
  Qi_x_pt <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .default_y_grid(c(list(Qi_ref_pt), Qi_x_pt))
  F_ref_pt <- qi_to_cdf(Qi_ref_pt, y_grid)

  Z_ref_boot <- replicate_z_at(boot_reps, cv_ref)
  B_         <- nrow(Z_ref_boot)
  F_diff_means <- lapply(Qi_x_pt, function(Qi) qi_to_cdf(Qi, y_grid) - F_ref_pt)

  band_list <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    Zb_x <- replicate_z_at(boot_reps, cv)
    Fd_b <- vapply(seq_len(B_), function(b) {
      qi_to_cdf(decode_z_to_Qi_vec(Zb_x[b, ], Ji), y_grid) -
      qi_to_cdf(decode_z_to_Qi_vec(Z_ref_boot[b, ], Ji), y_grid)
    }, numeric(length(y_grid)))
    list(lo = apply(Fd_b, 1, quantile, probs = alpha / 2),
         hi = apply(Fd_b, 1, quantile, probs = 1 - alpha / 2))
  })
  y_lim <- range(unlist(band_list), unlist(F_diff_means), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = y_lim,
         xlab = "y", ylab = "F(y) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    abline(h = 0, col = "red", lty = 2)
    if (!is.null(threshold_lines)) {
      abline(v = threshold_lines, col = mims_threshold_cols, lty = 4, lwd = 1.5)
    }
    lines(y_grid, band_list[[i]]$lo, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, band_list[[i]]$hi, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_diff_means[[i]], col = "black", lwd = 4)
  }
  invisible(NULL)
}


## ---------- Moments family ----------

.moments_from_Qi <- function(Qi) {
  mu <- mean(Qi)
  s2 <- mean((Qi - mu)^2)
  sd <- sqrt(s2)
  c(mean = mu, variance = s2,
    skewness = mean((Qi - mu)^3) / sd^3,
    kurtosis = mean((Qi - mu)^4) / sd^4)
}

.compute_moment_curve <- function(fit_result, covariate_name, x_grid,
                                   age_ref, pir_ref, male_ref, Ji) {
  M <- matrix(NA_real_, nrow = length(x_grid), ncol = 4,
              dimnames = list(NULL, c("mean", "variance", "skewness", "kurtosis")))
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
    M[j, ] <- .moments_from_Qi(Qi)
  }
  M
}

.compute_moment_bands <- function(fit_result, boot_reps, covariate_name, x_grid,
                                   age_ref, pir_ref, male_ref, Ji,
                                   alpha, contrast = FALSE) {
  J_ <- length(x_grid)
  B_ <- length(boot_reps)
  arr <- array(NA_real_, dim = c(J_, 4, B_))
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  cv_ref  <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  if (contrast) Z_ref_boot <- replicate_z_at(boot_reps, cv_ref)

  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    Zb <- replicate_z_at(boot_reps, cv)
    for (b in seq_len(B_)) {
      Qi_b <- decode_z_to_Qi_vec(Zb[b, ], Ji)
      m_b  <- .moments_from_Qi(Qi_b)
      if (contrast) {
        Qi_ref_b <- decode_z_to_Qi_vec(Z_ref_boot[b, ], Ji)
        m_b <- m_b - .moments_from_Qi(Qi_ref_b)
      }
      arr[j, , b] <- m_b
    }
  }
  lo <- apply(arr, c(1, 2), quantile, probs = alpha / 2)
  hi <- apply(arr, c(1, 2), quantile, probs = 1 - alpha / 2)
  list(lo = lo, hi = hi)
}

.plot_moment_panels <- function(x_grid, M_point, M_bands = NULL,
                                 moments, covariate_name, contrast = FALSE) {
  moment_labels <- c(mean = "Mean", variance = "Variance",
                     skewness = "Skewness", kurtosis = "Kurtosis")
  old_par <- par(mfrow = c(1, length(moments)), mar = c(4, 4.5, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (m in moments) {
    j <- match(m, colnames(M_point))
    y_vals <- M_point[, j]
    if (!is.null(M_bands)) y_vals <- c(y_vals, M_bands$lo[, j], M_bands$hi[, j])
    y_lim <- range(y_vals, if (contrast) 0 else NULL, finite = TRUE)
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = covariate_name,
         ylab = paste0(moment_labels[m], if (contrast) " contrast" else ""),
         main = moment_labels[m])
    if (contrast) abline(h = 0, col = "red", lty = 2)
    if (!is.null(M_bands)) {
      lines(x_grid, M_bands$lo[, j], col = "black", lwd = 0.5, lty = 3)
      lines(x_grid, M_bands$hi[, j], col = "black", lwd = 0.5, lty = 3)
    }
    lines(x_grid, M_point[, j], col = "black", lwd = 3)
  }
}

plot_conditional_moments <- function(fit_result, covariate_name,
                                      x_grid = NULL, n_grid = 50,
                                      age_ref = 40, pir_ref = 1, male_ref = 0,
                                      Ji = NULL,
                                      moments = c("mean", "variance",
                                                  "skewness", "kurtosis")) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  M <- .compute_moment_curve(fit_result, covariate_name, x_grid,
                             age_ref, pir_ref, male_ref, Ji)
  .plot_moment_panels(x_grid, M, NULL, moments, covariate_name, contrast = FALSE)
  invisible(NULL)
}

plot_conditional_moments_faceted <- function(fit_result, covariate_name,
                                              x_grid = NULL, n_grid = 50,
                                              age_ref = 40, pir_ref = 1, male_ref = 0,
                                              Ji = NULL,
                                              moments = c("mean", "variance",
                                                          "skewness", "kurtosis"),
                                              alpha = 0.05,
                                              boot_reps = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for band plots.")
  M  <- .compute_moment_curve(fit_result, covariate_name, x_grid,
                              age_ref, pir_ref, male_ref, Ji)
  Mb <- .compute_moment_bands(fit_result, boot_reps, covariate_name, x_grid,
                              age_ref, pir_ref, male_ref, Ji, alpha,
                              contrast = FALSE)
  .plot_moment_panels(x_grid, M, Mb, moments, covariate_name, contrast = FALSE)
  invisible(NULL)
}

plot_contrast_moments <- function(fit_result, covariate_name,
                                   x_grid = NULL, n_grid = 50,
                                   age_ref = 40, pir_ref = 1, male_ref = 0,
                                   Ji = NULL,
                                   moments = c("mean", "variance",
                                               "skewness", "kurtosis")) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  M_x   <- .compute_moment_curve(fit_result, covariate_name, x_grid,
                                 age_ref, pir_ref, male_ref, Ji)
  M_ref <- .compute_moment_curve(fit_result, covariate_name, ref_val,
                                 age_ref, pir_ref, male_ref, Ji)
  M_diff <- sweep(M_x, 2, M_ref, "-")
  .plot_moment_panels(x_grid, M_diff, NULL, moments, covariate_name, contrast = TRUE)
  invisible(NULL)
}

plot_contrast_moments_faceted <- function(fit_result, covariate_name,
                                           x_grid = NULL, n_grid = 50,
                                           age_ref = 40, pir_ref = 1, male_ref = 0,
                                           Ji = NULL,
                                           moments = c("mean", "variance",
                                                       "skewness", "kurtosis"),
                                           alpha = 0.05,
                                           boot_reps = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for band plots.")
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  M_x   <- .compute_moment_curve(fit_result, covariate_name, x_grid,
                                 age_ref, pir_ref, male_ref, Ji)
  M_ref <- .compute_moment_curve(fit_result, covariate_name, ref_val,
                                 age_ref, pir_ref, male_ref, Ji)
  M_diff <- sweep(M_x, 2, M_ref, "-")
  Mb     <- .compute_moment_bands(fit_result, boot_reps, covariate_name, x_grid,
                                  age_ref, pir_ref, male_ref, Ji, alpha,
                                  contrast = TRUE)
  .plot_moment_panels(x_grid, M_diff, Mb, moments, covariate_name, contrast = TRUE)
  invisible(NULL)
}


## ---------- Threshold-crossing ----------

plot_threshold_crossing <- function(fit_result, covariate_name, threshold_y,
                                     x_grid = NULL, n_grid = 50,
                                     age_ref = 40, pir_ref = 1, male_ref = 0,
                                     Ji = NULL, alpha = 0.05,
                                     boot_reps = NULL,
                                     pop_avg = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])

  F_curve <- numeric(length(x_grid))
  F_band  <- NULL
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
    F_curve[j] <- qi_to_cdf(Qi, threshold_y)
  }
  if (!is.null(boot_reps)) {
    B_     <- length(boot_reps)
    Fb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
      cv[[covariate_name]] <- x_grid[j]
      Zb <- replicate_z_at(boot_reps, cv)
      Fb_arr[j, ] <- vapply(seq_len(B_),
                            function(b) qi_to_cdf(decode_z_to_Qi_vec(Zb[b, ], Ji),
                                                  threshold_y),
                            numeric(1))
    }
    F_band <- list(lo = apply(Fb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Fb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(F_curve, F_band$lo, F_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, 0, 1, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("F(%g)", threshold_y),
       main = sprintf("Fraction below %g by %s", threshold_y, covariate_name))
  if (!is.null(F_band)) {
    lines(x_grid, F_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, F_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, F_curve, col = "black", lwd = 3)
  invisible(NULL)
}

plot_contrast_threshold_crossing <- function(fit_result, covariate_name, threshold_y,
                                              x_grid = NULL, n_grid = 50,
                                              age_ref = 40, pir_ref = 1, male_ref = 0,
                                              Ji = NULL, alpha = 0.05,
                                              boot_reps = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  F_ref <- qi_to_cdf(decode_z_to_Qi_vec(
                       compute_conditional_mean(fit_result, cv_ref), Ji),
                     threshold_y)

  F_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    F_curve[j] <- qi_to_cdf(decode_z_to_Qi_vec(
                              compute_conditional_mean(fit_result, cv), Ji),
                            threshold_y) - F_ref
  }

  F_band <- NULL
  if (!is.null(boot_reps)) {
    Z_ref_boot <- replicate_z_at(boot_reps, cv_ref)
    B_         <- nrow(Z_ref_boot)
    Fb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
      Zb_x <- replicate_z_at(boot_reps, cv)
      Fb_arr[j, ] <- vapply(seq_len(B_), function(b) {
        qi_to_cdf(decode_z_to_Qi_vec(Zb_x[b, ], Ji),       threshold_y) -
        qi_to_cdf(decode_z_to_Qi_vec(Z_ref_boot[b, ], Ji), threshold_y)
      }, numeric(1))
    }
    F_band <- list(lo = apply(Fb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Fb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(F_curve, F_band$lo, F_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("F(%g) contrast", threshold_y),
       main = sprintf("Contrast in fraction below %g by %s", threshold_y, covariate_name))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(F_band)) {
    lines(x_grid, F_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, F_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  lines(x_grid, F_curve, col = "black", lwd = 3)
  invisible(NULL)
}


## ---------- Quantile-crossing ----------

plot_quantile_crossing <- function(fit_result, covariate_name, p_star,
                                    x_grid = NULL, n_grid = 50,
                                    age_ref = 40, pir_ref = 1, male_ref = 0,
                                    Ji = NULL, alpha = 0.05,
                                    boot_reps = NULL,
                                    pop_avg = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])

  Q_curve <- numeric(length(x_grid))
  Q_band  <- NULL
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- decode_z_to_Qi_vec(compute_conditional_mean(fit_result, cv), Ji)
    Q_curve[j] <- qi_at_p(Qi, p_star)
  }
  if (!is.null(boot_reps)) {
    B_     <- length(boot_reps)
    Qb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
      cv[[covariate_name]] <- x_grid[j]
      Zb <- replicate_z_at(boot_reps, cv)
      Qb_arr[j, ] <- vapply(seq_len(B_),
                            function(b) qi_at_p(decode_z_to_Qi_vec(Zb[b, ], Ji),
                                                p_star),
                            numeric(1))
    }
    Q_band <- list(lo = apply(Qb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Qb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("Q(%g)", p_star),
       main = sprintf("Quantile at p = %g by %s", p_star, covariate_name))
  if (!is.null(Q_band)) {
    lines(x_grid, Q_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, Q_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, Q_curve, col = "black", lwd = 3)
  invisible(NULL)
}

plot_contrast_quantile_crossing <- function(fit_result, covariate_name, p_star,
                                             x_grid = NULL, n_grid = 50,
                                             age_ref = 40, pir_ref = 1, male_ref = 0,
                                             Ji = NULL, alpha = 0.05,
                                             boot_reps = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(Qi_ctx$payload[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Q_ref <- qi_at_p(decode_z_to_Qi_vec(
                     compute_conditional_mean(fit_result, cv_ref), Ji),
                   p_star)

  Q_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    Q_curve[j] <- qi_at_p(decode_z_to_Qi_vec(
                            compute_conditional_mean(fit_result, cv), Ji),
                          p_star) - Q_ref
  }

  Q_band <- NULL
  if (!is.null(boot_reps)) {
    Z_ref_boot <- replicate_z_at(boot_reps, cv_ref)
    B_         <- nrow(Z_ref_boot)
    Qb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
      Zb_x <- replicate_z_at(boot_reps, cv)
      Qb_arr[j, ] <- vapply(seq_len(B_), function(b) {
        qi_at_p(decode_z_to_Qi_vec(Zb_x[b, ], Ji),       p_star) -
        qi_at_p(decode_z_to_Qi_vec(Z_ref_boot[b, ], Ji), p_star)
      }, numeric(1))
    }
    Q_band <- list(lo = apply(Qb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Qb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("Q(%g) contrast", p_star),
       main = sprintf("Contrast in quantile at p = %g by %s",
                      p_star, covariate_name))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(Q_band)) {
    lines(x_grid, Q_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, Q_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  lines(x_grid, Q_curve, col = "black", lwd = 3)
  invisible(NULL)
}


## ---------- Driver ----------

Ji_interp <- length(Qi_ctx$payload[[1]])
B_interp  <- 100

amm_boots <- amm_boot(
  amm_fit         = amm,
  covariates_kept = covariates_amm[amm$keep, , drop = FALSE],
  B               = B_interp,
  n_cores         = 10,
  seed            = 12345
)

interp_dir <- file.path('artifacts', dir_art, 'plots')

## Per-covariate quantile levels for QF crossings / markings.
quantile_levels_by_cov <- list(
  age = c(q1 = 0.75, q2 = 0.99),
  pir = c(q1 = 0.75, q2 = 0.999)
)

## --- Latent-space effect curves
for (cov in c("age", "pir")) {
  path_plot <- file.path(interp_dir, str_glue("amm_latent_effect_{cov}.png"))
  png(path_plot, width = 1280, height = 960, pointsize = 14)
  plot_latent_effect_curves(amm, cov, band = TRUE,
                            boot_reps = amm_boots,
                            covariates_df = covariates_amm)
  dev.off()
}

## --- Decoded plots
for (cov in c("age", "pir")) {
  q_levels <- quantile_levels_by_cov[[cov]]
  q_cols   <- quantile_level_cols[names(q_levels)]
  q_labels <- sprintf("p = %g", q_levels)

  ## Reference QF / CDF (decoded at male = 0, age = 40, pir = 1)
  cv_ref_decode <- list(male = 0, age = 40, pir = 1)
  Qi_ref_decode <- decode_z_to_Qi_vec(
    compute_conditional_mean(amm, cv_ref_decode), Ji_interp)
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
  y_grid_ref <- .default_y_grid(list(Qi_ref_decode))
  F_ref_decode <- qi_to_cdf(Qi_ref_decode, y_grid_ref)
  plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)", main = "Reference CDF")
  lines(y_grid_ref, F_ref_decode, col = "gray60", lwd = 4)
  dev.off()

  ## QF family
  png(file.path(interp_dir, str_glue("amm_qf_conditional_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  plot_conditional_qf(amm, cov, Ji = Ji_interp,
                      quantile_lines = q_levels,
                      quantile_line_cols = q_cols,
                      quantile_line_labels = q_labels)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_conditional_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  plot_conditional_qf_faceted(amm, cov, Ji = Ji_interp,
                              boot_reps = amm_boots,
                              quantile_lines = q_levels,
                              quantile_line_cols = q_cols)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_contrast_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  plot_contrast_qf(amm, cov, Ji = Ji_interp,
                   quantile_lines = q_levels,
                   quantile_line_cols = q_cols,
                   quantile_line_labels = q_labels)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_qf_contrast_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  plot_contrast_qf_faceted(amm, cov, Ji = Ji_interp,
                           boot_reps = amm_boots,
                           quantile_lines = q_levels,
                           quantile_line_cols = q_cols)
  dev.off()

  ## CDF family
  png(file.path(interp_dir, str_glue("amm_cdf_conditional_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  plot_conditional_cdf(amm, cov, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_conditional_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  plot_conditional_cdf_faceted(amm, cov, Ji = Ji_interp,
                               boot_reps = amm_boots)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_contrast_{cov}.png")),
      width = 960, height = 720, pointsize = 14)
  plot_contrast_cdf(amm, cov, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_cdf_contrast_faceted_{cov}.png")),
      width = 1280, height = 480, pointsize = 14)
  plot_contrast_cdf_faceted(amm, cov, Ji = Ji_interp,
                            boot_reps = amm_boots)
  dev.off()

  ## Moments family
  png(file.path(interp_dir, str_glue("amm_moments_conditional_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  plot_conditional_moments(amm, cov, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_conditional_faceted_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  plot_conditional_moments_faceted(amm, cov, Ji = Ji_interp,
                                   boot_reps = amm_boots)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_contrast_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  plot_contrast_moments(amm, cov, Ji = Ji_interp)
  dev.off()
  png(file.path(interp_dir, str_glue("amm_moments_contrast_faceted_{cov}.png")),
      width = 1280, height = 360, pointsize = 14)
  plot_contrast_moments_faceted(amm, cov, Ji = Ji_interp,
                                boot_reps = amm_boots)
  dev.off()

  ## Threshold-crossing (one per MIMS threshold; absolute + contrast)
  for (thr_name in names(mims_thresholds)) {
    thr_y <- mims_thresholds[[thr_name]]
    png(file.path(interp_dir,
                  str_glue("amm_threshold_{thr_name}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    plot_threshold_crossing(amm, cov, thr_y, Ji = Ji_interp,
                            boot_reps = amm_boots)
    dev.off()
    png(file.path(interp_dir,
                  str_glue("amm_threshold_contrast_{thr_name}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    plot_contrast_threshold_crossing(amm, cov, thr_y, Ji = Ji_interp,
                                     boot_reps = amm_boots)
    dev.off()
  }

  ## Quantile-crossing (one per quantile level; absolute + contrast)
  for (q_name in names(q_levels)) {
    p_star <- q_levels[[q_name]]
    png(file.path(interp_dir,
                  str_glue("amm_quantile_{q_name}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    plot_quantile_crossing(amm, cov, p_star, Ji = Ji_interp,
                           boot_reps = amm_boots)
    dev.off()
    png(file.path(interp_dir,
                  str_glue("amm_quantile_contrast_{q_name}_{cov}.png")),
        width = 540, height = 540, pointsize = 14)
    plot_contrast_quantile_crossing(amm, cov, p_star, Ji = Ji_interp,
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
z_hat_male_0  <- as.numeric(compute_conditional_mean(amm, cv_male_0))
z_hat_male_1  <- as.numeric(compute_conditional_mean(amm, cv_male_1))
z_boot_male_0 <- replicate_z_at(amm_boots, cv_male_0)
z_boot_male_1 <- replicate_z_at(amm_boots, cv_male_1)

## Decode to Qi (point + B x Ji bootstrap matrices)
Qi_hat_male_0  <- decode_z_to_Qi_vec(z_hat_male_0, Ji_interp)
Qi_hat_male_1  <- decode_z_to_Qi_vec(z_hat_male_1, Ji_interp)
Qi_boot_male_0 <- t(apply(z_boot_male_0, 1,
                          function(z) decode_z_to_Qi_vec(z, Ji_interp)))
Qi_boot_male_1 <- t(apply(z_boot_male_1, 1,
                          function(z) decode_z_to_Qi_vec(z, Ji_interp)))

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
  alpha            = amm_male_alpha,
  main             = str_glue("AMM Decoded Conditional CDFs ({main_suffix})"),
  y_grid           = y_grid_male,
  thresholds       = mims_thresholds,
  threshold_cols   = mims_threshold_cols,
  threshold_labels = mims_threshold_labels
)
dev.off()

png(file.path(interp_dir, "amm_male_cond_cdfs_diff.png"),
    width = 960, height = 960, pointsize = 18)
plot_decoded_conditional_cdf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha            = amm_male_alpha,
  main             = str_glue("AMM Decoded Conditional CDF Difference ({main_suffix})"),
  y_grid           = y_grid_male,
  thresholds       = mims_thresholds,
  threshold_cols   = mims_threshold_cols,
  threshold_labels = mims_threshold_labels
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





## ----- Conditonal Mean Plots (Waterfall)

# ## IDEAS:
# ##  - Y-axis transform for visibility
# ##  - Only include most significant components in waterfall

# ## Extract coefficients and set up waterfall list
# B_hat <- coef(model)
# z_hat_0 <- as.numeric(B_hat["(Intercept)", ])
# z_hats <- list(z_hat_0)

# ## Populate waterfall list
# k_seq <- res_rb[order(-res_rb$t_stat), c('component')]
# for (k_idx in 1:length(k_seq)) {
#   k <- k_seq[k_idx]
#   z_hats[[k_idx + 1]] <- z_hats[[k_idx]]
#   z_hats[[k_idx + 1]][k] <- z_hats[[k_idx + 1]][k] + B_hat["x", k]
# }

# ## Decode to Qi-space
# Qi_hats <- decode_z_rot_to_Qi(z_hats, Ji)

# ## Plot waterfall
# path_plot <- file.path(
#   'artifacts', dir_art, 'plots',
#   str_glue('waterfall_{bin_pred}.png')
# )
# n_funs    <- length(Qi_hats)
# pi_grid   <- pi_grid_fun(Ji)
# grid_list <- rep(list(pi_grid), n_funs)
# ## Color ramp: black -> red, with translucent middle entries
# ramp      <- colorRampPalette(c("blue", "red"))(n_funs)
# ramp_rgb  <- col2rgb(ramp) / 255
# colors    <- rgb(ramp_rgb[1, ], ramp_rgb[2, ], ramp_rgb[3, ], alpha = 1)
# colors[1]      <- "blue"
# colors[n_funs] <- "red"
# widths            <- rep(0.5, n_funs)
# widths[1]         <- 4
# widths[n_funs]    <- 4
# types <- rep(1, n_funs)
# color_width_type_labels <- data.frame(
#   color = colors,
#   width = widths,
#   type  = types,
#   label = c('z0_hat', paste0('+ beta_1[', k_seq[1:(length(k_seq)-1)], ']') , 'z1_hat')
# )
# png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
# plot_funs(
#   fun_list  = Qi_hats,
#   grid_list = grid_list,
#   colors    = colors,
#   widths    = widths,
#   types     = types,
#   ylab      = "Q(p)",
#   color_width_type_labels = color_width_type_labels,
#   main      = str_glue("Waterfall: group 0 -> group 1 ({bin_pred})")
# )
# dev.off()








# ## ----- Parametric Bootstrap

# ## Extract fitted ingredients from multivariate regression
# X         <- cbind(1, x)
# B_hat     <- coef(model)
# E_hat     <- residuals(model)
# Sigma_hat <- crossprod(E_hat) / (N - 2)
# N_fit     <- nrow(E_hat)
# mu_hat    <- X %*% B_hat

# ## Compute conditional latent means for original dataset (one for each group)
# z_hat_list <- list(
#   as.numeric(B_hat["(Intercept)", ]),
#   as.numeric(B_hat["(Intercept)", ] + B_hat["x", ])
# )

# ## Parametric bootstrap
# S <- 1000
# z_hat_boot_lists <- list(vector('list', S), vector('list', S))
# beta1_boot <- matrix(NA_real_, nrow = S, ncol = K)
# set.seed(12345)
# for (s in 1:S) {

#   ## Simulate bootstrap dataset from fitted latent model
#   E_star <- MASS::mvrnorm(N_fit, mu = rep(0, ncol(B_hat)), Sigma = Sigma_hat)
#   Z_star <- mu_hat + E_star

#   ## Re-fit model to bootstrapped latent dataset
#   model_s <- lm(Z_star ~ x)
#   B_s     <- coef(model_s)

#   ## Capture beta_1 for sup-t bands
#   beta1_boot[s, ] <- B_s["x", ]

#   ## Compute conditional latent means (one for each group)
#   z_hat_boot_lists[[1]][[s]] <- as.numeric(B_s["(Intercept)", ])
#   z_hat_boot_lists[[2]][[s]] <- as.numeric(B_s["(Intercept)", ] + B_s["x", ])

# }


# ## ----- Component-Wise Testing via Parametric Bootstrap

# ## Pointwise SEs from the bootstrap distribution of beta_1
# beta1_hat  <- as.numeric(B_hat["x", ])
# beta1_ctr  <- sweep(beta1_boot, 2, beta1_hat, '-')
# se_boot    <- apply(beta1_ctr, 2, sd)

# ## Sup-t pivot and simultaneous critical value
# t_max_boot <- apply(abs(sweep(beta1_ctr, 2, se_boot, '/')), 1, max)
# level      <- 0.99
# q_supt     <- quantile(t_max_boot, level)

# ## Bands and component-wise FWER-adjusted p-values (rejects 0 iff band excludes 0)
# half       <- as.numeric(q_supt) * se_boot
# t_obs      <- abs(beta1_hat) / se_boot
# p_adj      <- sapply(t_obs, function(t) mean(t_max_boot >= t))
# bands_supt <- data.frame(
#   k     = seq_len(K),
#   est   = beta1_hat,
#   se    = se_boot,
#   lwr   = beta1_hat - half,
#   upr   = beta1_hat + half,
#   p_adj = p_adj
# )

# ## Append to the same file as the Hotelling output
# # sink(path, append = TRUE)
# # cat("\n\n----- Sup-t simultaneous bands for beta_1 (parametric bootstrap) -----\n")
# # cat(str_glue("level = {level}, S = {S}, sup-t critical value = {round(q_supt, 4)}\n\n"))
# # print(bands_supt, row.names = FALSE, digits = 4)
# # sink()

# ## ----- Decoding

# ## Set decoded grid length
# Ji <- 10080

# ## Decode original conditional means to Qi-space (one for each group)
# Qi_hat_list <- decode_z_to_Qi(z_hat_list, Ji)

# ## Decode bootstrapped conditional means to Qi-space (S for each group)
# Qi_hat_boot_lists <- vector('list', 2)
# for (g in 1:2) {
#   Qi_hat_boot_lists[[g]] <- decode_z_to_Qi(z_hat_boot_lists[[g]], Ji)
# }


# ## ----- Q-space interpretation

# ## Get bands
# band_0 <- get_pointwise_bands(Qi_hat_boot_lists[[1]], level = 0.99)
# band_1 <- get_pointwise_bands(Qi_hat_boot_lists[[2]], level = 0.99)

# ## Means of Q-data for comparison
# mask <- !is.na(df_cov[bin_pred])
# Q_0_mean <- colMeans(do.call(rbind, Q_ctx$payload[mask][x == 0]))
# Q_1_mean <- colMeans(do.call(rbind, Q_ctx$payload[mask][x == 1]))

# ## Plot
# path_plot <- file.path(
#   'artifacts', dir_art, 'plots', 'mc',
#   str_glue('Z-means_{bin_pred}.png')
# )
# png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
# col_0_mean <- 'black'
# col_1_mean <- 'red'
# pi_grid <- pi_grid_fun(Ji)
# y_max <- max(unlist(Qi_hat_boot_lists))
# # y_max <- 120
# plot(
#   NULL, 
#   xlim = c(0, 1), 
#   ylim = c(0, y_max),
#   xlab = "p",
#   ylab = "Q(p)"
# )
# lines(pi_grid, band_0$mean, col = col_0_mean)
# lines(pi_grid, band_0$lower, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_0$upper, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_1$mean, col = col_1_mean)
# lines(pi_grid, band_1$lower, col = col_1_mean, lty = 'dotted')
# lines(pi_grid, band_1$upper, col = col_1_mean, lty = 'dotted')
# lines(p_grid, Q_0_mean, col = col_0)
# lines(p_grid, Q_1_mean, col = col_1)
# dev.off()

# ## IDEA: Interpret specific quantiles!
# p <- 0.5
# Qp_0 <- sapply(Qi_hat_boot_lists[[1]], function(q) quantile(q, c(p)))
# Qp_1 <- sapply(Qi_hat_boot_lists[[2]], function(q) quantile(q, c(p)))
# quantile(Qp_0, c(0.01, 0.99))
# quantile(Qp_1, c(0.01, 0.99))


# ## ---------- Frequentist Group Comparison (via conditional Q-means) ---------- ##

# ## Set binary predictor
# bin_pred <- 'low_income'


# ## ----- Fit linear regression

# ## Fit linear model
# mask <- !is.na(df_cov[bin_pred])
# Z <- do.call(rbind, z_ctx$payload[mask])
# x <- df_cov[mask,bin_pred]
# model <- lm(Z ~ x)


# ## ----- Model diagnostics

# ## Setup
# N <- nrow(Z)
# K <- ncol(Z)
# resid <- residuals(lm(Z ~ x))


# ## Check for marginal normality via Q-Q plots
# nc <- ceiling(sqrt(K))
# nr <- ceiling(K / nc)
# path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-marg-qq.png')
# png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
# par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
# for (k in seq_len(K)) {
#   qqnorm(resid[, k], main = paste0("k = ", k), pch = 19, cex = 0.25, col = col_train)
#   qqline(resid[, k], col = "red")
# }
# dev.off()
# par(mfrow = c(1, 1))
# ## NOTE: Looks pretty good

# ## Check for marginal normality via histograms
# nc <- ceiling(sqrt(K))
# nr <- ceiling(K / nc)
# path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-marg-hist.png')
# png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
# par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
# for (k in seq_len(K)) {
#   hist(resid[,k], main = paste0("k = ", k))
# }
# dev.off()
# par(mfrow = c(1, 1))
# ## NOTE: Definitely looks good

# ## Check for multivariate normality via pairwise scatterplots
# n_plots <- choose(K, 2)
# nc <- ceiling(sqrt(n_plots))
# nr <- ceiling(n_plots / nc)
# path_plot <- file.path('artifacts', dir_art, 'diagnostics', 'group-comp_norm-mv-pw-plots.png')
# png(path_plot_tmp, width = 250*nc, height = 250*nr, pointsize = 12)
# par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
# for (k1 in 1:(K-1)) {
#   for (k2 in (k1+1):K) {
#     plot(
#       resid[,k1], resid[,k2], 
#       main = str_glue("k = {k1} vs. k = {k2}"),
#       pch = 19, cex = 0.25, col = col_train)
#   }
# }
# dev.off()
# par(mfrow = c(1, 1))
# ## NOTE: Looks good.

# ## Check for both univariate and multivariate normality via formal checks
# result <- mvn(resid, mvn_test = "mardia", univariate_test = "AD")
# print(result$multivariate_normality)
# print(result$univariate_normality)
# ## NOTES: Formal tests reveal departure from MVN. Large sample hypersensitivity.

# ## Check for multivariate normality via squared Mahalanobis distances
# # cov_resid  <- cov(resid)
# # mean_resid <- colMeans(resid)
# # d2 <- mahalanobis(resid, center = mean_resid, cov = cov_resid)
# # chi2_q <- qchisq(ppoints(N), df = K)
# # par(mar = c(4, 4, 2, 1))
# # plot(
# #   chi2_q, sort(d2),
# #   xlab = expression(chi[K]^2 ~ "quantiles"),
# #   ylab = "Sorted squared Mahalanobis distances",
# #   main = "Multivariate Normality: Chi-squared Q-Q Plot",
# #   pch = 19, cex = 0.5, 
# #   col = col_train
# # )
# # abline(0, 1, col = "red")
# ## NOTE: Mahalanobis plot suggests substantial deviation from MVN.

# ## Check for homescedasticity via covariance heatmaps
# cov0 <- cov(Z[x == 0, ])
# cov1 <- cov(Z[x == 1, ])
# K_dim    <- nrow(cov0)
# zlim     <- range(c(cov0, cov1))
# zmax     <- max(abs(zlim))
# breaks   <- seq(-zmax, zmax, length.out = 101)
# col_pal  <- colorRampPalette(c("#2166ac", "white", "#b2182b"))(100)

# plot_cov_heatmap <- function(M, title) {
#   image(
#     1:K_dim, 1:K_dim, t(M)[, K_dim:1],
#     zlim = c(-zmax, zmax), col = col_pal, breaks = breaks,
#     axes = FALSE, xlab = "", ylab = "", main = title
#   )
#   axis(1, at = 1:K_dim, labels = 1:K_dim, las = 1, tick = FALSE)
#   axis(2, at = 1:K_dim, labels = K_dim:1, las = 1, tick = FALSE)
#   for (i in 1:K_dim) for (j in 1:K_dim) if (i >= j) {
#     text(j, K_dim - i + 1, sprintf("%.2f", M[i, j]), cex = 0.7)
#   }
# }

# png(path_plot_tmp, width = 1200, height = 540, pointsize = 14)
# layout(matrix(c(1, 2, 3), nrow = 1), widths = c(4, 4, 1))
# par(mar = c(4, 4, 3, 1))
# plot_cov_heatmap(cov0, "Group 0 covariance")
# plot_cov_heatmap(cov1, "Group 1 covariance")
# par(mar = c(4, 0.5, 3, 4))
# legend_levels <- seq(-zmax, zmax, length.out = 100)
# image(
#   1, legend_levels, t(matrix(legend_levels, ncol = 1)),
#   col = col_pal, breaks = breaks, axes = FALSE, xlab = "", ylab = ""
# )
# axis(4, las = 1)
# dev.off()
# ## NOTE: Looks good

# ## Check for homoscedasticity via Box's M
# boxm_result <- boxM(Z, x)
# print(boxm_result)
# ## NOTE: Failed. Large sample hypersensitivity.


# ## ----- Hotelling's T-squared

# manova_result <- manova(model)
# summary(manova_result, test = "Hotelling-Lawley")
# summary.aov(manova_result)


# ## ----- Parametric bootstrap

# ## Extract fitted ingredients from multivariate regression
# X         <- cbind(1, x)
# B_hat     <- coef(model)
# E_hat     <- residuals(model)
# Sigma_hat <- crossprod(E_hat) / (N - 2)
# N_fit     <- nrow(E_hat)
# mu_hat    <- X %*% B_hat

# ## Compute conditional latent means for original dataset (one for each group)
# z_hat_list <- list(
#   as.numeric(B_hat["(Intercept)", ]),
#   as.numeric(B_hat["(Intercept)", ] + B_hat["x", ])
# )

# # ## Compute conditional latent means for parametrically boostrapped datasets
# # S <- 1000
# # z_hat_boot_lists <- list(vector('list', S), vector('list', S))
# # set.seed(12345)
# # for (s in 1:S) {

# #   ## Simulate bootstrap dataset from fitted latent model
# #   E_star <- MASS::mvrnorm(N_fit, mu = rep(0, ncol(B_hat)), Sigma = Sigma_hat)
# #   Z_star <- mu_hat + E_star

# #   ## Re-fit model to bootstrapped latent dataset
# #   model_s <- lm(Z_star ~ x)
# #   B_s     <- coef(model_s)

# #   ## Compute conditional latent means (one for each group)
# #   z_hat_boot_lists[[1]][[s]] <- as.numeric(B_s["(Intercept)", ])
# #   z_hat_boot_lists[[2]][[s]] <- as.numeric(B_s["(Intercept)", ] + B_s["x", ])

# # }

# ## For each bootstrap replicate, decode the full conditional distribution
# ## then average in Q-space
# S <- 100
# Ji <- 10080
# Qi_hat_boot_lists <- list(vector('list', S), vector('list', S))
# for (s in 1:S) {
#   if (s %% 10 == 0) {
#     print(str_glue("s = {s}"))
#   }

#   ## Fit model to bootstrapped dataset
#   E_star <- MASS::mvrnorm(N_fit, mu = rep(0, ncol(B_hat)), Sigma = Sigma_hat)
#   Z_star <- mu_hat + E_star
#   model_s <- lm(Z_star ~ x)
#   B_s <- coef(model_s)

#   ## Draw many z's from each group's conditional distribution and average after decoding
#   n_mc <- 500
#   z_draws_0 <- MASS::mvrnorm(n_mc, mu = B_s["(Intercept)",], Sigma = Sigma_hat)
#   z_draws_1 <- MASS::mvrnorm(n_mc, mu = B_s["(Intercept)",] + B_s["x",], Sigma = Sigma_hat)

#   Qi_draws_0 <- decode_z_to_Qi(split(z_draws_0, row(z_draws_0)), Ji)
#   Qi_draws_1 <- decode_z_to_Qi(split(z_draws_1, row(z_draws_1)), Ji)

#   ## Average in Q-space
#   Qi_hat_boot_lists[[1]][[s]] <- colMeans(do.call(rbind, Qi_draws_0))
#   Qi_hat_boot_lists[[2]][[s]] <- colMeans(do.call(rbind, Qi_draws_1))
# }


# ## ----- Decoding

# # ## Set decoded grid length
# # Ji <- 10080

# # ## Decode original conditional means to Qi-space (one for each group)
# # Qi_hat_list <- decode_z_to_Qi(z_hat_list, Ji)

# # ## Decode bootstrapped conditional means to Qi-space (S for each group)
# # Qi_hat_boot_lists <- vector('list', 2)
# # for (g in 1:2) {
# #   Qi_hat_boot_lists[[g]] <- decode_z_to_Qi(z_hat_boot_lists[[g]], Ji)
# # }


# ## ----- Q-space interpretation

# ## Get bands
# band_0 <- get_pointwise_bands(Qi_hat_boot_lists[[1]], level = 0.99)
# band_1 <- get_pointwise_bands(Qi_hat_boot_lists[[2]], level = 0.99)

# ## Means of Q-data for comparison
# mask <- !is.na(df_cov[bin_pred])
# Q_0_mean <- colMeans(do.call(rbind, Q_ctx$payload[mask][x == 0]))
# Q_1_mean <- colMeans(do.call(rbind, Q_ctx$payload[mask][x == 1]))

# ## Plot
# path_plot <- file.path(
#   'artifacts', dir_art, 'plots', 'mc',
#   str_glue('Q-means_{bin_pred}.png')
# )
# png(path_plot, width = 960, height = 960, pointsize = 18)
# col_0_mean <- 'black'
# col_1_mean <- 'red'
# pi_grid <- pi_grid_fun(Ji)
# y_max <- max(unlist(Qi_hat_boot_lists))
# # y_max <- 120
# plot(
#   NULL, 
#   xlim = c(0, 1), 
#   ylim = c(0, y_max),
#   xlab = "p",
#   ylab = "Q(p)"
# )
# lines(pi_grid, band_0$mean, col = col_0_mean)
# lines(pi_grid, band_0$lower, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_0$upper, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_1$mean, col = col_1_mean)
# lines(pi_grid, band_1$lower, col = col_1_mean, lty = 'dotted')
# lines(pi_grid, band_1$upper, col = col_1_mean, lty = 'dotted')
# lines(p_grid, Q_0_mean, col = col_0)
# lines(p_grid, Q_1_mean, col = col_1)
# dev.off()







# ## ---------- Bayesian Group Comparison ---------- ##


# ## Set binary predictor
# bin_pred <- 'gender'


# ## ----- Fit linear regression

# ## Fit linear model
# S <- 1000
# mask <- !is.na(df_cov[bin_pred])
# Z <- do.call(rbind, z_ctx$payload[mask])
# x <- df_cov[mask,bin_pred]
# samps <- fit_model_bin_pred(
#   Z, 
#   x,
#   n_chains = 1, 
#   burn_in = 2000, 
#   thin_interval = 5, 
#   n_samps = S,
#   seed = 12345
# )


# ## ----- Model Diagnostics


# ## ----- Joint Latent Test

# ## Extract mu and beta
# K <- ncol(Z)
# n_param <- ncol(samps)
# betas <- samps[,(n_param - 2*K + 1):(n_param - K)]
# mus <- samps[,(n_param - K + 1):n_param]

# ## SimBaS test
# out <- simbas(betas)
# print(out) ## Low scores indicate departures from null


# ## ----- Posterior Sampling of Conditional Means 

# ## Compute posterior conditional means
# z_hat_samps_0 <- mus
# z_hat_samps_1 <- mus + betas

# ## Compute posterior mean of posterior conditonal means
# z_hat_0 <- colMeans(z_hat_samps_0)
# z_hat_1 <- colMeans(z_hat_samps_1)
# z_hat_list <- list(z_hat_0, z_hat_1)

# ## Matrix to list for decoding
# z_hat_samps_0 <- asplit(z_hat_samps_0, MARGIN = 1)
# z_hat_samps_1 <- asplit(z_hat_samps_1, MARGIN = 1)
# z_hat_samps_lists <- list(z_hat_samps_0, z_hat_samps_1)


# ## ----- Decoding

# ## Set decoded grid length
# Ji <- 10080

# ## Decode original conditional means to Qi-space (one for each group)
# Qi_hat_list <- decode_z_to_Qi(z_hat_list, Ji)

# ## Decode bootstrapped conditional means to Qi-space (S for each group)
# Qi_hat_samps_lists <- vector('list', 2)
# for (g in 1:2) {
#   Qi_hat_samps_lists[[g]] <- decode_z_to_Qi(z_hat_samps_lists[[g]], Ji)
# }


# ## ----- Q-space interpretation

# ## Get bands
# band_0 <- get_pointwise_bands(Qi_hat_samps_lists[[1]])
# band_1 <- get_pointwise_bands(Qi_hat_samps_lists[[2]])

# ## Means of Q-data for comparison
# Q_0_mean <- colMeans(do.call(rbind, Q_ctx$payload[x == 0]))
# Q_1_mean <- colMeans(do.call(rbind, Q_ctx$payload[x == 1]))

# ## Plot
# png(path_plot_tmp, width = 960, height = 960, pointsize = 18)
# col_0_mean <- 'black'
# col_1_mean <- 'red'
# pi_grid <- pi_grid_fun(Ji)
# y_max <- max(unlist(Qi_hat_samps_lists))
# plot(
#   NULL, 
#   xlim = c(0, 1), 
#   ylim = c(0, y_max),
#   xlab = "p",
#   ylab = "Q(p)"
# )
# lines(pi_grid, band_0$mean, col = col_0_mean)
# lines(pi_grid, band_0$lower, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_0$upper, col = col_0_mean, lty = 'dotted')
# lines(pi_grid, band_1$mean, col = col_1_mean)
# lines(pi_grid, band_1$lower, col = col_1_mean, lty = 'dotted')
# lines(pi_grid, band_1$upper, col = col_1_mean, lty = 'dotted')
# lines(p_grid, Q_0_mean, col = col_0)
# lines(p_grid, Q_1_mean, col = col_1)
# dev.off()

