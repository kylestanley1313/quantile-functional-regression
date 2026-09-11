source('src/cot.R') 
source('src/model.R')
source('src/utils.R')


## Globals
dir_art <- 'demo_iceland_2'

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


## ========== Data Loading ========== ##

path <- file.path('data', 'processed', 'iceland_train.rds')
y_list <- readRDS(path)
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
J_aug <- 500


## ========== Representation Learning ========== ##

## To start, let's assume that we have already tuned the 
## pipeline hyperparamters: epsilon, alpha, and lambda. 
## We will actually tune them in the next subsection.
## Below shows you how to fit the pipeline. 

## ---------- Grid Definitions

## Construct the "common grid" that all EQFs will be smoothed onto
p_grid <- p_grid_fun(
  breaks = c(1/(Ji_max + 1), Ji_max/(Ji_max + 1)),
  interval_counts = c(Ji_max)
)

## Construct the "augmented grid", which augments the common grid with 
## a collection of evenly-spaced points. This grid is never used in 
## representation learning but is used later to approximate 
## pairwise Wasserstein distances and to compute generativity scores.
p_grid_aug <- c(p_grid, pi_grid_fun(J_aug - length(p_grid)))
p_grid_aug <- sort(unique(p_grid_aug))


## ---------- Encoder-Decoder Fitting

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = Ji_min),
    stage_wame( 
      K = 7,
      K_max = 40,               ## Automatically finds the smallest K < K_max that satisfies near-losslessness
      epsilon = 0.5,            ## Tuned tolerance level
      alpha = 0.01,             ## Tuned slippage rate
      V = 5,
      order_by = 'importance',  ## Change to order_by = "variance" to get old embeddings
      lambda = 0                ## Tuned shrinkage parameter
    ),
    stage_flow(
      n_layers = 16,
      hidden = 64,
      max_epochs = 1000,
      lr = 1e-3,
      val_frac = 0.25,
      path = str_glue('artifacts/{dir_art}/flow_iceland.pth')
    )
  ),
  supp_Y = seq(-30, 20, by = 0.0001),
  p_star = 0.5,
  y_star = NULL,
  y_min = NULL,
  seed = 12345
)

## Fitting
pipeline <- fit(pipeline, y_list)
path <- str_glue('artifacts/{dir_art}/pipe.rds')
saveRDS(pipeline, path)


## ---------- Representation Presentation

## New context
y_ctx <- new_context(
  payload = y_list,
  cache = pipeline$training$cache,
  meta = list()
)

## Encode/Decode stage-by-stage
Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
c_ctx_ <- decode(pipeline, z_ctx, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)
## Can perform full encode/decode operations:
##    z_ctx  <- encode(pipeline, y_ctx)
##    y_ctx_ <- decode(pipeline, z_ctx))


## -------- Visualize reconstructions

recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_ctx$payload[[i]]
  Q2 <- Qi_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

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

set.seed(12345)
plot_qi_recon_grid(
  Qi_orig_list  = Qi_ctx$payload,
  Qi_recon_list = Qi_ctx_$payload,
  recon_losses  = recon_losses,
  lower_mat     = lower_mat_3x5,
  upper_mat     = upper_mat_3x5,
  path          = file.path('artifacts', dir_art, 'plots',
                            'Qi-recon_grid_3x5.png')
)



## ---------- Visualize data in different spaces

## Plotting globals
mark_outliers <- TRUE
include_legend <- TRUE
k_from <- 1
k_to <- 3
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.25), col_train)

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



## ---------- Synthetic Data

## Generate synthetic data by simulating from a fitted latent 
## mean-covariance model
set.seed(12345)
Z <- do.call(rbind, z_ctx$payload)
mod <- fit_mean_cov(Z)
z_draws <- sample_mean_cov(N, mod)
Qi_draws <- decode_z_to_Qi(pipeline, z_draws, Ji = Ji_max)

## Plot Q
idx_shuff <- sample(1:(2*N))
col_data <- rgb(0, 0, 0, alpha = 0.25)
col_draws <- rgb(1, 0, 0, alpha = 0.25)
path_plot <- file.path(
  'artifacts', dir_art, 'plots',
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


## ========== Modeling ========== ##

## Fit a linear model with a (fake) binary covariate
set.seed(12345)
Z <- do.call(rbind, z_ctx$payload) ## Z-embeddings ("normalized" LQD scores)
x <- sample(0:1, size = nrow(Z), replace = TRUE)  ## binary covariate
mod <- lm(Z ~ x)

## Latent conditional means
z_hat_0 <- coef(mod)[1,]                 ## predicted value for x == 0
z_hat_1 <- coef(mod)[1,] + coef(mod)[2,] ## predicted value for x == 1

## Decode z_hats
z_hat_list <- list(
  z_hat_0,
  z_hat_1
)
Qi_hat_list <- decode_z_to_Qi(pipeline, z_hat_list, Ji = 91)

## Plot
pi_grid <- pi_grid_fun(91)
plot(pi_grid, Qi_hat_list[[1]], type = 'l')
lines(pi_grid, Qi_hat_list[[2]], col = 'red')


## ========== Dimension Selection ========== ##

## Before tuning epsilon/alpha/lambda in the next section, inspect how
## cross-validated reconstruction error and generativity vary with the
## latent dimension K. This mirrors scripts/assess_losslessness.R (CV loss
## by K) and scripts/assess_generativity_for_K.R (generativity by K), but
## is self-contained here and does not rely on anything defined below
## under Hyperparameter Tuning.

## Candidate K's
Ks <- 1:15

## Fits a fixed-K Iceland pipeline (no auto K-selection). Only attaches a
## flow stage when path_flow is supplied, since the CV loss sweep never
## needs one.
fit_pipeline_iceland <- function(y_list, K, path_flow = NULL) {
  stages <- list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = Ji_min),
    stage_wame(K = K, K_max = 40, lambda = 0, order_by = 'importance', loss = 'wasserstein')
  )
  if (!is.null(path_flow)) {
    stages[[4]] <- stage_flow(
      n_layers = 16,
      hidden = 64,
      max_epochs = 1000,
      lr = 1e-3,
      val_frac = 0.25,
      path = path_flow
    )
  }
  pipeline_K <- construct_pipeline(
    stages = stages,
    supp_Y = seq(-30, 20, by = 0.0001),
    p_star = 0.5,
    y_star = NULL,
    y_min = NULL,
    seed = 12345
  )
  fit(pipeline_K, y_list)
}


## ---------- Cross-validated reconstruction error by K

valid_losses_by_K <- compute_cv_losses(
  y_list,
  fit_pipeline_iceland,
  K_from = min(Ks),
  K_to = max(Ks),
  V = 5,
  seed = 12345
)
path <- file.path('artifacts', dir_art, 'dimension_selection', 'valid_losses_by_K.rds')
saveRDS(valid_losses_by_K, path)

## Reference pairwise Wasserstein distances, using the pipeline already
## fit above under Representation Learning.
pwds <- compute_sampled_pwds(
  y_list, pipeline,
  loss = 'wasserstein',
  p_grid_aug = p_grid_aug,
  N_samp = 500,
  seed = 12345
)

plot_losslessness(
  valid_losses_by_K,
  jitter_width = 0.2,
  epsilon = NULL, # 0.5,
  alpha = 0.01,
  plot_mean = FALSE,
  ylab = "Cross-Validated Wasserstein Error",
  ylab2 = "Pairwise Wasserstein Distance Percentile",
  xlab = "K",
  pairwise_distances = pwds,
  main = "Cross-Validated Wasserstein Reconstruction Error by K",
  path = file.path('artifacts', dir_art, 'plots', 'valid_losses.png')
)


## ---------- Generativity scores by K

S <- 5   ## number of data splits
R <- 20  ## number of repetitions per split
n_cores <- min(5, R)

gen_list_by_K <- list()
for (K in Ks) {
  print(str_glue("K = {K}"))
  pipeline_K <- fit_pipeline_iceland(
    y_list, K,
    path_flow = file.path('artifacts', dir_art, 'dimension_selection', str_glue('flow_K-{K}.pth'))
  )
  gen_list_by_K[[as.character(K)]] <- evaluate_unconditional_generativity(
    pipeline_K, y_list,
    S = S, R = R,
    frac_train = 0.5,
    J_aug = J_aug,
    ridge = 0,
    seed = 12345,
    n_cores = n_cores
  )
}
path <- file.path('artifacts', dir_art, 'dimension_selection', 'gen_res_by_K.rds')
saveRDS(gen_list_by_K, path)

## Plot generativity against K
scores_by_K <- lapply(Ks, function(K) {
  lapply(gen_list_by_K[[as.character(K)]]$splits, `[[`, "log_norm")
})
plot_generativity_boxes(
  groups      = as.character(Ks),
  scores      = scores_by_K,
  hline       = 0,
  xlab        = "K",
  ylab        = "log normalized generativity",
  main        = "Generativity (log-normalized) by K",
  path        = file.path('artifacts', dir_art, 'plots', 'gen_score_vs_K.png')
)

