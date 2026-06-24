source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## Globals
dir_art <- 'demo_iceland'

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
      K_max = 20,     ## Automatically finds the smallest K < K_max that satisfies near-losslessness
      epsilon = 0.5,  ## Tuned tolerance level
      alpha = 0.01,   ## Tuned slippage rate
      V = 5,
      lambda = 0      ## Tuned shrinkage parameter
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
  y_star = NULL,
  y_min = NULL,
  seed = 12345
)

## Fitting
pipeline <- fit(pipeline, y_list)
path <- 'artifacts/demo_iceland/pipe.rds'
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
  path          = file.path('artifacts', 'demo_iceland', 'plots',
                            'Qi-recon_grid_3x5.png')
)



## ---------- Visualize data in different spaces

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



## ---------- Synthetic Data

## Generate synthetic data by simulating from a fitted latent 
## mean-covariance model
set.seed(12345)
Z <- do.call(rbind, z_ctx$payload)
fit <- fit_mean_cov(Z)
z_draws <- draw_mean_cov(N, fit)
Qi_draws <- decode_z_to_Qi(pipeline, z_draws, Ji = Ji_max)

## Plot Q
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



## ========== Hyperparameter Tuning ========== ##

## Now let's see how we tuned those hyperparameters.
## We tune in two stages: 
##  (1) Inspect plot of cross-validated losses) to choose suitable epsilon and alpha for lambda = 0.
##  (2) For fixed epsilon and alpha, use generativity scoring to choose lambda.

## Returns fitted Iceland pipeline.
## We will reuse this wrapper throughout tuning. 
## Use Cases: 
##  1. Create pipeline with fixed K --> pass y_list, K, lambda (opt)
##  2. Create pipeline with auto-chosen K --> pass y_list, epsilon, alpha, lambda (opt)
fit_pipeline_iceland <- function(
  y_list,
  epsilon = NULL, 
  alpha = NULL, 
  lambda = NULL, 
  K = NULL, 
  K_max = 20, 
  path_flow = NULL
) {
  ## Validate arguments
  if (is.null(epsilon) && is.null(alpha) && is.null(K)) {
    stop("If epsilon and alpha are NULL, must pass K.")
  }
  if (!is.null(epsilon) && !is.null(alpha) && !is.null(K)) {
    warning("Passed K will override dimension derived from passed epsilon and alpha.")
  }

  ## Stages
  stages <- list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(
      p_grid = p_grid_fun(
        breaks = c(1/(91 + 1), 91/(91 + 1)),
        interval_counts = c(91)
      ), 
      Ji_min = 28
    ),
    stage_wame(
      epsilon = epsilon,
      alpha = alpha,
      lambda = lambda, 
      K = K, 
      K_max = K_max,
      loss = 'wasserstein'
    )
  )
  if (!is.null(path_flow)) {
    stages[[4]] <- stage_flow(
      n_layers = 16,
      max_epochs = 1000,
      lr = 1e-3,
      path = path_flow
    )
  }

  ## Fit pipeline
  pipeline <- construct_pipeline(
    stages = stages,
    supp_Y = seq(-30, 20, by = 0.0001),
    p_star = 0.5,
    y_star = NULL,
    y_min = NULL,
    seed = 12345
  )
  pipeline <- fit(pipeline, y_list)
  pipeline
}


## ---------- Choose epsilon and alpha via near-losslessness

## Compute validation losses for candidate K
valid_losses_by_K <- compute_cv_losses(
  y_list, fit_pipeline_iceland, 
  K_from = 1, 
  K_to = 20, 
  V = 5, 
  seed = 12345
)
path <- file.path('artifacts', dir_art, 'valid_losses.rds')
saveRDS(valid_losses_by_K, path)

## Wasserstein losses are not themselves interpretable.
## Compute pairwise wasserstein losses for a sample of
## observations for provide a reference. 
pwds <- compute_sampled_pwds(
  y_list, pipeline,
  loss = 'wasserstein',
  p_grid_aug = p_grid_aug,
  N_samp = 500,
  seed = 12345
)

## Plot losslessness
epsilon_star <- 0.5  ## set as desired
alpha_star <- 0.01   ## set as desired
plot_losslessness(
  valid_losses_by_K,
  jitter_width = 0.2,
  epsilon = epsilon_star,  
  alpha = alpha_star,   
  plot_mean = FALSE,
  ylab = "Cross-Validated Wasserstein Error",
  ylab2 = "Pairwise Wasserstein Distance Percentile",
  xlab = "K",
  K_star = NULL,
  pairwise_distances = pwds,
  main = "Cross-Validated Wasserstein Reconstruction Errors",
  path = file.path('artifacts', dir_art, 'plots', 'valid_losses.png')
)


## ---------- Choose lambda via generativity

## TODO: Streamline this code. 

## Shrinkage parameter candidates
lambdas <- c(0, 1e-3, 1e-2)

## Fit pipelines with these candidate lambdas
for (lambda in lambdas) {
  pipeline <- fit_pipeline_iceland(
    y_list, 
    epsilon = epsilon_star, 
    alpha = alpha_star, 
    lambda = lambda,
    path_flow = file.path('artifacts', dir_art, 'generativity', str_glue('flow_lambda-{lambda}.pth'))
  )
  path <- file.path('artifacts', dir_art, 'generativity', str_glue('pipe_lambda-{lambda}.rds'))
  saveRDS(pipeline, path)
}

## Evalute generativity for each lambda
S <- 5   ## number of data splits
R <- 20  ## number of repetitions per split
n_cores <- min(5, R)
gen_list <- list()
for (lambda in lambdas) {
  print(str_glue("lambda = {lambda}"))
  path_pipe <- file.path(
    'artifacts', dir_art, 'generativity',
    str_glue('pipe_lambda-{lambda}.rds')
  )
  pipeline <- readRDS(path_pipe)
  gen_list[[as.character(lambda)]] <- evaluate_pipeline_generativity(
    pipeline, y_list,
    S = S, R = R,
    frac_train = 0.5,
    J_aug = 500,
    ridge = 0,
    seed = 12345,
    n_cores = n_cores
  )
}
path_gen <- file.path('artifacts', dir_art, 'generativity', 'gen_res.rds')
saveRDS(gen_list, path_gen)

## Plot generativity and K against lambda
K_by_lambda <- sapply(lambdas, function(lambda) {
  pipe <- readRDS(file.path(
    'artifacts', dir_art, 'generativity',
    str_glue('pipe_lambda-{lambda}.rds')
  ))
  pipe$stages[[3]]$state$child_qg_pca$state$K
})
scores_by_lambda <- lapply(lambdas, function(lambda) {
  lapply(gen_list[[as.character(lambda)]]$splits, `[[`, "log_norm")
})
plot_generativity_boxes(
  groups      = as.character(lambdas),
  scores      = scores_by_lambda,
  Ks          = Ks,
  hline       = 0,
  # vline_group = as.character(lambda_star),
  xlab        = expression(lambda),
  ylab        = "log normalized generativity",
  main        = "Generativity (log-normalized) and K vs lambda",
  path        = file.path('artifacts', dir_art, 'plots', 'gen_score_vs_lambda.png')
)

