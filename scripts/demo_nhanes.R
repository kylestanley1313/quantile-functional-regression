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
dir_art <- 'demo_nhanes_v11'

## Latent representation used by the Group Comparison and AMM sections:
##   'z_rot' -> output of the PCA layer (stage 5)
##   'z'     -> output of the flow layer (stage 4)
##   'c'     -> output of the WAME layer (stage 3)
## Analysis output directories are suffixed for anything other than z_rot, so
## the two runs sit side by side under the same dir_art (and therefore share a
## single fitted pipeline -- see the pipeline load guard below).
latent <- 'z_rot'
suffix <- if (latent == 'z_rot') '' else str_glue('-{latent}')


## Load data
path <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path)
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
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100, extrap = 'ratio'),
    stage_wame(
      K = 5,
      K_max = 30,
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
  seed = 1234
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
pipeline <- set_rotation_signs(pipeline, flip = c(1, 2, 3, 4, 5)) ## TODO: Generalize this function to any output layer
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
  z = do.call(rbind, z_ctx$payload),
  c     = c_ctx_to_matrix(pipeline, c_ctx),
  stop(str_glue("Unknown latent: '{latent}'"))
)


## ----- Paper figure spec

## Render every paper figure at its final printed size (inches) and include it
## in LaTeX at that same width, so fig_pointsize is the printed text size in
## every figure regardless of panel count. fig_cex = 1 undoes mfrow's
## auto-shrink in multi-panel layouts. fig_cex_main / fig_cex_lab /
## fig_cex_axis are the baseline title / axis-label / tick-label sizes (each
## 80% of the one before), and fig_cex_legend the legend text size; tweak
## them per figure at the call site (e.g. cex_lab = 0.9 * fig_cex_lab) when
## text doesn't fit a small panel.
fig_res       <- 600
fig_pointsize <- 7
fig_cex       <- 1
fig_cex_lab   <- 1.0 
fig_cex_main  <- 1.2 * fig_cex_lab
fig_cex_axis  <- 0.8 * fig_cex_lab
fig_cex_legend <- fig_cex_lab
fig_png <- function(path, width_in, height_in, pointsize = fig_pointsize) {
  png(path, width = width_in, height = height_in, units = "in",
      res = fig_res, pointsize = pointsize)
}
## Square size (inches) of a pairs-plot figure (plot_embeddings) of n_k
## dimensions: fig_emb_panel_in per panel, so text, axes and points keep the
## same proportions to the panels whatever the number of plotted dimensions
## (5 dimensions -> 6.5in, full width).
fig_emb_panel_in <- 1.3
fig_emb_size <- function(n_k) fig_emb_panel_in * n_k


## ----- Dimension selection plot

## Joint K-selection figure: cross-validated reconstruction errors (bottom,
## from assess_losslessness.R) and median relative generativity
## (top, from assess_generativity_for_K.R), with the chosen K marked.
dir_loss <- file.path('artifacts', dir_art, 'losslessness')
dir_genK <- file.path('artifacts', dir_art, 'generativity_by_K')
dir_dim_plots <- file.path('artifacts', dir_art, 'plots', 'representation')
if (!dir.exists(dir_dim_plots)) dir.create(dir_dim_plots, recursive = TRUE)

loss_valid_by_K    <- readRDS(file.path(dir_loss, 'config-1_loss-valid.rds'))
pairwise_dist_norm <- readRDS(file.path(dir_loss, 'config-1_pairwise-dist-norm.rds'))
gen_res_by_K       <- readRDS(file.path(dir_genK, 'gen_res.rds'))
gen_scores_by_K <- lapply(gen_res_by_K, function(g) {
  unlist(lapply(g$splits, `[[`, "log_norm"), use.names = FALSE)
})

set.seed(1234)  # point jitter
fig_png(file.path(dir_dim_plots, 'K_selection.png'), 4.5, 3.6)
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
plot_dimension_selection(
  valid_losses_by_K  = loss_valid_by_K,
  gen_scores_by_K    = gen_scores_by_K,
  pairwise_distances = pairwise_dist_norm,
  K_star             = 5,
  epsilon_percentile = 2,  # epsilon = 1.34
  alpha              = 0.05,
  pt_col             = rgb(0, 0, 0, 0.1),
  pt_cex             = 0.1
)
dev.off()


## ----- Visualize reconstructions

## Create directory
dir_rep_plots <- file.path('artifacts', dir_art, 'plots', 'representation')
if (!dir.exists(dir_rep_plots)) dir.create(dir_rep_plots, recursive = TRUE)

## Compute reconstruction losses
recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_ctx$payload[[i]]
  Q2 <- Qi_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

set.seed(1234)

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

path_plot <- file.path(dir_rep_plots,
                       sprintf('Qi-recon_grid_3x5%s.png',
                               if (log_scale) '_log' else ''))
fig_png(path_plot, 6.5, 4)   # TODO: final printed size
plot_qi_recon_grid(
  Qi_orig_list  = Qi_ctx$payload,
  Qi_recon_list = Qi_ctx_$payload,
  recon_losses  = recon_losses,
  lower_mat     = lower_mat_3x5,
  upper_mat     = upper_mat_3x5,
  log_x_plus_1  = log_scale,
  share_ylim    = TRUE,
  cex_main      = fig_cex_main,
  cex_lab       = fig_cex_lab,
  cex           = fig_cex,
  cex_axis      = fig_cex_axis
)
dev.off()



## ----- Visualize data in different spaces

## Plotting globals
mark_outliers <- FALSE
include_legend <- FALSE
ks_plot <- 1:5   # components to plot (any subset, e.g. c(1, 3, 5))
w_emb   <- fig_emb_size(length(ks_plot))
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.05)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.05), col_train)

## Plot z-rot-embeddings
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_z-rot.png')
)
fig_png(path_plot, w_emb, w_emb)
Z_plot <- rbind(
  do.call(rbind, z_rot_ctx$payload)[idx_train, ks_plot, drop = FALSE],
  do.call(rbind, z_rot_ctx$payload)[idx_outliers, ks_plot, drop = FALSE]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(16, length.out = nrow(Z_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(Z_plot))
)
if (include_legend) {
  color_shape_size_labels <- data.frame(
    color = c(col_train, col_outlier),
    shape = c(16, 16),
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
  stats = FALSE,
  cex_main = fig_cex_main,
  cex_lab = fig_cex_lab,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend,
  ks = ks_plot
)
dev.off()

## Plot z-embeddings
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_z.png')
)
fig_png(path_plot, w_emb, w_emb)
Z_plot <- rbind(
  do.call(rbind, z_ctx$payload)[idx_train, ks_plot, drop = FALSE],
  do.call(rbind, z_ctx$payload)[idx_outliers, ks_plot, drop = FALSE]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(16, length.out = nrow(Z_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(Z_plot))
)
if (include_legend) {
  color_shape_size_labels = data.frame(
    color = c(col_train, col_outlier),
    shape = c(16, 16),
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
  stats = FALSE,
  cex_main = fig_cex_main,
  cex_lab = fig_cex_lab,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend,
  ks = ks_plot
)
dev.off()

## Plot c-embeddings
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_c.png')
)
fig_png(path_plot, w_emb, w_emb)
C_plot <- rbind(
  c_ctx_to_matrix(pipeline, c_ctx)[idx_train, ks_plot, drop = FALSE],
  c_ctx_to_matrix(pipeline, c_ctx)[idx_outliers, ks_plot, drop = FALSE]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(16, length.out = nrow(C_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(C_plot))
)
if (include_legend) {
  color_shape_size_labels = data.frame(
    color = c(col_train, col_outlier),
    shape = c(16, 16),
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
  stats = FALSE,
  cex_main = fig_cex_main,
  cex_lab = fig_cex_lab,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend,
  ks = ks_plot
)
dev.off()


## Plot Q
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_q.png')
)
fig_png(path_plot, 3.25, 3.25)   # TODO: final printed size
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
  color_width_type_labels = color_width_type_labels,
  cex.lab = fig_cex_lab,
  cex_main = fig_cex_main,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend
)
dev.off()


## Plot subsample of Q
## NOTE: For Jeff's slides
N_samp <- 500
if (!is.null(N_samp)) {
  set.seed(1234)

  ## Compute overall mean
  Q_mean <- colMeans(do.call(rbind, Q_ctx$payload))

  ## Subsample
  idx_samp <- sample(N, size = N_samp)
  Q_samp <- Q_ctx$payload[idx_samp]

  ## Plot
  path_plot <- file.path(
    dir_rep_plots,
    str_glue('data_q-subsamp.png')
  )
  fig_png(path_plot, 3.25, 3.25)   # TODO: final printed size
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
    color_width_type_labels = color_width_type_labels,
    cex.lab = fig_cex_lab,
    cex_main = fig_cex_main,
    cex = fig_cex,
    cex_axis = fig_cex_axis,
    legend_cex = fig_cex_legend
  )
  dev.off()
}


## Plot Qi
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_qi.png')
)
fig_png(path_plot, 3.25, 3.25)   # TODO: final printed size
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
  color_width_type_labels = color_width_type_labels,
  cex.lab = fig_cex_lab,
  cex_main = fig_cex_main,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend
)
dev.off()


## Plot Qi vs. Q (one subject)
i <- 1
p_idx_start <- 1
p_idx_stop <- 100
path_plot <- file.path(
  dir_rep_plots,
  str_glue('data_qi-v-q_i-{i}_p-{p_idx_start}-{p_idx_stop}.png')
)
pi_grid <- pi_grid_fun(Ji_vec[i])
Qi <- Qi_ctx$payload[[i]]
Q <- Q_ctx$payload[[i]]

## Restrict to p_grid[p_idx_start:p_idx_stop], and Qi to the same p-range
p_idx <- p_idx_start:p_idx_stop
p_sub <- p_grid[p_idx]
Q_sub <- Q[p_idx]
pi_keep <- pi_grid >= min(p_sub) & pi_grid <= max(p_sub)
pi_sub <- pi_grid[pi_keep]
Qi_sub <- Qi[pi_keep]

fig_png(path_plot, 3.25, 1.625)   # TODO: final printed size
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
plot(
  pi_sub, Qi_sub,
  type = 'l',
  lwd = 2,
  xlim = range(p_sub),
  ylim = range(c(0, Qi_sub, Q_sub)),
  xlab = "p",
  ylab = "Q(p)",
  main = bquote("Subject" ~ .(i) ~
                  "(" * p[.(p_idx_start)] ~ "to" ~ p[.(p_idx_stop)] * ")")
)
lines(
  p_sub, Q_sub,
  lwd = 2,
  col = rgb(0.9, 0, 0)
)
## Grid points sit on the x-axis line, with dotted guides up to Q
y_axis <- par('usr')[3]
segments(p_sub, y_axis, p_sub, Q_sub, col = 'gray50', lty = 'dotted')
points(
  x = p_sub,
  y = rep(y_axis, length(p_sub)),
  pch = 16,
  cex = 1,
  col = rgb(0, 0.9, 0),
  xpd = NA
)
dev.off()


## ----- Check generativity

## Unconditional covariate-blind generativity of the fitted pipeline (reference:
## all held-out subjects; latent model ignores covariates), with the same procedure
## and settings as assess_generativity_for_K.R (S splits x R replicates; per
## replicate: fit the latent mean-cov model on a random train half, draw |val|
## synthetic latents, decode, score G~ = log(T_sv / T_tv)). keep_draws = TRUE
## stores every replicate's synthetic latents (~0.17 MB per replicate at K = 5)
## so the examples below are the exact datasets behind boxplot points.
##
## The evaluation is cached in dir_gen; delete gen_res.rds to recompute after
## refitting the pipeline or changing the settings below. Each parallel worker
## holds its own copy of the pipeline (~500 MB) plus ~150 MB per OT solve.
gen_S       <- 5
gen_R       <- 20
gen_seed_    <- 1234
gen_n_cores <- 3
dir_gen <- file.path('artifacts', dir_art, 'plots', 'representation', 'generativity')
if (!dir.exists(dir_gen)) dir.create(dir_gen, recursive = TRUE)
path_gen <- file.path(dir_gen, 'gen_res.rds')
if (file.exists(path_gen)) {
  gen_res <- readRDS(path_gen)
} else {
  start <- Sys.time()
  gen_res <- evaluate_generativity(
    pipeline, y_list,
    S = gen_S, R = gen_R,
    frac_train       = 0.5,
    pool_subsample_n = NULL,
    J_aug            = 500,
    ridge            = 0,
    seed             = gen_seed_,
    n_cores          = gen_n_cores,
    keep_draws       = TRUE
  )
  print(Sys.time() - start)
  saveRDS(gen_res, path_gen)
}

## Pooled scores with their (split, replicate) of origin
gen_scores <- do.call(rbind, lapply(seq_along(gen_res$splits), function(s) {
  data.frame(split = s, rep = seq_along(gen_res$splits[[s]]$log_norm),
             log_norm = gen_res$splits[[s]]$log_norm)
}))

## Examples: stratify the pooled scores by rank into n_gen_examples equal
## strata (e.g. 10 -> deciles: ranks 1-10, 11-20, ..., 91-100 of S * R = 100;
## 4 -> quartiles: 1-25, 26-50, ...) and draw one replicate at random from each,
## so the examples span the typical to the worst synthetic datasets. Labels
## 1, ..., n_gen_examples follow the strata (lowest scores first) and are
## numbered on the boxplot.
n_gen_examples <- 4
n_gen_scores   <- nrow(gen_scores)
stopifnot(n_gen_examples <= n_gen_scores)
gen_strata <- split(seq_len(n_gen_scores),
                    ceiling(seq_len(n_gen_scores) * n_gen_examples / n_gen_scores))
set.seed(gen_seed_)
ranks  <- vapply(gen_strata, function(k) k[sample.int(length(k), 1)], integer(1))
gen_ex <- gen_scores[order(gen_scores$log_norm)[ranks], ]
gen_ex$label <- seq_len(nrow(gen_ex))

## Paneled paper figure (full width, 6.5in): the boxplot tall and thin on the
## left, the n_gen_examples = 4 QF-space examples in a 2 x 2 grid to its right.
## Each png is rendered at the size it is included at in LaTeX, so text sizes
## match across panels: 1.4 + 2 * 2.5 = 6.4in wide (0.1in for gaps), and the
## boxplot spans both example rows.
gen_fig_ex_w  <- 2.5
gen_fig_ex_h  <- 1.6
gen_fig_box_w <- 1.4
gen_fig_box_h <- 2 * gen_fig_ex_h

## Boxplot of the pooled scores (one box: the scores are unconditional)
K_fit <- pipeline$stages[[3]]$state$child_qg_pca$state$K
fig_png(file.path(dir_gen, str_glue('gen_scores_K-{K_fit}.png')), gen_fig_box_w, gen_fig_box_h)
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
plot_generativity_boxes(
  groups    = str_glue("K = {K_fit}"),
  scores    = list(lapply(gen_res$splits, `[[`, "log_norm")),
  hline     = 0,
  ylab      = "Relative generativity",
  main      = "",
  collapse  = TRUE,
  highlight = data.frame(group = 1, value = gen_ex$log_norm, label = gen_ex$label)
)
dev.off()

## Example plots: each selected replicate's held-out real QFs (black) vs its
## decoded synthetic QFs (green), titled with its number on the boxplot and
## its G~. Same plot style as the group-comparison generativity examples.
## Half-width paper figures.
##
## Each example is also shown in z-rot space (pairs plot of the first
## n_gen_k_plot components, same layout as data_z-rot.png): held-out real
## subjects vs the synthetic latents that produced the QFs. And in c space
## (gen-ex_{i}_c*.png, same layout as data_c.png): the same synthetic latents
## decoded back through the flow vs the held-out subjects' encoded c.
##
## gen_outlier_w2: synthetic QFs at least this W2 distance from the mean
## training-half QF are flagged (orange) in extra *_outliers.png versions of
## both plots, alongside the plain unflagged ones; NULL gives only the plain
## plots.
## For scale, the real subjects farthest from the mean QF sit at W2 ~ 20-25.
n_gen_k_plot   <- min(5, ncol(Z_latent))
gen_outlier_w2 <- 30
col_gen_real    <- adjustcolor("black",   0.25)
col_gen_synth   <- adjustcolor("#1B9E77", 0.25)
col_gen_outlier <- "#D95F02"
dir_gen_ex <- file.path(dir_gen, 'examples')
if (!dir.exists(dir_gen_ex)) dir.create(dir_gen_ex, recursive = TRUE)
if (!latent %in% c('z_rot', 'z')) {
  stop("z-rot view of generativity examples assumes latent 'z_rot' or 'z'.")
}
Z_rot_all <- do.call(rbind, z_rot_ctx$payload)
C_all     <- c_ctx_to_matrix(pipeline, c_ctx)

## Pairs plot of held-out real vs synthetic latents in one embedding space
## (z-rot or c), with outlying synthetic draws (is_out) drawn larger and on top.
plot_gen_ex_embeddings <- function(path, A_real, A_synth, is_out, main,
                                   leg_labels, leg_cols) {
  n_real <- nrow(A_real)
  ord_rs <- sample(n_real + sum(!is_out))   # interleave real / non-outlying synth
  A_plot <- rbind(A_real, A_synth[!is_out, , drop = FALSE])[ord_rs, ]
  colors <- c(rep(col_gen_real, n_real), rep(col_gen_synth, sum(!is_out)))[ord_rs]
  sizes  <- rep(0.5, nrow(A_plot))
  A_plot <- rbind(A_plot, A_synth[is_out, , drop = FALSE])   # outliers on top
  colors <- c(colors, rep(col_gen_outlier, sum(is_out)))
  sizes  <- c(sizes, rep(1, sum(is_out)))
  w_emb_ex <- fig_emb_size(n_gen_k_plot)
  fig_png(path, w_emb_ex, w_emb_ex)   # TODO: final printed size
  par(oma = c(0, 0, 1.5, 0))
  plot_embeddings(
    A_plot[, seq_len(n_gen_k_plot), drop = FALSE],
    colors = colors,
    shapes = rep(16, nrow(A_plot)),
    sizes  = sizes,
    color_shape_size_labels = NULL,
    stats = FALSE,
    cex_main = fig_cex_main,
    cex_lab = fig_cex_lab,
    cex = fig_cex,
    cex_axis = fig_cex_axis
  )
  ## plot_embeddings restores mfrow's automatic text shrink on exit; undo it
  ## so the title and legend match the other figures.
  par(cex = fig_cex)
  mtext(main, side = 3, outer = TRUE, cex = fig_cex_main)
  ## Legend in the empty upper-right triangle of the pairs grid
  par(fig = c(0.5, 1, 0.5, 1), new = TRUE, mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", legend = leg_labels, col = leg_cols, pch = 16,
         pt.cex = 1.2, cex = fig_cex_legend, bty = "n")
  dev.off()
}

## Marginal check of one rebuilt replicate (generativity_replicate()): for each
## quantile level p in p_show, side-by-side boxplots of Q_i(p) across the
## replicate's held-out real subjects (gray, left) and its synthetic draws
## (green, right). Each QF is linearly interpolated to p from the replicate's
## augmented p-grid. Compact panels for narrow figure columns: solid whiskers,
## y tick labels only (about 3, at the row-2 tick size; no axis labels, no x
## axis), "p = ..." as a small title over the plot region, and tight margins.
## Each panel has its own scale, so the caption should explain the colors and
## that panels are not on a common scale. outliers = FALSE omits outlier
## points (the axis then spans the whiskers). Files are gen-ex_{tag}_p-{p}.png
## in dir_out (created if needed).
plot_gen_replicate_quantiles <- function(ex, dir_out, tag,
                                         p_show = c(0.5, 0.75, 0.99),
                                         w, h, outliers = TRUE) {
  if (!dir.exists(dir_out)) dir.create(dir_out, recursive = TRUE)
  at_p <- function(Q_list) {
    do.call(rbind, lapply(Q_list, function(q) approx(ex$p_grid, q, xout = p_show, rule = 2)$y))
  }
  Q_real  <- at_p(ex$Q_val)
  Q_synth <- at_p(ex$Q_synth)
  for (j in seq_along(p_show)) {
    p <- p_show[j]
    fig_png(file.path(dir_out, str_glue('gen-ex_{tag}_p-{p}.png')), w, h)
    ## Tight margins: left only for the tick labels (parallel to the axis),
    ## almost none on the right; x range trimmed so the boxes sit close to the
    ## frame
    par(mar = c(0.4, 1.3, 1.2, 0.1), mgp = c(0, 0.2, 0), tcl = -0.15,
        cex = fig_cex, cex.axis = 0.8 * fig_cex_axis, las = 0)
    boxplot(list(Q_real[, j], Q_synth[, j]),
            col      = c(adjustcolor("black", 0.15), adjustcolor("#1B9E77", 0.4)),
            border   = c("black", "#1B9E77"),
            outline  = outliers, whisklty = 1,
            medlwd   = 1.5,   # as in the score boxplots (plot_generativity_boxes)
            outpch = 16, outcex = 0.3, boxwex = 0.7,
            xlim = c(0.55, 2.45), xaxt = 'n', yaxt = 'n')
    tk <- pretty(par('usr')[3:4], n = 3)   # about 3 ticks
    axis(2, at = tk[tk >= par('usr')[3] & tk <= par('usr')[4]])
    mtext(bquote(p == .(p)), side = 3, line = 0.25, cex = 0.8)
    dev.off()
  }
}

## All figures for one rebuilt replicate (generativity_replicate()): QF space,
## z-rot space and c space, each plain and (unless gen_outlier_w2 is NULL) with
## outliers highlighted. Files are gen-ex_{tag}{_z-rot,_c}{,_outliers}.png in
## dir_out. idx_full: rows of the full data (z_rot_ctx / c_ctx) of the
## replicate's held-out real subjects (ex$idx_val indexes the y_list passed to
## the evaluation, which may be a subset of the full data).
## highlight_outliers: which versions to draw -- FALSE (plain), TRUE (outliers
## highlighted, *_outliers.png) or c(FALSE, TRUE) (both; the default). Ignored
## (plain only) when gen_outlier_w2 is NULL.
## qf_w / qf_h: size (inches) of the QF-space panel. qf_text_scale multiplies
## the QF panel's title, axis-label and tick-label sizes (e.g. 0.8 for narrow
## figure columns). embeddings = FALSE skips the z-rot and c views.
plot_gen_replicate <- function(ex, main_gen, dir_out, tag,
                               idx_full = ex$idx_val,
                               highlight_outliers = c(FALSE, TRUE),
                               qf_w = gen_fig_ex_w, qf_h = gen_fig_ex_h,
                               qf_text_scale = 1, embeddings = TRUE) {
  is_out_w2 <- if (is.null(gen_outlier_w2)) rep(FALSE, length(ex$w2_synth)) else
    ex$w2_synth >= gen_outlier_w2

  if (embeddings) {
    ## z-rot space. With latent 'z_rot' or 'z', the synthetic draws live in the
    ## flow output (stage 4), so rotate them with the PCA stage; real held-out
    ## subjects come from the already-encoded z_rot.
    Z_rot_synth <- do.call(rbind, encode(
      pipeline,
      new_context(asplit(ex$z_draws, 1), pipeline$training$cache, list()),
      from = 4, to = 5
    )$payload)
    Z_rot_real <- Z_rot_all[idx_full, , drop = FALSE]

    ## c space (WAME output, stage 3): decode the synthetic flow-output draws
    ## back through the flow; real held-out subjects come from the
    ## already-encoded c. Both in the pipeline's c sign convention
    ## (c_ctx_to_matrix).
    C_synth <- c_ctx_to_matrix(pipeline, decode(
      pipeline,
      new_context(asplit(ex$z_draws, 1), pipeline$training$cache, list()),
      from = 4, to = 3
    ))
    C_real <- C_all[idx_full, , drop = FALSE]
  }

  ## Versions: plain, every synthetic draw in the same color (gen-ex_{tag}*.png),
  ## and/or with outliers highlighted (*_outliers.png), per highlight_outliers.
  ## Only the plain version when outlier flagging is disabled.
  hl_modes <- if (is.null(gen_outlier_w2)) FALSE else unique(highlight_outliers)
  for (hl in hl_modes) {
    is_out <- if (hl) is_out_w2 else rep(FALSE, length(is_out_w2))
    sfx    <- if (hl) "_outliers" else ""

    ## QF space. No legend at panel size (the LaTeX caption explains the
    ## colors); when highlighting, the outlier count goes in the title instead.
    main_gen_qf <- if (!hl) main_gen else {
      n_out <- sum(is_out)
      bquote(.(main_gen) * "," ~~ .(sprintf("%d %s", n_out, if (n_out == 1) "outlier" else "outliers")))
    }
    fig_png(file.path(dir_out, str_glue('gen-ex_{tag}{sfx}.png')), qf_w, qf_h)
    fig_par(cex_main = qf_text_scale * fig_cex_main,
            cex_lab  = qf_text_scale * fig_cex_lab, cex = fig_cex,
            cex_axis = qf_text_scale * fig_cex_axis)
    plot_generativity_example(
      ex,
      main = main_gen_qf,
      show_legend = FALSE,
      outlier_w2 = if (hl) gen_outlier_w2 else NULL,
      col_outlier = col_gen_outlier
    )
    dev.off()

    ## z-rot and c spaces
    if (!embeddings) next
    leg_labels <- c("real (val)", "synthetic")
    leg_cols   <- c("black", "#1B9E77")
    if (hl) {
      leg_labels <- c(leg_labels, sprintf("synthetic outlier (n = %d)", sum(is_out)))
      leg_cols   <- c(leg_cols, col_gen_outlier)
    }
    plot_gen_ex_embeddings(
      file.path(dir_out, str_glue('gen-ex_{tag}_z-rot{sfx}.png')),
      Z_rot_real, Z_rot_synth, is_out, main_gen, leg_labels, leg_cols
    )
    plot_gen_ex_embeddings(
      file.path(dir_out, str_glue('gen-ex_{tag}_c{sfx}.png')),
      C_real, C_synth, is_out, main_gen, leg_labels, leg_cols
    )
  }
}

## ----- Strata generativity figures (group comparison and AMM)
##
## The covariate-aware checks of an evaluate_generativity_strata() result are
## shown in two kinds of figure, assembled from separate pngs in LaTeX:
##
## Unconditional ("All"; plot_gen_unconditional): one full-width row --
##   the "All" boxplot (gen_unc_box_w), the QF panel of its median replicate
##   (gen_unc_qf_w) and its real vs synthetic quantile panels at gen_q_p
##   (gen_unc_q_w each), all gen_unc_h tall.
##
## Conditional (one per stratifying variable): three rows --
##   1. boxplot of the scores by stratum, full width, the median score of each
##      box marked (plot_generativity_strata(include_all = FALSE));
##   2. per box, directly below it, the QF panel of that median replicate,
##      titled G~ = ... (box identity left to the caption);
##   3. per box, below its QF panel, real vs synthetic boxplots of Q_i(p) at
##      gen_q_p (plot_gen_replicate_quantiles).
##   Each stratum gets a gen_cond_col_w-wide column (the figure is
##   n_strata * gen_cond_col_w wide, so a 2-stratum figure is not stretched
##   relative to a 5-stratum one): its row-2 panel fills the column, its row-3
##   panels a third of it each.
## Boxplots that belong together (a section's unconditional and conditional
## figures) share one y axis (generativity_ylim). gen_q_outliers toggles the
## outlier points of the quantile boxplots.
gen_q_p        <- c(0.5, 0.75, 0.99)
gen_q_outliers <- TRUE
gen_unc_h      <- 1.8   # TODO: final printed height
gen_unc_box_w  <- 1.1
gen_unc_qf_w   <- 2.4
gen_unc_q_w    <- (6.5 - gen_unc_box_w - gen_unc_qf_w) / length(gen_q_p)
gen_cond_col_w <- 6.5 / 5   # per-stratum column: 5 quintile strata span 6.5 in

## Median-score replicate of each box of an evaluate_generativity_strata()
## result (run with keep_draws = TRUE): the ceiling(n / 2)-th smallest of the
## box's S * R scores. Returns one row per box: group (box index), ref (box
## name), split, rep, log_norm and label (always 1; kept as the file-name
## suffix). Mark them on the boxplot with
## plot_generativity_strata(highlight = ex_df[, c("group", "log_norm")] ...).
select_gen_strata_examples <- function(gen_strata) {
  do.call(rbind, lapply(seq_along(gen_strata$gen), function(g) {
    res <- gen_strata$gen[[g]]
    sc  <- do.call(rbind, lapply(seq_along(res$splits), function(s) {
      data.frame(split = s, rep = seq_along(res$splits[[s]]$log_norm),
                 log_norm = res$splits[[s]]$log_norm)
    }))
    sc <- sc[order(sc$log_norm), ]
    data.frame(group = g, ref = names(gen_strata$gen)[g],
               sc[ceiling(nrow(sc) / 2), ], label = 1)
  }))
}

## Rows 2 and 3 for select_gen_strata_examples() rows: the QF panel
## (plot_gen_replicate, qf_w x qf_h, text at 0.8x, no z-rot / c views) and the
## row-3 quantile panels (plot_gen_replicate_quantiles at gen_q_p, q_w x q_h
## each; q_w defaults to qf_w / 3). Files are gen-ex_{tag}.png and
## gen-ex_{tag}_quantiles/gen-ex_{tag}_p-{p}.png in dir_out, with
## tag = {tag_prefix}_{tolower(ref)}_{label}. y_list_sub: the y_list passed to
## the evaluation; idx_full: rows of the full data behind y_list_sub.
## The QF panel's random curve draw order is seeded per example (seed), so
## every png is identical however the script is run. Returns (invisibly) the
## paths of the pngs for assemble_gen_strata_figure(): qf (row 2, one per box;
## the plain version when highlight_outliers includes FALSE) and q (row 3, a
## vector of gen_q_p panels per box).
plot_gen_strata_examples <- function(gen_strata, ex_df, y_list_sub, idx_full,
                                     dir_out, tag_prefix, qf_w, qf_h, q_h,
                                     q_w = qf_w / length(gen_q_p),
                                     highlight_outliers = FALSE,
                                     seed = gen_seed_) {
  if (!dir.exists(dir_out)) dir.create(dir_out, recursive = TRUE)
  sfx_qf  <- if (is.null(gen_outlier_w2) || FALSE %in% highlight_outliers) "" else "_outliers"
  paths_qf <- character(0)
  paths_q  <- list()
  for (j in seq_len(nrow(ex_df))) {
    set.seed(seed)
    e  <- ex_df[j, ]
    ex <- generativity_replicate(gen_strata$gen[[e$ref]], pipeline, y_list_sub,
                                 s = e$split, r = e$rep)
    main_gen <- bquote(tilde(G) == .(sprintf("%.2f", ex$log_norm)))
    tag_ex <- str_glue('{tag_prefix}_{tolower(e$ref)}_{e$label}')
    plot_gen_replicate(ex, main_gen, dir_out, tag = tag_ex,
                       idx_full = idx_full[ex$idx_val],
                       highlight_outliers = highlight_outliers,
                       qf_w = qf_w, qf_h = qf_h,
                       qf_text_scale = 0.8, embeddings = FALSE)
    dir_q <- file.path(dir_out, str_glue('gen-ex_{tag_ex}_quantiles'))
    plot_gen_replicate_quantiles(
      ex,
      dir_out  = dir_q,
      tag      = tag_ex,
      p_show   = gen_q_p,
      w        = q_w,
      h        = q_h,
      outliers = gen_q_outliers
    )
    paths_qf <- c(paths_qf, file.path(dir_out, str_glue('gen-ex_{tag_ex}{sfx_qf}.png')))
    paths_q[[j]] <- file.path(dir_q, str_glue('gen-ex_{tag_ex}_p-{gen_q_p}.png'))
  }
  invisible(list(qf = paths_qf, q = paths_q))
}

## The assembled three-row figure, for previews and slides (the paper builds
## the same layout in LaTeX from the separate pngs): row 1 the boxplot png
## (path_box), row 2 the QF panels and row 3 the quantile panels in box order
## (panels: the return value of plot_gen_strata_examples), stacked in one png
## at path_out. Every png is rendered at its printed size and resolution, so
## the rows line up as they do in LaTeX.
assemble_gen_strata_figure <- function(path_box, panels, path_out) {
  rd   <- magick::image_read
  rows <- c(rd(path_box),
            magick::image_append(rd(panels$qf)),
            magick::image_append(rd(unlist(panels$q))))
  magick::image_write(magick::image_append(rows, stack = TRUE), path_out)
  invisible(path_out)
}

## Split select_gen_strata_examples() rows into the unconditional ("All") row
## and the conditional rows, the latter with group renumbered to count strata
## only (for plot_generativity_strata(include_all = FALSE, highlight = ...)).
split_gen_strata_examples <- function(ex_df) {
  cond <- ex_df[ex_df$ref != "All", , drop = FALSE]
  cond$group <- cond$group - 1
  list(uncond = ex_df[ex_df$ref == "All", , drop = FALSE], cond = cond)
}

## Unconditional figure of an evaluate_generativity_strata() result gs (see
## "Strata generativity figures"): the "All" boxplot (median replicate marked,
## y range ylim) to path_box, the median replicate's QF and quantile panels to
## dir_ex (plot_gen_strata_examples, tag {tag_prefix}_all_1), and the
## assembled row to path_assembled. y_list_sub / idx_full as for
## plot_gen_strata_examples.
plot_gen_unconditional <- function(gs, ex_uncond, ylim, y_list_sub, idx_full,
                                   path_box, dir_ex, tag_prefix,
                                   path_assembled) {
  fig_png(path_box, gen_unc_box_w, gen_unc_h)
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
  plot_generativity_boxes(
    groups          = "All",
    scores          = list(lapply(gs$gen$All$splits, `[[`, "log_norm")),
    group_sublabels = list(bquote("(" * N == .(gs$n[["All"]]) * ")")),
    hline           = 0,
    ylab            = "Relative generativity",
    collapse        = TRUE,
    ylim            = ylim,
    highlight       = data.frame(group = 1, value = ex_uncond$log_norm)
  )
  dev.off()
  panels <- plot_gen_strata_examples(
    list(gen = gs$gen["All"]), ex_uncond,
    y_list_sub = y_list_sub, idx_full = idx_full,
    dir_out = dir_ex, tag_prefix = tag_prefix,
    qf_w = gen_unc_qf_w, qf_h = gen_unc_h,
    q_w = gen_unc_q_w, q_h = gen_unc_h
  )
  rd <- magick::image_read
  magick::image_write(
    magick::image_append(c(rd(path_box), rd(panels$qf), rd(unlist(panels$q)))),
    path_assembled
  )
  invisible(path_assembled)
}

for (j in seq_len(nrow(gen_ex))) {
  ex <- generativity_replicate(gen_res, pipeline, y_list,
                               s = gen_ex$split[j], r = gen_ex$rep[j])
  main_gen <- bquote(.(paste0("(", j, ")")) ~~ tilde(G) == .(sprintf("%.2f", ex$log_norm)))
  plot_gen_replicate(ex, main_gen, dir_gen_ex, tag = j)
}




## ========== Frechet Means ========== ##

## Common grid
Ji <- 7350
pi_grid <- pi_grid_fun(Ji)

## Q-mean (linearly interpolate each Qi onto the common grid, then average)
mean_q <- numeric(Ji)
for (Qi in Qi_ctx$payload) {
  mean_q <- mean_q + approx(
    x = pi_grid_fun(length(Qi)), y = Qi,
    xout = pi_grid,
    rule = 2
  )$y
}
mean_q <- mean_q / length(Qi_ctx$payload)

## Z-mean
mean_z <- colMeans(do.call(rbind, z_rot_ctx$payload))
mean_z <- decode_z_to_Qi(pipeline, list(mean_z), Ji = Ji)[[1]]

## F-mean (L2 in CDF space)
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
  dir_rep_plots,
  str_glue('frechet_nhanes.png')
)
fig_png(path_plot, 3.25, 3.25)   # TODO: final printed size
col_q <- 'red'
col_f <- 'green'
col_z <- 'blue'
fun_list <- c(
  Qi_ctx$payload,
  list(mean_q),
  list(mean_f),
  list(mean_z)
)
grid_list <- c(
  lapply(lengths(Qi_ctx$payload), pi_grid_fun),
  rep(list(pi_grid), 3)
)
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
  color_width_type_labels = color_width_type_labels,
  cex.lab = fig_cex_lab,
  cex_main = fig_cex_main,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend
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
      'male' = 'F',
      'low_income' = '> PL'
    )
  } else {
    name_map <- list(
      'old' = '>= 65',
      'male' = 'M',
      'low_income' = '<= PL'
    )
  }
  name_map[[cov]]
}

## Legend label with group sample size, e.g. "M (n = 4,012)"
fmt_n <- function(n) format(n, big.mark = ',', scientific = FALSE)
label_with_n <- function(label, n) str_glue('{label} (n = {fmt_n(n)})')


## Plot QFs color-coded by covariate level
for (bin_pred in c('old', 'male', 'low_income')) {

  ## Legend labels with per-group sample sizes (non-missing subjects only)
  x_bin <- df_cov[[bin_pred]][!is.na(df_cov[[bin_pred]])]
  bin_labels <- c(
    label_with_n(get_plot_name_for_cov(bin_pred, TRUE),  sum(x_bin == 0)),
    label_with_n(get_plot_name_for_cov(bin_pred, FALSE), sum(x_bin == 1))
  )

  ## Plot: QF
  set.seed(1234)
  path_plot <- file.path(
    dir_eda,
    str_glue('eda_Qi_{bin_pred}.png')
  )
  fig_png(path_plot, 2.15, 2.15)   # TODO: final printed size
  col_0 <- rgb(0, 0, 0, alpha = 0.05)
  col_1 <- rgb(1, 0, 0, alpha = 0.05)
  col_0_leg <- rgb(0, 0, 0)   # opaque legend keys; curves stay translucent
  col_1_leg <- rgb(1, 0, 0)
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
    color = c(col_0_leg, col_1_leg),
    width = c(2.5, 2.5),
    type  = c(1, 1),
    label = bin_labels
  )
  plot_funs(
    fun_list = fun_list[idx],
    grid_list = grid_list[idx],
    colors = colors[idx],
    widths = widths[idx],
    types = types[idx],
    color_width_type_labels = color_width_type_labels,
    cex.lab = fig_cex_lab,
    cex_main = fig_cex_main,
    cex = fig_cex,
    cex_axis = fig_cex_axis,
    legend_cex = fig_cex_legend
  )
  dev.off()

  ## Plot: z-rot-embeddings
  set.seed(1234)
  path_plot <- file.path(
    dir_eda,
    str_glue('eda_z-rot_{bin_pred}.png')
  )
  fig_png(path_plot, w_emb, w_emb)
  col_0 <- rgb(0, 0, 0, alpha = 0.25)
  col_1 <- rgb(1, 0, 0, alpha = 0.25)
  mask <- !is.na(df_cov[bin_pred])
  idx <- sample(1:sum(mask))
  Z_plot <- do.call(rbind, z_rot_ctx$payload)[mask, ks_plot, drop = FALSE]
  df_cov_to_plot <- df_cov[mask,]
  colors = rep(col_0, length.out = nrow(Z_plot))
  colors[df_cov_to_plot[bin_pred] == 1] <- col_1
  shapes <- c(
    rep(16, length.out = nrow(Z_plot))
  )
  sizes <- c(
    rep(0.5, length.out = nrow(Z_plot))
  )
  color_shape_size_labels = data.frame(
    color = c(col_0, col_1),
    shape = c(16, 16),
    size = c(0.5, 0.5),
    label = bin_labels
  )
  plot_embeddings(
    Z_plot, 
    colors = colors,
    shapes = shapes,
    sizes = sizes,
    color_shape_size_labels = color_shape_size_labels,
    stats = FALSE,
    cex_main = fig_cex_main,
    cex_lab = fig_cex_lab,
    cex = fig_cex,
    cex_axis = fig_cex_axis,
    legend_cex = fig_cex_legend,
    ks = ks_plot
  )
  dev.off()

}

## Plot QFs color-coded by continuous covariate (blue = low, red = high;
## linear in the covariate over its observed range), with a color bar legend.
cont_cov_labels <- c(age = 'Age', pir = 'PIR')
for (cont_pred in names(cont_cov_labels)) {

  ## Plot: QF
  set.seed(1234)
  path_plot <- file.path(
    dir_eda,
    str_glue('eda_Qi_{cont_pred}.png')
  )
  fig_png(path_plot, 2.15, 2.15)   # TODO: final printed size
  pal <- colorRampPalette(c("blue", "gray60", "red"))(101)
  val_to_col <- function(v, rng) {
    adjustcolor(pal[round((v - rng[1]) / diff(rng) * 100) + 1], alpha.f = 0.05)
  }
  mask <- !is.na(df_cov[[cont_pred]])
  idx <- sample(1:sum(mask))
  Q_to_plot <- Q_ctx$payload[mask]
  vals <- df_cov[[cont_pred]][mask]
  rng <- range(vals)
  fun_list <- c(
    Q_to_plot
  )
  grid_list <- rep(list(p_grid), length(fun_list))
  colors <- val_to_col(vals, rng)
  widths <- rep(1, length(fun_list))
  types <- rep(1, length(fun_list))
  ticks <- pretty(rng, n = 4)
  ticks <- ticks[ticks >= rng[1] & ticks <= rng[2]]
  plot_funs(
    fun_list = fun_list[idx],
    grid_list = grid_list[idx],
    colors = colors[idx],
    widths = widths[idx],
    types = types[idx],
    cex.lab = fig_cex_lab,
    cex_main = fig_cex_main,
    cex = fig_cex,
    cex_axis = fig_cex_axis,
    after_plot = function() {
      draw_colorbar(pal, rng, ticks,
                    title = label_with_n(cont_cov_labels[[cont_pred]],
                                         length(vals)),
                    cex = fig_cex_legend, vals = vals)
    }
  )
  dev.off()

}


## ========== Latent Component-Wise Traversals ========== ##

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
  fig_png(path_plot, 2.15, 2.15)   # TODO: final printed size
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
  ## Legend: quantile levels of latent coordinate k (named in the title)
  leg_args <- list(
    "topleft",
    legend  = sprintf("%.1f", k_star_quantiles),
    col     = line_cols,
    lwd     = 2,
    seg.len = 1.5,
    ncol    = 3,
    title   = "Quantile",
    cex     = 0.8*fig_cex_legend,
    bty     = "n"
  )

  ## Raise the top of ylim just enough that the legend clears every curve:
  ## measure the legend on a provisional window, find the highest curve point
  ## under it, then solve for the top that leaves the legend above that point.
  plot.new()
  plot.window(xlim = c(0, 1), ylim = y_lim)
  usr <- par("usr")
  leg <- do.call(legend, c(leg_args, plot = FALSE))
  under_leg <- pi_grid <= leg$rect$left + leg$rect$w
  y_max_under <- max(sapply(Qi_plot, function(q) max(q[under_leg])))
  f_leg <- leg$rect$h / diff(usr[3:4])   # legend height as a fraction of usr
  pad   <- 0.02                          # gap between legend and curves
  ext   <- 0.04                          # yaxs = "r" pads usr by 4% each side
  c_    <- ext - (1 + 2 * ext) * (f_leg + pad)
  y_top <- (y_max_under + c_ * y_lim[1]) / (1 + c_)
  y_lim[2] <- max(y_lim[2], y_top)

  ## Draw (axes, box and titles by hand, since plot() would start a new page)
  plot.window(xlim = c(0, 1), ylim = y_lim)
  axis(1)
  axis(2)
  box()
  title(
    main = str_glue("k = {k_star}"),
    xlab = "p",
    ylab = "Q(p) - Q_0.5(p)"
  )
  for (i in seq_along(Qi_plot)) {
    lines(pi_grid, Qi_plot[[i]], col = line_cols[i], lwd = 2)
  }
  do.call(legend, leg_args)
  dev.off()

}


## ========== Group Comparison ========== ##

## Set binary predictor
bin_pred <- 'male'
run_latent_diagnostics <- TRUE
run_generativity_diagnostics <- TRUE

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
  dir_diag_gc <- file.path(dir_gc, 'latent-diagnostics')
  if (!dir.exists(dir_diag_gc)) dir.create(dir_diag_gc, recursive = TRUE)
  binary_diagnostics(
    model, dir_diag_gc,
    open_png  = fig_png,
    col_train = col_train,
    gp_names  = c(gp0_name, gp1_name),
    pairs_width_in = 6.5,
    width_in  = 4.5,
    fig_args  = list(cex_main = fig_cex_main, cex_lab = fig_cex_lab,
                     cex = fig_cex, cex_axis = fig_cex_axis),
    cell_cex  = fig_cex_axis
  )
}


## Generativity diagnostics: covariate-aware scores, unconditional (all
## subjects) and conditional on each group, on shared axes. Scored against the
## full validation sets; keep_draws = TRUE so the examples below are the exact
## datasets behind boxplot points (same presentation as "Check generativity").
## The evaluation is cached in dir_gc; delete gen_res_{bin_pred}.rds to
## recompute after refitting the pipeline or changing the settings below.
if (run_generativity_diagnostics) {
  S <- 5
  R <- 20
  X <- matrix(x, ncol = 1)
  gc_labels <- c("0" = get_plot_name_for_cov(bin_pred, TRUE),
                 "1" = get_plot_name_for_cov(bin_pred, FALSE))
  path_gen_gc <- file.path(dir_gc, str_glue('gen_res_{bin_pred}.rds'))
  if (file.exists(path_gen_gc)) {
    gen_gc <- readRDS(path_gen_gc)
  } else {
    gen_gc <- evaluate_generativity_strata(
      pipeline        = pipeline,
      y_list          = y_list[mask],
      X               = X,
      strata          = x,
      fit_fun         = fit_binary,
      sample_fun      = sample_binary,
      S               = S,
      R               = R,
      val_subsample_n = NULL,
      n_cores         = 1,
      keep_draws      = TRUE
    )
    saveRDS(gen_gc, path_gen_gc)
  }

  ## Examples: the median-score replicate of each box, marked on the boxplots
  gen_ex_gc <- split_gen_strata_examples(select_gen_strata_examples(gen_gc))
  ylim_gc   <- generativity_ylim(gen_gc)   # shared by both figures
  dir_gen_ex_gc <- file.path(dir_gc, 'gen-examples')

  ## Unconditional figure (one row; see "Strata generativity figures"):
  ## gen_{bin_pred}_uncond.png, gen-examples/gen-ex_{bin_pred}_all_1*, and the
  ## assembled gen_{bin_pred}_uncond_assembled.png
  plot_gen_unconditional(
    gen_gc, gen_ex_gc$uncond, ylim_gc,
    y_list_sub     = y_list[mask],
    idx_full       = which(mask),   # rows of the full data behind y_list[mask]
    path_box       = file.path(dir_gc, str_glue('gen_{bin_pred}_uncond.png')),
    dir_ex         = dir_gen_ex_gc,
    tag_prefix     = bin_pred,
    path_assembled = file.path(dir_gc, str_glue('gen_{bin_pred}_uncond_assembled.png'))
  )

  ## Conditional figure (three rows): one gen_cond_col_w column per stratum
  ## (boxplot gc_fig_w = n_strata columns wide), each stratum's QF panel one
  ## column and its quantile panels a third of that, so each stratum's panels
  ## stack below it in LaTeX (every png rendered at its printed size).
  ## gc_highlight_outliers toggles the QF panel versions: FALSE plain, TRUE
  ## outliers highlighted (*_outliers.png), c(FALSE, TRUE) both.
  n_strata   <- length(gen_gc$levels)
  gc_ex_w    <- gen_cond_col_w
  gc_fig_w   <- n_strata * gc_ex_w
  gc_box_h   <- 1.8   # TODO: final printed height
  gc_ex_h    <- 1.4   # TODO: final printed height
  gc_q_h     <- 1.2   # TODO: final printed height
  gc_highlight_outliers <- FALSE

  path_gen_box_gc <- file.path(dir_gc, str_glue('gen_{bin_pred}_cond.png'))
  fig_png(path_gen_box_gc, gc_fig_w, gc_box_h)
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
  plot_generativity_strata(
    gen_gc,
    labels        = gc_labels,
    include_all   = FALSE,   # strata only; "All" is its own figure
    ylim          = ylim_gc,
    highlight     = gen_ex_gc$cond[, c("group", "log_norm")] |>
      setNames(c("group", "value")),   # marker only, no label
    align_columns = TRUE   # boxes centered over their example columns
  )
  dev.off()

  ## Rows 2-3: files gen-ex_{bin_pred}_{0|1}_1.png and
  ## gen-ex_{bin_pred}_{0|1}_1_quantiles/ in gen-examples/
  panels_gc <- plot_gen_strata_examples(
    gen_gc, gen_ex_gc$cond,
    y_list_sub = y_list[mask],
    idx_full   = which(mask),
    dir_out    = dir_gen_ex_gc,
    tag_prefix = bin_pred,
    qf_w = gc_ex_w, qf_h = gc_ex_h, q_h = gc_q_h,
    highlight_outliers = gc_highlight_outliers
  )

  ## Assembled three-row figure: gen_{bin_pred}_cond_assembled.png
  assemble_gen_strata_figure(
    path_gen_box_gc, panels_gc,
    file.path(dir_gc, str_glue('gen_{bin_pred}_cond_assembled.png'))
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
xlim_rb <- if (latent == 'z_rot') c(-0.5, 0.5) else NULL
fig_png(path_plot, 2.15, 2.15)   # top-left panel of the 2x3 traversal figure
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
par(mgp = c(1.9, 0.5, 0))   # keep the hat on beta_k clear of the tick labels
binary_plot_intervals(
  res_rb,
  xlim = xlim_rb,
  # main = str_glue("Joint Bands for Latent Regression Coefficients ({gp1_name})"),
  main = gp1_name,
  mar = c(3.3, 2.8, 2.2, 0.3),
  legend_in_margin = TRUE   # p-value legend in the right margin, clear of intervals
)
dev.off()

## MIMS activity-intensity thresholds (per-minute units) for CDF plots.
mims_thresholds <- list(
  list(
    label = "light",
    value = 15.05,
    col = 'darkgreen',
    eval_type = 'cdf',
    main = "Prop in sedentary"
  ),
  list(
    label = "mvpa",
    value = 19.61,
    col = 'darkblue',
    eval_type = 'survival',
    main = "Prop in MVPA"
  ),
  list(
    label = "strenuous",
    value = 50.00,
    col = 'darkred',
    eval_type = 'survival',
    main = "Prop in strenuous"
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

## ----- Latent Traversal: Mean-Difference

## For two groups the latent mean difference lies along the single direction
## u = beta_hat / ||beta_hat||. Sweep the projections s_i = Z_i^T u across their
## sample deciles, holding the orthogonal complement at the sample mean (the
## component-wise traversal in a basis whose first axis is u), and decode. This
## shows what the direction means at population scale; the two group means are
## overlaid (dashed) with their percentile ranks in s to show the effect size.
## u is chosen from the data, but Roy-Bose intervals cover every projection
## a^T beta jointly, so the interval for u^T beta keeps its simultaneous
## coverage.

## Roy-Bose interval for the projection onto the mean-difference direction
beta_hat <- coef(model)["x", ]
u_md     <- beta_hat / sqrt(sum(beta_hat^2))
res_rb_md <- binary_interval_projection(model, u_md)
print(res_rb_md)

## Append to the saved results
sink(file.path(dir_gc, str_glue('mod_{bin_pred}.txt')), append = TRUE)
print("========== Roy-Bose: mean-difference projection ==========")
print(res_rb_md)
sink()

## Projections onto u of the fitted sample, and of the two group means
Z_gc    <- fitted(model) + residuals(model)
s_md    <- as.numeric(Z_gc %*% u_md)
z_bar   <- colMeans(Z_gc)
s_bar   <- mean(s_md)
z_gp    <- list(coef(model)["(Intercept)", ],
                coef(model)["(Intercept)", ] + beta_hat)
s_gp    <- sapply(z_gp, function(z) sum(z * u_md))
pct_gp  <- 100 * ecdf(s_md)(s_gp)   # percentile rank of each group mean in s
print(setNames(round(pct_gp, 1), c(gp0_name, gp1_name)))

## Append the group-mean percentile ranks to the saved results
sink(file.path(dir_gc, str_glue('mod_{bin_pred}.txt')), append = TRUE)
print("========== Group-mean percentile ranks along u ==========")
print(setNames(round(pct_gp, 1), c(gp0_name, gp1_name)))
sink()

## Decode z(q) = z_bar + (Q_s(q) - s_bar) * u at the deciles of s, followed by
## the two group means projected onto the same line
q_md  <- seq(0.1, 0.9, by = 0.1)
s_pts <- c(quantile(s_md, q_md), s_gp)
Z_md  <- t(sapply(s_pts, function(s) z_bar + (s - s_bar) * u_md))
Ji    <- length(Qi_ctx$payload[[1]])
Qi_md <- decode_latent_to_Qi(
  pipeline,
  latent_list = asplit(Z_md, MARGIN = 1),
  Ji          = Ji
)

## Center each QF by subtracting the median (q = 0.5) QF
idx_med <- which(q_md == 0.5)
Qi_med  <- Qi_md[[idx_med]]
Qi_md   <- lapply(Qi_md, function(q) q - Qi_med)
idx_gp  <- length(q_md) + 1:2

## Plot (same layout as the component-wise traversals)
pi_grid <- pi_grid_fun(Ji)
path_plot <- file.path(dir_gc, str_glue('traversal-md_{bin_pred}.png'))
pal <- colorRampPalette(c("blue", "gray", "red"))(101)
line_cols <- pal[round(q_md * 100) + 1]
gp_ltys   <- c(2, 3)
y_lim <- range(unlist(Qi_md))
fig_png(path_plot, 2.15, 2.15)   # TODO: final printed size
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
## Legend: deciles of s, then the group means with their percentile ranks
leg_args <- list(
  "topleft",
  legend  = c(sprintf("%.1f", q_md),
              sprintf("%s (%.0f)", c(gp0_name, gp1_name), pct_gp)),
  col     = c(line_cols, "black", "black"),
  lty     = c(rep(1, length(q_md)), gp_ltys),
  lwd     = c(rep(2, length(q_md)), 1.5, 1.5),
  seg.len = 1.5,
  ncol    = 3,
  title   = "Quantile",
  cex     = 0.8*fig_cex_legend,
  bty     = "n"
)
## Raise the top of ylim so the legend clears every curve (as above)
plot.new()
plot.window(xlim = c(0, 1), ylim = y_lim)
usr <- par("usr")
leg <- do.call(legend, c(leg_args, plot = FALSE))
under_leg <- pi_grid <= leg$rect$left + leg$rect$w
y_max_under <- max(sapply(Qi_md, function(q) max(q[under_leg])))
f_leg <- leg$rect$h / diff(usr[3:4])
pad   <- 0.02
ext   <- 0.04
c_    <- ext - (1 + 2 * ext) * (f_leg + pad)
y_top <- (y_max_under + c_ * y_lim[1]) / (1 + c_)
y_lim[2] <- max(y_lim[2], y_top)

## Draw
plot.window(xlim = c(0, 1), ylim = y_lim)
axis(1)
axis(2)
box()
title(
  main = str_glue("Mean difference ({gp1_name})"),
  xlab = "p",
  ylab = "Q(p) - Q_0.5(p)"
)
for (i in seq_along(q_md)) {
  lines(pi_grid, Qi_md[[i]], col = line_cols[i], lwd = 2)
}
for (g in 1:2) {
  lines(pi_grid, Qi_md[[idx_gp[g]]], col = "black", lty = gp_ltys[g],
        lwd = 1.5)
}
do.call(legend, leg_args)
dev.off()


## ----- Conditional QF Plots

## --- Sampling distribution of conditional latent means

alpha      <- 0.05
mean_res   <- binary_sample_means(model, R = 1000, seed = 1234)
z_hat_0    <- mean_res$z_hat_0
z_hat_1    <- mean_res$z_hat_1
z_draws_0  <- mean_res$z_draws_0
z_draws_1  <- mean_res$z_draws_1

## Decode point estimates and sampled conditional means to Qi-space
Qi_hat_0   <- decode_latent_to_Qi(pipeline, list(z_hat_0), Ji)[[1]]
Qi_hat_1   <- decode_latent_to_Qi(pipeline, list(z_hat_1), Ji)[[1]]
Qi_draws_0 <- decode_latent_to_Qi(pipeline, z_draws_0, Ji)
Qi_draws_1 <- decode_latent_to_Qi(pipeline, z_draws_1, Ji)


## --- Paper figure settings for the conditional plots below
## Page layout (6.5 in text width):
##   row 1: QF contrast | survival contrast        (half width each)
##   row 2: quantile levels | thresholds           (half-width strips, 3 panels each)
##   row 3: moments                                (full-width strip, 4 panels)
## The strips are short (two estimates per panel). Groups are told apart by
## colour (black = gp0, red = gp1; state this in the caption), so the panels
## drop their x-axis group labels and y labels (the panel titles say what is
## plotted), and the margins shrink to match.
## Legends stay inside the axes: each single-panel plot moves its legend to a
## clear corner, or extends the y-axis to make room, rather than cover a curve.
## Quantile-level / threshold lines are labelled above the frame (one row,
## nudged sideways where labels would overlap), not in the legend.
gc_half_w   <- 3.25
gc_half_h   <- 2.15   # TODO: final printed height
gc_full_w   <- 6.5
gc_strip_h  <- 1.5    # TODO: final printed height of the panel strips
gc_lwd      <- 1.5    # estimate / legend / threshold line width (half page)
gc_size     <- 0.6    # interval bar, whisker and point scale (strips)
gc_xlim_2   <- c(0.6, 2.4)   # x range for the two group estimates per panel
## Panel margins (bottom/left shrink automatically with no labels). Row 2 fits
## the two-line threshold titles, and the quantile strip uses the same top so
## the two strips' frames line up side by side; moments have one-line titles.
gc_row2_mar   <- c(0, 3.2, 3.4, 0.5)
gc_moment_mar <- c(0, 3.2, 2.2, 0.5)

## Long labels and titles wrapped to fit the small panels and legends
gc_diff_label <- str_wrap(str_glue("{gp1_name} - {gp0_name}"), width = 20)
gc_thr_mains  <- str_wrap(sapply(mims_thresholds, `[[`, "main"), width = 12)
gc_q_mains    <- sprintf("Q(%g)", sapply(quantile_levels, `[[`, "value"))   # quantile-panel titles
gc_half_png <- function(path) {
  fig_png(path, gc_half_w, gc_half_h)
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
}
gc_strip_png <- function(path, width) {
  fig_png(path, width, gc_strip_h)
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
}


## --- Plot conditional QFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_qfs.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_qfs(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha = alpha,
  quantile_lines = quantile_levels,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  legend_cex = fig_cex_legend,
  lwd = gc_lwd
)
dev.off()


## --- Plot difference of conditional QFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_qfs-contrast.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_qf_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha = alpha,
  quantile_lines = quantile_levels,
  threshold_labels = "top",
  diff_label = gc_diff_label,
  legend_loc = 'bottomleft',
  legend_cex = fig_cex_legend,
  lwd = gc_lwd
)
dev.off()


## --- Shared y-grid for CDF plots
y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1)


## --- Plot conditional CDFs with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_cdf.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_cdfs(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha      = alpha,
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()


## --- Plot CDF difference with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_cdf-contrast.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_cdf_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha      = alpha,
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  diff_label = gc_diff_label,
  legend_loc = 'topright',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()


## --- Plot conditional survival functions (1-F(x)) with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_survival.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_survivals(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha      = alpha,
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  band_type  = 'pointwise',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()


## --- Plot survival function difference with bands

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_survival-contrast.png')
)
gc_half_png(path_plot)
plot_decoded_conditional_survival_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha      = alpha,
  y_grid     = y_grid,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  diff_label = gc_diff_label,
  band_type  = 'pointwise',
  legend_loc = 'bottomright',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()


## --- Plot thresholds with bands (one panel per threshold; row 2, right)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_threshold.png')
)
gc_strip_png(path_plot, gc_half_w)
plot_decoded_conditional_thresholds(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  thresholds   = mims_thresholds,
  eval_type    = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains  = gc_thr_mains,
  panel_ylabs  = rep("", length(mims_thresholds)),
  y_grid       = y_grid,
  alpha        = alpha,
  group_labels = c("", ""),
  cex          = fig_cex,
  mar          = gc_row2_mar,
  size         = gc_size,
  xlim         = gc_xlim_2
)
dev.off()


## --- Plot threshold constrasts with bands (one panel per threshold)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_threshold-contrast.png')
)
gc_strip_png(path_plot, gc_half_w)
plot_decoded_conditional_threshold_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  thresholds  = mims_thresholds,
  eval_type   = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains = gc_thr_mains,
  panel_ylabs = rep("", length(mims_thresholds)),
  y_grid      = y_grid,
  alpha       = alpha,
  diff_label  = "",
  cex         = fig_cex,
  mar         = gc_row2_mar,
  size        = gc_size
)
dev.off()


## --- Plot quantile levels with bands (one panel per quantile level; row 2, left)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_quantile-level.png')
)
gc_strip_png(path_plot, gc_half_w)
plot_decoded_conditional_quantile_levels(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  quantile_levels = quantile_levels,
  panel_mains     = gc_q_mains,
  panel_ylabs     = rep("", length(quantile_levels)),
  alpha           = alpha,
  group_labels    = c("", ""),
  cex             = fig_cex,
  mar             = gc_row2_mar,
  size            = gc_size,
  xlim            = gc_xlim_2
)
dev.off()


## --- Plot quantile-level contrasts with bands (one panel per quantile level)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_quantile-level-contrast.png')
)
gc_strip_png(path_plot, gc_half_w)
plot_decoded_conditional_quantile_level_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  quantile_levels = quantile_levels,
  panel_mains     = gc_q_mains,
  panel_ylabs     = rep("", length(quantile_levels)),
  alpha           = alpha,
  diff_label      = "",
  cex             = fig_cex,
  mar             = gc_row2_mar,
  size            = gc_size
)
dev.off()


## --- Plot moments with bands (4 panels, one per moment; row 3)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_moment.png')
)
gc_strip_png(path_plot, gc_full_w)
plot_decoded_conditional_moments(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha        = alpha,
  group_labels = c("", ""),
  panel_ylabs  = rep("", 4),
  cex          = fig_cex,
  mar          = gc_moment_mar,
  size         = gc_size,
  xlim         = gc_xlim_2
)
dev.off()


## --- Contrast in conditional moments (group 1 - group 0)

path_plot <- file.path(
  dir_gc,
  str_glue('gc_{bin_pred}_moment-contrast.png')
)
gc_strip_png(path_plot, gc_full_w)
plot_decoded_conditional_moment_diff(
  Qi_hat_0, Qi_hat_1, Qi_draws_0, Qi_draws_1,
  alpha       = alpha,
  diff_label  = "",
  panel_ylabs = rep("", 4),
  cex         = fig_cex,
  mar         = gc_moment_mar,
  size        = gc_size
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

run_latent_diagnostics <- TRUE
run_generativity_diagnostics <- TRUE

## Create directory
dir_amm_diags <- file.path(dir_amm, 'diagnostics')
if (!dir.exists(dir_amm_diags)) dir.create(dir_amm_diags)

## ----- Latent Diagnostics

## Latent residual diagnostics, styled as the group comparison's
## (latent-diagnostics/): residual Q-Q and pairwise scatter plots, and the
## residual covariance. Residuals R (N x K): R_i = Z_i - X_i beta_hat -
## W_i U_hat (amm$R_mat), over the AMM's fitted (complete-case) subjects.
if (run_latent_diagnostics) {

  dir_amm_latent <- file.path(dir_amm_diags, 'latent-diagnostics')
  if (!dir.exists(dir_amm_latent)) dir.create(dir_amm_latent, recursive = TRUE)
  fig_args_diag <- list(cex_main = fig_cex_main, cex_lab = fig_cex_lab,
                        cex = fig_cex, cex_axis = fig_cex_axis)
  R_amm <- amm$R_mat

  ## Per-column summary table (not a plot)
  summary_tbl <- amm_summary(amm)
  print(summary_tbl)
  write.csv(
    summary_tbl,
    file.path(dir_amm_diags, 'amm_summary.csv'),
    row.names = FALSE
  )

  ## (1) Residual Q-Q (diagonal) and pairwise scatter plots (lower triangle);
  ## full width, square
  fig_png(file.path(dir_amm_latent, 'resid-pairs.png'), 6.5, 6.5)
  plot_resid_pairs(R_amm, col_pt = col_train, point_cex = 0.5,
                   fig_args = fig_args_diag)
  dev.off()

  ## (2) Residual covariance Sigma_E_hat = R^T R / N, at the per-panel size of
  ## the group comparison's two-panel figure (4.5 in for 2 heatmaps + bar)
  Sigma_E_hat <- crossprod(R_amm) / nrow(R_amm)
  fig_png(file.path(dir_amm_latent, 'cov-heatmap.png'), 4.5 * 5 / 9, 2.3)
  plot_cov_heatmaps(list(Sigma_E_hat), list(expression(hat(Sigma)[E])),
                    cell_cex = fig_cex_axis, fig_args = fig_args_diag)
  dev.off()

}


## ----- Generativity Diagnostics 

## Covariate-aware generativity of the AMM, as in the group comparison: scores
## unconditional ("All") and conditional on strata of each AMM covariate --
## male (F / M) and the quantile strata of age and of pir (amm_gen_n_quantiles
## equal-probability strata, Q1, Q2, ..., labeled by their observed ranges) --
## scored against the full validation sets, keep_draws =
## TRUE so the examples are the exact datasets behind boxplot points.
## NOTE: Differences from the group comparison setting
##  - AMM fitting and sampling is aware of all covariates (male, age, pir),
##    whatever the stratifying variable
##  - complete-case sample, so the AMM sample size < group comparison's
## Speed: every evaluation uses the same sample, splits and seed, so each
## split's AMM is fit once and shared through amm_fit_cache (5 fits in total),
## and the unconditional "All" result is computed once and shown once, in its
## own figure (see "Strata generativity figures"). Results are cached in
## dir_amm_gen via
## cached_rds(), which
## recomputes when any setting in amm_gen_settings (or a quantile break, e.g.
## after changing amm_gen_n_quantiles) changes; delete the rds files to force a
## recompute after refitting the pipeline.
if (run_generativity_diagnostics) {

  dir_amm_gen <- file.path(dir_amm_diags, 'generativity')
  if (!dir.exists(dir_amm_gen)) dir.create(dir_amm_gen, recursive = TRUE)

  ## fit_fun for evaluate_generativity()
  fit_amm_nhanes <- function(Z, X) {
    fit_amm(Z, X, n_knots_age = n_knots_age, n_knots_pir = n_knots_pir)
  }

  ## Restrict to complete-case sample (male, age, pir all observed)
  mask_gen     <- complete.cases(covariates_amm)
  y_list_gen   <- y_list[mask_gen]
  X_gen        <- covariates_amm[mask_gen, , drop = FALSE]
  idx_gen      <- which(mask_gen)   # rows of the full data behind y_list_gen

  amm_gen_S    <- 5
  amm_gen_R    <- 20
  amm_gen_seed <- 12345
  amm_gen_examples <- TRUE   # rows 2-3 of the figures (redrawn every run)
  amm_gen_settings <- list(
    S = amm_gen_S, R = amm_gen_R, seed = amm_gen_seed, frac_train = 0.5,
    val_subsample_n = NULL, keep_draws = TRUE,
    n_knots_age = n_knots_age, n_knots_pir = n_knots_pir,
    latent = latent, dir_art = dir_art, N = length(y_list_gen)
  )

  ## Strata of each covariate. Continuous covariates: k equal-probability
  ## strata (k = amm_gen_n_quantiles; 5 = quintiles) over the complete-case
  ## sample (levels Q1-Qk, displayed as each stratum's observed range). Ties
  ## (integer age, pir top-coded at 5) make the strata only roughly equal in
  ## size; the boxplot sublabels give the actual sizes.
  amm_gen_n_quantiles <- 5
  quantile_strata <- function(x, k, digits) {
    br <- quantile(x, probs = seq(0, 1, length.out = k + 1), names = FALSE)
    if (anyDuplicated(br)) stop("Tied quantile breaks: ", toString(br))
    f  <- cut(x, br, include.lowest = TRUE, labels = paste0('Q', seq_len(k)))
    lab <- vapply(levels(f), function(l) {
      paste(formatC(range(x[f == l]), format = 'f', digits = digits),
            collapse = '–')
    }, character(1))
    list(strata = f, labels = lab, breaks = br)
  }
  amm_gen_strata <- list(
    male = list(
      strata = X_gen$male,
      labels = c("0" = get_plot_name_for_cov('male', TRUE),
                 "1" = get_plot_name_for_cov('male', FALSE)),
      breaks = NULL
    ),
    age = quantile_strata(X_gen$age, amm_gen_n_quantiles, digits = 0),
    pir = quantile_strata(X_gen$pir, amm_gen_n_quantiles, digits = 2)
  )

  ## Unconditional result, shared by every stratifying variable
  amm_fit_cache <- new.env()
  gen_amm_all <- cached_rds(
    file.path(dir_amm_gen, 'gen_res_all.rds'), amm_gen_settings,
    function() evaluate_generativity(
      pipeline, y_list_gen,
      X = X_gen, fit_fun = fit_amm_nhanes, sample_fun = sample_amm,
      ref_mask = NULL,
      S = amm_gen_S, R = amm_gen_R, seed = amm_gen_seed,
      val_subsample_n = NULL, n_cores = 1, keep_draws = TRUE,
      fit_cache = amm_fit_cache
    )
  )

  ## Conditional results, one per stratifying variable (all sharing "All")
  gen_amm_res <- lapply(setNames(nm = names(amm_gen_strata)), function(gen_var) {
    st <- amm_gen_strata[[gen_var]]
    cached_rds(
      file.path(dir_amm_gen, str_glue('gen_res_{gen_var}.rds')),
      c(amm_gen_settings, list(breaks = st$breaks)),
      function() evaluate_generativity_strata(
        pipeline        = pipeline,
        y_list          = y_list_gen,
        X               = X_gen,
        strata          = st$strata,
        fit_fun         = fit_amm_nhanes,
        sample_fun      = sample_amm,
        S               = amm_gen_S,
        R               = amm_gen_R,
        seed            = amm_gen_seed,
        val_subsample_n = NULL,
        n_cores         = 1,
        keep_draws      = TRUE,
        fit_cache       = amm_fit_cache,
        res_all         = gen_amm_all
      )
    )
  })

  ## One y axis for the unconditional and all conditional boxplots
  ylim_amm <- do.call(generativity_ylim, unname(gen_amm_res))
  dir_gen_ex_amm <- file.path(dir_amm_gen, 'gen-examples')

  ## Unconditional figure (one row; see "Strata generativity figures"), once
  ## for all variables: amm_gen_uncond.png, gen-examples/gen-ex_uncond_all_1*,
  ## and the assembled amm_gen_uncond_assembled.png
  if (amm_gen_examples) {
    plot_gen_unconditional(
      gen_amm_res[[1]],
      split_gen_strata_examples(select_gen_strata_examples(gen_amm_res[[1]]))$uncond,
      ylim_amm,
      y_list_sub     = y_list_gen,
      idx_full       = idx_gen,
      path_box       = file.path(dir_amm_gen, 'amm_gen_uncond.png'),
      dir_ex         = dir_gen_ex_amm,
      tag_prefix     = 'uncond',
      path_assembled = file.path(dir_amm_gen, 'amm_gen_uncond_assembled.png')
    )
  }

  ## Conditional figures (three rows, as in the group comparison): one
  ## gen_cond_col_w column per stratum (boxplot n_strata columns wide), each
  ## stratum's QF panel one column, its quantile panels a third of that
  amm_gen_box_h <- 1.8   # TODO: final printed height
  amm_gen_ex_h  <- 1.4   # TODO: final printed height
  amm_gen_q_h   <- 1.2   # TODO: final printed height

  for (gen_var in names(amm_gen_strata)) {
    st         <- amm_gen_strata[[gen_var]]
    gen_amm    <- gen_amm_res[[gen_var]]
    gen_ex_amm <- split_gen_strata_examples(select_gen_strata_examples(gen_amm))$cond
    n_strata   <- length(gen_amm$levels)
    amm_gen_fig_w <- n_strata * gen_cond_col_w

    path_gen_box_amm <- file.path(dir_amm_gen, str_glue('amm_gen_{gen_var}.png'))
    fig_png(path_gen_box_amm, amm_gen_fig_w, amm_gen_box_h)
    fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
            cex_axis = fig_cex_axis)
    plot_generativity_strata(
      gen_amm,
      labels        = st$labels,
      include_all   = FALSE,   # strata only; "All" is its own figure
      ylim          = ylim_amm,
      highlight     = gen_ex_amm[, c("group", "log_norm")] |>
        setNames(c("group", "value")),   # marker only, no label
      align_columns = TRUE   # boxes centered over their example columns
    )
    dev.off()

    ## Rows 2-3: files gen-ex_{gen_var}_{level}_1.png and
    ## gen-ex_{gen_var}_{level}_1_quantiles/ in gen-examples/
    if (amm_gen_examples) {
      panels_amm <- plot_gen_strata_examples(
        gen_amm, gen_ex_amm,
        y_list_sub = y_list_gen,
        idx_full   = idx_gen,
        dir_out    = dir_gen_ex_amm,
        tag_prefix = gen_var,
        qf_w = gen_cond_col_w, qf_h = amm_gen_ex_h,
        q_h  = amm_gen_q_h
      )

      ## Assembled three-row figure: amm_gen_{gen_var}_assembled.png
      assemble_gen_strata_figure(
        path_gen_box_amm, panels_amm,
        file.path(dir_amm_gen, str_glue('amm_gen_{gen_var}_assembled.png'))
      )
    }
  }

}


## ========== AMM Inference ========== ##

## Parametric-bootstrap LRTs of the null models in amm_hypotheses() against the
## full AMM (see .claude/summaries/amm-null-model-bootstrap.md)

run_amm_inference <- FALSE

## Create directory
dir_amm_inf <- file.path(dir_amm, 'inference')
if (!dir.exists(dir_amm_inf)) dir.create(dir_amm_inf)
path_amm_inf <- file.path(dir_amm_inf, 'amm_boot_lrt.rds')


## ---------- Driver ----------

B_inf <- 999  ## TODO: Increase this! (B = 999 -> smallest reportable p is <.001)

if (run_amm_inference) {

  ## Full model under ML for the no-effect tests; the REML fit (amm) serves
  ## the linearity tests
  amm_ml <- fit_amm(
    Z_amm, covariates_amm,
    n_knots_age = n_knots_age,
    n_knots_pir = n_knots_pir,
    method = "ML"
  )
  fit_alt <- list(REML = amm, ML = amm_ml)

  hyps <- amm_hypotheses()
  amm_lrts <- list()
  for (j in seq_along(hyps)) {
    h <- hyps[[j]]
    message(str_glue("==== AMM bootstrap LRT: {h$id} ===="))
    start <- Sys.time()
    amm_lrts[[h$id]] <- amm_boot_lrt(
      Z_amm, covariates_amm, h,
      fit_alt = fit_alt[[h$method]],
      B       = B_inf,
      n_cores = 5,
      seed    = 1234 + j
    )
    stop <- Sys.time()
    print(stop - start)
  }
  saveRDS(amm_lrts, path_amm_inf)
}


## ---------- Results ----------

if (file.exists(path_amm_inf)) {

  amm_lrts <- readRDS(path_amm_inf)
  cov_names_inf <- c(male = "Sex", age = "Age", pir = "Income")

  ## Summary table
  inference_tbl <- data.frame(
    test      = names(amm_lrts),
    covariate = unname(cov_names_inf[sapply(amm_lrts, `[[`, "covariate")]),
    null      = sapply(amm_lrts, `[[`, "label"),
    method    = sapply(amm_lrts, `[[`, "method"),
    T_obs     = sapply(amm_lrts, `[[`, "T_obs"),
    B_eff     = sapply(amm_lrts, `[[`, "B_eff"),
    n_fail    = sapply(amm_lrts, `[[`, "n_fail"),
    p_value   = sapply(amm_lrts, `[[`, "p_value"),
    p_upper   = sapply(amm_lrts, `[[`, "p_upper"),
    p_text    = sapply(amm_lrts, amm_format_p),
    delta_r2  = sapply(amm_lrts, `[[`, "delta_r2"),
    row.names = NULL
  )
  print(inference_tbl)
  write.csv(
    inference_tbl,
    file.path(dir_amm_inf, 'amm_inference_summary.csv'),
    row.names = FALSE
  )

  ## Per-column contributions Lambda_k to each T_obs
  per_k_tbl <- data.frame(
    k = seq_along(amm_lrts[[1]]$T_obs_per_k),
    sapply(amm_lrts, `[[`, "T_obs_per_k")
  )
  print(per_k_tbl)
  write.csv(
    per_k_tbl,
    file.path(dir_amm_inf, 'amm_inference_per_k.csv'),
    row.names = FALSE
  )

  ## Per-column share of latent variance explained (Delta R^2_k)
  per_k_r2_tbl <- data.frame(
    k = seq_along(amm_lrts[[1]]$delta_r2_per_k),
    sapply(amm_lrts, `[[`, "delta_r2_per_k")
  )
  print(per_k_r2_tbl)
  write.csv(
    per_k_r2_tbl,
    file.path(dir_amm_inf, 'amm_inference_per_k_r2.csv'),
    row.names = FALSE
  )

  ## LaTeX table: p-value and Delta R^2 (% of latent variance) per covariate
  p_cell  <- function(id) amm_format_p(amm_lrts[[id]], drop_leading_zero = TRUE)
  r2_cell <- function(id) {
    r2 <- round(100 * amm_lrts[[id]]$delta_r2, 2)
    sprintf("%.2f", if (r2 == 0) 0 else r2)   # avoid "-0.00"
  }
  inf_tbl_latex <- data.frame(
    "Null Model" = c("No effect", "Linear"),
    "Sex_p"      = c(p_cell("male_none"), "{---}"),
    "Sex_r2"     = c(r2_cell("male_none"), "{---}"),
    "Age_p"      = c(p_cell("age_none"), p_cell("age_linear")),
    "Age_r2"     = c(r2_cell("age_none"), r2_cell("age_linear")),
    "Income_p"   = c(p_cell("pir_none"), p_cell("pir_linear")),
    "Income_r2"  = c(r2_cell("pir_none"), r2_cell("pir_linear"))
  )
  sink(file.path(dir_amm_inf, 'amm_inf_latex.txt'))
  print(
    kbl(inf_tbl_latex,
        format = "latex",
        booktabs = TRUE,
        escape = FALSE,
        align = c("l", rep("S", 6)),
        caption = paste("Parametric bootstrap tests for the additive mixed",
                        "model, with the share of latent variance explained",
                        "by each dropped term ($\\Delta R^2$, \\%)"),
        col.names = c("Null Model",
                      rep(c("{$p$}", "{$\\Delta R^2$}"), 3))) %>%
      add_header_above(c(" " = 1, "Sex" = 2, "Age" = 2, "Income" = 2)) %>%
      kable_styling(latex_options = c("HOLD_position"))
  )
  sink()

  ## Bootstrap null distributions (half width)
  for (res in amm_lrts) {
    fig_png(file.path(dir_amm_inf, str_glue('amm_boot_hist_{res$id}.png')),
            3.25, 2.15)   # TODO: final printed height
    fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
            cex_axis = fig_cex_axis)
    amm_plot_boot_hist(
      res,
      main = str_glue("{cov_names_inf[[res$covariate]]}: {tolower(res$label)}")
    )
    dev.off()
  }

}


## ========== AMM Interpretation ========== ##

## Create directories for continuous variable interpretation
dir_amm_interp <- file.path(dir_amm, 'interpretation')
if (!dir.exists(dir_amm_interp)) dir.create(dir_amm_interp)


## ---------- Driver ----------

Ji_interp <- length(Qi_ctx$payload[[1]])
B_interp  <- 100  ## TODO: Increase this!

amm_boots <- amm_boot_interpret(
  amm,
  covariates_kept = covariates_amm[amm$keep, , drop = FALSE],
  B               = B_interp,
  n_cores         = 5,
  seed            = 1234
)

## Interpret continuous covariates
cov_to_plot <- c('age', 'pir') # c('age', 'pir')
# fam_to_plot <- c('latent', 'qf', 'survival')#, 'qf', 'cdf', 'survival', 'moment', 'threshold', 'quantile')
fam_to_plot <- c('quantile', 'threshold', 'moment')

## Paper figures: same sizes and styling as the group comparison (gc_*
## settings above). Overlaid QF / CDF / survival plots (and their contrasts)
## are half width; the absolute quantile-level / threshold curves are
## half-width strips and the moments a full-width strip; the latent effects
## are a full-width strip. Overlays use the covariate values below (the centre
## value is the contrast reference: age 40, PIR 1), coloured blue -> red, with
## a one-column legend titled by the covariate; quantile-level / threshold
## lines are labelled above the frame. Faceted and contrast-over-covariate
## versions are kept at their working sizes.
amm_x_values   <- list(age = c(10, 25, 40, 55, 70),
                       pir = c(0.25, 0.5, 1, 3, 5))
amm_cov_labels <- c(age = "Age", pir = "PIR")
amm_legend_ncol <- 1
## Strip margins: bottom fits the x tick labels (the strips drop their x labels;
## the covariate is named in the caption); tops as for the group-comparison
## strips (two-line titles in row 2, one-line for moments and the latent strip)
amm_row2_mar   <- c(1.8, 3.2, 3.4, 0.5)
amm_moment_mar <- c(1.8, 3.2, 2.2, 0.5)
amm_latent_mar <- c(1.8, 3.2, 2.2, 0.5)

for (cov in cov_to_plot) {

  ## Create covariate directory
  dir_amm_cov <- file.path(dir_amm_interp, cov)
  if (!dir.exists(dir_amm_cov)) dir.create(dir_amm_cov)

  x_vals    <- amm_x_values[[cov]]
  cov_label <- amm_cov_labels[[cov]]

  ## Reference QF / CDF (decoded at male = 0, age = 40, pir = 1)
  cv_ref_decode <- list(male = 0, age = 40, pir = 2)
  Qi_ref_decode <- .decode_z_vec(pipeline,
    amm_compute_conditional_mean(amm, cv_ref_decode), Ji_interp)
  pi_grid_ref <- pi_grid_fun(Ji_interp)

  ## Latent effect plots (one panel per latent dimension; full-width strip)
  if ('latent' %in% fam_to_plot) {
    K_amm <- ncol(amm$beta_hat)
    path_plot <- file.path(dir_amm_cov, str_glue("amm_{cov}_latent-effect.png"))
    gc_strip_png(path_plot, gc_full_w)
    amm_plot_latent_effects(amm, cov, band = TRUE, joint_over = 'x_k',
                            boot_reps = amm_boots,
                            ncol = K_amm,
                            panel_mains = sprintf("k = %d", seq_len(K_amm)),
                            xlab = "", ylab = "",
                            cex = fig_cex, mar = amm_latent_mar, mgp = NULL,
                            lwd = gc_lwd)
    dev.off()
  }

  ## QF Reference
  if ('qf' %in% fam_to_plot) {
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf_reference.png")))
    plot_funs(
      fun_list  = list(Qi_ref_decode),
      grid_list = list(pi_grid_ref),
      colors    = "gray60",
      widths    = gc_lwd,
      types     = 1,
      ylab      = "Q(p)",
      main      = "",
      cex.lab   = fig_cex_lab,
      cex_main  = fig_cex_main,
      cex       = fig_cex,
      cex_axis  = fig_cex_axis
    )
    dev.off()
  }

  ## CDF Reference
  if ('cdf' %in% fam_to_plot) {
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf_reference.png")))
    par(mar = .trimmed_mar(par("mar"), "", "y", "F(y)"))
    y_grid_ref <- .amm_default_y_grid(list(Qi_ref_decode))
    F_ref_decode <- qi_to_cdf(Qi_ref_decode, y_grid_ref)
    plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
        xlab = "y", ylab = "F(y)")
    lines(y_grid_ref, F_ref_decode, col = "gray60", lwd = gc_lwd)
    dev.off()
  }

  ## Survival Reference
  if ('survival' %in% fam_to_plot) {
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival_reference.png")))
    par(mar = .trimmed_mar(par("mar"), "", "y", "1 - F(y)"))
    y_grid_ref <- .amm_default_y_grid(list(Qi_ref_decode))
    S_ref_decode <- .amm_qi_to_survival(Qi_ref_decode, y_grid_ref)
    plot(NULL, xlim = range(y_grid_ref), ylim = c(0, 1),
        xlab = "y", ylab = "1 - F(y)")
    lines(y_grid_ref, S_ref_decode, col = "gray60", lwd = gc_lwd)
    dev.off()
  }

  ## QF family
  if ('qf' %in% fam_to_plot) {
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf.png")))
    amm_plot_conditional_qf(amm, cov, pipeline, Ji = Ji_interp,
                            x_values = x_vals,
                            quantile_lines = quantile_levels,
                            main = "", legend_title = cov_label,
                            legend_ncol = amm_legend_ncol,
                            legend_cex = fig_cex_legend, lwd = gc_lwd,
                            line_labels = "top")
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_qf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                    boot_reps = amm_boots,
                                    quantile_lines = quantile_levels)
    dev.off()
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_qf-contrast.png")))
    amm_plot_contrast_qf(amm, cov, pipeline, Ji = Ji_interp,
                         x_values = x_vals,
                         quantile_lines = quantile_levels,
                         main = "", legend_title = cov_label,
                         legend_ncol = amm_legend_ncol,
                         legend_cex = fig_cex_legend, lwd = gc_lwd,
                         line_labels = "top")
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
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf.png")))
    amm_plot_conditional_cdf(amm, cov, pipeline, Ji = Ji_interp,
                             x_values = x_vals,
                             threshold_lines = mims_thresholds,
                             main = "", legend_title = cov_label,
                             legend_ncol = amm_legend_ncol,
                             legend_cex = fig_cex_legend, lwd = gc_lwd,
                             line_labels = "top")
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_cdf_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                    threshold_lines = mims_thresholds,
                                    boot_reps = amm_boots)
    dev.off()
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_cdf-contrast.png")))
    amm_plot_contrast_cdf(amm, cov, pipeline, Ji = Ji_interp,
                          x_values = x_vals,
                          threshold_lines = mims_thresholds,
                          main = "", legend_title = cov_label,
                          legend_ncol = amm_legend_ncol,
                          legend_cex = fig_cex_legend, lwd = gc_lwd,
                          line_labels = "top")
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
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival.png")))
    amm_plot_conditional_survival(amm, cov, pipeline, Ji = Ji_interp,
                                  x_values = x_vals,
                                  threshold_lines = mims_thresholds,
                                  main = "", legend_title = cov_label,
                                  legend_ncol = amm_legend_ncol,
                                  legend_cex = fig_cex_legend, lwd = gc_lwd,
                                  line_labels = "top")
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_conditional_survival_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                          threshold_lines = mims_thresholds,
                                          boot_reps = amm_boots)
    dev.off()
    gc_half_png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-contrast.png")))
    amm_plot_contrast_survival(amm, cov, pipeline, Ji = Ji_interp,
                               x_values = x_vals,
                               threshold_lines = mims_thresholds,
                               main = "", legend_title = cov_label,
                               legend_ncol = amm_legend_ncol,
                               legend_cex = fig_cex_legend, lwd = gc_lwd,
                               line_labels = "top")
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_survival-contrast-faceted.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_contrast_survival_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                      threshold_lines = mims_thresholds,
                                      boot_reps = amm_boots)
    dev.off()
  }

  ## Moments family (joint bands over the covariate; full-width strip)
  if ('moment' %in% fam_to_plot) {
    gc_strip_png(file.path(dir_amm_cov, str_glue("amm_{cov}_moments.png")), gc_full_w)
    amm_plot_conditional_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                         boot_reps = amm_boots,
                                         panel_ylabs = rep("", 4),
                                         xlab = "",
                                         cex = fig_cex, mar = amm_moment_mar,
                                         lwd = gc_lwd)
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_moments-contrast.png")),
        width = 1280, height = 240, pointsize = 14)
    amm_plot_contrast_moments_faceted(amm, cov, pipeline, Ji = Ji_interp,
                                      boot_reps = amm_boots)
    dev.off()
  }

  ## Threshold-crossing (one panel per MIMS threshold; absolute half-width
  ## strip + working-size contrast)
  if ('threshold' %in% fam_to_plot) {
    gc_strip_png(file.path(dir_amm_cov, str_glue("amm_{cov}_threshold.png")), gc_half_w)
    amm_plot_threshold_crossing_panels(
      amm, cov, mims_thresholds, pipeline,
      panel_mains = gc_thr_mains,
      panel_ylabs = rep("", length(mims_thresholds)),
      xlab        = "",
      cex         = fig_cex,
      mar         = amm_row2_mar,
      lwd         = gc_lwd,
      Ji          = Ji_interp,
      boot_reps   = amm_boots
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

  ## Quantile-crossing (one panel per quantile level; absolute half-width
  ## strip + working-size contrast)
  if ('quantile' %in% fam_to_plot) {
    gc_strip_png(file.path(dir_amm_cov, str_glue("amm_{cov}_quantile.png")), gc_half_w)
    amm_plot_quantile_crossing_panels(
      amm, cov, quantile_levels, pipeline,
      panel_mains = gc_q_mains,
      panel_ylabs = rep("", length(quantile_levels)),
      xlab        = "",
      cex         = fig_cex,
      mar         = amm_row2_mar,
      lwd         = gc_lwd,
      Ji          = Ji_interp,
      boot_reps   = amm_boots
    )
    dev.off()
    png(file.path(dir_amm_cov, str_glue("amm_{cov}_quantile-contrast.png")),
        width = 1280, height = 320, pointsize = 14)
    amm_plot_quantile_crossing_panels(
      amm, cov, quantile_levels, pipeline,
      panel_mains = gc_q_mains,
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
amm_male_pir_ref <- 2
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
## Paper figures: same sizes, layout and styling as the group comparison
## (gc_* settings above). Reference values for the caption: age =
## amm_male_age_ref, pir = amm_male_pir_ref; black = gp0 (female), red = gp1.
amm_male_diff_label <- str_wrap(str_glue("{gp1_name} - {gp0_name}"), width = 20)

## QF and QF contrast (half width)
gc_half_png(file.path(dir_amm_cov, "amm_male_qf.png"))
plot_decoded_conditional_qfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha        = amm_male_alpha,
  quantile_lines        = quantile_levels,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  legend_cex   = fig_cex_legend,
  lwd          = gc_lwd
)
dev.off()
gc_half_png(file.path(dir_amm_cov, "amm_male_qf-contrast.png"))
plot_decoded_conditional_qf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  quantile_lines      = quantile_levels,
  threshold_labels = "top",
  diff_label = amm_male_diff_label,
  legend_loc = 'bottomleft',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()

## CDF and CDF contrast (half width)
y_grid_male <- default_y_grid_two_group(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1
)
gc_half_png(file.path(dir_amm_cov, "amm_male_cdf.png"))
plot_decoded_conditional_cdfs(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha        = amm_male_alpha,
  y_grid       = y_grid_male,
  thresholds   = mims_thresholds,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  legend_cex   = fig_cex_legend,
  lwd          = gc_lwd
)
dev.off()
gc_half_png(file.path(dir_amm_cov, "amm_male_cdf-contrast.png"))
plot_decoded_conditional_cdf_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  diff_label = amm_male_diff_label,
  legend_loc = 'topright',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()

## Survival function and survival function contrast (half width)
gc_half_png(file.path(dir_amm_cov, "amm_male_survival.png"))
plot_decoded_conditional_survivals(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha        = amm_male_alpha,
  y_grid       = y_grid_male,
  thresholds   = mims_thresholds,
  threshold_labels = "top",
  group_labels = c(gp0_name, gp1_name),
  band_type    = 'pointwise',
  legend_cex   = fig_cex_legend,
  lwd          = gc_lwd
)
dev.off()
gc_half_png(file.path(dir_amm_cov, "amm_male_survival-contrast.png"))
plot_decoded_conditional_survival_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha      = amm_male_alpha,
  y_grid     = y_grid_male,
  thresholds = mims_thresholds,
  threshold_labels = "top",
  diff_label = amm_male_diff_label,
  band_type  = 'pointwise',
  legend_loc = 'bottomright',
  legend_cex = fig_cex_legend,
  lwd        = gc_lwd
)
dev.off()

## Moments and moment contrast (full-width strips)
gc_strip_png(file.path(dir_amm_cov, "amm_male_moment.png"), gc_full_w)
plot_decoded_conditional_moments(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha        = amm_male_alpha,
  group_labels = c("", ""),
  panel_ylabs  = rep("", 4),
  cex          = fig_cex,
  mar          = gc_moment_mar,
  size         = gc_size,
  xlim         = gc_xlim_2
)
dev.off()
gc_strip_png(file.path(dir_amm_cov, "amm_male_moment-contrast.png"), gc_full_w)
plot_decoded_conditional_moment_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  alpha       = amm_male_alpha,
  diff_label  = "",
  panel_ylabs = rep("", 4),
  cex         = fig_cex,
  mar         = gc_moment_mar,
  size        = gc_size
)
dev.off()

## Threshold probabilities and threshold contrast (half-width strips)
gc_strip_png(file.path(dir_amm_cov, "amm_male_threshold.png"), gc_half_w)
plot_decoded_conditional_thresholds(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  thresholds   = mims_thresholds,
  eval_type    = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains  = gc_thr_mains,
  panel_ylabs  = rep("", length(mims_thresholds)),
  y_grid       = y_grid_male,
  alpha        = amm_male_alpha,
  group_labels = c("", ""),
  cex          = fig_cex,
  mar          = gc_row2_mar,
  size         = gc_size,
  xlim         = gc_xlim_2
)
dev.off()
gc_strip_png(file.path(dir_amm_cov, "amm_male_threshold-contrast.png"), gc_half_w)
plot_decoded_conditional_threshold_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  thresholds  = mims_thresholds,
  eval_type   = sapply(mims_thresholds, `[[`, "eval_type"),
  panel_mains = gc_thr_mains,
  panel_ylabs = rep("", length(mims_thresholds)),
  y_grid      = y_grid_male,
  alpha       = amm_male_alpha,
  diff_label  = "",
  cex         = fig_cex,
  mar         = gc_row2_mar,
  size        = gc_size
)
dev.off()

## Quantile levels and quantile-level contrast (half-width strips)
gc_strip_png(file.path(dir_amm_cov, "amm_male_quantile-level.png"), gc_half_w)
plot_decoded_conditional_quantile_levels(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  quantile_levels = quantile_levels,
  panel_mains     = gc_q_mains,
  panel_ylabs     = rep("", length(quantile_levels)),
  alpha           = amm_male_alpha,
  group_labels    = c("", ""),
  cex             = fig_cex,
  mar             = gc_row2_mar,
  size            = gc_size,
  xlim            = gc_xlim_2
)
dev.off()
gc_strip_png(file.path(dir_amm_cov, "amm_male_quantile-level-contrast.png"), gc_half_w)
plot_decoded_conditional_quantile_level_diff(
  Qi_hat_male_0, Qi_hat_male_1, Qi_boot_male_0, Qi_boot_male_1,
  quantile_levels = quantile_levels,
  panel_mains     = gc_q_mains,
  panel_ylabs     = rep("", length(quantile_levels)),
  alpha           = amm_male_alpha,
  diff_label      = "",
  cex             = fig_cex,
  mar             = gc_row2_mar,
  size            = gc_size
)
dev.off()
