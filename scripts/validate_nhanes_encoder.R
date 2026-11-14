library(MASS)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## Globals
dir_art <- 'demo_nhanes_v11'
epsilon <- 1.34  ## TODO: Change this!!!

## Canonical pipeline (the one demo_nhanes.R chose). K_star follows.
path_pipe_star <- file.path('artifacts', dir_art, 'pipe_nhanes.rds')
pipeline_star  <- readRDS(path_pipe_star)
K_star         <- pipeline_star$stages[[3]]$state$child_qg_pca$state$K

## Load datasets used by every section below.
y_nhanes <- readRDS(file.path('data', 'processed', 'nhanes-fda-mims_v2.rds'))
y_chop   <- readRDS(file.path('data', 'processed', 'chop-mims_v7.rds'))

## Output directory
dir_val <- file.path('artifacts', dir_art, 'validation')
if (!dir.exists(dir_val)) dir.create(dir_val, recursive = TRUE)


## ----- Paper figure spec (mirrors demo_nhanes.R)

## Render every paper figure at its final printed size (inches) and include it
## in LaTeX at that same width, so fig_pointsize is the printed text size in
## every figure regardless of panel count. fig_cex = 1 undoes mfrow's
## auto-shrink in multi-panel layouts. fig_cex_main / fig_cex_lab /
## fig_cex_axis are the baseline title / axis-label / tick-label sizes, and
## fig_cex_legend the legend text size.
fig_res        <- 600
fig_pointsize  <- 7
fig_cex        <- 1
fig_cex_lab    <- 1.0
fig_cex_main   <- 1.2 * fig_cex_lab
fig_cex_axis   <- 0.8 * fig_cex_lab
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



## ---------- Losslessness: NHANES vs CHOP at K_star ---------- ##

compute_recon_loss <- function(pipeline, y_list) {
  y_ctx <- new_context(
    payload = y_list,
    cache   = pipeline$training$cache,
    meta    = list(Ji_vec = lengths(y_list))
  )
  Qi_list      <- encode(pipeline, y_ctx, from = 0, to = 1)$payload
  ctx_top      <- encode(pipeline, y_ctx)   # default to = n_stages
  Qi_reco_list <- decode(pipeline, ctx_top, from = pipeline$n_stages, to = 1)$payload
  losses       <- numeric(length(Qi_list))
  for (i in seq_along(Qi_list)) {
    dp         <- 1 / (length(Qi_list[[i]]) + 1)
    losses[i]  <- wasserstein(Qi_list[[i]], Qi_reco_list[[i]], dp)
  }
  losses
}

## NHANES held-out subset matched to CHOP size, so both columns have the same
## sample size and the jitter clouds are visually comparable.
losses_nhanes  <- compute_recon_loss(pipeline_star, y_nhanes)
losses_chop    <- compute_recon_loss(pipeline_star, y_chop)

## Half-width paper figure
set.seed(1234)   # reproducible jitter
fig_png(file.path(dir_val, str_glue('chop-losses_K-{K_star}.png')),
        3.25, 2.15)   # TODO: final printed size
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
plot_losslessness_groups(
  losses_list  = list(losses_nhanes, losses_chop),
  group_labels = c("NHANES", "CHOP"),
  epsilon      = c(epsilon),
  annotate_idx = c(1, 2),
  jitter_width = 0.1,
  ylab         = "Wasserstein error",
  main         = str_glue("Reconstruction error (K = {K_star})"),
  legend_cex   = 0.8 * fig_cex_legend
)
dev.off()


## ---------- Embedding plots at K_star ---------- ##

ks_plot   <- 1:5   # components to plot (any subset, e.g. c(1, 3, 5))
w_emb     <- fig_emb_size(length(ks_plot))
col_train <- rgb(0, 0, 0, alpha = 0.25)

y_ctx_chop <- new_context(
  payload = y_chop,
  cache   = pipeline_star$training$cache,
  meta    = list()
)
Qi_chop <- encode(pipeline_star, y_ctx_chop, from = 0, to = 1)$payload
c_chop <- do.call(rbind,
  encode(pipeline_star, y_ctx_chop, from = 0, to = 3)$payload$c_list)
z_chop <- do.call(rbind,
  encode(pipeline_star, y_ctx_chop, from = 0, to = 4)$payload)

emb_list <- list(z = z_chop, c = c_chop)
for (space in names(emb_list)) {
  E_plot <- emb_list[[space]][, ks_plot, drop = FALSE]
  fig_png(file.path(dir_val, str_glue('chop-{space}_K-{K_star}.png')),
          w_emb, w_emb)
  plot_embeddings(
    E_plot,
    colors     = rep(col_train, nrow(E_plot)),
    shapes     = rep(16, nrow(E_plot)),
    sizes      = rep(0.5, nrow(E_plot)),
    stats      = FALSE,
    cex_main   = fig_cex_main,
    cex_lab    = fig_cex_lab,
    cex        = fig_cex,
    cex_axis   = fig_cex_axis,
    legend_cex = fig_cex_legend,
    ks         = ks_plot
  )
  dev.off()
}



## ---------- Generativity: NHANES vs CHOP ---------- ##

S       <- 5
R       <- 20
n_cores <- min(5, R)

## NHANES gets subsampled to CHOP's size BEFORE each split so both arms see
## matched-size splits and the comparison is apples-to-apples.
gen_nhanes <- evaluate_generativity(
  pipeline_star, y_nhanes,
  S = S, R = R,
  frac_train       = 0.5,
  pool_subsample_n = length(y_chop),
  J_aug            = 500,
  ridge            = 0,
  seed             = 1234,
  n_cores          = n_cores
)
gen_chop <- evaluate_generativity(
  pipeline_star, y_chop,
  S = S, R = R,
  frac_train       = 0.5,
  pool_subsample_n = NULL,
  J_aug            = 500,
  ridge            = 0,
  seed             = 1234,
  n_cores          = n_cores
)
saveRDS(
  list(nhanes = gen_nhanes, chop = gen_chop),
  file.path(dir_val, str_glue('gen_nhanes_vs_chop_K-{K_star}.rds'))
)

## Half-width paper figure
fig_png(file.path(dir_val, str_glue('chop-gen_K-{K_star}.png')),
        3.25, 2.15)   # TODO: final printed size
fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
        cex_axis = fig_cex_axis)
plot_generativity_boxes(
  groups      = c("NHANES", "CHOP"),
  scores      = list(
    lapply(gen_nhanes$splits, `[[`, "log_norm"),
    lapply(gen_chop$splits,   `[[`, "log_norm")
  ),
  Ks          = NULL,
  vline_group = NULL,
  hline       = 0,
  ylab        = "Relative generativity",
  main        = str_glue("Generativity (K = {K_star})"),
  collapse = TRUE
)
dev.off()



## ---------- Synthetic vs Val Qi overlays ---------- ##
## Visual analog of the generativity boxplots above: same train/val split,
## fit mean-cov on train, draw |val| synthetic z's, decode to Qi, then
## overlay against the held-out real Qi. One plot per dataset.

set.seed(1234)

## Stage 1: prepare per-dataset Qi (real-val + synth). NHANES is subsampled to
## CHOP's size BEFORE the split so both panels see matched-size train/val.
plot_data <- list()
for (label in c("nhanes", "chop")) {
  y_data <- if (label == "nhanes") {
    y_nhanes[sample(length(y_nhanes), length(y_chop))]
  } else {
    y_chop
  }
  N_d <- length(y_data)
  Ji  <- max(lengths(y_data))

  n_train <- floor(0.5 * N_d)
  perm    <- sample(N_d)
  idx_tr  <- perm[seq_len(n_train)]
  idx_val <- perm[(n_train + 1):N_d]

  fit      <- fit_mean_cov(encode_y_to_z(pipeline_star, y_data[idx_tr]), ridge = 0)
  ############
  # Z <- encode_y_to_z(pipeline_star, y_chop[idx_tr])
  # z_norms <- apply(Z, MARGIN = 1, function(z) sqrt(sum(z^2)))
  # mask <- z_norms < 15
  # fit <- fit_mean_cov(Z[mask,], ridge = 0)
  ############
  z_draws  <- sample_mean_cov(length(idx_val), fit)
  Qi_draws <- decode_z_to_Qi(pipeline_star, z_draws, Ji = Ji)

  y_ctx_val <- new_context(
    payload = y_data[idx_val],
    cache   = pipeline_star$training$cache,
    meta    = list(Ji_vec = lengths(y_data[idx_val]))
  )
  Qi_val <- encode(pipeline_star, y_ctx_val, from = 0, to = 1)$payload

  plot_data[[label]] <- list(
    Qi_val   = Qi_val,
    Qi_draws = Qi_draws,
    Ji_val   = lengths(y_data[idx_val]),
    Ji_synth = Ji
  )
}

## Shared y-axis range across both panels.
ylim_shared <- range(unlist(lapply(plot_data, function(pd) {
  c(unlist(pd$Qi_val), unlist(pd$Qi_draws))
})), finite = TRUE)

## Stage 2: plot each panel with the shared ylim.
col_real  <- rgb(0, 0, 0, alpha = 0.25)
col_synth <- rgb(1, 0, 0, alpha = 0.25)
for (label in names(plot_data)) {
  pd        <- plot_data[[label]]
  n_real    <- length(pd$Qi_val)
  n_synth   <- length(pd$Qi_draws)
  fun_list  <- c(pd$Qi_val, pd$Qi_draws)
  grid_list <- c(
    lapply(pd$Ji_val,                   pi_grid_fun),
    lapply(rep(pd$Ji_synth, n_synth),   pi_grid_fun)
  )
  colors <- c(rep(col_real, n_real), rep(col_synth, n_synth))
  widths <- rep(0.25, length(fun_list))
  types  <- rep(1,    length(fun_list))
  shuff     <- sample(length(fun_list))
  fun_list  <- fun_list[shuff]
  grid_list <- grid_list[shuff]
  colors    <- colors[shuff]
  cwt_lbl <- data.frame(
    color = c(col_real, col_synth),
    width = c(2.5, 2.5),
    type  = c(1, 1),
    label = c('real (val)', 'synth')
  )
  fig_png(file.path(dir_val, str_glue('synth_qi_{label}_K-{K_star}.png')),
          3.25, 3.25)   # TODO: final printed size
  plot_funs(
    fun_list   = fun_list,
    grid_list  = grid_list,
    ylim       = ylim_shared,
    colors     = colors,
    widths     = widths,
    types      = types,
    color_width_type_labels = cwt_lbl,
    main       = str_glue("{toupper(label)} (K = {K_star})"),
    cex.lab    = fig_cex_lab,
    cex_main   = fig_cex_main,
    cex        = fig_cex,
    cex_axis   = fig_cex_axis,
    legend_cex = fig_cex_legend
  )
  dev.off()
}


##########################
# Z <- encode_y_to_z(pipeline_star, y_chop[idx_tr])
# z_norms <- apply(Z, MARGIN = 1, function(z) sqrt(sum(z^2)))
# hist(z_norms)
##########################


##### Concordance Computation for Jeff #####

# K_star <- 8
# path <- file.path('artifacts', art_dir, str_glue('pipe_nhanes-{K_star}.rds'))
# pipeline <- readRDS(path)

# ## Encode/Decode
# y_ctx    <- new_context(
#   payload = y_list,
#   cache   = pipeline_star$training$cache,
#   meta    = list()
# )
# Qi_list <- encode(pipeline, y_ctx, from = 0, to = 1)$payload
# Qi_reco_list <- decode(pipeline, encode(pipeline, y_ctx), from = 3, to = 1)$payload

# ## Compute concordances
# concordances <- numeric(length(Qi_list))
# for (i in 1:length(Qi_list)) {
#   concordances[i] <- sqrt(1 - one_minus_sqconc(Qi_list[[i]], Qi_reco_list[[i]]))
# }
# mean(concordances > 0.99)
# mean(concordances > 0.997)
############################################