library(MASS)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## Globals
art_dir_nhanes <- 'demo_nhanes'
art_dir <- 'validate_nhanes_3'
epsilon <- 1.25

## Canonical pipeline (the one demo_nhanes.R chose). K_star follows.
path_pipe_star <- file.path('artifacts', art_dir_nhanes, 'pipe_nhanes.rds')
pipeline_star  <- readRDS(path_pipe_star)
K_star         <- pipeline_star$stages[[5]]$state$K

## Load datasets used by every section below.
y_nhanes <- readRDS(file.path('data', 'processed', 'nhanes_v1_nofilter.rds'))
y_chop   <- readRDS(file.path('data', 'processed', 'chop-mims_v3_nofilter.rds'))



## ---------- Losslessness: NHANES vs CHOP at K_star ---------- ##

compute_recon_loss <- function(pipeline, y_list) {
  y_ctx <- new_context(
    payload = y_list,
    cache   = pipeline$training$cache,
    meta    = list(Ji_vec = lengths(y_list))
  )
  Qi_list      <- encode(pipeline, y_ctx, from = 0, to = 2)$payload
  ctx_top      <- encode(pipeline, y_ctx)   # default to = n_stages
  Qi_reco_list <- decode(pipeline, ctx_top, from = pipeline$n_stages, to = 2)$payload
  loss_scale   <- pipeline$training$meta$loss_scale
  losses       <- numeric(length(Qi_list))
  for (i in seq_along(Qi_list)) {
    dp         <- 1 / (length(Qi_list[[i]]) + 1)
    losses[i]  <- wasserstein(Qi_list[[i]], Qi_reco_list[[i]], dp) / loss_scale
  }
  losses
}

## NHANES held-out subset matched to CHOP size, so both columns have the same
## sample size and the jitter clouds are visually comparable.
set.seed(12345)
idx_nhanes_val <- sample(length(y_nhanes), length(y_chop))
losses_nhanes  <- compute_recon_loss(pipeline_star, y_nhanes)# y_nhanes[idx_nhanes_val])
losses_chop    <- compute_recon_loss(pipeline_star, y_chop)

plot_losslessness_groups(
  losses_list  = list(losses_nhanes, losses_chop),
  group_labels = c("NHANES", "CHOP"),
  epsilon      = c(epsilon),
  annotate_idx = c(1, 2),
  jitter_width = 0.1,
  ylab         = "Wasserstein Error",
  main         = str_glue("Reconstruction Error (K = {K_star})"),
  path         = file.path('artifacts', art_dir,
                           str_glue('chop-losses_K-{K_star}.png'))
)

## NOTES: 
##  - Decide whether to include subsample of NHANES
##  - Write percentage of CHOP < eps as well? Should be even higher than 95.3% (that was on validation sets)



## ---------- Embedding plots at K_star ---------- ##

k_start   <- 1
k_stop    <- 3
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_flag <- rgb(1, 0, 0)

y_ctx_chop <- new_context(
  payload = y_chop,
  cache   = pipeline_star$training$cache,
  meta    = list()
)
Qi_chop <- encode(pipeline_star, y_ctx_chop, from = 0, to = 2)$payload
c_chop <- do.call(rbind,
  encode(pipeline_star, y_ctx_chop, from = 0, to = 5)$payload$c_list)
z_chop <- do.call(rbind,
  encode(pipeline_star, y_ctx_chop, from = 0, to = 6)$payload)

png(file.path('artifacts', art_dir, str_glue('chop-z_K-{K_star}.png')),
    width = 960, height = 960, pointsize = 18)
plot_embeddings(
  z_chop[, k_start:k_stop],
  stats  = FALSE,
  colors = rep(col_train, length(y_chop))
)
dev.off()

png(file.path('artifacts', art_dir, str_glue('chop-c_K-{K_star}.png')),
    width = 960, height = 960, pointsize = 18)
plot_embeddings(
  c_chop[, k_start:k_stop],
  stats  = FALSE,
  colors = rep(col_train, length(y_chop))
)
dev.off()



## ---------- Generativity: NHANES vs CHOP ---------- ##

S       <- 10
R       <- 50
n_cores <- min(5, R)

## NHANES gets subsampled to CHOP's size BEFORE each split so both arms see
## matched-size splits and the comparison is apples-to-apples.
gen_nhanes <- assess_generativity_split(
  pipeline_star, y_nhanes,
  S = S, R = R,
  frac_train  = 0.5,
  subsample_n = length(y_chop),
  J_aug       = 500,
  ridge       = 0,
  seed        = 12345,
  n_cores     = n_cores
)
gen_chop <- assess_generativity_split(
  pipeline_star, y_chop,
  S = S, R = R,
  frac_train  = 0.5,
  subsample_n = NULL,
  J_aug       = 500,
  ridge       = 0,
  seed        = 12345,
  n_cores     = n_cores
)
saveRDS(
  list(nhanes = gen_nhanes, chop = gen_chop),
  file.path('artifacts', art_dir,
            str_glue('gen_nhanes_vs_chop_K-{K_star}.rds'))
)

plot_generativity_boxes(
  groups      = c("NHANES", "CHOP"),
  scores      = list(
    lapply(gen_nhanes$splits, `[[`, "log_norm"),
    lapply(gen_chop$splits,   `[[`, "log_norm")
  ),
  Ks          = NULL,
  vline_group = NULL,
  hline       = 0,
  ylab        = "log normalized generativity",
  main        = str_glue("Generativity (K = {K_star})"),
  path        = file.path('artifacts', art_dir,
                          str_glue('chop-gen_K-{K_star}.png'))
)



## ---------- Synthetic vs Val Qi overlays ---------- ##
## Visual analog of the generativity boxplots above: same train/val split,
## fit mean-cov on train, draw |val| synthetic z's, decode to Qi, then
## overlay against the held-out real Qi. One plot per dataset.

set.seed(12345)

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

  fit      <- fit_mean_cov(encode_to_Z(pipeline_star, y_data[idx_tr]), ridge = 0)
  z_draws  <- draw_mean_cov(length(idx_val), fit)
  Qi_draws <- decode_z_draws(z_draws, pipeline_star, Ji = Ji)$Qi

  y_ctx_val <- new_context(
    payload = y_data[idx_val],
    cache   = pipeline_star$training$cache,
    meta    = list(Ji_vec = lengths(y_data[idx_val]))
  )
  Qi_val <- encode(pipeline_star, y_ctx_val, from = 0, to = 2)$payload

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
  png(file.path('artifacts', art_dir,
                str_glue('synth_qi_{label}_K-{K_star}.png')),
      width = 960, height = 960, pointsize = 18)
  plot_funs(
    fun_list  = fun_list,
    grid_list = grid_list,
    ylim      = ylim_shared,
    colors    = colors,
    widths    = widths,
    types     = types,
    color_width_type_labels = cwt_lbl,
    main      = str_glue("Synthetic vs Real Val Qi ({toupper(label)}, K = {K_star})")
  )
  dev.off()
}



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
# Qi_list <- encode(pipeline, y_ctx, from = 0, to = 2)$payload
# Qi_reco_list <- decode(pipeline, encode(pipeline, y_ctx), from = 5, to = 2)$payload

# ## Compute concordances
# concordances <- numeric(length(Qi_list))
# for (i in 1:length(Qi_list)) {
#   concordances[i] <- sqrt(1 - one_minus_sqconc(Qi_list[[i]], Qi_reco_list[[i]]))
# }
# mean(concordances > 0.99)
# mean(concordances > 0.997)
############################################