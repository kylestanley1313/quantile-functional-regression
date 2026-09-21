library(clue)
library(MASS)
library(parallel)
library(transport)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## ==================== Fit Pipelines ==================== ##

## Quantlet ordering for the WAME/QG-PCA stage: "variance" (default, PCA
## explained-variance order) or "importance" (linearized loss-reduction
## order -- see stage_wame()'s order_by argument in src/cot.R).
order_by <- "importance"

## Lambda held fixed while K is swept explicitly (bypassing the CV-based
## near-lossless K-selection in choose_near_lossless_K_qg_pca).
lambda <- 0

## Candidate Ks
Ks <- 1:20

## Load data
path <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
dir_art <- file.path('artifacts', 'demo_nhanes_jsm_2026', 'generativity_K')
y_list <- readRDS(path)
# y_list <- y_list[1:1000]
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
Ji_min <- min(Ji_vec)
y_max <- max(unlist(y_list))
J_aug <- 500
path_plot_tmp = "scratch/plots/tmp_plot.png"

## Define grid
p_grid <- p_grid_fun(
  breaks = c(1/(7350 + 1), 0.95, 7350/(7350 + 1)),
  interval_counts = c(51, 50)
)

for (K in Ks) {

  ## Construct pipeline
  pipeline <- construct_pipeline(
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_wame(
        K = K,
        K_max = 50,
        epsilon = 1.45,
        alpha = 0.05,
        V = 5,
        lambda = lambda,
        order_by = order_by
      ),
      stage_flow(
        n_layers = 32,
        hidden = 64,
        max_epochs = 1000,
        lr = 1e-3,
        path = file.path(dir_art, str_glue('flow_K-{K}_order-{order_by}.pth'))
      )
    ),
    supp_Y = c(0, seq(0.006, 500, by = 0.001)),
    p_star = 0,
    y_star = 0,
    y_min = 0,
    seed = gen_seed()
  )

  ## Fitting
  pipeline <- fit(pipeline, y_list)
  path <- file.path(dir_art, str_glue('pipe_K-{K}_order-{order_by}.rds'))
  saveRDS(pipeline, path)

}



## ==================== Generativity Helpers ==================== ##

## All generativity helpers live in src/utils.R:
##   decode_z_to_Qi, encode_y_to_Qi_aug, decode_z_to_Qi_aug, compute_generativity,
##   generativity_score, encode_y_to_z, fit_mean_cov, sample_mean_cov,
##   evaluate_unconditional_generativity, plot_generativity_boxes.




## ==================== Assess Generativity ==================== ##

## Settings
Ks <- 1:20
S          <- 5
R          <- 20
n_cores <- min(4, R)

## Load data
path_y <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path_y)

## Compute split-normalized generativity per K
gen_list <- list()
for (K in Ks) {
  print(str_glue("K = {K}"))

  path_pipe <- file.path(dir_art, str_glue('pipe_K-{K}_order-{order_by}.rds'))
  pipeline <- readRDS(path_pipe)

  start <- Sys.time()
  gen_list[[as.character(K)]] <- evaluate_unconditional_generativity(
    pipeline, y_list,
    S = S, R = R,
    frac_train       = 0.5,
    pool_subsample_n = NULL,
    J_aug            = 500,
    ridge            = 0,
    seed             = 12345,
    n_cores          = n_cores
  )

  path_gen <- file.path(dir_art, str_glue('gen_res_K-{K}_order-{order_by}.rds'))
  saveRDS(gen_list[[as.character(K)]], path_gen)
  print(Sys.time() - start)
}


path_gen <- file.path(dir_art, 'gen_res.rds')
saveRDS(gen_list, path_gen)


## ---------- Plot Generativity against K

Ks <- 1:20
gen_res <- readRDS(path_gen)

## Per-K: list (length S) of per-split log_norm vectors (length R).
scores_by_K <- lapply(Ks, function(K) {
  lapply(gen_res[[as.character(K)]]$splits, `[[`, "log_norm")
})

plot_idx   <- seq(1, length(Ks))
path_plot <- file.path(dir_art, 'gen_score_vs_K.png')
plot_generativity_boxes(
  groups      = as.character(Ks[plot_idx]),
  scores      = scores_by_K[plot_idx],
  hline       = 0,
  xlab        = "K",
  ylab        = "log normalized generativity",
  main        = str_glue("Generativity (log-normalized) vs K (lambda = {lambda})"),
  path        = path_plot
)
