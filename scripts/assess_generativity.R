library(clue)
library(MASS)
library(parallel)
library(transport)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## ==================== Fit Pipelines ==================== ##

## Candidate lambdas
# lambdas    <- c(0, 0.001,  0.01, 0.1, 1, 10)
lambdas <- c(
  # 0,
  # sqrt(10)^-6,
  sqrt(10)^-5,
  # sqrt(10)^-4,
  sqrt(10)^-3,
  # sqrt(10)^-2,
  sqrt(10)^-1,
  # sqrt(10)^0,
  sqrt(10)^1
  # sqrt(10)^2
)

## Load data
path <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
dir_art <- file.path('artifacts', 'demo_nhanes_v4', 'generativity')
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

for (lambda in lambdas) {

  ## Construct pipeline
  pipeline <- construct_pipeline(
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_wame(
        K_max = 30,
        epsilon = 1.86,
        alpha = 0.05,
        V = 5,
        lambda = lambda
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = file.path(dir_art, str_glue('flow_lambda-{lambda}.pth'))
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
  path <- file.path(dir_art, str_glue('pipe_lambda-{lambda}.rds'))
  saveRDS(pipeline, path)

}



## ==================== Generativity Helpers ==================== ##

## All generativity helpers live in src/utils.R:
##   decode_z_to_Qi, encode_y_to_Qi_aug, decode_z_to_Qi_aug, compute_generativity,
##   generativity_score, encode_y_to_z, fit_mean_cov, draw_mean_cov,
##   evaluate_pipeline_generativity, plot_generativity_boxes.




## ==================== Assess Generativity ==================== ##

## Settings
# lambdas    <- c(0, 0.001, 0.01, 0.1, 1, 10)
lambdas <- c(
  0,
  sqrt(10)^-6,
  sqrt(10)^-5,
  sqrt(10)^-4,
  sqrt(10)^-3,
  sqrt(10)^-2,
  sqrt(10)^-1,
  sqrt(10)^0,
  sqrt(10)^1,
  sqrt(10)^2
)
S          <- 5
R          <- 20
n_cores <- min(4, R)

## Load data
path_y <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path_y)

## Compute split-normalized generativity per lambda
gen_list <- list()
for (lambda in lambdas) {
  print(str_glue("lambda = {lambda}"))

  path_pipe <- file.path(dir_art, str_glue('pipe_lambda-{lambda}.rds'))
  pipeline <- readRDS(path_pipe)

  start <- Sys.time()
  gen_list[[as.character(lambda)]] <- evaluate_pipeline_generativity(
    pipeline, y_list,
    S = S, R = R,
    frac_train  = 0.5,
    subsample_n = NULL,
    J_aug       = 500,
    ridge       = 0,
    seed        = 12345,
    n_cores     = n_cores
  )
  print(Sys.time() - start)
}


###################
tmp <- readRDS("artifacts/demo_nhanes_v4/generativity/gen_res_old.rds")
tmp2 <- vector('list', 10)
tmp2[[1]] <- tmp[[1]]
tmp2[[2]] <- tmp[[2]]
tmp2[[3]] <- gen_list[[1]]
tmp2[[4]] <- tmp[[3]]
tmp2[[5]] <- gen_list[[2]]
tmp2[[6]] <- tmp[[4]]
tmp2[[7]] <- gen_list[[3]]
tmp2[[8]] <- tmp[[5]]
tmp2[[9]] <- gen_list[[4]]
tmp2[[10]] <- tmp[[6]]
names(tmp2) <- as.character(lambdas)
gen_list_old <- gen_list
###################

path_gen <- file.path(dir_art, 'gen_res.rds')
saveRDS(gen_list, path_gen)


## ---------- Plot Generativity and K against Lambda

# lambdas <- c(0, 0.001, 0.01, 0.1, 1, 10)
lambda_star <- 0.1   # tuned lambda for the vertical reference line
lambdas <- c(
  0,
  sqrt(10)^-6,
  sqrt(10)^-5,
  sqrt(10)^-4,
  sqrt(10)^-3,
  sqrt(10)^-2,
  sqrt(10)^-1,
  sqrt(10)^0,
  sqrt(10)^1,
  sqrt(10)^2
)
Ks <- sapply(lambdas, function(lambda) {
  pipe <- readRDS(file.path(dir_art, str_glue('pipe_lambda-{lambda}.rds')))
  pipe$stages[[3]]$state$child_qg_pca$state$K
})
gen_res <- readRDS(path_gen)

## Per-lambda: list (length S) of per-split log_norm vectors (length R).
scores_by_lambda <- lapply(lambdas, function(lambda) {
  lapply(gen_res[[as.character(lambda)]]$splits, `[[`, "log_norm")
})

## Trim lambda = 0 (often pathological) -- mirrors prior behavior.
plot_idx   <- seq(1, length(lambdas))
path_plot <- file.path(dir_art, 'gen_score_vs_lambda.png')
plot_generativity_boxes(
  groups      = as.character(lambdas[plot_idx]),
  scores      = scores_by_lambda[plot_idx],
  Ks          = Ks[plot_idx],
  hline       = 0,
  vline_group = as.character(lambda_star),
  xlab        = expression(lambda),
  ylab        = "log normalized generativity",
  main        = "Generativity (log-normalized) and K vs lambda",
  path        = path_plot
)



