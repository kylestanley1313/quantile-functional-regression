## ============================================================================
## Generativity and qualifying dimension by WAME penalty lambda (NHANES)
## ============================================================================
##
## For each lambda: cross-validated losslessness gives the qualifying
## dimension K*(lambda) (smallest K whose (1 - alpha) quantile of CV
## Wasserstein error is at most epsilon); the NHANES encoder-decoder is then
## fit at K*(lambda) and its unconditional generativity evaluated. Pipeline,
## CV and generativity settings (and seeds) match demo_nhanes.R, so
## lambda = 0 is the demo's own fit and generativity run, and every lambda is
## scored on the same splits and replicate seeds.
##
## Sections
##   1. CV losslessness per lambda (cached per lambda) and K*(lambda)
##   2. Pipeline fit and generativity at K*(lambda) (cached; run serially)
##   3. Generativity boxplots by lambda, with K*(lambda) on the right axis
##
## Runtime (uncached): CV ~20-30 min per batch of lambdas (parallel over
## lambda x fold); ~15 min per lambda for the fit and generativity.
## Outputs: artifacts/demo_nhanes_v11/generativity_by_lambda (caches in cache/).

library(stringr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


## ==================== Settings ==================== ##

dir_out   <- file.path('artifacts', 'demo_nhanes_v11', 'generativity_by_lambda')
dir_cache <- file.path(dir_out, 'cache')
for (d in c(dir_out, dir_cache)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

## Penalty grid (lambda = 0 first)
lambdas <- c(0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)

## Losslessness (same CV procedure as assess_losslessness.R)
alpha      <- 0.05
epsilon    <- 1.34
Ks_cv      <- 1:15
K_max      <- 30
V_cv       <- 5
fold_seed  <- 1234
cv_workers <- 5        # parallel CV tasks (one lambda x fold each)

## Generativity (same settings as demo_nhanes.R's "Check generativity")
gen_S         <- 5
gen_R         <- 20
gen_eval_seed <- 1234     # not gen_seed: that name is a function in src/utils.R
gen_n_cores   <- 3

## Data, grid and support (as in demo_nhanes.R)
path_y <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path_y)
N      <- length(y_list)
p_grid <- p_grid_fun(
  breaks = c(1/(7350 + 1), 0.95, 7350/(7350 + 1)),
  interval_counts = c(51, 50)
)
supp_Y <- c(0, seq(0.006, 500, by = 0.001))

## lambda = 0 at K = 5 is the demo's pipeline and generativity run
dir_demo   <- file.path('artifacts', 'demo_nhanes_v11')
path_pipe0 <- file.path(dir_demo, 'pipe_nhanes.rds')
path_gen0  <- file.path(dir_demo, 'plots', 'representation', 'generativity', 'gen_res.rds')
K0         <- 5


## ==================== Paper figure spec (from demo_nhanes.R) ==================== ##

fig_res        <- 600
fig_pointsize  <- 7
fig_cex        <- 1
fig_cex_lab    <- 1.0
fig_cex_main   <- 1.2 * fig_cex_lab
fig_cex_axis   <- 0.8 * fig_cex_lab
fig_png <- function(path, width_in, height_in, pointsize = fig_pointsize) {
  png(path, width = width_in, height = height_in, units = "in",
      res = fig_res, pointsize = pointsize)
}
fig_par_paper <- function() {
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
}
fig_half_w <- 3.25
fig_half_h <- 2.6

## Per element, so labels aren't padded to a common width
lambda_lab <- function(l) vapply(l, function(x) format(x, scientific = FALSE, drop0trailing = TRUE), "")


## ==================== Helpers ==================== ##

## Full pipeline (as in demo_nhanes.R) with penalty lambda at K
build_pipeline <- function(K, lambda, flow_path) {
  construct_pipeline(
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100, extrap = 'ratio'),
      stage_wame(K = K, K_max = K_max, V = 5, lambda = lambda, order_by = 'importance'),
      stage_flow(n_layers = 16, hidden = 64, max_epochs = 1000, lr = 1e-3, path = flow_path),
      stage_pca_rotation()
    ),
    supp_Y = supp_Y, p_star = 0, y_star = 0, y_min = 0, seed = 1234
  )
}

## One CV task (lambda, fold): fit the K-independent prefix on the training
## folds, then for each K fit only the wame stage and score the validation
## subjects (same procedure as compute_cv_losses in assess_losslessness.R).
## Self-contained so it can run in a fresh worker.
cv_task <- function(v, lambda, a) {
  setwd(a$repo_root)
  suppressMessages({
    source('src/cot.R'); source('src/model.R'); source('src/utils.R')
  })
  y_list <- readRDS(a$path_y)
  set.seed(a$fold_seed)
  folds     <- sample(rep(seq_len(a$V), length.out = length(y_list)))
  idx_train <- which(folds != v)
  idx_valid <- which(folds == v)

  prefix <- construct_pipeline(
    stages = list(stage_eqf_sgrid(), stage_eqf_cgrid(p_grid = a$p_grid, extrap = 'ratio')),
    supp_Y = a$supp_Y, p_star = 0, y_star = 0, y_min = 0, seed = 1234
  )
  prefix     <- fit(prefix, y_list[idx_train])
  ctx_train  <- encode(prefix, new_context(y_list[idx_train], prefix$training$cache), from = 0, to = 2)
  ctx1_valid <- encode(prefix, new_context(y_list[idx_valid], prefix$training$cache), from = 0, to = 1)
  data_valid <- ctx1_valid$payload
  ctx_valid  <- encode(prefix, ctx1_valid, from = 1, to = 2)
  loss_fun   <- loss_to_fun[['wasserstein']]

  losses <- matrix(NA_real_, length(idx_valid), length(a$Ks))
  for (ki in seq_along(a$Ks)) {
    st    <- fit(stage_wame(K = a$Ks[ki], K_max = a$K_max, lambda = lambda,
                            order_by = 'importance'), ctx_train)
    ctx_v <- decode(st, encode(st, ctx_valid))
    reco  <- decode(prefix, ctx_v, from = 2, to = 1)$payload
    for (j in seq_along(idx_valid)) {
      losses[j, ki] <- loss_fun(data_valid[[j]], reco[[j]], 1 / (length(data_valid[[j]]) + 1))
    }
  }
  list(v = v, lambda = lambda, idx_valid = idx_valid, losses = losses)
}


## ==================== 1. CV losslessness and K*(lambda) ==================== ##

## Cached per lambda (CV losses do not depend on epsilon), so extending the
## grid only runs the new values
path_cv   <- function(l) file.path(dir_cache, sprintf('cv_lambda-%s.rds', lambda_lab(l)))
cv_cached <- function(l) file.exists(path_cv(l)) && all(as.character(Ks_cv) %in% names(readRDS(path_cv(l))))
todo      <- lambdas[!vapply(lambdas, cv_cached, TRUE)]
if (length(todo) > 0) {
  message(sprintf('CV for lambda = %s', paste(lambda_lab(todo), collapse = ', ')))
  cv_args <- list(repo_root = getwd(), path_y = path_y, fold_seed = fold_seed, V = V_cv,
                  p_grid = p_grid, supp_Y = supp_Y, K_max = K_max, Ks = Ks_cv)
  tasks <- expand.grid(v = seq_len(V_cv), lambda = todo)
  t0 <- Sys.time()
  old_plan <- future::plan(future::multisession, workers = cv_workers)
  out <- future.apply::future_lapply(
    seq_len(nrow(tasks)),
    function(i) cv_task(tasks$v[i], tasks$lambda[i], cv_args),
    future.seed = NULL,
    future.globals = c('cv_task', 'tasks', 'cv_args')
  )
  future::plan(old_plan)
  print(Sys.time() - t0)
  for (l in todo) {
    L <- matrix(NA_real_, N, length(Ks_cv))
    for (o in out[vapply(out, `[[`, 0, 'lambda') == l]) L[o$idx_valid, ] <- o$losses
    saveRDS(setNames(lapply(seq_along(Ks_cv), function(ki) L[, ki]), Ks_cv), path_cv(l))
  }
}
cv_losses <- setNames(lapply(lambdas, function(l) readRDS(path_cv(l))[as.character(Ks_cv)]),
                      as.character(lambdas))

## Qualifying dimension: smallest K whose (1 - alpha) quantile of CV error is
## at most epsilon
q_err  <- sapply(cv_losses, function(L) sapply(L, quantile, 1 - alpha))   # K x lambda
K_star <- apply(q_err, 2, function(q) { k <- which(q <= epsilon); if (length(k)) Ks_cv[min(k)] else NA })
tab_K  <- data.frame(
  lambda          = lambdas,
  K_star          = K_star,
  q_err_at_K_star = sapply(seq_along(lambdas), function(j) {
    if (is.na(K_star[j])) NA else q_err[match(K_star[j], Ks_cv), j]
  })
)
cat(sprintf('\nepsilon = %g, alpha = %g\n', epsilon, alpha))
print(tab_K)
if (!isTRUE(K_star[1] == K0)) {
  warning(sprintf('K*(lambda = 0) = %s, but the demo pipeline uses K = %d.', K_star[1], K0))
}
write.csv(tab_K, file.path(dir_out, 'qualifying_dimension_by_lambda.csv'), row.names = FALSE)
write.csv(data.frame(K = Ks_cv, q_err, check.names = FALSE),
          file.path(dir_out, 'cv_error_quantile_by_lambda.csv'), row.names = FALSE)


## ==================== 2. Fits and generativity at K*(lambda) ==================== ##

path_pipe <- function(l, K) file.path(dir_cache, sprintf('pipe_lambda-%s_K-%d.rds', lambda_lab(l), K))
path_flow <- function(l, K) file.path(dir_cache, sprintf('flow_lambda-%s_K-%d.pth', lambda_lab(l), K))
path_gen  <- function(l, K) {
  if (l == 0 && K == K0) return(path_gen0)
  file.path(dir_cache, sprintf('gen_lambda-%s_K-%d.rds', lambda_lab(l), K))
}

## Serial over lambda (each generativity run holds n_cores pipeline copies)
for (j in seq_along(lambdas)) {
  l <- lambdas[j]; K <- K_star[j]
  if (is.na(K)) { message(sprintf('lambda = %s: no qualifying K in Ks_cv; skipped', lambda_lab(l))); next }
  if (file.exists(path_gen(l, K))) next
  message(sprintf('========== lambda = %s, K = %d ==========', lambda_lab(l), K))

  if (file.exists(path_pipe(l, K))) {
    pipe <- readRDS(path_pipe(l, K))
  } else {
    t0   <- Sys.time()
    pipe <- fit(build_pipeline(K, l, path_flow(l, K)), y_list)
    saveRDS(pipe, path_pipe(l, K))
    print(Sys.time() - t0)
  }

  ## Same seed as the demo -> same splits and replicate seeds
  t0 <- Sys.time()
  g  <- evaluate_unconditional_generativity(
    pipe, y_list, S = gen_S, R = gen_R, frac_train = 0.5, pool_subsample_n = NULL,
    J_aug = 500, ridge = 0, seed = gen_eval_seed, n_cores = gen_n_cores, keep_draws = TRUE
  )
  saveRDS(g, path_gen(l, K))
  print(Sys.time() - t0)
  rm(pipe, g); gc()
}


## ==================== 3. Generativity by lambda ==================== ##

avail <- which(!is.na(K_star) & mapply(function(l, K) !is.na(K) && file.exists(path_gen(l, K)),
                                       lambdas, K_star))
lam   <- lambdas[avail]
Ks_a  <- K_star[avail]
labs  <- lambda_lab(lam)
gens  <- lapply(seq_along(lam), function(j) readRDS(path_gen(lam[j], Ks_a[j])))

## Pairing check: T_tv depends only on the split, so equal T_tv across lambda
## confirms the same splits as the lambda = 0 run
for (j in seq_along(lam)[-1]) for (s in seq_len(gen_S)) {
  if (!isTRUE(all.equal(gens[[j]]$splits[[s]]$T_tv, gens[[1]]$splits[[s]]$T_tv))) {
    warning(sprintf('lambda = %s, split %d: splits differ from lambda = %s; scores are unpaired.',
                    labs[j], s, labs[1]))
  }
}

## Generativity scores by lambda, with K*(lambda) on the right axis
scores <- lapply(gens, function(g) lapply(g$splits, `[[`, 'log_norm'))
fig_png(file.path(dir_out, 'gen_scores_vs_lambda.png'), fig_half_w, fig_half_h)
fig_par_paper()
plot_generativity_boxes(
  groups   = labs,
  scores   = scores,
  Ks       = Ks_a,
  hline    = 0,
  xlab     = expression(lambda),
  ylab     = "Log norm. generativity",
  collapse = TRUE
)
dev.off()

## Summary table
ln   <- lapply(scores, unlist)
summ <- data.frame(
  lambda       = lam,
  K_star       = Ks_a,
  cv_err_q     = sapply(seq_along(lam), function(j) q_err[match(Ks_a[j], Ks_cv), match(lam[j], lambdas)]),
  gen_median   = sapply(ln, median),
  gen_q90      = sapply(ln, quantile, 0.9),
  gen_frac_gt1 = sapply(ln, function(x) mean(x > 1))
)
write.csv(summ, file.path(dir_out, 'summary_by_lambda.csv'), row.names = FALSE)
print(format(summ, digits = 3))

message('Done. Outputs in ', dir_out)
