library(dplyr)
library(lmerTest)
library(MASS)
library(MuMIn)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')  



## ========== Representation Learning ========== ##

dataset <- 'chop-mims-day'

if (dataset == 'chop-mims-day') {

  ## Globals
  dir_art <- file.path('artifacts', 'analyze_chop-mims-days')

  ## Load day-level data
  path <- file.path('data', 'processed', 'chop-mims-day_v1.rds')
  y_list <- readRDS(path)
  y_list <- y_list[lengths(y_list) >= 7]  ## remove subjects with less than one week of data

  ## Convert day-level data to unnested list
  tmp <- list()
  for (i in 1:length(y_list)) {
    for (d in 1:length(y_list[[i]])) {
      tmp[[length(tmp)+1]] <- y_list[[i]][[d]]
    } 
  }
  y_list <- tmp
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))

  ## Define grid
  J_max <- 1000
  p_grid <- p_grid_fun_2(
    breaks = c(1/(J_max + 1), 0.95, J_max/(J_max + 1)),
    interval_counts = c(51, 50)
  )

  ## Construct pipeline
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'identity', y_shift = 0),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 20,
        epsilon = 0.25,
        alpha = 0.05,
        V = 5,
        lambda = 1e-4
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = file.path(dir_art, 'flow.pth')
      )
    ),
    supp_Y = c(0, 0.6, seq(0.601, 30000, by = 0.001)),
    p_star = 0,
    y_star = 0,
    y_min = 0,
    # loss = 'one_minus_concordance',
    loss = 'wasserstein',
    loss_scale = 'median_pairwise_distance',
    loss_scale_samp_rate = 0.1,
    seed = 12345
  )

} else {
  stop("Invalid dataset.")
}

## Fitting
pipeline <- fit(pipeline, y_list)
path_pipe <- file.path(dir_art, 'pipe.pth')
saveRDS(pipeline, path_pipe)

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
c_ctx_ <- decode(pipeline, z_ctx, from = 6, to = 5)
G_Q_star_ctx_ <- decode(pipeline, c_ctx_, from = 5, to = 4)
Q_ctx_ <- decode(pipeline, G_Q_star_ctx_, from = 4, to = 3)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 3, to = 2)
Ty_ctx_ <- decode(pipeline, Qi_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Ty_ctx_, from = 1, to = 0)

## Plot stages
col_recon <- rgb(0, 1, 0, alpha = 0.5)
for (i in 1:2) {
  pi_grid <- pi_grid_fun(Ji_vec[[i]])
  y_max_i <- max(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
  y_min_i <- min(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
  Ty_max_i <- max(c(Ty_ctx$payload[[i]], Ty_ctx_$payload[[i]]))
  Ty_min_i <- min(c(Ty_ctx$payload[[i]], Ty_ctx_$payload[[i]]))
  breaks_y <- seq(y_min_i, y_max_i, length.out = 50)
  breaks_Ty <- seq(Ty_min_i, Ty_max_i, length.out = 50)
  
  par(mfrow=c(2,4))
  h <- hist(y_ctx$payload[[i]], breaks = breaks_y)
  hist(y_ctx_$payload[[i]], add = TRUE, col = col_recon, breaks = breaks_y)
  h <- hist(Ty_ctx$payload[[i]], breaks = breaks_Ty)
  hist(Ty_ctx_$payload[[i]], add = TRUE, col = col_recon, breaks = breaks_Ty)
  plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1))
  lines(pi_grid, Qi_ctx_$payload[[i]], type = 'l', col = col_recon)
  plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1), col = 'gray')
  lines(p_grid, Q_ctx$payload[[i]], type = 'l')
  lines(p_grid, Q_ctx_$payload[[i]], type = 'l', col = col_recon)
  plot(p_grid, G_Q_star_ctx$payload$G_list[[i]], type = 'l', xlim = c(0, 1))
  lines(p_grid, G_Q_star_ctx_$payload$G_list[[i]], type = 'l', col = col_recon)
  plot(c_ctx$payload$c_list[[i]])
  points(c_ctx_$payload$c_list[[i]], col = col_recon)
  plot(z_ctx$payload[[i]])
}

## Plotting
col_train <- rgb(0, 0, 0, alpha = 0.5)
col_outlier <- rgb(1, 0, 0, alpha = 0.5) # rgb(1, 1, 0.75)
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
path_png <- file.path('scratch', 'plots', str_glue('tmp_plot.png'))
fun_list <- c(
  pipeline$training$meta$Q_list[-idx_outliers],
  pipeline$training$meta$Q_list[idx_outliers]
)
grid_list <- rep(list(p_grid), length(fun_list))
colors = c(
  rep(col_train, length.out = N - N_outlier),
  rep(col_outlier, length.out = N_outlier)
)
widths <- rep(1, length(fun_list))
types <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_train, col_outlier),
  width = c(1, 1),
  type  = c(1, 1),
  label = c('Q-train', 'outlier')
)
plot_funs(
  fun_list = fun_list,
  grid_list = grid_list,
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels,
  path = path_png
)


## ----- Save Quantities

path_Q <- file.path(dir_art, str_glue('Q.rds'))
path_C <- file.path(dir_art, str_glue('C.rds'))
path_Z <- file.path(dir_art, str_glue('Z.rds'))
path_E <- file.path(dir_art, str_glue('E.rds'))
path_p_grid <- file.path(dir_art, str_glue('p_grid.rds'))
Q <- do.call(rbind, Q_ctx$payload)
C <- do.call(rbind, c_ctx$payload$c_list)
Z <- do.call(rbind, z_ctx$payload)
E <- pipeline$stages[[5]]$state$E
p_grid <- pipeline$stages[[3]]$state$p_grid
saveRDS(Q, path_Q)
saveRDS(C, path_C)
saveRDS(Z, path_Z)
saveRDS(E, path_E)
saveRDS(p_grid, path_p_grid)


## ----- Weighted Q-PCA

## Perform Q-PCA
sqrt_w <- pipeline$stages[[3]]$state$sqrt_w
K <- pipeline$stages[[5]]$state$K
Qw <- sweep(Q, 2, sqrt_w, FUN = "*")
Qpc_res <- prcomp(Qw, center = TRUE, scale. = FALSE)
Qpc <- Qpc_res$x[, 1:K, drop = FALSE]
Vw <- Qpc_res$rotation[, 1:K]
L <- sweep(Vw, 1, sqrt_w, FUN = "/")

## Save quantities 
path_Qpc <- file.path(dir_art, str_glue('Qpc.rds'))
path_L <- file.path(dir_art, str_glue('L.rds'))
path_Qpc_res <- file.path(dir_art, str_glue('Qpc_res.rds'))
saveRDS(Qpc, path_Qpc)
saveRDS(L, path_L)
saveRDS(Qpc_res, path_Qpc_res)



## ========== Data Loading + Processing ========== ##

## Globals
dir_art <- file.path('artifacts', 'analyze_chop-mims-days')

## ----- Data Loading

## Paths (from process_chop.R)
path_y <- file.path('data', 'processed', 'chop-mims-day_v1.rds')
path_sleep <- file.path('data', 'processed', 'chop-sleep-day_v1.rds')
path_nonwear <- file.path('data', 'processed', 'chop-nonwear-day_v1.rds')
path_mvpa <- file.path('data', 'processed', 'chop-mvpa-day_v1.rds')
path_misc <- file.path('data', 'processed', 'chop_misc-preds.rds')

## Paths (from pipeline)
path_Q <- file.path(dir_art, str_glue('Q.rds'))
path_C <- file.path(dir_art, str_glue('C.rds'))
path_Z <- file.path(dir_art, str_glue('Z.rds'))
path_E <- file.path(dir_art, str_glue('E.rds'))
path_p_grid <- file.path(dir_art, str_glue('p_grid.rds'))

## Paths (other)
path_Qpc <- file.path(dir_art, str_glue('Qpc.rds'))

## Load data
y_list <- readRDS(path_y)
y_list <- y_list[lengths(y_list) >= 7]  ## remove subjects with less than one week of data
df_sleep <- readRDS(path_sleep)
df_nonwear <- readRDS(path_nonwear)
df_mvpa <- readRDS(path_mvpa)
df_misc <- readRDS(path_misc)
Q <- readRDS(path_Q)
C <- readRDS(path_C)
Z <- readRDS(path_Z)
E <- readRDS(path_E)
p_grid <- readRDS(path_p_grid)
Qpc <- readRDS(path_Qpc)

## Define factor variables
df_misc$age_cat <- as.factor(df_misc$age_cat)
df_misc$sex <- as.factor(df_misc$sex)


## ----- Derived Variables

## Scalar summaries
df_a <- data.frame(
  sub_id = character(),
  date = character()
)
for (sub_id in names(y_list)) {
  for (date in names(y_list[[sub_id]])) {
    a <- y_list[[sub_id]][[date]]
    row <- data.frame(
      sub_id = sub_id,
      date = date,
      a_mean = mean(a),
      a_10 = quantile(a, 0.1),
      a_20 = quantile(a, 0.2),
      a_30 = quantile(a, 0.3),
      a_40 = quantile(a, 0.4),
      a_50 = quantile(a, 0.5),
      a_60 = quantile(a, 0.6),
      a_70 = quantile(a, 0.7),
      a_80 = quantile(a, 0.8),
      a_90 = quantile(a, 0.9),
      a_95 = quantile(a, 0.95),
      a_99 = quantile(a, 0.99),
      a_995 = quantile(a, 0.995),
      a_999 = quantile(a, 0.999)
    )
    df_a <- rbind(df_a, row)
  }
}

## Nonwear
df_nonwear$t_nonwear <- log((df_nonwear$nonwear + 1e-3) / (1 - df_nonwear$nonwear + 1e-3))

## Sleep
##  - Add lagged sleep data
df_sleep <- df_sleep %>%
  arrange(sub_id, date) %>%
  group_by(sub_id) %>%
  mutate(
    lag_1_date = dplyr::lag(date),
    is_lag_1 = as.integer(as.Date(date) - as.Date(lag_1_date)) == 1L,
    sleep_dur_lag_1 = ifelse(is_lag_1, dplyr::lag(sleep_dur), NA_real_),
    sleep_eff_lag_1 = ifelse(is_lag_1, dplyr::lag(sleep_eff), NA_real_)
  ) %>%
  ungroup() %>%
  dplyr::select(-lag_1_date, -is_lag_1)


## ----- Dataframe Assembly

make_block <- function(mat, prefix) {
  out <- as.data.frame(mat)
  names(out) <- paste0(prefix, "_", seq_len(ncol(mat)))
  out
}

## Assemble master dataframe
df <- cbind(
  df_a,
  make_block(C, 'c'), 
  make_block(Z, 'z'),
  make_block(Qpc, 'q')
)
df <- df %>%
  inner_join(df_sleep, by = c('sub_id', 'date')) %>%
  inner_join(df_nonwear, by = c('sub_id', 'date')) %>%
  inner_join(df_mvpa, by = c('sub_id', 'date')) %>%
  inner_join(df_misc, by = c('sub_id'))
df$sub_id <- as.factor(df$sub_id)

## Save dataframe
path_df <- file.path(dir_art, 'df.rds')
saveRDS(df, path_df)


## ========== EDA ========== ##

resp <- 'sleep_dur'
preds <- c(
  'a_50',
  'a_99',
  paste0('c_', 1:K),
  paste0('z_', 1:K),
  paste0('q_', 1:K),
  't_nonwear',
  'sleep_dur',
  'sleep_eff'
)
for (pred in preds) {
  plot(df[[pred]], df[[resp]], main = str_glue("{resp} vs. {pred}"))
}



## ========== Modeling Helpers ========== ##

## ----- CV

make_x_y <- function(formula, data) {
  y <- data[[all.vars(formula)[1]]]
  X <- model.matrix(formula, data)[, -1, drop = FALSE]  # drop intercept
  list(X = X, y = y)
}


########
# df = df
# formula = f_full
# fit_fun  = mf$fit
# pred_fun = mf$pred
# V = 2
# seed = gen_seed()
# subject_var = 'sub_id'
########

cv_oos_errors <- function(
    df,
    formula,
    fit_fun,
    pred_fun,
    subject_var = "sub_id",
    V = 5,
    seed = gen_seed()
) {
  N <- nrow(df)
  subjects <- unique(df[[subject_var]])
  n_subj   <- length(subjects)
  
  set.seed(seed)
  subj_folds <- sample(rep(seq_len(V), length.out = n_subj))
  names(subj_folds) <- subjects
  folds <- subj_folds[as.character(df[[subject_var]])]   # expand to row level
  
  y     <- df[[all.vars(formula)[1]]]
  y_hat <- rep(NA_real_, N)
  
  for (v in seq_len(V)) {
    test_idx  <- which(folds == v)
    train_idx <- setdiff(seq_len(N), test_idx)
    fit <- fit_fun(formula, df[train_idx, , drop = FALSE])
    y_hat[test_idx] <- pred_fun(fit, df[test_idx, , drop = FALSE])
  }
  
  data.frame(row = seq_len(N), y = y, y_hat = y_hat, oos_error = y - y_hat)
}


## ----- LME

fit_lme <- function(formula, data, subject_var = "sub_id") {
  ## Inject random intercept for subject into whatever fixed-effects formula
  ## was passed, without the caller needing to specify it explicitly.
  lmerTest::lmer(
    update(formula, paste("~ . + (1 |", subject_var, ")")),
    data = data,
    REML = TRUE
  )
}

pred_lme <- function(fit, newdata) {
  ## re.form = NA: use only fixed effects for new subjects (no BLUP).
  ## For subjects seen in training, use re.form = NULL to include BLUPs.
  predict(fit, newdata = newdata, allow.new.levels = TRUE, re.form = NA)
}


## ----- GAMM

fit_gamm <- function(formula, data, subject_var = "sub_id") {
  ## Add subject random effect as a RE smoother — stays within gam().
  f_re <- update(formula, paste("~ . + s(", subject_var, ", bs = 're')"))
  mgcv::gam(f_re, data = data, method = "REML", select = TRUE)
}

pred_gamm <- function(fit, newdata) {
  predict(fit, newdata = newdata, type = "response")
}



## ----- Formulas

get_model_funs <- function(model_type) {
  if (model_type == "lme") {
    list(fit = fit_lme, pred = pred_lme, r_sq = function(m) MuMIn::r.squaredGLMM(m)[1, "R2m"])
    
  } else if (model_type == "gamm") {
    list(fit = fit_gamm, pred = pred_gamm, r_sq = function(m) summary(m)$r.sq)
    
  } else {
    stop("Unknown model_type")
  }
}

make_formula <- function(
    response, 
    predictors, 
    model_type,
    factor_vars
) {
  ## Intercept-only (skip later for non-LM models)
  if (length(predictors) == 1 && predictors == "1") {
    return(as.formula(paste(response, "~ 1")))
  }
  
  if (model_type %in% c("lme", "rf", "svm", "enet", "xgb")) {
    rhs <- paste(predictors, collapse = " + ")
    
  } else if (model_type == "gamm") {
    pred_terms <- sapply(predictors, function(pred) {
      if (pred %in% factor_vars) pred else str_glue("s({pred})")
    })
    rhs <- paste(pred_terms, collapse = " + ")
    
  } else {
    stop("Unknown model_type")
  }
  
  as.formula(paste(response, "~", rhs))
}

formula_to_string <- function(f) {
  paste(
    as.character(f)[2],
    as.character(f)[1],
    as.character(f)[3]
  )
}

pred_block <- function(prefix, K) {
  paste0(prefix, "_", seq_len(K))
}



## ========== Modeling ========== ##

## Globals
response <- 'sleep_dur_lag_1'
preds_common <- c('age_cat', 'sex', 'bmiz')

## Load data and pipeline
path_pipe <- file.path(dir_art, 'pipe.pth')
path_df <- file.path(dir_art, 'df.rds')
pipeline <- readRDS(path_pipe)
K <- pipeline$stages[[5]]$state$K
df <- readRDS(path_df)

## Predictors
models <- list(
  
  ref_mean = list(
    preds = c('1'),
    f_str_abbr = '--'
  ),
  ref_common_preds = list(
    preds = c(preds_common),
    f_str_abbr = 'x'
  ),
  
  
  scale_1 = list(
    preds = c(preds_common, "mvpa_rate"),
    f_str_abbr = 'x,mvpa'
  ),
  scale_2 = list(
    preds = c(preds_common, "a_10"),
    f_str_abbr = 'x,a_10'
  ),
  scale_3 = list(
    preds = c(preds_common, "a_20"),
    f_str_abbr = 'x,a_20'
  ),
  scale_4 = list(
    preds = c(preds_common, "a_30"),
    f_str_abbr = 'x,a_30'
  ),
  scale_5 = list(
    preds = c(preds_common, "a_40"),
    f_str_abbr = 'x,a_40'
  ),
  scale_6 = list(
    preds = c(preds_common, "a_50"),
    f_str_abbr = 'x,a_50'
  ),
  scale_7 = list(
    preds = c(preds_common, "a_60"),
    f_str_abbr = 'x,a_60'
  ),
  scale_8 = list(
    preds = c(preds_common, "a_70"),
    f_str_abbr = 'x,a_70'
  ),
  scale_9 = list(
    preds = c(preds_common, "a_80"),
    f_str_abbr = 'x,a_80'
  ),
  scale_10 = list(
    preds = c(preds_common, "a_90"),
    f_str_abbr = 'x,a_90'
  ),
  scale_11 = list(
    preds = c(preds_common, "a_95"),
    f_str_abbr = 'x,a_95'
  ),
  scale_12 = list(
    preds = c(preds_common, "a_99"),
    f_str_abbr = 'x,a_99'
  ),
  scale_13 = list(
    preds = c(preds_common, "a_995"),
    f_str_abbr = 'x,a_995'
  ),
  scale_14 = list(
    preds = c(preds_common, "a_999"),
    f_str_abbr = 'x,a_999'
  ),
  
  dist_Qpc = list(
    preds = c(preds_common, pred_block('q', K)),
    f_str_abbr = 'x,q'
  ),
  dist_C = list(
    preds = c(preds_common, pred_block('c', K)),
    f_str_abbr = 'x,c'
  ),
  
  dist_Qpc_miss = list(
    preds = c(preds_common, 't_nonwear', pred_block('q', K)),
    f_str_abbr = 'x,q,nw'
  ),
  dist_C_miss = list(
    preds = c(preds_common, 't_nonwear', pred_block('c', K)),
    f_str_abbr = 'x,c,nw'
  )
  
)

## Dataframe omissions
df <- na.omit(df)

## Compute OOS Errors
mf <- get_model_funs('lme')
factor_vars <- names(Filter(is.factor, df))
results <- data.frame(
  oos_err = numeric(0),
  r_sq    = numeric(0),
  f_str   = character(0),
  preds   = character(0)
)
for (i in seq_along(models)) {
  print(str_glue("i = {i}/{length(models)}"))
  set.seed(12345)
  preds <- models[[i]]$preds
  
  ## Fit model
  f_full <- make_formula(response, preds, 'lme', factor_vars)
  mod_full <- mf$fit(f_full, df)
  models[[i]]$model_full <- mod_full
  
  ## Cross validation
  cv_full <- cv_oos_errors(
    df, f_full,
    fit_fun  = mf$fit,
    pred_fun = mf$pred,
    V = 10,
    seed = gen_seed()
  )
  
  ## Save results
  results <- rbind(
    results,
    data.frame(
      oos_err = sqrt(mean(cv_full$oos_error^2)),
      marg_r_sq    = mf$r_sq(mod_full),
      f_str   = formula_to_string(f_full),
      preds = models[[i]]$f_str_abbr
    )
  )
  
}
results$oos_err <- round(results$oos_err, digits = 4)
results$marg_r_sq <- round(results$marg_r_sq, digits = 4)
print(results[,c('oos_err', 'marg_r_sq', 'preds')])

## Summaries
for (i in 1:length(models)) {
  out <- coef(summary(models[[i]]$model_full))[, c(1, 2, 4, 5)]
  if (is.null(nrow(out))) {
    dim(out) <- c(1, 4)
  }
  pv <- out[, 4]
  stars <- symnum(pv, cutpoints = c(0, .001, .01, .05, 1),
                  symbols = c("***", "**", "*", ""))
  out <- cbind(round(out, 4), sig = format(stars))
  print(noquote(out))
}


## Response: sleep_dur
#       oos_err marg_r_sq   preds
# R2m    1.6111    0.0000      --
# R2m1   1.6079    0.0101       x
# R2m2   1.6077    0.0117  x,mvpa
# R2m3   1.6075    0.0126  x,a_50
# R2m4   1.6060    0.0136  x,a_95
# R2m5   1.6072    0.0113  x,a_99
# R2m6   1.6076    0.0107 x,a_995
# R2m7   1.6082    0.0102 x,a_999
# R2m8   1.6027    0.0241     x,q
# R2m9   1.5937    0.0400     x,c
# R2m10  1.6038    0.0243  x,q,nw
# R2m11  1.5958    0.0401  x,c,nw

## NOTES
##  - OOS error is larger for day-level analysis than for subject-level analysis.
##    This is likely because day-to-day sleep variation is higher than 
##    subject-to-subject sleep variation.
##  - P-values estimated via Satterthwaite approximation.
##  - PA-based predictors are more significant in day-level analysis. 

## TODO: 
##  - Plan out fancier models. 
