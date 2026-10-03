library(dplyr)
library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')  


## ========== Setup ========== ##

dir_art <- file.path('artifacts', 'analyze_chop-mims-subs_v2')

## Latent dimensions
K <- 4
K_qpca <- 4

## Load data
path <- file.path('data', 'processed', 'chop-mims_v1.rds')
y_list <- readRDS(path)
N <- length(y_list)
Ji_vec <- lengths(y_list)
Ji_max <- max(Ji_vec)
y_max <- max(unlist(y_list))

## Define grid
p_grid <- p_grid_fun(
  breaks = c(1/(12000 + 1), 0.95, 12000/(12000 + 1)),
  interval_counts = c(51, 50)
)

## ========== Representation Learning (WAME) ========== ##

## Construct pipeline
pipeline <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_wame(
      K = K,
      K_max = 20,
      lambda = 0,
      order_by = 'importance'
    ),
    stage_flow(
      n_layers = 16,
      hidden = 64,
      max_epochs = 1000,
      lr = 1e-3,
      path = file.path(dir_art, 'flow.pth')
    )
  ),
  supp_Y = c(0, 0.6, seq(0.601, 500, by = 0.001)),
  p_star = 0,
  y_star = 0,
  y_min = 0,
  seed = 12345
)

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
Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
c_ctx_ <- decode(pipeline, z_ctx, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)


## ----- Visualize reconstructions

## Compute reconstruction losses
recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_ctx$payload[[i]]
  Q2 <- Qi_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

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
  path          = file.path(dir_art, 'plots', 'representation',
                            sprintf('Qi-recon_grid_3x5%s.png',
                                    if (log_scale) '_log' else ''))
)


## ----- Visualize data in different spaces

## Plotting globals
mark_outliers <- TRUE
include_legend <- TRUE
k_from <- 1
k_to <- 4
idx_outliers <- pipeline$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.25), col_train)

## Plot z-embeddings
path_plot <- file.path(
  dir_art, 'plots', 'representation',
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
  dir_art, 'plots', 'representation',
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


## ----- Save Quantities

path_Q <- file.path(dir_art, str_glue('Q.rds'))
path_C <- file.path(dir_art, str_glue('C.rds'))
path_Z <- file.path(dir_art, str_glue('Z.rds'))
path_E <- file.path(dir_art, str_glue('E.rds'))
path_p_grid <- file.path(dir_art, str_glue('p_grid.rds'))
Q <- do.call(rbind, Q_ctx$payload)
C <- do.call(rbind, c_ctx$payload$c_list)
Z <- do.call(rbind, z_ctx$payload)
E <- pipeline$stages[[3]]$state$child_qg_pca$state$E
p_grid <- pipeline$training$cache$p_grid
saveRDS(Q, path_Q)
saveRDS(C, path_C)
saveRDS(Z, path_Z)
saveRDS(E, path_E)
saveRDS(p_grid, path_p_grid)


## ========== Representation Learning (Q-PCA) ========== ##

## Construct pipeline
pipeline_qpca <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid),
    stage_q_pca(K = K_qpca)
  ),
  supp_Y = c(0, 0.6, seq(0.601, 500, by = 0.001)),
  p_star = 0,
  y_star = 0,
  y_min  = 0,
  seed   = 12345
)

## Fitting
pipeline_qpca <- fit(pipeline_qpca, y_list)
path_pipe <- file.path(dir_art, 'pipe_qpca.pth')
saveRDS(pipeline_qpca, path_pipe)

## New context
y_qpca_ctx <- new_context(
  payload = y_list,
  cache = pipeline_qpca$training$cache,
  meta = list()
)

## Encode/Decode
Qi_qpca_ctx <- encode(pipeline_qpca, y_qpca_ctx, from = 0, to = 1)
Q_qpca_ctx <- encode(pipeline_qpca, Qi_qpca_ctx, from = 1, to = 2)
q_qpca_ctx <- encode(pipeline_qpca, Q_qpca_ctx, from = 2, to = 3)
Q_qpca_ctx_ <- decode(pipeline_qpca, q_qpca_ctx, from = 3, to = 2)
Qi_qpca_ctx_ <- decode(pipeline_qpca, Q_qpca_ctx_, from = 2, to = 1)
y_qpca_ctx_ <- decode(pipeline_qpca, Qi_qpca_ctx_, from = 1, to = 0)


## ----- Visualize reconstructions

## Compute reconstruction losses
recon_losses <- numeric(N)
for (i in 1:N) {
  Q1 <- Qi_qpca_ctx$payload[[i]]
  Q2 <- Qi_qpca_ctx_$payload[[i]]
  w <- get_quadrature_weights(pi_grid_fun(length(Q1)))
  recon_losses[i] <- wasserstein(Q1, Q2, w)
}

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
  path          = file.path(dir_art, 'plots', 'representation',
                            sprintf('Qi-qpca-recon_grid_3x5%s.png',
                                    if (log_scale) '_log' else ''))
)


## ----- Visualize data in different spaces

## Plotting globals
mark_outliers <- TRUE
include_legend <- TRUE
k_from <- 1
k_to <- 4
idx_outliers <- pipeline_qpca$training$meta$idx_outliers
N_outlier <- length(idx_outliers)
idx_train <- if (N_outlier > 0) -idx_outliers else seq_len(N)
col_train <- rgb(0, 0, 0, alpha = 0.25)
col_outlier <- ifelse(mark_outliers, rgb(1, 0.8, 0.5, alpha = 0.25), col_train)

## Plot q-embeddings
path_plot <- file.path(
  dir_art, 'plots', 'representation',
  str_glue('data_q_qpca.png')
)
png(path_plot, width = 960, height = 960, pointsize = 18)
Qpc_plot <- rbind(
  do.call(rbind, q_qpca_ctx$payload)[idx_train,k_from:k_to],
  do.call(rbind, q_qpca_ctx$payload)[idx_outliers,k_from:k_to]
)
colors <- c(
  rep(col_train, N - N_outlier),
  rep(col_outlier, N_outlier)
)
shapes <- c(
  rep(19, length.out = nrow(Qpc_plot))
)
sizes <- c(
  rep(0.5, length.out = nrow(Qpc_plot))
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
  Qpc_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)
dev.off()


## ----- Save Quantities

Qpc <- do.call(rbind, q_qpca_ctx$payload)
path_Qpc <- file.path(dir_art, str_glue('Qpc.rds'))
saveRDS(Qpc, path_Qpc)



## ========== Data Loading + Processing ========== ##

## ----- Data Loading

## Paths (from process_chop.R)
path_y <- file.path('data', 'processed', 'chop-mims_v1.rds')
path_sleep <- file.path('data', 'processed', 'chop-sleep_v1.rds')
path_nonwear <- file.path('data', 'processed', 'chop-nonwear_v1.rds')
path_mvpa <- file.path('data', 'processed', 'chop-mvpa_v1.rds')
path_misc <- file.path('data', 'processed', 'chop_misc-preds.rds')

## Paths (from pipelines)
path_Q <- file.path(dir_art, str_glue('Q.rds'))
path_C <- file.path(dir_art, str_glue('C.rds'))
path_Z <- file.path(dir_art, str_glue('Z.rds'))
path_E <- file.path(dir_art, str_glue('E.rds'))
path_Qpc <- file.path(dir_art, str_glue('Qpc.rds'))
path_p_grid <- file.path(dir_art, str_glue('p_grid.rds'))

## Load data
y_list <- readRDS(path_y)
df_sleep <- readRDS(path_sleep)
df_nonwear <- readRDS(path_nonwear)
df_mvpa <- readRDS(path_mvpa)
df_misc <- readRDS(path_misc)
Q <- readRDS(path_Q)
C <- readRDS(path_C)
Z <- readRDS(path_Z)
E <- readRDS(path_E)
Qpc <- readRDS(path_Qpc)
p_grid <- readRDS(path_p_grid)

## Define factor variables
## Binary (0/1 numeric, so coefficients print as `old` / `male` without a
## level suffix): old = age > 12; male = sex code 2 (1 = female)
df_misc$old <- as.integer(df_misc$age_cat == '>12')
df_misc$male <- as.integer(df_misc$sex == 2)
df_misc$age_cat <- NULL
df_misc$sex <- NULL


## ----- Derived Variables

## Scalar summaries
df_a <- data.frame(sub_id = names(y_list))
df_a$a_mean <- unlist(lapply(y_list, mean))
df_a$a_10 <- unlist(lapply(y_list, function(y) quantile(y, c(0.1))))
df_a$a_20 <- unlist(lapply(y_list, function(y) quantile(y, c(0.2))))
df_a$a_30 <- unlist(lapply(y_list, function(y) quantile(y, c(0.3))))
df_a$a_40 <- unlist(lapply(y_list, function(y) quantile(y, c(0.4))))
df_a$a_50 <- unlist(lapply(y_list, function(y) quantile(y, c(0.5))))
df_a$a_60 <- unlist(lapply(y_list, function(y) quantile(y, c(0.6))))
df_a$a_70 <- unlist(lapply(y_list, function(y) quantile(y, c(0.7))))
df_a$a_80 <- unlist(lapply(y_list, function(y) quantile(y, c(0.8))))
df_a$a_90 <- unlist(lapply(y_list, function(y) quantile(y, c(0.9))))
df_a$a_95 <- unlist(lapply(y_list, function(y) quantile(y, c(0.95))))
df_a$a_96 <- unlist(lapply(y_list, function(y) quantile(y, c(0.97))))
df_a$a_99 <- unlist(lapply(y_list, function(y) quantile(y, c(0.99))))
df_a$a_995 <- unlist(lapply(y_list, function(y) quantile(y, c(0.995))))
df_a$a_999 <- unlist(lapply(y_list, function(y) quantile(y, c(0.999))))
df_a$a_tail95_mean <- unlist(lapply(y_list, function(y) mean(y[y >= quantile(y, c(0.95))])))
df_a$a_tail90_mean <- unlist(lapply(y_list, function(y) mean(y[y >= quantile(y, c(0.90))])))
df_a$a_tail85_mean <- unlist(lapply(y_list, function(y) mean(y[y >= quantile(y, c(0.85))])))
df_a$a_above60mims <- unlist(lapply(y_list, function(y) mean(y >= 60)))

## Nonwear
df_nonwear$t_nonwear <- log((df_nonwear$nonwear + 1e-3) / (1 - df_nonwear$nonwear + 1e-3))

## Sleep
df_sleep$good_sleep <- df_sleep$sleep_dur >= 8

## Misc
## NOTE: Dichotomized impact variable leads to models with better predictive 
## performance then trichotomized impact models and continuous impact variables.
imp_high_50 <- median(df_misc$impact23_hr_day)
df_misc$imp <- as.integer(df_misc$impact23_hr_day >= imp_high_50)



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
  inner_join(df_sleep, by = 'sub_id') %>%
  inner_join(df_nonwear, by = 'sub_id') %>%
  inner_join(df_mvpa, by = 'sub_id') %>%
  inner_join(df_misc, by = 'sub_id')
df <- na.omit(df)

## Save dataframe
path_df <- file.path(dir_art, 'df.rds')
saveRDS(df, path_df)


## ========== EDA ========== ##

## Questions
##  - Univariate summaries for the following variables
##     * Age
##     * Sex
##     * BMI
##     * Impact Activities
##     * 

# resps <- c(
#   # 'a_99'
#   # 'bmiz'
#   'subtot_bmd_age_z_all'
#   # 'hip_neck_bmd_age_z_all',
#   # 'spine_bmd_age_z_all',
#   # 'spine_bmad_age_z_all',
#   # 'tot_hip_bmd_age_z_all',
#   # 'radius13_bmd_age_z_all',
#   # 'udradius_bmd_age_z_all'

# )
# preds <- c(
#   # 'bmiz',
#   # 'a_50',
#   # 'a_60',
#   # 'a_70',
#   # 'a_80',
#   # 'a_90',
#   # 'a_95',
#   # 'a_99'
#   # paste0('c_', 1:K),
#   # paste0('z_', 1:K),
#   # paste0('q_', 1:K),
#   # 't_nonwear',
#   # 'sleep_dur'
#   'imp'
# )
# par(mfrow = c(1,1))
# for (resp in resps) {
#   for (pred in preds) {
#     x <- df[[pred]]
#     y <- df[[resp]]
#     plot(
#       x, y, 
#       # main = str_glue("{resp} vs. {pred}"), xlim = c(0,2)
#       main = str_glue("{resp} vs. {pred} | cor = {round(cor(x,y), 3)}")
#     )
#   }
# }



## ========== Modeling Helpers ========== ##

## ----- CV

make_x_y <- function(formula, data) {
  y <- data[[all.vars(formula)[1]]]
  X <- model.matrix(formula, data)[, -1, drop = FALSE]  # drop intercept
  list(X = X, y = y)
}

cv_oos_errors <- function(
    df,
    formula,
    fit_fun,
    pred_fun,
    V = 5,
    seed = gen_seed(),
    folds = NULL
) {
  ## folds: optional precomputed fold assignment (length nrow(df)); pass the
  ## same vector to every model so their errors are on identical splits
  N <- nrow(df)
  if (is.null(folds)) {
    set.seed(seed)
    folds <- sample(rep(seq_len(V), length.out = N))
  }
  V <- max(folds)
  
  y <- df[[all.vars(formula)[1]]]
  y_hat <- rep(NA_real_, N)
  
  for (v in seq_len(V)) {
    test_idx  <- which(folds == v)
    train_idx <- setdiff(seq_len(N), test_idx)
    
    fit <- fit_fun(formula, df[train_idx, , drop = FALSE])
    
    y_hat[test_idx] <- pred_fun(
      fit,
      df[test_idx, , drop = FALSE]
    )
  }
  
  data.frame(
    row = seq_len(N),
    y = y,
    y_hat = y_hat,
    oos_error = y - y_hat
  )
}


## ----- LM

fit_lm <- function(formula, data) {
  lm(formula, data = data)
}

pred_lm <- function(fit, newdata) {
  predict(fit, newdata = newdata)
}

get_sig_lm_predictors <- function(mod, factor_vars, alpha = 0.05) {

  coefs <- summary(mod)$coefficients
  p_vals <- coefs[-1, 4]     # drop intercept
  coef_nm <- rownames(coefs)[-1]

  preds_sig <- character(0)

  for (v in all.vars(delete.response(terms(mod)))) {
    if (v %in% factor_vars) {
      idx <- startsWith(coef_nm, paste0(v))
      if (any(p_vals[idx] < alpha)) {
        preds_sig <- c(preds_sig, v)
      }
    } else {
      if (v %in% coef_nm && p_vals[coef_nm == v] < alpha) {
        preds_sig <- c(preds_sig, v)
      }
    }
  }

  unique(preds_sig)
}

## Return the term in the model with the highest "effective" p-value if it
## exceeds alpha, else NULL. Operates at the term level (so interaction terms
## like "a:b" are first-class candidates). The effective p-value of a term is
## the minimum p-value across all coefficient columns that belong to that
## term (per model.matrix() "assign") — matching the any-significant-keeps-it
## convention. Marginality is enforced: a main-effect term is excluded from
## removal as long as any retained interaction term contains it as a
## component (e.g., "b" cannot be dropped while "a:b" is in the model).
get_worst_lm_predictor <- function(mod, alpha = 0.05) {

  coefs  <- summary(mod)$coefficients
  p_vals <- coefs[, 4]

  assign_vec  <- attr(model.matrix(mod), "assign")
  term_labels <- attr(terms(mod), "term.labels")

  ## Per-term effective p-value (min across the term's coef columns)
  pred_p <- vapply(seq_along(term_labels), function(k) {
    idx <- which(assign_vec == k)
    if (length(idx) == 0) return(NA_real_)
    pv <- p_vals[idx]
    pv <- pv[!is.na(pv)]
    if (length(pv) == 0) NA_real_ else min(pv)
  }, numeric(1))
  names(pred_p) <- term_labels

  ## Marginality: block main effects that appear in any retained interaction
  is_interaction <- grepl(":", term_labels, fixed = TRUE)
  if (any(is_interaction)) {
    blocked <- unique(unlist(
      strsplit(term_labels[is_interaction], ":", fixed = TRUE)
    ))
    pred_p[!is_interaction & term_labels %in% blocked] <- NA_real_
  }

  pred_p <- pred_p[!is.na(pred_p)]
  if (length(pred_p) == 0) return(NULL)

  worst_p <- max(pred_p)
  if (worst_p < alpha) return(NULL)
  names(pred_p)[which.max(pred_p)]
}


get_model_funs <- function(model_type) {
  if (model_type == "lm") {
    list(fit = fit_lm, pred = pred_lm, r_sq = function(m) summary(m)$adj.r.squared)
    
  } else if (model_type == "gam") {
    list(fit = fit_gam, pred = pred_gam, r_sq = function(m) summary(m)$r.sq)
    
  } else if (model_type == "rf") {
    list(fit = fit_rf, pred = pred_rf, r_sq = function(m) NA_real_)
    
  } else if (model_type == "svm") {
    list(fit = fit_svm, pred = pred_svm, r_sq = function(m) NA_real_)
    
  } else if (model_type == "enet") {
    list(fit = fit_enet, pred = pred_enet, r_sq = function(m) NA_real_)
    
  } else if (model_type == "xgb") {
    list(fit = fit_xgb, pred = pred_xgb, r_sq = function(m) NA_real_)
    
  } else {
    stop("Unknown model_type")
  }
}


## ----- Formulas


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
  
  if (model_type %in% c("lm", "rf", "svm", "enet", "xgb")) {
    rhs <- paste(predictors, collapse = " + ")
    
  } else if (model_type == "gam") {
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

pred_block <- function(prefix, K, modifier = NULL) {
  if (is.null(modifier)) {
    paste0(prefix, "_", seq_len(K))
  } else {
    paste0(prefix, "_", seq_len(K), ":", modifier)
  }
}


## ========== Modeling ========== ##

## Globals
response <- 'subtot_bmd_age_z_all'
preds_common <- c('old', 'male', 'bmiz')

## Load data and pipeline
path_pipe <- file.path(dir_art, 'pipe.pth')
path_pipe_qpca <- file.path(dir_art, 'pipe_qpca.pth')
path_df <- file.path(dir_art, 'df.rds')
pipeline <- readRDS(path_pipe)
pipeline_qpca <- readRDS(path_pipe_qpca)
K <- pipeline$stages[[3]]$state$child_qg_pca$state$K
stopifnot(K_qpca == K)  ## the q_* predictor blocks below assume Q-PCA dim == WAME dim
df <- readRDS(path_df)

## Predictors
models <- list(
  
  ## Reference
  # ref_mean = list(
  #   preds = c('1'),
  #   f_str_abbr = '--'
  # ),
  # ref_common_preds = list(
  #   preds = c(preds_common),
  #   f_str_abbr = 'x'
  # ),
  ref_common_preds_imp = list(
    preds = c(preds_common, 'imp'),
    f_str_abbr = 'x,imp'
  ),

  ## Scalar PA
  # scale_0 = list(
  #   preds = c(preds_common, "a_mean"),
  #   f_str_abbr = 'x,a_mean'
  # ),
  # scale_1 = list(
  #   preds = c(preds_common, "mvpa_rate"),
  #   f_str_abbr = 'x,mvpa'
  # ),
  # scale_2 = list(
  #   preds = c(preds_common, "a_50"),
  #   f_str_abbr = 'x,a_50'
  # ),
  # scale_3 = list(
  #   preds = c(preds_common, "a_60"),
  #   f_str_abbr = 'x,a_60'
  # ),
  # scale_4 = list(
  #   preds = c(preds_common, "a_70"),
  #   f_str_abbr = 'x,a_70'
  # ),
  # scale_5 = list(
  #   preds = c(preds_common, "a_80"),
  #   f_str_abbr = 'x,a_80'
  # ),
  # scale_6 = list(
  #   preds = c(preds_common, "a_90"),
  #   f_str_abbr = 'x,a_90'
  # ),
  # scale_7 = list(
  #   preds = c(preds_common, "a_95"),
  #   f_str_abbr = 'x,a_95'
  # ),
  # scale_8 = list(
  #   preds = c(preds_common, "a_99"),
  #   f_str_abbr = 'x,a_99'
  # ),
  # scale_9 = list(
  #   preds = c(preds_common, "a_995"),
  #   f_str_abbr = 'x,a_995'
  # ),
  # scale_10 = list(
  #   preds = c(preds_common, "a_999"),
  #   f_str_abbr = 'x,a_999'
  # ),
  # scale_11_a = list(
  #   preds = c(preds_common, "a_tail95_mean"),
  #   f_str_abbr = 'x,a_tail95_mean'
  # ),
  # scale_11_b = list(
  #   preds = c(preds_common, "a_tail90_mean"),
  #   f_str_abbr = 'x,a_tail90_mean'
  # ),
  # scale_11_c = list(
  #   preds = c(preds_common, "a_tail85_mean"),
  #   f_str_abbr = 'x,a_tail85_mean'
  # ),
  # scale_12 = list(
  #   preds = c(preds_common, "a_above60mims"),
  #   f_str_abbr = 'x,a_above60mims'
  # ),
  # scale_13 = list(
  #   preds = c(preds_common, "a_96", "a_995"),
  #   f_str_abbr = 'x,a_96,a_995'
  # ),
  # scale_14 = list(
  #   preds = c(preds_common, "a_96"),
  #   f_str_abbr = 'x,a_96'
  # ),
  
  ## Scalar PA + Impact
  scale_imp_0 = list(
    preds = c(preds_common, "a_mean", "imp", "a_mean:imp"),
    f_str_abbr = 'x,a_mean,imp'
  ),
  scale_imp_1 = list(
    preds = c(preds_common, "mvpa_rate", "imp", "mvpa_rate:imp"),
    f_str_abbr = 'x,mvpa,imp'
  ),
  scale_imp_2 = list(
    preds = c(preds_common, "a_50", "imp", "a_50:imp"),
    f_str_abbr = 'x,a_50,imp'
  ),
  # scale_imp_3 = list(
  #   preds = c(preds_common, "a_60", "imp", "a_60:imp"),
  #   f_str_abbr = 'x,a_60,imp'
  # ),
  # scale_imp_4 = list(
  #   preds = c(preds_common, "a_70", "imp", "a_70:imp"),
  #   f_str_abbr = 'x,a_70,imp'
  # ),
  # scale_imp_5 = list(
  #   preds = c(preds_common, "a_80", "imp", "a_80:imp"),
  #   f_str_abbr = 'x,a_80,imp'
  # ),
  # scale_imp_6 = list(
  #   preds = c(preds_common, "a_90", "imp", "a_90:imp"),
  #   f_str_abbr = 'x,a_90,imp'
  # ),
  scale_imp_7 = list(
    preds = c(preds_common, "a_95", "imp", "a_95:imp"),
    f_str_abbr = 'x,a_95,imp'
  ),
  scale_imp_8 = list(
    preds = c(preds_common, "a_99", "imp", "a_99:imp"),
    f_str_abbr = 'x,a_99,imp'
  ),
  scale_imp_9 = list(
    preds = c(preds_common, "a_995", "imp", "a_995:imp"),
    f_str_abbr = 'x,a_995,imp'
  ),
  scale_imp_10 = list(
    preds = c(preds_common, "a_999", "imp", "a_999:imp"),
    f_str_abbr = 'x,a_999,imp'
  ),
  # scale_imp_11_a = list(
  #   preds = c(preds_common, "a_tail95_mean", "imp", "a_tail95_mean:imp"),
  #   f_str_abbr = 'x,a_tail95_mean,imp'
  # ),
  # scale_imp_11_b = list(
  #   preds = c(preds_common, "a_tail90_mean", "imp", "a_tail90_mean:imp"),
  #   f_str_abbr = 'x,a_tail90_mean,imp'
  # ),
  # scale_imp_11_c = list(
  #   preds = c(preds_common, "a_tail85_mean", "imp", "a_tail85_mean:imp"),
  #   f_str_abbr = 'x,a_tail85_mean,imp'
  # ),
  # scale_imp_12 = list(
  #   preds = c(preds_common, "a_above60mims", "imp", "a_above60mims:imp"),
  #   f_str_abbr = 'x,a_above60mims,imp'
  # ),
  # scale_imp_13 = list(
  #   preds = c(preds_common, "a_96", "a_995", "imp", "a_96:imp", "a_995:imp"),
  #   f_str_abbr = 'x,a_96,a_995,imp'
  # ),
  # scale_imp_14 = list(
  #   preds = c(preds_common, "a_96", "imp", "a_96:imp"),
  #   f_str_abbr = 'x,a_96,imp'
  # ),
  

  ## Distributional PA
  # dist_Qpc = list(
  #   preds = c(preds_common, pred_block('q', K)),
  #   f_str_abbr = 'x,q'
  # ),
  # dist_C = list(
  #   preds = c(preds_common, pred_block('c', K)),
  #   f_str_abbr = 'x,c'
  # ),
  # dist_Z_imp = list(
  #   preds = c(preds_common, pred_block('z', K)),
  #   f_str_abbr = 'x,z'
  # ),
  
  ## Distributional PA + Impact
  dist_Qpc_imp = list(
    preds = c(preds_common, 'imp', pred_block('q', K), pred_block('q', K, 'imp')),
    f_str_abbr = 'x,q,imp'
  ),
  dist_C_imp = list(
    preds = c(preds_common, 'imp', pred_block('c', K), pred_block('c', K, 'imp')),
    f_str_abbr = 'x,c,imp'
  ),
  dist_Z_imp = list(
    preds = c(preds_common, 'imp', pred_block('z', K), pred_block('z', K, 'imp')),
    f_str_abbr = 'x,z,imp'
  )
  
  # dist_Qpc_miss = list(
  #   preds = c(preds_common, 't_nonwear', pred_block('q', K)),
  #   f_str_abbr = 'x,q,nw'
  # ),
  # dist_C_miss = list(
  #   preds = c(preds_common, 't_nonwear', pred_block('c', K)),
  #   f_str_abbr = 'x,c,nw'
  # ),
  # dist_Z_miss = list(
  #   preds = c(preds_common, 't_nonwear', pred_block('z', K)),
  #   f_str_abbr = 'x,z,nw'
  # )
  
)


## Compute OOS Errors
##
## Every model is cross-validated on one shared fold assignment (folds_cv), so
## per-child errors are paired across models. Two errors (two results rows)
## per model:
##  - full: the full formula, refit in each training fold
##  - reduced: the pruning strategy (prune_pval_all) rerun inside each
##    training fold, so selection never sees the held-out fold (this avoids
##    the optimism of selecting on all the data and then CV-ing the selected
##    formula)
## models[[i]]$model_red is the pruned model on all data when pruning drops
## something (but not everything).
## oos_err_low/high: 95% CI for MAE (mean +/- 1.96 SE of per-child absolute
## errors). Paired differences vs. the baseline model (delta_*) compare
## per-child absolute errors on the same folds: full vs. baseline full,
## reduced vs. baseline reduced.
## Reduced rows also describe the pruning: n_terms (terms kept on all data /
## candidate terms), mean_terms_cv (average terms kept per training fold),
## pa_keep_share (share of training folds keeping at least one PA term;
## strictly between 0 and 1 = PA selection unstable across folds) and kept
## (terms kept on all data).
mf <- get_model_funs('lm')
factor_vars <- names(Filter(is.factor, df))
V_cv <- N
set.seed(12345)
folds_cv <- sample(rep(seq_len(V_cv), length.out = nrow(df)))
baseline_model <- 'ref_common_preds_imp'
fit_reduced <- TRUE

## ----- Pruning strategy

## prune_backward_pval: p-value backward elimination. Repeatedly drops the
## worst term (get_worst_lm_predictor, alpha = 0.05; main effects kept while
## their interactions remain) until all remaining terms are significant.
## Returns list(preds, model): the terms kept (character(0) if all dropped)
## and the last fitted model.
norm_term <- function(t) vapply(
  strsplit(t, ":", fixed = TRUE),
  function(p) paste(sort(p), collapse = ":"),
  character(1)
)
prune_backward_pval <- function(preds, data) {
  preds_curr <- preds
  mod_curr <- mf$fit(make_formula(response, preds_curr, 'lm', factor_vars), data)
  repeat {
    worst <- get_worst_lm_predictor(mod_curr, alpha = 0.05)
    if (is.null(worst)) break                  # all predictors significant
    preds_curr <- preds_curr[norm_term(preds_curr) != norm_term(worst)]
    if (length(preds_curr) == 0) break         # nothing left to fit
    mod_curr <- mf$fit(make_formula(response, preds_curr, 'lm', factor_vars), data)
  }
  list(preds = preds_curr, model = mod_curr)
}

## Strategy 'pval_all': p-value backward elimination over all terms,
## covariates included
prune_pval_all <- function(preds, data) {
  prune_backward_pval(preds, data)
}

## is_pa_term: TRUE for terms involving a PA variable (anything outside the
## covariates old, male, bmiz and imp)
is_pa_term <- function(t) vapply(strsplit(t, ':', fixed = TRUE), function(v) {
  !all(v %in% c(preds_common, 'imp'))
}, logical(1))

## err_row: one results row from a vector of per-child OOS errors; prune is
## NULL for full rows, else list(preds, kept_cv, n_candidates) for reduced rows
err_row <- function(err, adj_r_sq, f_str, preds, reduced, prune = NULL) {
  ae <- abs(err)
  se <- sd(ae) / sqrt(length(ae))
  data.frame(
    oos_err_mean  = mean(ae),
    oos_err_low   = mean(ae) - 1.96 * se,
    oos_err_high  = mean(ae) + 1.96 * se,
    adj_r_sq      = adj_r_sq,
    f_str         = f_str,
    preds         = preds,
    reduced       = reduced,
    n_terms       = if (is.null(prune)) NA_character_ else
      as.character(str_glue('{length(prune$preds)}/{prune$n_candidates}')),
    mean_terms_cv = if (is.null(prune)) NA_real_ else mean(lengths(prune$kept_cv)),
    pa_keep_share = if (is.null(prune)) NA_real_ else
      mean(vapply(prune$kept_cv, function(k) any(is_pa_term(k)), logical(1))),
    kept          = if (is.null(prune)) NA_character_ else
      paste(prune$preds, collapse = ',')
  )
}

results <- data.frame()
row_err <- list()   ## per-child OOS errors for each row of results
for (i in seq_along(models)) {
  print(str_glue("{i}/{length(models)}"))
  preds <- models[[i]]$preds

  ## Full model: fit on all data; CV on the shared folds
  f_full <- make_formula(response, preds, 'lm', factor_vars)
  mod_full <- mf$fit(f_full, df)
  models[[i]]$model_full <- mod_full
  cv_full <- cv_oos_errors(df, f_full, fit_fun = mf$fit, pred_fun = mf$pred,
                           folds = folds_cv)
  models[[i]]$cv_err_full <- cv_full$oos_error
  results <- rbind(results, err_row(
    cv_full$oos_error, mf$r_sq(mod_full), formula_to_string(f_full),
    models[[i]]$f_str_abbr, reduced = FALSE
  ))
  row_err[[nrow(results)]] <- cv_full$oos_error

  if (fit_reduced) {
    ## Pruned model on all data (for interpretation and n_terms / kept)
    pruned <- prune_pval_all(preds, df)
    p_red <- if (length(pruned$preds) > 0) pruned$preds else '1'
    f_red <- make_formula(response, p_red, 'lm', factor_vars)
    mod_red <- mf$fit(f_red, df)
    if (length(pruned$preds) > 0 && length(pruned$preds) < length(preds)) {
      models[[i]]$model_red <- mod_red
    }

    ## Pruning CV'd on the shared folds: prune inside each training fold (an
    ## all-pruned fold falls back to intercept only); kept_cv records the
    ## terms kept in each fold
    kept_cv <- list()
    fit_pruned <- function(formula, data) {
      p_sel <- prune_pval_all(preds, data)$preds
      kept_cv[[length(kept_cv) + 1]] <<- p_sel
      if (length(p_sel) == 0) p_sel <- '1'
      mf$fit(make_formula(response, p_sel, 'lm', factor_vars), data)
    }
    cv_red <- cv_oos_errors(df, f_full, fit_fun = fit_pruned, pred_fun = mf$pred,
                            folds = folds_cv)
    models[[i]]$cv_err_red <- cv_red$oos_error
    models[[i]]$kept_cv_red <- kept_cv
    results <- rbind(results, err_row(
      cv_red$oos_error, mf$r_sq(mod_red), formula_to_string(f_red),
      models[[i]]$f_str_abbr, reduced = TRUE,
      prune = list(preds = pruned$preds, kept_cv = kept_cv,
                   n_candidates = length(preds))
    ))
    row_err[[nrow(results)]] <- cv_red$oos_error
  }
}

## Paired differences vs. the baseline (same children, same folds):
## d_j = |e_model,j| - |e_baseline,j|; mean, SE and 95% CI over children.
## Negative delta = lower error than the baseline.
base <- models[[baseline_model]]
stopifnot(!is.null(base$cv_err_full))
paired <- do.call(rbind, lapply(seq_len(nrow(results)), function(r) {
  e_base <- if (results$reduced[r]) base$cv_err_red else base$cv_err_full
  d <- abs(row_err[[r]]) - abs(e_base)
  se <- sd(d) / sqrt(length(d))
  data.frame(delta_mean = mean(d), delta_se = se,
             delta_low = mean(d) - 1.96 * se, delta_high = mean(d) + 1.96 * se)
}))
results <- cbind(results, paired)

num_cols <- c('oos_err_mean', 'oos_err_low', 'oos_err_high', 'adj_r_sq',
              'delta_mean', 'delta_se', 'delta_low', 'delta_high', 'pa_keep_share')
results[num_cols] <- lapply(results[num_cols], round, digits = 3)
results$mean_terms_cv <- round(results$mean_terms_cv, 2)
print_full <- TRUE
print_reduced <- TRUE
print_cols <- c('preds', 'oos_err_mean', 'delta_mean', 'delta_low', 'delta_high')
if (print_full) {
  print(results[!results$reduced, print_cols])
}
if (print_reduced) {
  print(results[results$reduced, c(print_cols, 'n_terms', 'mean_terms_cv',
                                   'pa_keep_share')])
}

## Summaries
print_full <- TRUE
print_reduced <- FALSE
for (i in 1:length(models)) {
  if (print_full) {
    print(summary(models[[i]]$model_full))
  }
  if (print_reduced) {
    print(summary(models[[i]]$model_red))
  }
}



## ========== Model Interpretation ========== ##

## ----- Interpretation: Q PCA (weight function)

## Get model
## NOTE: If you were to use the reduced model to compute the weight
## function, the corresponding uncertainty bands would be too narrow.
## This is because the reduced model effectively fixes insigificant 
## coefficients, which themselves carry uncertainty, at zero.
model <- models[['dist_Qpc_imp']]$model_full
summary(model)

## qpca_joint_band: estimate of the linear functional L %*% alpha (one value per
## row of L), where alpha_k are the q_k coefficients at a given level of
## imp, together with bootstrap-calibrated joint (max-t) 95% bands.
## Accounts for any q_k:imp interaction retained in the model.
## Returns NULL if no q_k terms are retained.
qpca_joint_band <- function(model, L, imp,
                            col_prefix = 'q',
                            imp_var = 'imp',
                            imp_dummy = '',  ## imp is 0/1 numeric: coef named `imp`
                            B = 5000, seed = 12345) {

  K <- ncol(L)
  coef_names <- names(coef(model))
  q_names <- paste0(col_prefix, '_', 1:K)
  imp_dummy_name <- paste0(imp_var, imp_dummy)

  ## R names an interaction column after the order its variables first
  ## appear in the overall formula, not the order written in the ":" term
  ## itself -- so look up both orderings rather than assuming one.
  find_inter_name <- function(term) {
    cand <- c(
      paste0(term, ':', imp_dummy_name),
      paste0(imp_dummy_name, ':', term)
    )
    cand[cand %in% coef_names][1]
  }
  main_names <- ifelse(q_names %in% coef_names, q_names, NA_character_)
  inter_names <- vapply(q_names, find_inter_name, character(1))

  ## Contrast matrix mapping model coefficients to alpha_1, ..., alpha_K
  ## at the given impact level (main effect if present, plus interaction if
  ## present and imp == 1). q_k's with no contributing coefficient have
  ## alpha_k fixed at 0, so they are dropped (along with their column of L)
  ## to keep Sigma_alpha non-singular.
  M <- matrix(
    0, nrow = K, ncol = length(coef_names),
    dimnames = list(q_names, coef_names)
  )
  for (k in 1:K) {
    if (!is.na(main_names[k])) {
      M[k, main_names[k]] <- 1
    }
    if (imp == 1 && !is.na(inter_names[k])) {
      M[k, inter_names[k]] <- 1
    }
  }
  keep <- rowSums(M != 0) > 0
  if (!any(keep)) {
    return(NULL)
  }
  M <- M[keep, , drop = FALSE]
  L <- L[, keep, drop = FALSE]
  K <- sum(keep)
  alpha <- as.vector(M %*% coef(model))
  Sigma_alpha <- M %*% vcov(model) %*% t(M)

  ## Estimate L %*% alpha and pointwise SEs
  est <- as.vector(L %*% alpha)
  se <- sqrt(diag(L %*% Sigma_alpha %*% t(L)))
  pos <- se > 0

  ## Joint (max-t) bands via parametric bootstrap
  set.seed(seed)
  Rchol <- chol(Sigma_alpha)
  M_boot <- numeric(B)
  for (b in 1:B) {
    z <- rnorm(K)
    alpha_b <- alpha + t(Rchol) %*% z
    est_b <- L %*% alpha_b
    M_boot[b] <- max(abs(est_b - est)[pos] / se[pos])
  }
  c95 <- quantile(M_boot, 0.95)
  list(est = est, low = est - c95 * se, high = est + c95 * se)
}

## create_weight_plot: plots the estimated Q-PCA weight function
## alpha(p) = sum_k beta_k e_k(p) (so BMD ~ int alpha(p) (Q(p) - mu(p)) dp)
## together with bootstrap-calibrated joint (max-t) 95% bands, at a given
## level of imp; alpha(p) reduces to the same curve at both impact
## levels when no q_k:imp interaction survived.
create_weight_plot <- function(model, L, p_grid, plim, imp, path,
                                width = 960, height = 960, pointsize = 18,
                                ...) {

  band <- qpca_joint_band(model, L, imp, ...)
  if (is.null(band)) {
    warning('No q_k terms retained in model; alpha(p) is identically 0. ',
            'Skipping plot: ', path)
    return(invisible(NULL))
  }
  alpha_p <- band$est
  alpha_p_low <- band$low
  alpha_p_high <- band$high

  ## Plot joint bands
  p_idx <- which(p_grid >= plim[1] & p_grid <= plim[2])
  png(path, width = width, height = height, pointsize = pointsize)
  plot(
    NULL,
    xlab = 'p', ylab = 'alpha(p)',
    xlim = c(min(p_grid[p_idx]), max(p_grid[p_idx])),
    ylim = c(min(alpha_p_low[p_idx]), max(alpha_p_high[p_idx]))
  )
  draw_band(p_grid[p_idx], alpha_p_low[p_idx], alpha_p_high[p_idx])
  abline(h = 0, lty = 'dashed', col = 'red')
  lines(p_grid[p_idx], alpha_p[p_idx], lwd = 3)
  dev.off()
}

## Basis mapping q-shifts to Q(p)-shifts (shared across impact/plim combos)
qpca_state <- pipeline_qpca$stages[[3]]$state
L <- qpca_state$E
colnames(L) <- paste0('q_', 1:ncol(L))

## Generate weight function plots
plims <- list(`1` = c(0, 1), `2` = c(0.9, 1))
for (plim_id in names(plims)) {
  plim <- plims[[plim_id]]
  for (imp in c(0, 1)) {
    create_weight_plot(
      model = model, L = L, p_grid = p_grid, plim = plim, imp = imp,
      path = str_glue('{dir_art}/plots/q-pca/alpha_plim-{plim_id}_imp-{imp}.png')
    )
  }
}

## create_tail_shift_plot: plots the predicted change in BMD from raising the
## upper tail of the QF, Q(p) for p >= p0, by delta (MIMS), as a function of p0:
##   Delta(p0) = delta * int_{p0}^{1} alpha(p) dp,
## with bootstrap-calibrated joint (max-t) 95% bands. A tail shift keeps the QF
## monotone, so each p0 corresponds to a valid distribution. w are the
## quadrature weights on p_grid used to compute the q_k scores.
## compute_tail_shift_band returns the estimate and joint band (one value per
## p0 in p_grid), or NULL if no q_k terms are retained.
compute_tail_shift_band <- function(model, L, w, imp, delta = 1, ...) {
  ## Row j of L_tail maps alpha to delta * sum_{i >= j} w_i alpha(p_i)
  L_tail <- delta * apply(w * L, 2, function(x) rev(cumsum(rev(x))))
  qpca_joint_band(model, L_tail, imp, ...)
}

create_tail_shift_plot <- function(model, L, w, p_grid, plim, imp, path,
                                    delta = 1,
                                    width = 960, height = 960, pointsize = 18,
                                    ...) {

  band <- compute_tail_shift_band(model, L, w, imp, delta, ...)
  if (is.null(band)) {
    warning('No q_k terms retained in model; tail shift effect is identically 0. ',
            'Skipping plot: ', path)
    return(invisible(NULL))
  }

  ## Plot joint bands
  p_idx <- which(p_grid >= plim[1] & p_grid <= plim[2])
  png(path, width = width, height = height, pointsize = pointsize)
  plot(
    NULL,
    xlab = 'p0',
    ylab = str_glue('Change in BMD (when Q(p) raised by {delta} for all p >= p0)'),
    xlim = c(min(p_grid[p_idx]), max(p_grid[p_idx])),
    ylim = c(min(band$low[p_idx]), max(band$high[p_idx]))
  )
  draw_band(p_grid[p_idx], band$low[p_idx], band$high[p_idx])
  abline(h = 0, lty = 'dashed', col = 'red')
  lines(p_grid[p_idx], band$est[p_idx], lwd = 3)
  dev.off()
}

## Quadrature weights used by the Q-PCA encoder
w_qpca <- pipeline_qpca$training$cache$sqrt_w^2
stopifnot(length(w_qpca) == length(p_grid))

## Generate upper tail shift plots
delta <- 10  ## MIMS; placeholder
plims <- list(`1` = c(0, 1), `2` = c(0.8, 1))
for (plim_id in names(plims)) {
  plim <- plims[[plim_id]]
  for (imp in c(0, 1)) {
    create_tail_shift_plot(
      model = model, L = L, w = w_qpca, p_grid = p_grid, plim = plim,
      imp = imp, delta = delta,
      path = str_glue('{dir_art}/plots/q-pca/tail-shift_plim-{plim_id}_imp-{imp}.png')
    )
  }
}


## ----- Interpretation: activity substitution curves (Q-PCA)

## Activity substitution: replace a reference child's least active delta
## fraction of time with activity of intensity Q(p0), for p0 in [delta, 1].
## The substituted QF is
##   Q~(p) = Q(p + delta)  for p in [0, p0 - delta]
##           Q(p0)         for p in (p0 - delta, p0]
##           Q(p)          for p in (p0, 1],
## and the predicted change in BMD is
##   Delta(p0, delta) = int_0^1 alpha(p) {Q~(p) - Q(p)} dp
##                    = int_0^{p0 - delta} alpha(p) {Q(p + delta) - Q(p)} dp
##                      + int_{p0 - delta}^{p0} alpha(p) {Q(p0) - Q(p)} dp.
## For small delta, Delta ~ delta * int_{Q(0)}^{Q(p0)} alpha(F(y)) dy: alpha
## accumulated along the intensity scale from sedentary up to Q(p0).
## The reference Q for each impact group is the pointwise mean of the group's
## QFs (the QF of their Wasserstein barycenter). Delta is linear in the q_k
## coefficients, so joint bands come from qpca_joint_band (conditional on the
## Q-PCA basis and the reference QF). Integrals are computed on an augmented
## p-grid (p_grid plus evenly spaced points), onto which the Q-PCA basis E is
## linearly interpolated, so that small delta spans many grid points.

## NOTE: (WAME Analogue)
## The WAME model is also linear in its embeddings c, even though the WAME
## encoder c(.) is nonlinear. For a fixed fitted pipeline,
##   Delta_WAME(p0) = sum_k beta_k^(g) {c_k(Q~_{p0}) - c_k(Q_ref)},
## where beta^(g) are the c_k coefficients at impact level g (main effects plus
## c_k:imp interactions when g = 1). Encoding each Q~_{p0} once and stacking
## the differences c(Q~_{p0}) - c(Q_ref) as rows gives a matrix that plays the
## role of L_sub below, so the same vcov-based max-t bands apply (conditional
## on the fitted encoder). To also reflect encoder (and reference) estimation
## uncertainty, use a nonparametric bootstrap: resample children with
## replacement B times (B ~ 200-500); in each replicate refit the WAME stage
## (the flow stage is not needed), re-encode, refit the lm, recompute the
## reference QFs and Delta*(p0); then form pointwise percentile bands or
## max-t bands from the 95th percentile of max_{p0} |Delta* - Delta| / sd*(p0).
## Delta is a difference of predictions, so it is invariant to rotations or
## sign flips of the embeddings across refits and no alignment is needed.

## Augmented grid, quadrature weights and Q-PCA basis on it
J_aug <- 500
p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(J_aug - length(p_grid)))))
w_aug <- get_quadrature_weights(p_grid_aug)
E_aug <- apply(qpca_state$E, 2, function(e) {
  approx(p_grid, e, xout = p_grid_aug, rule = 2)$y
})

## Reference QFs: pointwise mean, on the augmented grid, of the QFs of the
## children in each impact group (children in df, matched to y_list by sub_id)
Qi_aug <- do.call(rbind, encode_y_to_Qi_aug(pipeline_qpca, y_list, p_grid_aug))
Q_ref_sub <- lapply(c(`0` = 0, `1` = 1), function(g) {
  idx <- match(df$sub_id[df$imp == g], names(y_list))
  stopifnot(!anyNA(idx))
  colMeans(Qi_aug[idx, , drop = FALSE])
})

## compute_substitution_band: Delta(p0, delta) for each p0 in p_grid_aug with
## p0 >= delta, with joint (max-t) 95% bands. Returns the qpca_joint_band list
## plus p0 and Q_ref(p0), or NULL if no q_k terms are retained.
compute_substitution_band <- function(model, E_aug, w_aug, p_grid_aug, Q_ref,
                                      imp, delta, ...) {
  p0 <- p_grid_aug[p_grid_aug >= delta]
  Q_shift <- approx(p_grid_aug, Q_ref, xout = p_grid_aug + delta, rule = 2)$y
  Q_p0 <- approx(p_grid_aug, Q_ref, xout = p0, rule = 2)$y

  ## Row j: Q~ - Q_ref for p0 = p0[j], on p_grid_aug
  D <- t(vapply(seq_along(p0), function(j) {
    ifelse(p_grid_aug <= p0[j] - delta, Q_shift - Q_ref,
           ifelse(p_grid_aug <= p0[j], Q_p0[j] - Q_ref, 0))
  }, numeric(length(p_grid_aug))))

  ## Row j of L_sub maps the q_k coefficients to sum_i w_i D_ji alpha(p_i)
  L_sub <- sweep(D, 2, w_aug, '*') %*% E_aug
  band <- qpca_joint_band(model, L_sub, imp, ...)
  if (is.null(band)) return(NULL)
  c(band, list(p0 = p0, Q_p0 = Q_p0))
}

## draw_substitution_panel: plots a substitution band over the p0 (>= delta)
## in plim whose intensity Q_ref(p0) lies in qlim. With x_intensity = TRUE
## the x-axis is the substituted intensity Q* = Q_ref(p0) in MIMS (by default
## spanning qlim; p0 itself is not shown) and the usual top margin suffices.
## With FALSE the x-axis is p0, with Q* on a top axis, and a top margin of
## about 5 lines is expected. Stretches of the bottom axis where the joint
## band excludes zero are drawn in sig_col.
sub_panel_idx <- function(band, delta, plim = c(0, 1), qlim = c(-Inf, Inf)) {
  which(band$p0 >= max(plim[1], delta) & band$p0 <= min(plim[2], 1) &
          band$Q_p0 >= qlim[1] & band$Q_p0 <= qlim[2])
}
draw_substitution_panel <- function(band, delta, plim = c(0, 1),
                                    qlim = c(-Inf, Inf),
                                    xlim = NULL, ylim = NULL, main = '',
                                    ylab = 'Change in BMD Z-score',
                                    x_intensity = TRUE, sig_col = 'red') {
  idx <- sub_panel_idx(band, delta, plim, qlim)
  x <- if (x_intensity) band$Q_p0[idx] else band$p0[idx]
  lab_int <- expression(Q^'*' ~ '(MIMS)')
  lab_p0 <- expression(p[0])
  if (is.null(xlim)) {
    xlim <- if (x_intensity && all(is.finite(qlim))) qlim else range(x)
  }
  if (is.null(ylim)) ylim <- range(c(band$low[idx], band$high[idx]))
  plot(
    NULL, xlim = xlim, ylim = ylim,
    xlab = if (x_intensity) lab_int else lab_p0, ylab = ylab,
    main = if (x_intensity) main else ''
  )
  draw_band(x, band$low[idx], band$high[idx])
  abline(h = 0, lty = 'dashed', col = 'red')
  lines(x, band$est[idx], lwd = 3)

  ## Bottom axis in sig_col where the joint band excludes zero; each point
  ## covers the half-gaps to its neighbors so isolated points still show
  sig <- band$low[idx] > 0 | band$high[idx] < 0
  if (any(sig)) {
    n <- length(x)
    mids <- (x[-1] + x[-n]) / 2
    left <- c(x[1], mids)
    right <- c(mids, x[n])
    runs <- rle(sig)
    ends <- cumsum(runs$lengths)
    starts <- ends - runs$lengths + 1
    usr <- par('usr')
    for (r in which(runs$values)) {
      segments(left[starts[r]], usr[3], right[ends[r]], usr[3],
               col = sig_col, lwd = 5, lend = 'butt', xpd = NA)
    }
  }

  ## p0 on the x-axis: Q* on a top axis at the bottom-axis tick positions (a
  ## p0 = 1 tick is labeled with Q_ref at the last grid point)
  if (!x_intensity) {
    at <- axTicks(1)
    at <- at[at >= min(band$p0) & at <= 1]
    q_at <- approx(band$p0, band$Q_p0, xout = at, rule = 2)$y
    axis(3, at = at, labels = signif(q_at, 2))
    mtext(lab_int, side = 3, line = 2)
    title(main = main, line = 3.5)
  }
}

## Generate activity substitution plots
delta_sub <- 0.01     ## proportion of least active time substituted
qlims_sub <- list(`1` = c(-Inf, Inf), `2` = c(0, 100))   ## Q(p0) ranges (MIMS)
for (imp in c(0, 1)) {
  band_sub <- compute_substitution_band(
    model = model, E_aug = E_aug, w_aug = w_aug, p_grid_aug = p_grid_aug,
    Q_ref = Q_ref_sub[[as.character(imp)]], imp = imp, delta = delta_sub
  )
  if (is.null(band_sub)) {
    warning('No q_k terms retained in model; skipping substitution plots (imp = ', imp, ')')
    next
  }
  for (qlim_id in names(qlims_sub)) {
    png(str_glue('{dir_art}/plots/q-pca/substitution_qlim-{qlim_id}_imp-{imp}.png'),
        width = 960, height = 960, pointsize = 18)
    par(mar = c(4.5, 4.5, 3, 1))
    draw_substitution_panel(
      band_sub, delta = delta_sub, qlim = qlims_sub[[qlim_id]],
      main = str_glue('Activity substitution (imp = {imp}, delta = {delta_sub})')
    )
    dev.off()
  }
}


## ----- Interpretation: effect plots (Q-PCA and WAME)

## compute_effect_curves: decoded quantile curves (rows of Q) obtained by
## shifting embedding k_star across its marginal decile range (probs) while
## holding all other embedding dims and covariates at preds_star / the
## embedding median, along with the predicted response at each curve. ref_idx
## is the row for the 0.5 decile. If contrast = TRUE, the 0.5 decile curve is
## subtracted from every curve.
compute_effect_curves <- function(model, embeddings, k_star, preds_star,
                                  decode_fn, col_prefix,
                                  probs = seq(0.1, 0.9, by = 0.1),
                                  contrast = FALSE) {

  ref_idx <- which(abs(probs - 0.5) < 1e-8)
  if (length(ref_idx) != 1) stop('probs must contain 0.5')

  K <- ncol(embeddings)
  emb_median <- matrix(apply(embeddings, 2, median), nrow = 1)
  deltas <- quantile(embeddings[,k_star], probs) - emb_median[,k_star]

  ## Create df_new
  for (k in 1:K) {
    preds_star[[str_glue('{col_prefix}_{k}')]] <- emb_median[,k]
  }
  k_star_col <- str_glue('{col_prefix}_{k_star}')
  df_new <- do.call(rbind, lapply(deltas, function(delta) {
    row <- preds_star
    row[[k_star_col]] <- preds_star[1,k_star_col] + delta
    row
  }))

  ## Predict at df_new
  resp_new <- predict(model, df_new)

  ## Map embedding shifts to Q-space
  shifts <- as.matrix(df_new[,paste0(col_prefix, '_', 1:K)])
  Q_plot <- decode_fn(shifts)
  if (contrast) {
    Q_plot <- sweep(Q_plot, 2, Q_plot[ref_idx,])
  }

  list(Q = Q_plot, resp = resp_new, ref_idx = ref_idx)
}

## create_effect_plot: plots predicted response (color) against the curves
## from compute_effect_curves. The color ramp is centered (gray) at the
## response for the 0.5 decile and is symmetric about it. If contrast = TRUE,
## the 0.5 decile curve is subtracted from every curve (so the 0.5 decile
## curve becomes a horizontal line at zero).
create_effect_plot <- function(model, embeddings, k_star, plim, preds_star,
                                path, decode_fn, col_prefix, p_grid,
                                probs = seq(0.1, 0.9, by = 0.1),
                                contrast = FALSE,
                                legend_title = 'BD',
                                y_labs = c(1, 10, 100),
                                n_cols = 100,
                                width = 500, height = 400) {

  curves <- compute_effect_curves(
    model = model, embeddings = embeddings, k_star = k_star,
    preds_star = preds_star, decode_fn = decode_fn, col_prefix = col_prefix,
    probs = probs, contrast = contrast
  )
  Q_plot <- curves$Q
  resp_new <- curves$resp
  ref_idx <- curves$ref_idx
  p_idx <- which(p_grid >= plim[1] & p_grid <= plim[2])

  # Normalize response to [0,1], centered at the 0.5 decile response and
  # symmetric about it
  resp_ref <- resp_new[ref_idx]
  resp_rad <- max(abs(resp_new - resp_ref))
  resp_scaled <- if (resp_rad > 0) {
    0.5 + (resp_new - resp_ref) / (2 * resp_rad)
  } else {
    rep(0.5, length(resp_new))
  }

  # Create color ramp (odd length so the gray midpoint is an exact color)
  if (n_cols %% 2 == 0) n_cols <- n_cols + 1
  col_fun <- colorRampPalette(c("blue", "gray", "red"))
  cols <- col_fun(n_cols)

  # Map each curve to a color
  curve_cols <- cols[round(resp_scaled * (n_cols - 1)) + 1]

  # Plot first curve to initialize axes
  png(path, width = width, height = height)
  plot(
    p_grid[p_idx],
    Q_plot[1,p_idx],
    type = "l",
    col = curve_cols[1],
    lwd = 2,
    xlab = "p",
    ylab = if (contrast) "Q(p) - Q[0.5](p)" else "Q(p)",
    ylim = range(Q_plot[,p_idx]),
    yaxt = if (contrast) 's' else 'n',
    main = str_glue('{col_prefix}_{k_star}')
  )
  if (!contrast) {
    axis(side = 2, at = y_labs, labels = y_labs)  ## identity: Y-axis transform removed from pipeline
  }

  # Add remaining curves
  for (i in 2:nrow(Q_plot)) {
    lines(
      p_grid[p_idx],
      Q_plot[i,p_idx],
      col = curve_cols[i],
      lwd = 2
    )
  }


  usr <- par("usr")

  # Legend height = 25% of plot height
  legend_height <- 0.25 * diff(usr[3:4])
  legend_width  <- 0.05 * diff(usr[1:2])

  # Position in top-left with small margin
  x_left  <- usr[1] + 0.05 * diff(usr[1:2])
  x_right <- x_left + legend_width

  y_top    <- usr[4] - 0.05 * diff(usr[3:4])
  y_bottom <- y_top - legend_height

  # Sequence for rectangles
  y_seq <- seq(y_bottom, y_top, length.out = n_cols)

  # Draw gradient
  for (i in 1:(n_cols - 1)) {
    rect(
      x_left,  y_seq[i],
      x_right, y_seq[i + 1],
      col = cols[i],
      border = NA
    )
  }

  # Draw border box
  rect(x_left, y_bottom, x_right, y_top)

  # Add ticks (3 ticks: ref - rad, ref (0.5 decile), ref + rad)
  tick_vals  <- resp_ref + c(-1, 0, 1) * resp_rad
  tick_pos   <- y_bottom + c(0, 0.5, 1) * legend_height

  segments(x_right, tick_pos,
           x_right + 0.01 * diff(usr[1:2]), tick_pos)

  text(x_right + 0.015 * diff(usr[1:2]),
       tick_pos,
       labels = sprintf("%.2f", tick_vals),
       adj = 0)

  # Legend title
  text((x_left + x_right)/2,
       y_top + 0.03 * diff(usr[3:4]),
       legend_title,
       cex = 0.8)

  dev.off()
}

## Decoders: map embedding shifts back to Q-space for each embedding type
decode_qpc <- function(shifts) {
  shift_ctx <- new_context(
    payload = asplit(shifts, 1),
    cache = pipeline_qpca$training$cache,
    meta = list()
  )
  do.call(rbind, decode(pipeline_qpca, shift_ctx, from = 3, to = 2)$payload)
}

decode_c <- function(shifts) {
  c_shifts_list <- asplit(shifts, MARGIN = 1)
  c_shifts_ctx <- new_context(
    payload = list(
      c_list = c_shifts_list,
      Q_star_list = replicate(length(c_shifts_list), pipeline$training$cache$Q_star, simplify = FALSE)
    ),
    cache = pipeline$training$cache,
    meta = list()
  )
  do.call(rbind, decode(pipeline, c_shifts_ctx, from = 3, to = 2)$payload)
}

## Models and base covariates
model_qpc <- models[['dist_Qpc_imp']]$model_full
model_c <- models[['dist_C_imp']]$model_full
preds_star_base <- data.frame(
  old = c(0),
  male = c(0),
  bmiz = c(0.5),
  imp = c(1)
)

## model_embs: indices k (in 1:K) whose embedding direction {col_prefix}_k
## appears in any term of the model, as a main effect or in an interaction
model_embs <- function(model, col_prefix, K) {
  term_vars <- unlist(strsplit(attr(terms(model), 'term.labels'), ':'))
  which(paste0(col_prefix, '_', 1:K) %in% term_vars)
}

## Generate effect plots
k_qpc <- model_embs(model_qpc, 'q', ncol(Qpc))
k_c <- model_embs(model_c, 'c', ncol(C))
plims <- list(`1` = c(0, 1), `2` = c(0.9, 1))
for (plim_id in names(plims)) {
  plim <- plims[[plim_id]]
  for (imp in c(0, 1)) {
    preds_star <- preds_star_base
    preds_star$imp <- imp
    for (contrast in c(FALSE, TRUE)) {
      suffix <- if (contrast) '-contrast' else ''
      for (k_star in k_qpc) {
        create_effect_plot(
          model = model_qpc, embeddings = Qpc, k_star = k_star, plim = plim,
          preds_star = preds_star, decode_fn = decode_qpc,
          col_prefix = 'q', p_grid = p_grid, contrast = contrast,
          path = str_glue('{dir_art}/plots/q-pca/q{k_star}{suffix}_plim-{plim_id}_imp-{imp}.png')
        )
      }
      for (k_star in k_c) {
        create_effect_plot(
          model = model_c, embeddings = C, k_star = k_star, plim = plim,
          preds_star = preds_star, decode_fn = decode_c,
          col_prefix = 'c', p_grid = p_grid, contrast = contrast,
          path = str_glue('{dir_art}/plots/wame/c{k_star}{suffix}_plim-{plim_id}_imp-{imp}.png')
        )
      }
    }
  }
}


## ========== Poster Plots ========== ##

dir_poster <- file.path(dir_art, 'plots', 'poster')
dir.create(dir_poster, recursive = TRUE, showWarnings = FALSE)

## Poster resolution: plot sizes below are given in px at 72 dpi (which sets
## the layout); poster_png renders them at poster_scale x the pixel density
## (res = 72 * poster_scale), so text, lines and layout keep their proportions
poster_scale <- 4
poster_png <- function(path, width, height, pointsize = 12) {
  png(path, width = round(width * poster_scale), height = round(height * poster_scale),
      res = 72 * poster_scale, pointsize = pointsize)
}


## ----- Single QF (median, mean and MVPA rate as QF summaries)

## Subject: the QF closest (L2 on p_grid) to the pointwise median QF among
## subjects whose QF reaches the MVPA threshold
y_thresh <- 19.6
Q_med <- apply(Q, 2, median)
reaches <- apply(Q, 1, max) > y_thresh
i_qf <- which(reaches)[which.min(rowSums(sweep(Q[reaches, , drop = FALSE], 2, Q_med)^2))]
Q_i <- Q[i_qf, ]

## Q(0.5) and Q_inv(19.6) = inf{p : Q(p) >= 19.6}, linearly interpolated
Q_half <- approx(p_grid, Q_i, xout = 0.5)$y
j <- which(Q_i >= y_thresh)[1]
p_thresh <- if (j == 1) p_grid[1] else
  approx(Q_i[(j - 1):j], p_grid[(j - 1):j], xout = y_thresh)$y

## Same 400 x 300 layout at res_scale x the pixel density (res = 72 * res_scale)
res_scale <- 4
png(file.path(dir_poster, 'qf_single.png'),
    width = 400 * res_scale, height = 250 * res_scale,
    res = 72 * res_scale, pointsize = 10)
## Tight top (no title) and bottom margins; left/right at R defaults
par(mar = c(3, 4.1, 0.5, 2.1))
plot(
  p_grid, Q_i, type = 'l', lwd = 4, col = 'black',
  xlab = '', ylab = 'Q(p) (MIMS)', main = '',
  xlim = c(0, 1), ylim = c(0, 1.05 * max(Q_i)),
  xaxs = 'i', yaxs = 'i', xaxt = 'n', yaxt = 'n'
)
title(xlab = 'Quantile level p', line = 1.9)
axis(1, at = c(0, 0.5, 1), labels = c('0', '0.5', '1'))
axis(2, at = y_thresh, labels = as.character(y_thresh), las = 1)
segments(0.5, 0, 0.5, Q_half, col = 'gray50', lty = 'dotted', lwd = 2)
segments(0.5, Q_half, 0, Q_half, col = 'gray50', lty = 'dotted', lwd = 2)
segments(0, y_thresh, p_thresh, y_thresh, col = 'gray50', lty = 'dotted', lwd = 2)
segments(p_thresh, y_thresh, p_thresh, 0, col = 'gray50', lty = 'dotted', lwd = 2)
text(0.02, y_thresh, 'MVPA cutpoint (19.6 MIMS)', adj = c(0, -0.5),
     cex = 0.8, col = 'gray30')
## MVPA rate: the interval [Q_inv(19.6), 1] marked along the x-axis
segments(p_thresh, 0, 1, 0, lwd = 6, col = 'gray20', lend = 'butt', xpd = NA)
text((p_thresh + 1) / 2, 0, 'MVPA rate', adj = c(0.5, -0.8), cex = 0.8, col = 'gray20')
legend(
  'topleft',
  legend = expression(
    median == Q(0.5),
    mean == integral(Q(p) * dp, 0, 1),
    'MVPA rate' == 1 - Q^{-1} * (19.6)
  ),
  bty = 'n', y.intersp = 1.6, cex = 1.25
)
dev.off()


## ----- All QFs

## plot_funs draws onto the device opened by poster_png (no path given)
poster_png(file.path(dir_poster, 'qf_all.png'), width = 400, height = 300)
plot_funs(
  fun_list = asplit(Q, 1),
  grid_list = rep(list(p_grid), nrow(Q)),
  xlab = 'Quantile level p', ylab = 'Q(p) (MIMS)', main = '',
  colors = rep(rgb(0, 0, 0, alpha = 0.5), nrow(Q))
)
dev.off()


## ----- Embedding heatmaps (Q-PCA and WAME)

## plot_emb_heatmap: N x K heatmap of embeddings (row i = subject i, top to
## bottom), blue (-max|emb|) -> white (0) -> red (+max|emb|)
plot_emb_heatmap <- function(emb, path, n_cols = 201,
                             width = 200, height = 400, pointsize = 10) {
  N_emb <- nrow(emb)
  K_emb <- ncol(emb)
  lim <- max(abs(emb))
  cols <- colorRampPalette(c('blue', 'white', 'red'))(n_cols)
  poster_png(path, width = width, height = height, pointsize = pointsize)
  ## No tick labels, so keep margins tight and axis titles next to the image
  par(mar = c(1.6, 1.6, 0.5, 0.5), mgp = c(0.4, 0, 0))
  image(
    x = 1:K_emb, y = 1:N_emb, z = t(emb[N_emb:1, , drop = FALSE]),
    zlim = c(-lim, lim), col = cols,
    xlab = 'k', ylab = 'Subject', xaxt = 'n', yaxt = 'n'
  )
  dev.off()
}

plot_emb_heatmap(Qpc, file.path(dir_poster, 'heatmap_qpc.png'))
plot_emb_heatmap(C, file.path(dir_poster, 'heatmap_c.png'))


## ----- QFs -> heatmaps composite

## Stitches the existing qf_all (top) and heatmap (bottom left: C, bottom
## right: Q-PC) PNGs, each cropped to its drawn content (no resampling), with
## labeled arrows
## crop_white: drop the white margins of an image (any pixel darker than
## near-white counts as content), keeping pad px around the content
crop_white <- function(img, pad = 0) {
  ink <- apply(img[, , 1:3, drop = FALSE] < 0.99, c(1, 2), any)
  r <- range(which(rowSums(ink) > 0))
  c_ <- range(which(colSums(ink) > 0))
  r <- c(max(1, r[1] - pad), min(dim(img)[1], r[2] + pad))
  c_ <- c(max(1, c_[1] - pad), min(dim(img)[2], c_[2] + pad))
  img[r[1]:r[2], c_[1]:c_[2], , drop = FALSE]
}
img_qf <- crop_white(png::readPNG(file.path(dir_poster, 'qf_all.png')))
img_c <- crop_white(png::readPNG(file.path(dir_poster, 'heatmap_c.png')))
img_q <- crop_white(png::readPNG(file.path(dir_poster, 'heatmap_qpc.png')))
img_w <- function(img) dim(img)[2]
img_h <- function(img) dim(img)[1]

## Pixel sizes at 72 dpi, scaled to the source images' poster_scale density
pad <- 10 * poster_scale        ## outer border
gap_x <- 100 * poster_scale     ## horizontal gap between heatmaps
arrow_h <- 200 * poster_scale   ## vertical space for arrows
row_w <- img_w(img_c) + gap_x + img_w(img_q)
hm_h <- max(img_h(img_c), img_h(img_q))
W <- max(row_w, img_w(img_qf)) + 2 * pad
H <- img_h(img_qf) + arrow_h + hm_h + 2 * pad

## Positions (pixel coordinates, origin bottom left); heatmap tops aligned
x_qf <- (W - img_w(img_qf)) / 2
x_c <- (W - row_w) / 2
x_q <- x_c + img_w(img_c) + gap_x
y_top_hm <- pad + hm_h
y_bot_qf <- H - pad - img_h(img_qf)

png(file.path(dir_poster, 'qf_to_heatmaps.png'), width = W, height = H,
    res = 72 * poster_scale, pointsize = 20)
par(mar = c(0, 0, 0, 0))
plot.new()
plot.window(xlim = c(0, W), ylim = c(0, H), xaxs = 'i', yaxs = 'i')
rasterImage(img_qf, x_qf, y_bot_qf, x_qf + img_w(img_qf), H - pad, interpolate = FALSE)
rasterImage(img_c, x_c, y_top_hm - img_h(img_c), x_c + img_w(img_c), y_top_hm,
            interpolate = FALSE)
rasterImage(img_q, x_q, y_top_hm - img_h(img_q), x_q + img_w(img_q), y_top_hm,
            interpolate = FALSE)

## Arrows from just below qf_all to just above the top center of each heatmap
x_from <- W / 2 + c(-1, 1) * 0.15 * img_w(img_qf)
x_to <- c(x_c + img_w(img_c) / 2, x_q + img_w(img_q) / 2)
y_from <- y_bot_qf - 15 * poster_scale
y_to <- y_top_hm + 15 * poster_scale
arrows(x_from, y_from, x_to, y_to, lwd = 3, length = 0.2)
text((x_from + x_to) / 2, (y_from + y_to) / 2, labels = c('WAME', 'Q-PCA'),
     pos = c(2, 4), offset = 0.8, font = 1.5)
dev.off()


## ----- Out-of-sample error table (LaTeX)

## One row per PA summary. The two change columns give the paired change in
## CV MAE relative to the covariates-only model ('None'): for the full models
## (each model's full formula) and for the pruned models (prune_pval_all,
## rerun inside each training fold, compared with the pruned None model).
## Values are mean (95% CI) over children of |error_model| - |error_None| on
## the shared folds; negative = lower error than None. The None row shows its
## MAE instead. In each change column the most negative change is bold. A
## suggested caption is written as comments at the top of the .tex file. Edit
## table_spec to change rows. Requires \usepackage{booktabs}.
table_spec <- data.frame(
  model = c('ref_common_preds_imp', 'scale_imp_0', 'scale_imp_1',
            'scale_imp_8', 'dist_C_imp', 'dist_Qpc_imp'),
  label = c('None (Baseline)', 'Mean (\\texttt{a\\_mean})', 'MVPA (\\texttt{a\\_mvpa})', '99th pct. (\\texttt{a\\_99})', 'WAME (\\texttt{c})', 'Q-PCA (\\texttt{q})'),
  type = c('--', 'Scalar', 'Scalar', 'Scalar',
           'Distributional', 'Distributional')
)
stopifnot(all(table_spec$model %in% names(models)), fit_reduced)

table_vals <- do.call(rbind, lapply(seq_len(nrow(table_spec)), function(r) {
  abbr <- models[[table_spec$model[r]]]$f_str_abbr
  full <- results[results$preds == abbr & !results$reduced, ]
  pr <- results[results$preds == abbr & results$reduced, ]
  stopifnot(nrow(full) == 1, nrow(pr) == 1)
  data.frame(full_mae = full$oos_err_mean, pruned_mae = pr$oos_err_mean,
             full_d = full$delta_mean, full_lo = full$delta_low,
             full_hi = full$delta_high,
             pruned_d = pr$delta_mean, pruned_lo = pr$delta_low,
             pruned_hi = pr$delta_high)
}))
is_pa_row <- table_spec$type != '--'

## Cell text: change (95% CI), 3 decimals, true minus signs; bold = most
## negative change among PA rows; the None row shows its MAE as the
## reference
num <- function(x) sub('^-', '$-$', sprintf('%.3f', x))
fmt_col <- function(d, lo, hi, mae) {
  txt <- num(d)
  best <- is_pa_row & d == min(d[is_pa_row])
  txt[best] <- paste0('\\textbf{', txt[best], '}')
  txt <- paste0(txt, ' (', num(lo), ', ', num(hi), ')')
  txt[!is_pa_row] <- paste0('Baseline (MAE = ', sprintf('%.3f', mae[!is_pa_row]), ')')
  txt
}
full_txt <- fmt_col(table_vals$full_d, table_vals$full_lo, table_vals$full_hi,
                    table_vals$full_mae)
pruned_txt <- fmt_col(table_vals$pruned_d, table_vals$pruned_lo,
                      table_vals$pruned_hi, table_vals$pruned_mae)

body <- paste0(
  paste(table_spec$label, table_spec$type, full_txt, pruned_txt, sep = ' & '),
  ' \\\\'
)

## Suggested caption (written as LaTeX comments above the tabular)
cv_desc <- if (V_cv == nrow(df)) 'leave-one-out' else paste0(V_cv, '-fold')
table_caption <- strwrap(paste0(
  'Out-of-sample prediction of total-body BMD Z-score. The baseline model ',
  'includes age group, sex, BMI Z-score and high-impact activity; each other ',
  'row adds the listed physical-activity (PA) summary and its interaction ',
  'with high-impact activity. $\\Delta$MAE is the paired difference in ',
  cv_desc, ' cross-validated mean absolute error from the baseline, averaged ',
  'over children (negative = lower error than the baseline), with 95\\% CI. ',
  'Full: all terms retained. Pruned: backward selection ($p < 0.05$) ',
  'repeated within each cross-validation fit, applied to the baseline as ',
  'well. Bold: largest reduction in each column.'
), width = 76)

table_tex <- c(
  '% Suggested caption:',
  paste('%', table_caption),
  '\\begin{tabular}{llcc}',
  '\\toprule',
  paste0('PA summary & Type & ',
         '$\\Delta$ Full MAE vs.\\ baseline (95\\% CI) & ',
         '$\\Delta$ Pruned MAE vs.\\ baseline (95\\% CI) \\\\'),
  '\\midrule',
  body,
  '\\bottomrule',
  '\\end{tabular}'
)
writeLines(table_tex, file.path(dir_poster, 'oos_table.tex'))
cat(table_tex, sep = '\n')


## ----- Coefficient intervals (Q-PCA and WAME)

## Orientation: FALSE = horizontal intervals stacked vertically in each plot,
## plots stitched side by side; TRUE = vertical intervals laid out left to
## right in each plot, plots stitched top to bottom
coef_horizontal <- TRUE

## Plotting-area height (px) in the stacked layout. The in-box p-value legend
## sits in a band reserved above the highest interval, so it cannot overlap
## any CI at any height (binary_plot_intervals stops if the plot is too short
## for the band)
plot_height_coef <- 130

## plot_coef_intervals: (unadjusted) 95% t confidence intervals and p-values
## for all non-intercept coefficients of an lm fit, drawn in the style of the
## Roy-Bose interval plots (binary_plot_intervals). plot_width (and, when
## horizontal, plot_height) give the size (px) of the plotting area itself;
## the PNG size adds whatever the margins need (coefficient labels, p-value
## legend), so plots with different labels/legends share the same plotting
## area. When not horizontal, the PNG height is fixed at height.
plot_coef_intervals <- function(model, path, main, show_legend = TRUE,
                                horizontal = FALSE,
                                plot_width = if (horizontal) 450 else 150,
                                plot_height = 200,
                                height = 600, pointsize = 14) {
  ci <- confint(model, level = 0.95)
  p_vals <- summary(model)$coefficients[, 'Pr(>|t|)']
  keep <- rownames(ci) != '(Intercept)'
  ci_df <- data.frame(
    estimate = coef(model)[keep],
    lower = ci[keep, 1],
    upper = ci[keep, 2],
    p_adj = p_vals[rownames(ci)[keep]]
  )
  ## Measure margins (in lines) on a scratch PNG device with the same font
  ## metrics: the label margin (left, or bottom when horizontal) just wide
  ## enough for the longest coefficient label, and room on the right for the
  ## p-value legend (if shown; when horizontal it sits inside the plot box)
  tmp <- tempfile(fileext = '.png')
  png(tmp, width = 100, height = height, pointsize = pointsize)
  csi <- par('csi')
  mar_lab <- max(strwidth(rownames(ci_df), units = 'inches')) / csi + 1.5
  mar_right <- if (show_legend && !horizontal) max(1, binary_legend_margin_in() / csi) else 1
  dev.off()
  unlink(tmp)

  ## PNG size = margins + plotting area, in px at 72 dpi (poster_png scales)
  if (horizontal) {
    mar <- c(mar_lab, 4.5, 3, mar_right)
    height <- plot_height + (mar[1] + mar[3]) * csi * 72
  } else {
    mar <- c(4.5, mar_lab, 3, mar_right)
  }
  width <- plot_width + (mar[2] + mar[4]) * csi * 72
  poster_png(path, width = width, height = height, pointsize = pointsize)
  binary_plot_intervals(
    ci_df,
    main = main,
    labels = rownames(ci_df),
    xlab = if (horizontal) '' else 'Estimate (95% CI)',
    ylab = if (horizontal) 'Estimate (95% CI)' else '',
    p_lab = 'p',
    mar = mar,
    legend_in_margin = TRUE,
    show_legend = show_legend,
    horizontal = horizontal
  )
  dev.off()
}

## The p-value legend goes on one plot only: Q-PCA (right) when side by side,
## WAME (top) when stacked
plot_coef_intervals(
  models[['dist_Qpc_imp']]$model_full,
  path = file.path(dir_poster, 'coefs_qpca.png'),
  main = 'Q-PCA Model',
  show_legend = TRUE,
  horizontal = coef_horizontal,
  plot_height = plot_height_coef
)
plot_coef_intervals(
  models[['dist_C_imp']]$model_full,
  path = file.path(dir_poster, 'coefs_wame.png'),
  main = 'WAME Model',
  show_legend = TRUE,
  horizontal = coef_horizontal,
  plot_height = plot_height_coef
)

## Stitch at native pixel sizes: side by side (WAME left, Q-PCA right, so the
## Q-PCA legend sits at the right edge; tops aligned) or, when horizontal,
## stacked (WAME top, Q-PCA bottom; left edges aligned)
img_coef_q <- png::readPNG(file.path(dir_poster, 'coefs_qpca.png'))
img_coef_c <- png::readPNG(file.path(dir_poster, 'coefs_wame.png'))
gap_coef <- 20 * poster_scale   ## gap between plots (px at 72 dpi, scaled)
w_c <- dim(img_coef_c)[2]; h_c <- dim(img_coef_c)[1]
w_q <- dim(img_coef_q)[2]; h_q <- dim(img_coef_q)[1]
if (coef_horizontal) {
  W_coef <- max(w_c, w_q)
  H_coef <- h_c + gap_coef + h_q
  pos_c <- c(0, H_coef - h_c)   ## (left, bottom)
  pos_q <- c(0, 0)
} else {
  W_coef <- w_c + gap_coef + w_q
  H_coef <- max(h_c, h_q)
  pos_c <- c(0, H_coef - h_c)
  pos_q <- c(w_c + gap_coef, H_coef - h_q)
}

png(file.path(dir_poster, 'coefs_combined.png'), width = W_coef, height = H_coef,
    res = 72 * poster_scale)
par(mar = c(0, 0, 0, 0))
plot.new()
plot.window(xlim = c(0, W_coef), ylim = c(0, H_coef), xaxs = 'i', yaxs = 'i')
rasterImage(img_coef_c, pos_c[1], pos_c[2], pos_c[1] + w_c, pos_c[2] + h_c,
            interpolate = FALSE)
rasterImage(img_coef_q, pos_q[1], pos_q[2], pos_q[1] + w_q, pos_q[2] + h_q,
            interpolate = FALSE)
dev.off()


## ----- Effect contrasts (WAME and Q-PCA, by impact)

## plot_effect_contrasts: 2 x 2 effect contrast panels (rows: the two
## components in k_stars; columns: low / high impact) plus a right-margin
## panel with the shared color ramp beside a boxplot of observed BMD, on a
## common BMD axis. comp_label prefixes the panel titles (e.g. 'C' -> 'C1').
plot_effect_contrasts <- function(model, embeddings, decode_fn, col_prefix,
                                  k_stars, comp_label, path,
                                  plim = c(0, 1), n_cols = 101, gamma = 1.5) {

  ## Panels in layout order (top left, top right, bottom left, bottom right)
  panels <- list()
  for (k_star in k_stars) {
    for (imp in c(0, 1)) {
      imp_lab <- if (imp == 0) 'low impact' else 'high impact'
      panels[[length(panels) + 1]] <- list(
        k_star = k_star, imp = imp,
        main = str_glue('{comp_label}{k_star}, {imp_lab}')
      )
    }
  }
  p_idx <- which(p_grid >= plim[1] & p_grid <= plim[2])
  for (i in seq_along(panels)) {
    preds_star <- preds_star_base
    preds_star$imp <- panels[[i]]$imp
    panels[[i]]$curves <- compute_effect_curves(
      model = model, embeddings = embeddings, k_star = panels[[i]]$k_star,
      preds_star = preds_star, decode_fn = decode_fn, col_prefix = col_prefix,
      contrast = TRUE
    )
  }

  ## Shared color scale: blue (min) -> gray (midpoint) -> red (max) over the
  ## predicted BMD of all curves in all panels; shared y-limits across panels.
  ## gamma > 1 widens the gray band (symmetric about the midpoint); 1 = linear
  resp_all <- unlist(lapply(panels, function(pan) pan$curves$resp))
  resp_rng <- range(resp_all)
  cols <- colorRampPalette(c('blue', 'gray', 'red'))(n_cols)
  resp_to_col <- function(resp) {
    scaled <- if (diff(resp_rng) > 0) (resp - resp_rng[1]) / diff(resp_rng) else 0.5
    u <- 2 * scaled - 1                             ## -1 (min) .. 0 (mid) .. 1 (max)
    scaled <- 0.5 + 0.5 * sign(u) * abs(u)^gamma
    cols[round(scaled * (n_cols - 1)) + 1]
  }
  ylim_panels <- range(unlist(lapply(panels, function(pan) pan$curves$Q[, p_idx])))

  poster_png(path, width = 900, height = 600, pointsize = 20)
  layout(matrix(c(1, 2, 5, 3, 4, 5), nrow = 2, byrow = TRUE),
         widths = c(1, 1, 0.5))

  ## Axis titles only on the outer panels: y on the left column, x on the
  ## bottom row. The most extreme decile curves (first and last) are bold.
  par(mar = c(4.5, 4.5, 3, 1))
  for (i in seq_along(panels)) {
    pan <- panels[[i]]
    Q_pan <- pan$curves$Q
    curve_cols <- resp_to_col(pan$curves$resp)
    plot(
      NULL,
      xlim = range(p_grid[p_idx]), ylim = ylim_panels,
      xlab = if (i > 2) 'Quantile level p' else '',
      ylab = if (i %% 2 == 1) expression(Q(p) - Q[0.5](p) ~ '(MIMS)') else '',
      main = pan$main
    )
    n_curves <- nrow(Q_pan)
    for (r in seq_len(n_curves)) {
      lines(p_grid[p_idx], Q_pan[r, p_idx], col = curve_cols[r],
            lwd = if (r %in% c(1, n_curves)) 3 else 1.5)
    }
  }

  ## Right margin: shared color ramp (left) beside boxplot of observed BMD
  ## (right), on a common BMD axis
  bmd <- df[[response]]
  par(mar = c(4.5, 4.5, 3, 1))
  plot(
    NULL,
    xlim = c(0.3, 2.7), ylim = range(c(bmd, resp_rng)),
    xaxt = 'n', xlab = '', ylab = 'Predicted / observed BMD Z-score', main = ''
  )
  y_seq <- seq(resp_rng[1], resp_rng[2], length.out = n_cols + 1)
  y_mid <- (y_seq[-1] + y_seq[-(n_cols + 1)]) / 2
  rect(0.6, y_seq[-(n_cols + 1)], 1.2, y_seq[-1], col = resp_to_col(y_mid), border = NA)
  rect(0.6, resp_rng[1], 1.2, resp_rng[2])
  boxplot(bmd, at = 2, add = TRUE, axes = FALSE, boxwex = 0.9)
  ## mtext (unlike axis) never drops labels in a narrow panel
  mtext(c('Pred', 'Obs'), side = 1, at = c(0.9, 2), line = 1, cex = 0.7)
  dev.off()
}

k_stars_c <- c(1, 3)
plot_effect_contrasts(
  model = model_c, embeddings = C, decode_fn = decode_c, col_prefix = 'c',
  k_stars = k_stars_c, comp_label = 'C',
  path = file.path(dir_poster, str_glue('effect_contrasts_c{k_stars_c[1]}_c{k_stars_c[2]}.png'))
)

k_stars_q <- c(1, 3)
plot_effect_contrasts(
  model = model_qpc, embeddings = Qpc, decode_fn = decode_qpc, col_prefix = 'q',
  k_stars = k_stars_q, comp_label = 'Q',
  path = file.path(dir_poster, str_glue('effect_contrasts_q{k_stars_q[1]}_q{k_stars_q[2]}.png'))
)


## ----- Tail shifts (Q-PCA, by impact)

delta_poster <- 10         ## MIMS
plim_tail <- c(0, 1)       ## p0 range shown (e.g. c(0.8, 1) for the right tail)
p_idx_tail <- which(p_grid >= plim_tail[1] & p_grid <= plim_tail[2])
tail_panels <- list(
  list(imp = 0, main = 'Low impact'),
  list(imp = 1, main = 'High impact')
)
for (i in seq_along(tail_panels)) {
  tail_panels[[i]]$band <- compute_tail_shift_band(
    model = models[['dist_Qpc_imp']]$model_full, L = L, w = w_qpca,
    imp = tail_panels[[i]]$imp, delta = delta_poster
  )
}
stopifnot(!any(sapply(tail_panels, function(pan) is.null(pan$band))))

## Shared y-limits across panels (over the plotted p0 range)
ylim_tail <- range(unlist(lapply(tail_panels, function(pan) {
  c(pan$band$low[p_idx_tail], pan$band$high[p_idx_tail])
})))

## Same canvas size, pointsize and panel margins as effect_contrasts_c1_c3.png
poster_png(file.path(dir_poster, 'tail_shift.png'), width = 600, height = 300, pointsize = 12)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))
for (pan in tail_panels) {
  plot(
    NULL,
    xlim = range(p_grid[p_idx_tail]), ylim = ylim_tail,
    xlab = expression(p[0]),
    ylab = str_glue('Change in BMD (Q + {delta_poster} above p0)'),
    main = pan$main
  )
  draw_band(p_grid[p_idx_tail], pan$band$low[p_idx_tail], pan$band$high[p_idx_tail])
  abline(h = 0, lty = 'dashed', col = 'red')
  lines(p_grid[p_idx_tail], pan$band$est[p_idx_tail], lwd = 3)
}
dev.off()


## ----- Activity substitution curves (Q-PCA, by impact)

delta_sub_poster <- 0.01   ## proportion of least active time substituted
qlim_sub_poster <- c(0, 90)    ## intensity Q(p0) range shown (MIMS)
x_intensity_poster <- TRUE      ## FALSE: p0 on the bottom axis, Q* on top
sub_panels <- list(
  list(imp = 0, main = 'Low impact'),
  list(imp = 1, main = 'High impact')
)
for (i in seq_along(sub_panels)) {
  sub_panels[[i]]$band <- compute_substitution_band(
    model = models[['dist_Qpc_imp']]$model_full, E_aug = E_aug, w_aug = w_aug,
    p_grid_aug = p_grid_aug, Q_ref = Q_ref_sub[[as.character(sub_panels[[i]]$imp)]],
    imp = sub_panels[[i]]$imp, delta = delta_sub_poster
  )
}
stopifnot(!any(sapply(sub_panels, function(pan) is.null(pan$band))))

## Shared y-limits across panels (over the plotted intensity range); with
## intensity on the x-axis, both panels span qlim_sub_poster
sub_idx <- lapply(sub_panels, function(pan) {
  sub_panel_idx(pan$band, delta_sub_poster, qlim = qlim_sub_poster)
})
ylim_sub <- range(unlist(Map(function(pan, idx) {
  c(pan$band$low[idx], pan$band$high[idx])
}, sub_panels, sub_idx)))

## Same canvas size and pointsize as tail_shift.png (taller top margin only
## when p0 is on the x-axis, for the Q* top axis)
poster_png(file.path(dir_poster, 'activity_substitution.png'), width = 600, height = 300, pointsize = 12)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, if (x_intensity_poster) 3 else 5, 1))
for (pan in sub_panels) {
  draw_substitution_panel(
    pan$band, delta = delta_sub_poster, qlim = qlim_sub_poster,
    ylim = ylim_sub, main = pan$main,
    ylab = 'Change in BMD Z-score',
    x_intensity = x_intensity_poster
  )
  ## delta as a small in-panel annotation
  legend('topleft', legend = as.expression(bquote(delta == .(delta_sub_poster))),
         bty = 'n', cex = 0.8)
}
dev.off()


##########################################################################
##########################################################################
######################### OLD CODE #######################################
##########################################################################
##########################################################################



## ---------- Interpretation: Q-G PCA (based on Local Linear Smoothing with Gaussian Kernel)

## Model summary
model <- models[['dist_C_imp']]$model_red
summary(model)

## Set k_star, target c_k* values, and Gaussian kernel bandwidth
## (Silverman's rule of thumb: h = 1.06 * sd(x) * n^(-1/5))
k_star      <- 6
k_star_col  <- str_glue('c_{k_star}')
quant_levels <- seq(0.1, 0.9, by = 0.1)
c_k_stars   <- quantile(df[[k_star_col]], probs = quant_levels)
bandwidth   <- 1.06 * sd(df[[k_star_col]]) * nrow(df)^(-1/5)
p_idx       <- 1:length(p_grid)

## Fixed covariates (same hardcoded values as before)
covariates_star <- data.frame(
  old           = 0,
  male          = 0,
  bmiz          = 0.5,
  imp           = 1
)

## Build df_new: one row per c_k* via Nadaraya-Watson kernel-weighted
## means with a Gaussian kernel:
##   w_i = exp( -(c_{i,k_star} - c_k*)^2 / (2 * h^2) )
##   c_bar_{k'} = sum(w_i * c_{i,k'}) / sum(w_i)
c_cols <- paste0('c_', 1:K)
C_mat  <- as.matrix(df[, c_cols, drop = FALSE])
df_new <- data.frame()
for (c_k_star in c_k_stars) {
  w <- exp(-(df[[k_star_col]] - c_k_star)^2 / (2 * bandwidth^2))
  c_means             <- colSums(w * C_mat) / sum(w)
  c_means[k_star_col] <- c_k_star
  row <- cbind(covariates_star, as.list(c_means))
  df_new <- rbind(df_new, row)
}

####################
# col1 <- 'c_5'
# col2 <- 'c_8'
# plot(df[[col1]], df[[col2]])
# for (idx in 1:length(c_k_stars)) {
#   points(df_new[idx,col1], df_new[idx,col2], pch = 19, col = 'red')
# }
####################

## Predict at df_new
resp_new <- predict(model, df_new)

## Map c-shifts to Q-space
c_shifts_list <- asplit(df_new[,paste0('c_', 1:K)], MARGIN = 1)
c_shifts_ctx <- new_context(
  payload = list(
    c_list = c_shifts_list,
    Q_star_list = replicate(length(c_shifts_list), pipeline$training$cache$Q_star, simplify = FALSE)
  ),
  cache = pipeline$training$cache,
  meta = list()
)
out <- decode(pipeline, c_shifts_ctx, from = 3, to = 2)
Q_plot <- do.call(rbind, out$payload)

# Normalize response to [0,1]
resp_scaled <- (resp_new - min(resp_new)) /
  (max(resp_new) - min(resp_new))

# Create color ramp
col_fun <- colorRampPalette(c("blue", "gray", "red"))
n_cols <- 100
cols <- col_fun(n_cols)

# Map each curve to a color
curve_cols <- cols[ceiling(resp_scaled * (n_cols - 1)) + 1]

# Plot first curve to initialize axes
png(str_glue('artifacts/analyze_chop-mims-subs/plots/c{k_star}_bd-all_imp-2_pa-c.png'), width = 500, height = 400)
plot(
  p_grid[p_idx],
  Q_plot[1,p_idx],
  type = "l",
  col = curve_cols[1],
  lwd = 1,
  xlab = "p",
  ylab = "Q(p)",
  ylim = range(Q_plot),
  yaxt = 'n',
  main = str_glue('c_{k_star}')
)
y_labs <- c(1, 10, 100)
y_ticks <- y_labs  ## identity: Y-axis transform removed from pipeline
axis(side = 2, at = y_ticks, labels = y_labs)

# Add remaining curves
for (i in 2:nrow(Q_plot)) {
  lines(
    p_grid[p_idx],
    Q_plot[i,p_idx],
    col = curve_cols[i],
    lwd = 2
  )
}


usr <- par("usr")

# Legend height = 25% of plot height
legend_height <- 0.25 * diff(usr[3:4])
legend_width  <- 0.05 * diff(usr[1:2])

# Position in top-left with small margin
x_left  <- usr[1] + 0.05 * diff(usr[1:2])
x_right <- x_left + legend_width

y_top    <- usr[4] - 0.05 * diff(usr[3:4])
y_bottom <- y_top - legend_height

# Sequence for rectangles
y_seq <- seq(y_bottom, y_top, length.out = n_cols)

# Draw gradient
for (i in 1:(n_cols - 1)) {
  rect(
    x_left,  y_seq[i],
    x_right, y_seq[i + 1],
    col = cols[i],
    border = NA
  )
}

# Draw border box
rect(x_left, y_bottom, x_right, y_top)

# Add ticks (3 ticks: min, mid, max)
resp_min <- min(resp_new)
resp_max <- max(resp_new)
resp_mid <- (resp_min + resp_max) / 2

tick_vals  <- c(resp_min, resp_mid, resp_max)
tick_pos   <- y_bottom + 
  (tick_vals - resp_min) /
  (resp_max - resp_min) *
  legend_height

segments(x_right, tick_pos,
         x_right + 0.01 * diff(usr[1:2]), tick_pos)

text(x_right + 0.015 * diff(usr[1:2]),
     tick_pos,
     labels = sprintf("%.2f", tick_vals),
     adj = 0)

# Legend title
text((x_left + x_right)/2,
     y_top + 0.03 * diff(usr[3:4]),
     "BD",
     cex = 0.8)

dev.off()



## ---------- Interpretation: Q-G PCA ("Biomarker" vs. BD)

## Model summary
model <- models[['dist_C_imp']]$model_red
summary(model)

k_stars <- c(4, 5, 6)
plot_list <- list()
for (k_star in k_stars) {

  ## Set k_star, target c_k* values, and Gaussian kernel bandwidth
  ## (Silverman's rule of thumb: h = 1.06 * sd(x) * n^(-1/5))
  k_star_col  <- str_glue('c_{k_star}')
  quant_levels <- seq(0.1, 0.9, by = 0.1)
  c_k_stars   <- quantile(df[[k_star_col]], probs = quant_levels)
  bandwidth   <- 1.06 * sd(df[[k_star_col]]) * nrow(df)^(-1/5)
  p_idx       <- 1:length(p_grid)

  ## Fixed covariates (same hardcoded values as before)
  covariates_star <- data.frame(
    old           = 0,
    male          = 0,
    bmiz          = 0.5,
    imp           = 1
  )

  ## Build df_new: one row per c_k* via Nadaraya-Watson kernel-weighted
  ## means with a Gaussian kernel:
  ##   w_i = exp( -(c_{i,k_star} - c_k*)^2 / (2 * h^2) )
  ##   c_bar_{k'} = sum(w_i * c_{i,k'}) / sum(w_i)
  c_cols <- paste0('c_', 1:K)
  C_mat  <- as.matrix(df[, c_cols, drop = FALSE])
  df_new <- data.frame()
  for (c_k_star in c_k_stars) {
    w <- exp(-(df[[k_star_col]] - c_k_star)^2 / (2 * bandwidth^2))
    c_means             <- colSums(w * C_mat) / sum(w)
    c_means[k_star_col] <- c_k_star
    row <- cbind(covariates_star, as.list(c_means))
    df_new <- rbind(df_new, row)
  }

  ####################
  # col1 <- 'c_5'
  # col2 <- 'c_8'
  # plot(df[[col1]], df[[col2]])
  # for (idx in 1:length(c_k_stars)) {
  #   points(df_new[idx,col1], df_new[idx,col2], pch = 19, col = 'red')
  # }
  ####################

  ## Predict at df_new
  resp_new <- predict(model, df_new)

  plot_list[[as.character(k_star)]] <- list(
    resp = resp_new,
    pred = df_new[[str_glue('c_{k_star}')]]
  )

}


k_star <- 6
png(str_glue('artifacts/analyze_chop-mims-subs/plots/c{k_star}_bd-all_imp-2_pa-c_biomarker.png'), width = 500, height = 400)
plot(
  plot_list[[as.character(k_star)]]$pred,
  plot_list[[as.character(k_star)]]$resp,
  xlab = str_glue("c_{k_star}"),
  ylab = "BD",
  type = 'l'
)
points(
  plot_list[[as.character(k_star)]]$pred,
  plot_list[[as.character(k_star)]]$resp,
  pch = 19
)
dev.off()

print('STOP!')

## NOTES
##  - Sleep is significant without BMI, but insignificant with BMI.
##     * This does not mean sleep has now effect on BD. Can you devise
##       a mediation analysis to tease out the small effect?
##  - Scalar right-tail PA summaries are significant. 
##     * PA tells us something beyond BMI
##     * Idea: Right-tail PA is high impact which leads to higher BD.
##     * Message: Right-tail PA as proxy for high-impact activity.
##  - Full distributional predictor models have slightly higher OOS error
##    than models with a single right-tail PA summary. 
##     * Q: Do we achieve lower OOS error using a reduced distributional
##       predictor model?


## IDEAS
##  - Mediation analysis to tease out effect of sleep.
##     * Know that high levels of right-tail PA correlate with BD
##     * Mediation questions
##        - What is the direct effect of sleep on BD
##        - What is the indirect effect of sleep on BD through PA
##     * Start with scalar mediation analysis as proof of concept. 







##########################################################################
##########################################################################
######################### MEDIATION ANALYSES #############################
##########################################################################
##########################################################################


## ========== Mediation Analysis Iterations ========== ##

## Variables
##  IV: PA
##  DV: BD
##  ME: Sleep

library(mediation)

pa_vars <- c(
  'mvpa_rate',
  'a_mean', 
  # 'a_10',
  # 'a_20',
  'a_30',
  'a_40',
  'a_50',
  'a_60',
  'a_70',
  'a_80',
  'a_90',
  'a_95',
  'a_99',
  'a_995',
  'a_999'
)

bd_vars <- c(
  'subtot_bmd_age_z_all',
  'hip_neck_bmd_age_z_all',
  'spine_bmd_age_z_all',
  'spine_bmad_age_z_all',
  'tot_hip_bmd_age_z_all',
  'radius13_bmd_age_z_all',
  'udradius_bmd_age_z_all'
)

meds_out <- list()
for (bd_var in bd_vars) {
  meds_out[[bd_var]] <- list()
  for (pa_var in pa_vars) {
    print(str_glue("========== BD = {bd_var} | PA = {pa_var} =========="))

    ## Step 1: Model for the mediator (Sleep ~ PA)
    f_str_m <- str_glue('sleep_dur ~ {pa_var} + bmiz')
    model_m <- lm(f_str_m, data = df)

    ## Step 2: Model for the outcome (BD ~ Sleep + PA)
    f_str_y <- str_glue('{bd_var} ~ sleep_dur + {pa_var} + bmiz')
    model_y <- lm(f_str_y, data = df)

    ## Step 3: Run mediation analysis
    med_out <- mediate(
      model_m,          # mediator model
      model_y,          # outcome model
      treat = pa_var,   # independent variable
      mediator = 'sleep_dur',    # mediator
      boot = TRUE,               # use bootstrapping
      sims = 1000                # number of bootstrap samples
    )
    meds_out[[bd_var]][[pa_var]] <- med_out
    print(summary(med_out))

  }
}


## ========== Mediation Analysis Iterations ========== ##

## Variables
##  IV: Sleep
##  DV: BD
##  ME: PA

library(mediation)

pa_vars <- c(
  'mvpa_rate',
  'a_mean', 
  'a_10',
  'a_20',
  'a_30',
  'a_40',
  'a_50',
  'a_60',
  'a_70',
  'a_80',
  'a_90',
  'a_95',
  'a_99',
  'a_995',
  'a_999'
)

bd_vars <- c(
  'subtot_bmd_age_z_all',
  'hip_neck_bmd_age_z_all',
  'spine_bmd_age_z_all',
  'spine_bmad_age_z_all',
  'tot_hip_bmd_age_z_all',
  'radius13_bmd_age_z_all',
  'udradius_bmd_age_z_all'
)

meds_out <- list()
for (bd_var in bd_vars) {
  meds_out[[bd_var]] <- list()
  for (pa_var in pa_vars) {
    print(str_glue("========== BD = {bd_var} | PA = {pa_var} =========="))

    ## Step 1: Model for the mediator (PA ~ Sleep)
    f_str_m <- str_glue('{pa_var} ~ sleep_dur + bmiz')
    model_m <- lm(f_str_m, data = df)

    ## Step 2: Model for the outcome (BD ~ Sleep + PA)
    f_str_y <- str_glue('{bd_var} ~ sleep_dur + {pa_var} + bmiz')
    model_y <- lm(f_str_y, data = df)

    ## Step 3: Run mediation analysis
    med_out <- mediate(
      model_m,          # mediator model
      model_y,          # outcome model
      treat = "sleep_dur",  # independent variable
      mediator = pa_var,    # mediator
      boot = TRUE,          # use bootstrapping
      sims = 1000           # number of bootstrap samples
    )
    meds_out[[bd_var]][[pa_var]] <- med_out
    print(summary(med_out))

  }
}





## ========== Mediation Analysis Walkthrough ========== ##

## Setup
##  - IV: Sleep
##  - DV: BD

## Sleep weakly negatively associated with BD
##  --> More sleep associated with lower BD
m <- lm('subtot_bmd_age_z_all ~ sleep_dur', data = df)
summary(m)

## BMI strongly positively associated with BD 
##  --> Heavier kids have high BD
m <- lm('subtot_bmd_age_z_all ~ bmiz', data = df)
summary(m)

## BMI moderately negatively associated with BD
##  --> Heaver kids sleep less
m <- lm('sleep_dur ~ bmiz', data = df)
summary(m)

## NOTE: BMI associated with Sleep and BD --> confounder

## No Sleep effect when BMI included as confounder (Step 1 in B+K)
m <- lm('subtot_bmd_age_z_all ~ sleep_dur + bmiz', data = df)
summary(m)

## But PA (controlling for BMI) moderately positively associated with BD 
##  --> Kids with more intense activity in their tails (maybe higher impact?) have high BD
m <- lm('subtot_bmd_age_z_all ~ a_90 + bmiz', data = df)
summary(m)

## And Sleep is weakly positively associated with PA
##  --> More sleep means higher levels of peak PA
m <- lm('a_90 ~ sleep_dur', data = df)
summary(m)

## Even when controlling for BMI
m <- lm('a_90 ~ sleep_dur + bmiz', data = df)
summary(m)

## Q: Even though sleep is not a significant predictor of BD when controlling for BMI, 
##    could it have an indirect effect through PA?

## ----- Mediation Analysis
library(mediation)

## Step 1: Model for the mediator (PA ~ Sleep)
model.m <- lm(a_90 ~ sleep_dur + bmiz, data = df)

## Step 2: Model for the outcome (BD ~ Sleep + PA)
model.y <- lm(subtot_bmd_age_z_all ~ sleep_dur + a_90 + bmiz, data = df)

## Step 3: Run mediation analysis
med.out <- mediate(
  model.m,          # mediator model
  model.y,          # outcome model
  treat = "sleep_dur",  # independent variable
  mediator = "a_90",    # mediator
  boot = TRUE,          # use bootstrapping
  sims = 1000           # number of bootstrap samples
)
summary(med.out)

## NOTE: There is a really small indirect effect of sleep through PA on BD
##  - Can this improve with day-level PA data?
##  - Can this improve with distributional mediator?


## ========== Mediation Analysis 1 ========== ##

## Model via Baron and Kenny: 
##     BD ~ S (intervention) + PA (mediator) + BMI (confounder)

## -----------------------------------------------
##              BD Variable      S P-val   S Sig   
## -----------------------------------------------
##     subtot_bmd_age_z_all     0.05337        .        
##   hip_neck_bmd_age_z_all     0.0428         *        
##      spine_bmd_age_z_all     0.1372                
##     spine_bmad_age_z_all     0.2505                
##    tot_hip_bmd_age_z_all     0.01862        *        
##   radius13_bmd_age_z_all     0.000656     ***        
##   udradius_bmd_age_z_all     0.0281         *
m1 <- lm('subtot_bmd_age_z_all ~ sleep_dur + bmiz', data = df)
summary(m1)
summary(m1)


## ---------------------------------------
##              PA Variable        S Sig 
## ---------------------------------------
##                mvpa_rate            .
##                   a_mean            .
##                     a_50                    
##                     a_60            .         
##                     a_70            .            
##                     a_80            .      
##                     a_90            .  
##                     a_95            .     
##                     a_99
##                    a_995
##                    a_999
m2 <- lm('a_90 ~ sleep_dur + bmiz', data = df)
summary(m2)
summary(m2)


## ---------------------------------------------------------
##              BD Variable   PA Variable   PA Sig   S Sig        
## ----------------------------------------------------------
##     subtot_bmd_age_z_all          a_90       
##   hip_neck_bmd_age_z_all          a_90
##      spine_bmd_age_z_all          a_90   
##     spine_bmad_age_z_all          a_90   
##    tot_hip_bmd_age_z_all          a_90  
##   radius13_bmd_age_z_all          a_90
##   udradius_bmd_age_z_all          a_90
m3 <- lm('subtot_bmd_age_z_all ~ sleep_dur + a_90 + bmiz', data = df)
summary(m3)
summary(m3)

## ========== Mediation Analysis 2 ========== ##

## Study 1 via `mediation`: 
##     BD ~ S (intervention) + PA (mediator)

library(mediation)

# Step 1: Model for the mediator (PA ~ Sleep)
model.m <- lm(a_90 ~ sleep_dur, data = df)

# Step 2: Model for the outcome (BD ~ Sleep + PA)
model.y <- lm(subtot_bmd_age_z_all ~ sleep_dur + a_90, data = df)

# Step 3: Run mediation analysis
med.out <- mediate(
  model.m,          # mediator model
  model.y,          # outcome model
  treat = "sleep_dur",  # independent variable
  mediator = "a_90",    # mediator
  boot = TRUE,          # use bootstrapping
  sims = 1000           # number of bootstrap samples
)

summary(med.out)


## ========== Mediation Analysis 3 ========== ##

## Model via `mediation`: 
##     BD ~ S (intervention) + PA (mediator) + BMI (confounder)

library(mediation)

# Step 1: Model for the mediator (PA ~ Sleep + BMI)
model.m <- lm(a_90 ~ sleep_dur + bmiz, data = df)

# Step 2: Model for the outcome (BD ~ Sleep + PA + BMI)
model.y <- lm(subtot_bmd_age_z_all ~ sleep_dur + a_90 + bmiz, data = df)

# Step 3: Run mediation analysis
med.out <- mediate(
  model.m,
  model.y,
  treat = "sleep_dur",
  mediator = "a_90",
  boot = TRUE,
  sims = 1000
)

summary(med.out)

## ========== Mediation Analysis 4 ========== ##

## Model via `mediation`: 
##     BD ~ S (intervention) + PA (distributional mediator) + BMI (confounder)

library(lavaan)


## ----- Attempt 1

## NOTE: Sleep-to-PA equality constraint

model <- '
  # Mediator models
  q_1 ~ a*sleep_dur + bmiz
  q_2 ~ a*sleep_dur + bmiz
  q_3 ~ a*sleep_dur + bmiz
  q_4 ~ a*sleep_dur + bmiz
  q_5 ~ a*sleep_dur + bmiz
  q_6 ~ a*sleep_dur + bmiz
  q_7 ~ a*sleep_dur + bmiz
  q_8 ~ a*sleep_dur + bmiz

  # Outcome model
  subtot_bmd_age_z_all ~ b1*q_1 + b2*q_2 + b3*q_3 + b4*q_4 +
                         b5*q_5 + b6*q_6 + b7*q_7 + b8*q_8 +
                         c*sleep_dur + bmiz

  # Indirect effects
  ind1 := a*b1
  ind2 := a*b2
  ind3 := a*b3
  ind4 := a*b4
  ind5 := a*b5
  ind6 := a*b6
  ind7 := a*b7
  ind8 := a*b8

  # Total indirect effect
  total_ind := ind1 + ind2 + ind3 + ind4 + ind5 + ind6 + ind7 + ind8
'

fit <- sem(model, data = df, se = "bootstrap", bootstrap = 1000)
summary(fit, ci = TRUE)


## ----- Attempt 2

## NOTE: No Sleep-to-PA equality constraint

model <- '
  # Mediator models (each q_k gets its own sleep coefficient)
  q_1 ~ a1*sleep_dur + bmiz
  q_2 ~ a2*sleep_dur + bmiz
  q_3 ~ a3*sleep_dur + bmiz
  q_4 ~ a4*sleep_dur + bmiz
  q_5 ~ a5*sleep_dur + bmiz
  q_6 ~ a6*sleep_dur + bmiz
  q_7 ~ a7*sleep_dur + bmiz
  q_8 ~ a8*sleep_dur + bmiz

  # Outcome model
  subtot_bmd_age_z_all ~ b1*q_1 + b2*q_2 + b3*q_3 + b4*q_4 +
                         b5*q_5 + b6*q_6 + b7*q_7 + b8*q_8 +
                         c*sleep_dur + bmiz

  # Indirect effects
  ind1 := a1*b1
  ind2 := a2*b2
  ind3 := a3*b3
  ind4 := a4*b4
  ind5 := a5*b5
  ind6 := a6*b6
  ind7 := a7*b7
  ind8 := a8*b8

  # Total indirect effect
  total_ind := ind1 + ind2 + ind3 + ind4 + ind5 + ind6 + ind7 + ind8
'

fit <- sem(model, data = df, se = "bootstrap", bootstrap = 1000)
summary(fit, ci = TRUE)



## ========== Mediation Analysis 2 ========== ##

## Study 1: BD ~ S (intervention) + PA (mediator) + BMI (???)

## -----------------------------------------------
##              BD Variable      S P-val   S Sig   
## -----------------------------------------------
##     subtot_bmd_age_z_all     0.05337        .        
##   hip_neck_bmd_age_z_all     0.0428         *        
##      spine_bmd_age_z_all     0.1372                
##     spine_bmad_age_z_all     0.2505                
##    tot_hip_bmd_age_z_all     0.01862        *        
##   radius13_bmd_age_z_all     0.000656     ***        
##   udradius_bmd_age_z_all     0.0281         *
m1 <- lm('radius13_bmd_age_z_all ~ sleep_dur + bmiz', data = df)
summary(m1)

## ---------------------------------------
##              PA Variable        S Sig 
## ---------------------------------------
##                mvpa_rate            .
##                   a_mean            .
##                     a_50                    
##                     a_60            .         
##                     a_70            .            
##                     a_80            .      
##                     a_90            .  
##                     a_95            .     
##                     a_99
##                    a_995
##                    a_999
m2 <- lm('a_90 ~ sleep_dur + bmiz', data = df)
summary(m2)


## ---------------------------------------------------------
##              BD Variable   PA Variable   PA Sig   S Sig        
## ----------------------------------------------------------
##     subtot_bmd_age_z_all          a_90       
##   hip_neck_bmd_age_z_all          a_90
##      spine_bmd_age_z_all          a_90   
##     spine_bmad_age_z_all          a_90   
##    tot_hip_bmd_age_z_all          a_90  
##   radius13_bmd_age_z_all          a_90
##   udradius_bmd_age_z_all          a_90
m3 <- lm('radius13_bmd_age_z_all ~ sleep_dur + a_90 + bmiz', data = df)
  summary(m3)
