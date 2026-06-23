library(dplyr)
library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')  



## ========== Representation Learning ========== ##

dataset <- 'chop-mims-sub'

if (dataset == 'chop-mims-sub') {

  ## Globals
  dir_art <- file.path('artifacts', 'analyze_chop-mims-subs')

  ## Load data
  path <- file.path('data', 'processed', 'chop-mims_v1.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))

  ## Define grid
  J_max <- 10000
  p_grid <- p_grid_fun(
    breaks = c(1/(J_max + 1), 0.95, J_max/(J_max + 1)),
    interval_counts = c(51, 50)
  )

  ## Construct pipeline
  pipeline <- construct_pipeline(
    stages = list(
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_wame(
        K_max = 20,
        epsilon = 0.25,
        alpha = 0.05,
        V = 5,
        lambda = 1e-6
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = file.path(dir_art, 'flow.pth')
      )
    ),
    supp_Y = c(0, 0.6, seq(0.601, 2000, by = 0.001)),
    p_star = 0,
    y_star = 0,
    y_min = 0,
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
Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
c_ctx_ <- decode(pipeline, z_ctx, from = 4, to = 3)
Q_ctx_ <- decode(pipeline, c_ctx_, from = 3, to = 2)
Qi_ctx_ <- decode(pipeline, Q_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Qi_ctx_, from = 1, to = 0)

## Plot stages
col_recon <- rgb(0, 0, 1, alpha = 0.5)
i <- 0
i <- i + 1
pi_grid <- pi_grid_fun(Ji_vec[[i]])
y_max_i <- max(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
y_min_i <- min(c(y_ctx$payload[[i]], y_ctx_$payload[[i]]))
breaks_y <- seq(y_min_i, y_max_i, length.out = 50)

png(path_plot_tmp, width = 1000, height = 500)
par(mfrow=c(2,3))
h <- hist(y_ctx$payload[[i]], breaks = breaks_y)
hist(y_ctx_$payload[[i]], add = TRUE, col = col_recon, breaks = breaks_y)
plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1))
lines(pi_grid, Qi_ctx_$payload[[i]], type = 'l', col = col_recon)
plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1), col = 'gray')
lines(p_grid, Q_ctx$payload[[i]], type = 'l')
lines(p_grid, Q_ctx_$payload[[i]], type = 'l', col = col_recon)
plot(c_ctx$payload$c_list[[i]])
points(c_ctx_$payload$c_list[[i]], col = col_recon)
plot(z_ctx$payload[[i]])
dev.off()



## Plot reconstructions
par(mfrow = c(1,1))
K_plot_min <- 5
  
## Plot z-embeddings
png(path_plot_tmp)
Z_plot <- do.call(rbind, z_ctx$payload)
Z_plot <- Z_plot[,1:min(ncol(Z_plot), K_plot_min)]
plot_embeddings(Z_plot)
dev.off()

## Plot c-embeddings
png(path_plot_tmp, width = 2000, height = 2000)
C_plot <- do.call(rbind, c_ctx$payload$c_list)
C_plot <- C_plot[,1:min(ncol(C_plot), K_plot_min)]
plot_embeddings(C_plot)
dev.off()

## Plot Smooth EQFs on common grid
png(path_plot_tmp)
plot_funs(
  fun_list = Q_ctx$payload,
  grid_list = rep(list(p_grid), N)
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



## ----- Weighted Q-PCA

## Perform Q-PCA
sqrt_w <- pipeline$training$cache$sqrt_w
K <- pipeline$stages[[3]]$state$child_qg_pca$state$K
Qw <- sweep(Q, 2, sqrt_w, FUN = "*")
Qpc_res <- prcomp(Qw, center = TRUE, scale. = FALSE)
Qpc <- Qpc_res$x[, 1:K, drop = FALSE]
Vw <- Qpc_res$rotation[, 1:K]
L <- sweep(Vw, 1, sqrt_w, FUN = "/")
L <- sweep(L, 2, Qpc_res$sdev[1:K], FUN = "*")

## Save quantities 
path_Qpc <- file.path(dir_art, str_glue('Qpc.rds'))
path_L <- file.path(dir_art, str_glue('L.rds'))
path_Qpc_res <- file.path(dir_art, str_glue('Qpc_res.rds'))
saveRDS(Qpc, path_Qpc)
saveRDS(L, path_L)
saveRDS(Qpc_res, path_Qpc_res)


## ========== Data Loading + Processing ========== ##

## Globals
dir_art <- file.path('artifacts', 'analyze_chop-mims-subs')

## ----- Data Loading

## Paths (from process_chop.R)
path_y <- file.path('data', 'processed', 'chop-mims_v1.rds')
path_sleep <- file.path('data', 'processed', 'chop-sleep_v1.rds')
path_nonwear <- file.path('data', 'processed', 'chop-nonwear_v1.rds')
path_mvpa <- file.path('data', 'processed', 'chop-mvpa_v1.rds')
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
df_a$a_99 <- unlist(lapply(y_list, function(y) quantile(y, c(0.99))))
df_a$a_995 <- unlist(lapply(y_list, function(y) quantile(y, c(0.995))))
df_a$a_999 <- unlist(lapply(y_list, function(y) quantile(y, c(0.999))))

## Nonwear
df_nonwear$t_nonwear <- log((df_nonwear$nonwear + 1e-3) / (1 - df_nonwear$nonwear + 1e-3))

## Sleep
df_sleep$good_sleep <- df_sleep$sleep_dur >= 8

## Misc
imp_trans <- log((df_misc$impact23_hr_day) / (df_misc$impact01_hr_day + 1e-3) + 1)
imp_33 <- quantile(tmp_imp, c(0.33))
imp_50 <- quantile(tmp_imp, c(0.5))
imp_66 <- quantile(tmp_imp, c(0.66))
df_misc$impact_trans <- imp_trans
df_misc$impact_cat_2 <- as.factor(sapply(imp_trans, function(x) {
  if (x < imp_50) 0 else 1
}))
df_misc$impact_cat_3 <- as.factor(sapply(imp_trans, function(x) {
  if (x < imp_33) 0
  else if (x < imp_66) 1
  else 2
}))


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

resps <- c(
  # 'a_99'
  # 'bmiz'
  'subtot_bmd_age_z_all'
  # 'hip_neck_bmd_age_z_all',
  # 'spine_bmd_age_z_all',
  # 'spine_bmad_age_z_all',
  # 'tot_hip_bmd_age_z_all',
  # 'radius13_bmd_age_z_all',
  # 'udradius_bmd_age_z_all'

)
preds <- c(
  # 'bmiz',
  # 'a_50',
  # 'a_60',
  # 'a_70',
  # 'a_80',
  # 'a_90',
  # 'a_95',
  # 'a_99'
  # paste0('c_', 1:K),
  # paste0('z_', 1:K),
  # paste0('q_', 1:K),
  # 't_nonwear',
  # 'sleep_dur'
  'impact_cat_2'
)
par(mfrow = c(1,1))
for (resp in resps) {
  for (pred in preds) {
    x <- df[[pred]]
    y <- df[[resp]]
    plot(
      x, y, 
      # main = str_glue("{resp} vs. {pred}"), xlim = c(0,2)
      main = str_glue("{resp} vs. {pred} | cor = {round(cor(x,y), 3)}")
    )
  }
}



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
    seed = gen_seed()
) {
  N <- nrow(df)
  set.seed(seed)
  folds <- sample(rep(seq_len(V), length.out = N))
  
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
preds_common <- c('age_cat', 'sex', 'bmiz')

## Load data and pipeline
path_pipe <- file.path(dir_art, 'pipe.pth')
path_df <- file.path(dir_art, 'df.rds')
pipeline <- readRDS(path_pipe)
K <- pipeline$stages[[3]]$state$child_qg_pca$state$K
df <- readRDS(path_df)

## Predictors
models <- list(
  
  ## Reference
  ref_mean = list(
    preds = c('1'),
    f_str_abbr = '--'
  ),
  ref_common_preds = list(
    preds = c(preds_common),
    f_str_abbr = 'x'
  ),
  ref_common_preds_imp = list(
    preds = c(preds_common, 'impact_cat_2'),
    f_str_abbr = 'x,imp'
  ),

  ## Scalar PA
  scale_1 = list(
    preds = c(preds_common, "mvpa_rate"),
    f_str_abbr = 'x,mvpa'
  ),
  scale_2 = list(
    preds = c(preds_common, "a_50"),
    f_str_abbr = 'x,a_50'
  ),
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
  scale_7 = list(
    preds = c(preds_common, "a_95"),
    f_str_abbr = 'x,a_95'
  ),
  scale_8 = list(
    preds = c(preds_common, "a_99"),
    f_str_abbr = 'x,a_99'
  ),
  # scale_9 = list(
  #   preds = c(preds_common, "a_995"),
  #   f_str_abbr = 'x,a_995'
  # ),
  # scale_10 = list(
  #   preds = c(preds_common, "a_999"),
  #   f_str_abbr = 'x,a_999'
  # ),
  
  ## Scalar PA + Impact
  scale_imp_1 = list(
    preds = c(preds_common, "mvpa_rate", "impact_cat_2", "mvpa_rate:impact_cat_2"),
    f_str_abbr = 'x,mvpa,imp'
  ),
  scale_imp_2 = list(
    preds = c(preds_common, "a_50", "impact_cat_2", "a_50:impact_cat_2"),
    f_str_abbr = 'x,a_50,imp'
  ),
  # scale_imp_3 = list(
  #   preds = c(preds_common, "a_60", "impact_cat_2", "a_60:impact_cat_2"),
  #   f_str_abbr = 'x,a_60,imp'
  # ),
  # scale_imp_4 = list(
  #   preds = c(preds_common, "a_70", "impact_cat_2", "a_70:impact_cat_2"),
  #   f_str_abbr = 'x,a_70,imp'
  # ),
  # scale_imp_5 = list(
  #   preds = c(preds_common, "a_80", "impact_cat_2", "a_80:impact_cat_2"),
  #   f_str_abbr = 'x,a_80,imp'
  # ),
  # scale_imp_6 = list(
  #   preds = c(preds_common, "a_90", "impact_cat_2", "a_90:impact_cat_2"),
  #   f_str_abbr = 'x,a_90,imp'
  # ),
  scale_imp_7 = list(
    preds = c(preds_common, "a_95", "impact_cat_2", "a_95:impact_cat_2"),
    f_str_abbr = 'x,a_95,imp'
  ),
  scale_imp_8 = list(
    preds = c(preds_common, "a_99", "impact_cat_2", "a_99:impact_cat_2"),
    f_str_abbr = 'x,a_99,imp'
  ),
  # scale_imp_9 = list(
  #   preds = c(preds_common, "a_995", "impact_cat_2", "a_995:impact_cat_2"),
  #   f_str_abbr = 'x,a_995,imp'
  # ),
  # scale_imp_10 = list(
  #   preds = c(preds_common, "a_999", "impact_cat_2", "a_999:impact_cat_2"),
  #   f_str_abbr = 'x,a_999,imp'
  # ),

  ## Distributional PA
  dist_Qpc = list(
    preds = c(preds_common, pred_block('q', K)),
    f_str_abbr = 'x,q'
  ),
  dist_C = list(
    preds = c(preds_common, pred_block('c', K)),
    f_str_abbr = 'x,c'
  ),
  # dist_Z_imp = list(
  #   preds = c(preds_common, pred_block('z', K)),
  #   f_str_abbr = 'x,z'
  # ),
  
  ## Distributional PA + Impact
  dist_Qpc_imp = list(
    preds = c(preds_common, 'impact_cat_2', pred_block('q', K), pred_block('q', K, 'impact_cat_2')),
    f_str_abbr = 'x,q,imp'
  ),
  dist_C_imp = list(
    preds = c(preds_common, 'impact_cat_2', pred_block('c', K), pred_block('c', K, 'impact_cat_2')),
    f_str_abbr = 'x,c,imp'
  )
  # dist_Z_imp = list(
  #   preds = c(preds_common, 'impact_cat_2', pred_block('z', K), pred_block('z', K, 'impact_cat_2')),
  #   f_str_abbr = 'x,z,imp'
  # )
  
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
mf <- get_model_funs('lm')
factor_vars <- names(Filter(is.factor, df))
results <- data.frame(
  oos_err = numeric(0),
  r_sq    = numeric(0),
  f_str   = character(0),
  preds   = character(0),
  reduced = logical(0)
)
fit_reduced <- TRUE
# worst_list <- list()
for (i in seq_along(models)) {
  print(str_glue("{i}/{length(models)}"))
  set.seed(12345)
  preds <- models[[i]]$preds
  
  ## Fit model
  f_full <- make_formula(response, preds, 'lm', factor_vars)
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
      adj_r_sq    = mf$r_sq(mod_full),
      f_str   = formula_to_string(f_full),
      preds = models[[i]]$f_str_abbr,
      reduced = FALSE
    )
  )

  ## Step-wise pruning (drop the worst predictor one at a time)
  if (fit_reduced) {
    preds_curr <- preds
    mod_curr   <- mod_full
    repeat {
      worst <- get_worst_lm_predictor(mod_curr, alpha = 0.05)
      # worst_list[[length(worst_list)+1]] <- worst
      if (is.null(worst)) break                  # all predictors significant
      norm_term <- function(t) vapply(
        strsplit(t, ":", fixed = TRUE),
        function(p) paste(sort(p), collapse = ":"),
        character(1)
      )
      preds_curr <- preds_curr[norm_term(preds_curr) != norm_term(worst)]
      if (length(preds_curr) == 0) break         # nothing left to fit
      f_curr   <- make_formula(response, preds_curr, "lm", factor_vars)
      mod_curr <- mf$fit(f_curr, df)
    }

    if (length(preds_curr) > 0 && length(preds_curr) < length(preds)) {
      f_red   <- make_formula(response, preds_curr, "lm", factor_vars)
      mod_red <- mod_curr
      models[[i]]$model_red <- mod_red
      cv_red <- cv_oos_errors(
        df, f_red,
        fit_fun  = fit_lm,
        pred_fun = pred_lm,
        V = 10,
        seed = 12345
      )
      results <- rbind(
        results,
        data.frame(
          oos_err  = sqrt(mean(cv_red$oos_error^2)),
          adj_r_sq = mf$r_sq(mod_red),
          f_str    = formula_to_string(f_red),
          preds    = models[[i]]$f_str_abbr,
          reduced  = TRUE
        )
      )
    }
  }
}
results$oos_err <- round(results$oos_err, digits = 3)
results$adj_r_sq <- round(results$adj_r_sq, digits = 3)
print_full <- FALSE
if (!print_full) {
  results_ <- results[results$reduced,]
  print(results_[,c('oos_err', 'adj_r_sq', 'preds', 'reduced')])
} else {
  print(results[,c('oos_err', 'adj_r_sq', 'preds', 'reduced')])
}


## Summaries
print_full <- FALSE
print_reduced <- TRUE
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
model <- models[['dist_Qpc_imp']]$model_red
summary(model)

## Extract fit quantities
coeff_names <- names(model$coefficients[startsWith(names(model$coefficients), 'q_')])
alpha <- model$coefficients[coeff_names]
Sigma_alpha <- vcov(model)[coeff_names, coeff_names, drop = FALSE]

## Estimate alpha(p)
colnames(L) <- paste0('q_', 1:K)
L_ <- L[, coeff_names, drop = FALSE]
alpha_p <- L_ %*% alpha
alpha_p_se <- sqrt(diag(L_ %*% Sigma_alpha %*% t(L_)))

## Pointwise bands
alpha_p_low <- alpha_p - 2*alpha_p_se
alpha_p_high <- alpha_p + 2*alpha_p_se

## Plot pointwise bands
par(mfrow = c(1,1))
idx <- 1:length(p_grid)
plot(p_grid[idx], alpha_p[idx], type = 'l')
lines(p_grid[idx], alpha_p_low[idx], col = 'gray')
lines(p_grid[idx], alpha_p_high[idx], col = 'gray')
abline(a = 0, b = 0, lty = 'dotted')

## Joint bands
B <- 5000
Rchol <- chol(Sigma_alpha)
M <- numeric(B)
set.seed(12345)
for (b in 1:B) {
  z <- rnorm(length(alpha))
  alpha_b <- alpha + t(Rchol) %*% z
  alpha_b_p <- L_ %*% alpha_b
  M[b] <- max(abs(alpha_b_p - alpha_p) / alpha_p_se)
}
c95 <- quantile(M, 0.95)
alpha_p_low <- alpha_p - c95 * alpha_p_se
alpha_p_high <- alpha_p + c95 * alpha_p_se

## Plot joint bands
idx <- 1:length(p_grid)
png('artifacts/analyze_chop-mims-subs/plots/alpha_bd-all_imp-2_pa-q.png', width = 500, height = 400)
plot(
  p_grid[idx], alpha_p[idx], type = 'l',
  xlab = 'p', ylab = 'alpha(p)'
)
lines(p_grid[idx], alpha_p_low[idx], col = 'gray')
lines(p_grid[idx], alpha_p_high[idx], col = 'gray')
abline(a = 0, b = 0, lty = 'dotted')
dev.off()


## ----- Interpretation: Q PCA (effect plot)

## Set k_star and deltas
# k_star <- 1
# deltas <- seq(-5, 5, by = 1)
k_star <- 2
deltas <- seq(-3, 3, by = 1)
p_idx <- 1:length(p_grid)

## Get scores of Q_center
Q_mean <- colMeans(do.call(rbind, pipeline$training$meta$Q_list))
sqrt_w <- pipeline$training$cache$sqrt_w
path_Qpc_res <- file.path(dir_art, str_glue('Qpc_res.rds'))
Qpc_res <- readRDS(path_Qpc_res)
Qw_mean  <- Q_mean * sqrt_w
Qw_mean_ctr  <- Qw_mean - Qpc_res$center
Vw <- Qpc_res$rotation[, 1:K]
Q_mean_scores  <- Qw_mean_ctr %*% Vw

## Create df_new
preds_star <- data.frame(
  age_cat = c('<=12'),
  sex = c(1),
  bmiz = c(0.5)
)
for (k in 1:K) {
  preds_star[[str_glue('q_{k}')]] <- Q_mean_scores[,k]
}
df_new <- data.frame(matrix(nrow = 0, ncol = ncol(preds_star)))
for (delta in deltas) {
  row <- preds_star
  row[[str_glue('q_{k_star}')]] <- preds_star[1,c(str_glue('q_{k_star}'))] + delta
  df_new <- rbind(df_new, row)
}
df_new$age_cat <- as.factor(df_new$age_cat)
df_new$sex <- as.factor(df_new$sex)

## Predict at df_new
resp_new <- predict(model, df_new)

## Map q-shifts to Q-space
q_shifts <- as.matrix(df_new[,paste0('q_', 1:K)])
Qw_rec <- q_shifts %*% t(Qpc_res$rotation[,1:K])
Qw_rec <- sweep(
  Qw_rec, 2,
  Qpc_res$center,
  "+"
)
Q_plot <- sweep(Qw_rec, 2, sqrt_w, "/")

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
png(str_glue('artifacts/analyze_chop-mims-subs/plots/q{k_star}_bd-all_pa-q.png'), width = 500, height = 400)
plot(
  p_grid[p_idx],
  Q_plot[1,p_idx],
  type = "l",
  col = curve_cols[1],
  lwd = 2,
  xlab = "p",
  ylab = "Q(p)",
  ylim = range(Q_plot),
  yaxt = 'n',
  main = str_glue('q_{k_star}')
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


## ---------- Interpretation: Q-G PCA (based on means)

## Model summary
model <- models[['dist_C_imp']]$model_red
summary(model)

## Set k_star and deltas
# k_star <- 4
# deltas <- seq(-1, 1, by = 0.25)
k_star <- 5
deltas <- seq(-2, 2, by = 0.5)
# k_star <- 6
# deltas <- seq(-2, 2, by = 0.5)
p_idx <- 1:length(p_grid)

# vec <- df$c_5
# plot(df$c_4, vec)
# points(mean(df$c_4), mean(vec), pch = 19, col = 'red')


## Create df_new
out <- qg_pca(
  Q_obs = colMeans(do.call(rbind, pipeline$training$meta$Q_list)),
  E = pipeline$stages[[3]]$state$child_qg_pca$state$E,
  G_center = pipeline$stages[[3]]$state$child_qg_pca$state$G_center,
  p_grid = pipeline$training$cache$p_grid,
  p_star = pipeline$training$cache$p_star,
  Q_star = pipeline$training$cache$Q_star,
  sqrt_w = pipeline$training$cache$sqrt_w,
  lambda = pipeline$stages[[3]]$state$child_qg_pca$state$lambda
)
preds_star <- data.frame(
  age_cat = c('<=12'),
  sex = c(1),
  bmiz = 0.5,
  impact_cat_2 = c(1)
)
for (k in 1:length(out$c)) {
  preds_star[[str_glue('c_{k}')]] <- out$c[k]
}
df_new <- data.frame(matrix(nrow = 0, ncol = ncol(preds_star)))
for (delta in deltas) {
  row <- preds_star
  row[[str_glue('c_{k_star}')]] <- preds_star[1,c(str_glue('c_{k_star}'))] + delta
  df_new <- rbind(df_new, row)
}
df_new$age_cat <- as.factor(df_new$age_cat)
df_new$sex <- as.factor(df_new$sex)
df_new$impact_cat_2 <- as.factor(df_new$impact_cat_2)

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
png(str_glue('artifacts/analyze_chop-mims-subs/plots/c{k_star}_bd-all_pa-c.png'), width = 500, height = 400)
plot(
  p_grid[p_idx],
  Q_plot[1,p_idx],
  type = "l",
  col = curve_cols[1],
  lwd = 2,
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
  age_cat       = factor('<=12'),
  sex           = factor(1),
  bmiz          = 0.5,
  impact_cat_2  = factor(1)
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
    age_cat       = factor('<=12'),
    sex           = factor(1),
    bmiz          = 0.5,
    impact_cat_2  = factor(1)
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
