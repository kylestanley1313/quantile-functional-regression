library(e1071)
library(glmnet)
library(MASS)
library(mgcv)
library(randomForest)
library(xgboost)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


par(mfrow=c(1, 2))
plot(hist(rnorm(100)))
plot(hist(rnorm(100)))

## ---------- Colors ---------- ##

col_recon <- rgb(0, 1, 0, alpha = 0.5)
col_train <- 'black'
col_mean <- 'red'
col_draw <- 'gray'


## ---------- Representation Learning ---------- ##

## Globals
dataset <- 'enmo'
pipe_type <- 'flow'
pipe_name <- str_glue('{pipe_type}_{dataset}')
n_stage_plots <- 3
plot_recons <- TRUE
plot_qf_visual <- FALSE


set.seed(12345)
if (pipe_name == 'flow_mims') {
  
  ## Load data
  path <- file.path('data', 'processed', 'chop-mims_v1.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, 100,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(y_trans = 'log', y_shift = 100),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K = NULL,
        K_max = 20,
        epsilon = 0.01,
        alpha = 0.05,
        V = 5,
        lambda = 1e-6
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_chop-mims.pth'
      )
    ),
    supp_Y = c(0, 0.6, seq(0.601, 4000, by = 0.001)),  ## TODO: Improve this?
    p_star = 0,
    y_star = y_star,
    y_min = -0.01,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else if (pipe_name == 'flow_enmo') {

  ## Load data
  path <- file.path('data', 'processed', 'chop-enmo_v1.rds')
  y_list <- readRDS(path)
  N <- length(y_list)
  Ji_vec <- lengths(y_list)
  Ji_max <- max(Ji_vec)
  y_max <- max(unlist(y_list))
  
  ## Define grid
  p_grid <- p_grid_fun(
    y_list, 200,
    p_left = 0.05, J_left = 50,
    p_right = 0.95, J_right = 50
  )
  
  ## Construct pipeline
  y_star <- 0
  pipeline <- construct_pipeline(
    stages = list(
      stage_y_axis(
        y_trans = 'log',
        y_shift = 1e-3
      ),
      stage_eqf_sgrid(),
      stage_eqf_cgrid(p_grid = p_grid),
      stage_lqd(),
      stage_qg_pca(
        K_max = 20,
        epsilon = 0.01,
        alpha = 0.05,
        V = 5,
        lambda = 1e-3
      ),
      stage_flow(
        n_layers = 16,
        max_epochs = 1000,
        lr = 1e-3,
        path = 'artifacts/flow_chop-enmo.pth'
      )
    ),
    supp_Y = seq(0, 2*y_max, by = 0.0001),
    p_star = 0,
    y_star = y_star,
    y_min = 0,
    loss = 'one_minus_sqcor',
    seed = gen_seed()
  )
  
} else {
  stop("Invalid pipe_name!")
}

## Fitting
pipeline <- fit(pipeline, y_list)

## Save pipeline
path <- file.path('artifacts', str_glue('demo_chop'), str_glue('{pipe_name}.rds'))
saveRDS(pipeline, path)

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
for (i in 1:n_stage_plots) {
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

## Plot reconstructions
if (plot_recons) {
  par(mfrow = c(1,1))
  K_plot_min <- 4
  
  ## Plot z-embeddings
  Z_plot <- do.call(rbind, z_ctx$payload)
  Z_plot <- Z_plot[,1:min(ncol(Z_plot), K_plot_min)]
  plot_embeddings(Z_plot)
  
  ## Plot z-embeddings
  C_plot <- do.call(rbind, c_ctx$payload$c_list)
  C_plot <- C_plot[,1:min(ncol(C_plot), K_plot_min)]
  plot_embeddings(C_plot)
  
  ## Plot Smooth EQFs on common grid
  plot_funs(
    fun_list = Q_ctx$payload,
    grid_list = rep(list(p_grid), N)
  )
}


if (dataset == 'enmo' & plot_qf_visual) {
  Qi <- Qi_ctx$payload[[1]]
  
  ## Prepare Y-axis transform
  stage_y_axis_state <- pipeline$stages[[1]]$state
  y_trans_fun <- y_trans_to_fun[[stage_y_axis_state$y_trans]]
  y_shift <- stage_y_axis_state$y_shift
  
  ## Points to plot
  mvpa_rate <- mean(Qi >= y_trans_fun(0.100, shift = y_shift))
  p1 <- 0.5
  p2 <- 0.99
  p3 <- 1 - mvpa_rate
  Q1 <- quantile(Qi, c(0.5))
  Q2 <- quantile(Qi, c(0.99))
  Q3 <- y_trans_fun(0.100, shift = y_shift)
  y1 <- y_trans_fun(Q1, shift = y_shift, TRUE)
  y2 <- y_trans_fun(Q2, shift = y_shift, TRUE)
  y3 <- y_trans_fun(Q3, shift = y_shift, TRUE)
  
  ## Main Plot
  pi_grid <- pi_grid_fun(length(Qi))
  plot(
    pi_grid, Qi, type = 'l',
    xlab = 'p', ylab = 'Q(p)',
    yaxt = 'n'
  )
  y_labs <- c(0.001, 0.01, 0.1, 1)
  y_ticks <- y_trans_fun(y_labs, shift = y_shift) 
  axis(side = 2, at = y_ticks, labels = y_labs)
  
  ## Segments/Labels
  col_guide <- "gray60"
  lty_guide <- 3  # dotted
  
  ## (0.5, Q(0.5))
  points(p1, Q1, pch = 19, cex = 0.7)
  segments(0, Q1, p1, Q1, col = col_guide, lty = lty_guide)
  segments(p1, min(Qi), p1, Q1, col = col_guide, lty = lty_guide)
  
  ## (0.99, Q(0.99))
  points(p2, Q2, pch = 19, cex = 0.7)
  segments(0,  Q2, p2, Q2, col = col_guide, lty = lty_guide)
  segments(p2, min(Qi), p2, Q2, col = col_guide, lty = lty_guide)
  
  ## (1 - mvpa_rate, 0.100)
  points(p3, Q3, pch = 19, cex = 0.7)
  segments(0,  Q3, p3, Q3, col = col_guide, lty = lty_guide)
  segments(p3, min(Qi), p3, Q3, col = col_guide, lty = lty_guide)
  
  text(p1, Q1, labels = sprintf("(%.2f, %.3f)", p1, y1),
       pos = NULL, adj = c(1.1, -0.5), col = "black")
  
  text(p2, Q2, labels = sprintf("(%.2f, %.3f)", p2, y2),
       pos = NULL, adj = c(1.1, -0.5), col = "black")
  
  text(p3, Q3, labels = sprintf("(%.2f, %.3f)", p3, y3),
       pos = NULL, adj = c(1.1, -0.5), col = "black")
  
}



## ----- Save Quantities

path_Q <- file.path('artifacts', str_glue('demo_chop'), str_glue('Q_{pipe_name}.rds'))
path_C <- file.path('artifacts', str_glue('demo_chop'), str_glue('C_{pipe_name}.rds'))
path_Z <- file.path('artifacts', str_glue('demo_chop'), str_glue('Z_{pipe_name}.rds'))
path_E <- file.path('artifacts', str_glue('demo_chop'), str_glue('E_{pipe_name}.rds'))
path_p_grid <- file.path('artifacts', str_glue('demo_chop'), str_glue('p_grid_{pipe_name}.rds'))
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



## ========== Analysis ========== ##

## ----- Variables ----- #

## Responses
path_avg_sleep_dur <- file.path('data', 'processed', 'chop-sleep_avg-dur_v1.rds')
path_avg_sleep_eff <- file.path('data', 'processed', 'chop-sleep_avg-eff_v1.rds')
path_subs_sleep <- file.path('data', 'processed', 'chop-sleep_v1_subs.rds')
avg_sleep_dur <- unlist(readRDS(path_avg_sleep_dur))
avg_sleep_eff <- unlist(readRDS(path_avg_sleep_eff))
subs_sleep <- unlist(readRDS(path_subs_sleep))

## Predictors
path_y <- file.path('data', 'processed', str_glue('chop-{dataset}_v1.rds'))
path_Q <- file.path('artifacts', str_glue('demo_chop'), str_glue('Q_{pipe_name}.rds'))
path_C <- file.path('artifacts', str_glue('demo_chop'), str_glue('C_{pipe_name}.rds'))
path_Z <- file.path('artifacts', str_glue('demo_chop'), str_glue('Z_{pipe_name}.rds'))
path_subs_y <- file.path('data', 'processed', str_glue('chop-{dataset}_v1_subs.rds'))
y_list <- readRDS(path_y)
Q <- readRDS(path_Q)
C <- readRDS(path_C)
Z <- readRDS(path_Z)
subs_y <- unlist(readRDS(path_subs_y))

## Nonwear
path_nonwear <- file.path('data', 'processed', 'chop-nonwear_v1.rds')
path_nonwear_hourly <- file.path('data', 'processed', 'chop-nonwear-hourly_v1.rds')
path_subs <- file.path('data', 'processed', 'chop-nonwear_v1_subs.rds')
nonwear <- readRDS(path_nonwear)
nonwear_hourly <- readRDS(path_nonwear_hourly)
subs_nonwear <- unlist(readRDS(path_subs))

## MVPA predictors
path_mvpa <- file.path('data', 'processed', 'chop-mvpa_v1.rds')
df_mvpa <- readRDS(path_mvpa)
subs_mvpa <- df_mvpa$subject_id

## Other predictors
path_misc_preds <- file.path('data', 'processed', 'chop_misc-preds.rds')
df_misc_preds <- readRDS(path_misc_preds)
subs_misc_preds <- df_misc_preds$subject_id

## Align responses and predictors
subs_common <- Reduce(intersect, list(subs_sleep, subs_y, subs_nonwear, subs_mvpa, subs_misc_preds))
idx_sleep <- match(subs_common, subs_sleep)
idx_y <- match(subs_common, subs_y)
idx_nonwear <- match(subs_common, subs_nonwear)
idx_mvpa <- match(subs_common, subs_mvpa)
idx_misc_preds <- match(subs_common, subs_misc_preds)
avg_sleep_dur <- avg_sleep_dur[idx_sleep]
avg_sleep_eff <- avg_sleep_eff[idx_sleep]
y_list <- y_list[idx_y]
Q <- Q[idx_y, , drop = FALSE]
C <- C[idx_y, , drop = FALSE]
Z <- Z[idx_y, , drop = FALSE]
nonwear <- nonwear[idx_nonwear, 1 , drop = FALSE]
nonwear_hourly <- nonwear_hourly[idx_nonwear, , drop = FALSE]
df_mvpa <- df_mvpa[idx_mvpa, , drop = FALSE]
df_misc_preds <- df_misc_preds[idx_misc_preds, , drop = FALSE]
subs_sleep <- subs_common
subs_y <- subs_common
subs_nonwear <- subs_common
subs_mvpa <- subs_common
subs_misc_preds <- subs_common

## Derived variables
K <- pipeline$stages[[5]]$state$K  ## TODO: De-hardcode 5
y_mean <- unlist(lapply(y_list, mean))
y_50 <- unlist(lapply(y_list, function(y) quantile(y, c(0.5))))
y_85 <- unlist(lapply(y_list, function(y) quantile(y, c(0.85))))
y_90 <- unlist(lapply(y_list, function(y) quantile(y, c(0.9))))
y_95 <- unlist(lapply(y_list, function(y) quantile(y, c(0.95))))
y_99 <- unlist(lapply(y_list, function(y) quantile(y, c(0.99))))
y_995 <- unlist(lapply(y_list, function(y) quantile(y, c(0.995))))
y_999 <- unlist(lapply(y_list, function(y) quantile(y, c(0.999))))
t_nonwear <- log((nonwear + 1e-3) / (1 - nonwear + 1e-3))

## Weight Q-PCA
sqrt_w <- pipeline$stages[[3]]$state$sqrt_w
Qw <- sweep(Q, 2, sqrt_w, FUN = "*")
Qpc_res <- prcomp(Qw, center = TRUE, scale. = FALSE)
Qpc <- Qpc_res$x[, 1:K, drop = FALSE]
Vw <- Qpc_res$rotation[, 1:K]
L <- sweep(Vw, 1, sqrt_w, FUN = "/")
path_L <- file.path('artifacts', str_glue('demo_chop'), str_glue('L_{pipe_name}.rds'))
saveRDS(L, path_L)
path_Qpc_res <- file.path('artifacts', str_glue('demo_chop'), str_glue('Qpc_res_{pipe_name}.rds'))
saveRDS(Qpc_res, path_Qpc_res)

## Missingness adjustment
out_C <- lm(C ~ nonwear_hourly)
C_miss <- out_C$residuals
out_Z <- lm(Z ~ nonwear_hourly)
Z_miss <- out_Z$residuals

## Make dataframe
make_block <- function(mat, prefix) {
  out <- as.data.frame(mat)
  names(out) <- paste0(prefix, "_", seq_len(ncol(mat)))
  out
}
df <- data.frame(
  age_cat = as.factor(df_misc_preds$age_cat),
  sex = as.factor(df_misc_preds$sex),
  bmiz = df_misc_preds$bmiz,
  fmi_all_z = df_misc_preds$fmi_all_z,
  
  avg_sleep_dur = avg_sleep_dur,
  avg_sleep_eff = avg_sleep_eff,
  
  a_mean = y_mean,
  a_50 = y_50,
  a_85 = y_85,
  a_90 = y_90,
  a_95 = y_95,
  a_99 = y_99,
  a_995 = y_995,
  a_999 = y_999,
  mvpa = df_mvpa$mvpa_rate,
  nw = t_nonwear
)
df <- cbind(
  df,
  make_block(C, "c"),
  make_block(Z, "z"),
  make_block(Qpc, "q"),
  make_block(C_miss, "c_nw"),
  make_block(Z_miss, "z_nw")
)
factor_vars <- names(Filter(is.factor, df))

## Save dataframe
path <- file.path('artifacts', 'demo_chop', str_glue('df-{dataset}.rds'))
saveRDS(df, path)


## ---------- EDA ---------- #

y <- 'bmiz'
par(mfrow = c(3,3))
for (var in names(df)) {
  plot(df[[var]], df[[y]], main = str_glue("{var} vs. {y}"))
}


## ---------- Model Exploration ---------- #

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

get_sig_lm_predictors <- function(mod, factor_vars, alpha = 0.1) {
  
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



## ----- GAM

fit_gam <- function(formula, data) {
  gam(
    formula,
    data = data,
    method = "REML",
    select = TRUE
  )
}

pred_gam <- function(fit, newdata) {
  predict(fit, newdata = newdata, type = "response")
}


## ----- Random Forest

fit_rf <- function(formula, data) {
  randomForest(
    formula,
    data = data,
    importance = FALSE
  )
}

pred_rf <- function(fit, newdata) {
  predict(fit, newdata = newdata)
}


## ----- SVM

fit_svm <- function(formula, data) {
  tune.out <- tune(
    svm,
    formula,
    data = data,
    ranges = list(
      cost = 2^(-2:4),
      gamma = 2^(-4:1)
    ),
    scale = TRUE
  )
  tune.out$best.model
}

pred_svm <- function(fit, newdata) {
  predict(fit, newdata = newdata)
}


## ----- Elastic Net

fit_enet <- function(formula, data, alpha = 0.5) {
  xy <- make_x_y(formula, data)
  
  cv <- cv.glmnet(
    x = xy$X,
    y = xy$y,
    alpha = alpha,
    standardize = TRUE
  )
  
  list(
    model = cv,
    coef_names = colnames(xy$X)
  )
}

pred_enet <- function(fit, newdata, formula) {
  X_new <- model.matrix(formula, newdata)[, fit$coef_names, drop = FALSE]
  as.numeric(predict(fit$model, newx = X_new, s = "lambda.min"))
}


## ----- XGBoost

fit_xgb <- function(formula, data) {
  xy <- make_x_y(formula, data)
  
  dtrain <- xgb.DMatrix(xy$X, label = xy$y)
  
  model <- xgb.train(
    data = dtrain,
    objective = "reg:squarederror",
    nrounds = 300,
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    verbose = 0
  )
  
  list(
    model = model,
    coef_names = colnames(xy$X)
  )
}


pred_xgb <- function(fit, newdata, formula) {
  X_new <- model.matrix(formula, newdata)[, fit$coef_names, drop = FALSE]
  predict(fit$model, xgb.DMatrix(X_new))
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

pred_block <- function(prefix, K) {
  paste0(prefix, "_", seq_len(K))
}

response <- 'fmi_all_z'
preds_common <- c('age_cat', 'sex')
preds_miss <- c('t_nonwear')
preds_list <- list(
  
  ## Base
  c("1"),
  c(preds_common),
  c(preds_common, preds_miss),
  
  ## Scalar
  c(preds_common, "x_mean"),
  c(preds_common, "x_90"),
  c(preds_common, "x_95"),
  c(preds_common, "x_99"),
  c(preds_common, c("x_90", "x_95", "x_99")),
  
  ## Distributional
  c(preds_common, pred_block("C", K)),
  c(preds_common, pred_block("Z", K)),
  c(preds_common, pred_block("Qpc", K)),
  c(preds_common, pred_block("Z_miss", K)),
  c(preds_common, pred_block("C_miss", K)),
  
  ## Scalar + Missing
  c(preds_common, preds_miss, "x_mean"),
  c(preds_common, preds_miss, "x_90"),
  c(preds_common, preds_miss, "x_95"),
  c(preds_common, preds_miss, "x_99"),
  c(preds_common, preds_miss, c("x_90", "x_95", "x_99")),
  
  ## Distributional + Missing
  c(preds_common, preds_miss),
  c(preds_common, preds_miss, pred_block("C", K)),
  c(preds_common, preds_miss, pred_block("Z", K)),
  c(preds_common, preds_miss, pred_block("Qpc", K)),
  c(preds_common, preds_miss, pred_block("Z_miss", K)),
  c(preds_common, preds_miss, pred_block("C_miss", K))
  
)


## ----- Prediction

get_model_funs <- function(model_type) {
  if (model_type == "lm") {
    list(fit = fit_lm, pred = pred_lm, r_sq = function(m) summary(m)$adj.r.squared)
    
  } else if (model_type == "gam") {
    list(fit = fit_gam, pred = pred_gam, r_sq = function(m) summary(m)$r.sq)
    
  } else if (model_type == "rf") {
    list(fit = fit_rf, pred = pred_rf, r_sq = function(m) NA_real_)
    
  } else if (model_type == "svm") {
    list(fit = fit_svm, pred = pred_svm, r_sq = function(m) NA_real_)
    
  }else if (model_type == "enet") {
    list(fit = fit_enet, pred = pred_enet, r_sq = function(m) NA_real_)
    
  } else if (model_type == "xgb") {
    list(fit = fit_xgb, pred = pred_xgb, r_sq = function(m) NA_real_)
    
  } else {
    stop("Unknown model_type")
  }
}



## Select model type
model_type <- 'lm'   
mf <- get_model_funs(model_type)

## Fit models
results <- data.frame(
  oos_err = numeric(0),
  r_sq    = numeric(0),
  f_str   = character(0)
)
for (i in seq_along(preds_list)) {
  print(str_glue("{strrep('=====', 10)}"))
  
  ## Get formula
  preds <- preds_list[[i]]
  if (model_type %in% c("rf", "svm", "enet", "xgb") && identical(preds, "1")) {
    next
  }
  f_full <- make_formula(response, preds, model_type, factor_vars)
  
  ## ----- Full model
  print(str_glue("f_full = {formula_to_string(f_full)}"))
  mod_full <- mf$fit(f_full, df)
  
  cv_full <- cv_oos_errors(
    df, f_full,
    fit_fun  = mf$fit,
    pred_fun = mf$pred,
    V = 10,
    seed = 12345
  )
  
  results <- rbind(
    results,
    data.frame(
      oos_err = mean(cv_full$oos_error^2),
      r_sq    = mf$r_sq(mod_full),
      f_str   = formula_to_string(f_full)
    )
  )
  
  ## ----- Stepwise pruning (LM ONLY)
  if (model_type == "lm") {
    
    preds_sig <- get_sig_lm_predictors(mod_full, factor_vars)
    
    if (length(preds_sig) > 0 &&
        length(preds_sig) < length(coef(mod_full)) - 1) {
      
      f_red <- make_formula(response, preds_sig, "lm", factor_vars)
      print(str_glue("f_red = {formula_to_string(f_red)}"))
      mod_red <- fit_lm(f_red, df)
      
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
          oos_err = mean(cv_red$oos_error^2),
          r_sq    = summary(mod_red)$adj.r.squared,
          f_str   = formula_to_string(f_red)
        )
      )
    }
  }
}
results <- results[order(results$oos_err), ]
path <- file.path(
  'artifacts', 'demo_chop', 'model-fits', 
  str_glue('data-{dataset}_resp-{response}_mod-{model_type}.csv')
)
write.csv(results, path, row.names = FALSE)
rownames(results) <- NULL
print(rbind(
  results[1:5, ],
  results[results$f_str == formula_to_string(make_formula(response, c(preds_common, preds_miss), model_type, factor_vars)), ],
  results[results$f_str == formula_to_string(make_formula(response, preds_common, model_type, factor_vars)), ],
  results[results$f_str == formula_to_string(make_formula(response, "1", model_type, factor_vars)), ]
))





