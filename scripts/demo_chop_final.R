library(stringr)

source('src/cot.R')
source('src/utils.R')


## ---------- HELPERS ---------- ##


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

pred_block <- function(prefix, K) {
  paste0(prefix, "_", seq_len(K))
}


## ----- Dataset Settings

dataset_settings <- list(
  mims = list(
    path_df = 'artifacts/demo_chop/df-mims.rds',
    path_pipe = 'artifacts/demo_chop/flow_mims.rds'
  ),
  enmo = list(
    path_df = 'artifacts/demo_chop/df-enmo.rds',
    path_pipe = 'artifacts/demo_chop/flow_enmo.rds'
  )
)


## ---------- Model Fitting ---------- ##


## Globals
dataset <- 'enmo'
response <- 'avg_sleep_dur'
preds_common <- c('age_cat', 'sex', 'bmiz')

## Load data and pipeline
df <- readRDS(dataset_settings[[dataset]]$path_df)
pipeline <- readRDS(dataset_settings[[dataset]]$path_pipe)
K <- pipeline$stages[[5]]$state$K

## Load/extract other quantities
path_L <- file.path('artifacts', 'demo_chop', str_glue('L_flow_{dataset}.rds'))
L <- readRDS(path_L)
E <- pipeline$stages[[5]]$state$E
p_grid <- pipeline$training$cache$p_grid

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
    preds = c(preds_common, "mvpa"),
    f_str_abbr = 'x,mvpa'
  ),
  # scale_2 = list(
  #   preds = c(preds_common, "a_mean"),
  #   f_str_abbr = 'x,a_mean'
  # ),
  scale_3 = list(
    preds = c(preds_common, "a_50"),
    f_str_abbr = 'x,a_50'
  ),
  # scale_4 = list(
  #   preds = c(preds_common, "a_85"),
  #   f_str_abbr = 'x,a_85'
  # ),
  scale_5 = list(
    preds = c(preds_common, "a_95"),
    f_str_abbr = 'x,a_95'
  ),
  scale_6 = list(
    preds = c(preds_common, "a_99"),
    f_str_abbr = 'x,a_99'
  ),
  scale_7 = list(
    preds = c(preds_common, "a_995"),
    f_str_abbr = 'x,a_995'
  ),
  scale_8 = list(
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
    preds = c(preds_common, 'nw', pred_block('q', K)),
    f_str_abbr = 'x,q,nw'
  ),
  dist_C_miss = list(
    preds = c(preds_common, 'nw', pred_block('c', K)),
    f_str_abbr = 'x,c,nw'
  )
  
)

## Compute OOS Errors
mf <- get_model_funs('lm')
results <- data.frame(
  oos_err = numeric(0),
  r_sq    = numeric(0),
  f_str   = character(0),
  preds   = character(0)
)
for (i in seq_along(models)) {
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
      oos_err = mean(cv_full$oos_error^2),
      adj_r_sq    = mf$r_sq(mod_full),
      f_str   = formula_to_string(f_full),
      preds = models[[i]]$f_str_abbr
    )
  )
  
  ## Step-wise pruning
  # preds_sig <- get_sig_lm_predictors(mod_full, factor_vars)
  # if (length(preds_sig) > 0 &&
  #     length(preds_sig) < length(coef(mod_full)) - 1) {
  #   
  #   f_red <- make_formula(response, preds_sig, "lm", factor_vars)
  #   mod_red <- fit_lm(f_red, df)
  #   models[[i]]$model_red <- mod_red
  #   
  #   cv_red <- cv_oos_errors(
  #     df, f_red,
  #     fit_fun  = fit_lm,
  #     pred_fun = pred_lm,
  #     V = 10,
  #     seed = 12345
  #   )
  #   
  #   results <- rbind(
  #     results,
  #     data.frame(
  #       oos_err = mean(cv_red$oos_error^2),
  #       adj_r_sq    = mf$r_sq(mod_red),
  #       f_str   = formula_to_string(f_red)
  #     )
  #   )
  # } else {
  #   results <- rbind(
  #     results,
  #     data.frame(
  #       oos_err = NA,
  #       adj_r_sq    = NA,
  #       f_str   = NA
  #     )
  #   )
  # }
}
# results <- results[order(results$oos_err), ]
results$oos_err <- round(results$oos_err, digits = 3)
results$adj_r_sq <- round(results$adj_r_sq, digits = 3)
print(results[,c('oos_err', 'adj_r_sq', 'preds')])

## Summaries
for (i in 1:length(models)) {
  # print("===== Full =====")
  print(summary(models[[i]]$model_full))
  # print("===== Reduced =====")
  # print(summary(models[[i]]$model_red))
}


## ---------- Model Interpretation ---------- ##


## ----- Interpretation: Q PCA (weight function)

## Get model
model <- models[['dist_Qpc']]$model_full
summary(model)

## Extract fit quantities
coeff_names <- names(model$coefficients[startsWith(names(model$coefficients), 'q_')])
alpha <- model$coefficients[coeff_names]
Sigma_alpha <- vcov(model)[coeff_names, coeff_names]

## Estimate alpha(p)
colnames(L) <- paste0('q_', 1:K)
L_ <- L[,coeff_names]
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
plot(
  p_grid[idx], alpha_p[idx], type = 'l',
  xlab = 'p', ylab = 'alpha(p)'
)
lines(p_grid[idx], alpha_p_low[idx], col = 'gray')
lines(p_grid[idx], alpha_p_high[idx], col = 'gray')
abline(a = 0, b = 0, lty = 'dotted')


## ----- Interpretation: Q PCA (effect plot)

## Set k_star and deltas
# k_star <- 2
# deltas <- seq(-0.01, 0.01, by = 0.002)
k_star <- 6
deltas <- seq(-0.001, 0.001, by = 0.0002)
p_idx <- 50:length(p_grid)

## Get scores of Q_center
Q_mean <- colMeans(do.call(rbind, pipeline$training$meta$Q_list))
sqrt_w <- pipeline$stages[[3]]$state$sqrt_w
path_Qpc_res <- file.path('artifacts', str_glue('demo_chop'), str_glue('Qpc_res_flow_{dataset}.rds'))
Qpc_res <- readRDS(path_Qpc_res)
Qw_mean  <- Q_mean * sqrt_w
Qw_mean_ctr  <- Qw_mean - Qpc_res$center
Vw <- Qpc_res$rotation[, 1:K]
Q_mean_scores  <- Qw_mean_ctr %*% Vw

## Create df_new
preds_star <- data.frame(
  age_cat = c('<=12'),
  sex = c(1)
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

## Prepare Y-axis transform
stage_y_axis_state <- pipeline$stages[[1]]$state
y_trans_fun <- y_trans_to_fun[[stage_y_axis_state$y_trans]]
y_shift <- stage_y_axis_state$y_shift

# Plot first curve to initialize axes
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
y_ticks <- y_trans_fun(y_labs, shift = y_shift)
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
     "BMIz",
     cex = 0.8)


## ---------- Interpretation: Q-G PCA

## Model summary
model <- models[['dist_C']]$model_full
summary(model)

## Set k_star and deltas
k_star <- 4
deltas <- seq(-1, 1, by = 0.2)
p_idx <- 1:length(p_grid)

## Create df_new
out <- qg_pca(
  Q_obs = colMeans(do.call(rbind, pipeline$training$meta$Q_list)),
  E = pipeline$stages[[5]]$state$E,
  G_center = pipeline$stages[[5]]$state$G_center,
  p_grid = pipeline$training$cache$p_grid,
  p_star = pipeline$training$cache$p_star,
  Q_star = pipeline$training$cache$Q_star,
  sqrt_w = pipeline$training$cache$sqrt_w,
  lambda = pipeline$stages[[5]]$state$lambda
)
preds_star <- data.frame(
  age_cat = c('<=12'),
  sex = c(1)
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

## Predict at df_new
resp_new <- predict(model, df_new)

## Map c-shifts to Q-space
c_shifts_list <- asplit(df_new[,paste0('c_', 1:K)], MARGIN = 1)
dc_shifts_ctx <- new_context(
  payload = list(
    c_list = c_shifts_list,
    Q_star_list = replicate(length(c_shifts_list), pipeline$training$cache$Q_star, simplify = FALSE)
  ),
  cache = pipeline$training$cache,
  meta = list()
)
out <- decode(pipeline, c_shifts_ctx, from = 5, to = 3)
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

## Prepare Y-axis transform
stage_y_axis_state <- pipeline$stages[[1]]$state
y_trans_fun <- y_trans_to_fun[[stage_y_axis_state$y_trans]]
y_shift <- stage_y_axis_state$y_shift

# Plot first curve to initialize axes
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
y_ticks <- y_trans_fun(y_labs, shift = y_shift)
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
     "BMIz",
     cex = 0.8)



