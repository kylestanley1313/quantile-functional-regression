## ============================================================
## src/model/amm.R
## Additive mixed model for latent covariates: estimation,
## inference, diagnostics, and interpretation.
## Naming convention: amm_verb() / amm_verb_qualifier()
## Internal helpers: .amm_*()
## ============================================================

library(future)
library(future.apply)
library(nlme)
library(splines)


## ==================== Estimation ====================

## O'Sullivan (Demmler-Reinsch) spline design matrix for x with n_knots
## interior knots and an identity-penalized random-effects reparameterization.
## Returns a matrix W whose columns are the penalized basis functions; basis
## attributes (knots, range, Z_scaled) are stored as attributes.
.amm_build_dr_design <- function(x, n_knots = 5, range_x = NULL) {
  if (is.null(range_x)) range_x <- range(x)
  a <- range_x[1]; b <- range_x[2]

  knots_interior <- as.numeric(quantile(
    unique(x),
    probs = seq(0, 1, length.out = n_knots + 2)[-c(1, n_knots + 2)]
  ))

  all_knots <- c(rep(a, 4), knots_interior, rep(b, 4))
  B <- splineDesign(knots = all_knots, x = x, ord = 4, outer.ok = TRUE)

  n_grid  <- 401
  grid    <- seq(a, b, length.out = n_grid)
  B2      <- splineDesign(knots = all_knots, x = grid, ord = 4, derivs = 2,
                          outer.ok = TRUE)
  h       <- (b - a) / (n_grid - 1)
  w_simp  <- rep(0, n_grid)
  w_simp[1]                          <- h / 3
  w_simp[n_grid]                     <- h / 3
  w_simp[seq(2, n_grid - 1, by = 2)] <- 4 * h / 3
  w_simp[seq(3, n_grid - 1, by = 2)] <- 2 * h / 3
  Omega <- t(B2) %*% (w_simp * B2)

  eig <- eigen(Omega, symmetric = TRUE)
  d   <- eig$values
  P   <- eig$vectors
  ord <- order(d)
  d   <- d[ord]; P <- P[, ord]

  Z_Omega  <- P[, -(1:2)]
  d_nz     <- d[-(1:2)]
  Z_scaled <- Z_Omega %*% diag(1 / sqrt(d_nz))

  W <- B %*% Z_scaled
  attr(W, "knots_interior") <- knots_interior
  attr(W, "all_knots")      <- all_knots
  attr(W, "Z_scaled")       <- Z_scaled
  attr(W, "range")          <- range_x
  W
}


## Fit an additive mixed model to each column of Z separately:
##   Z_k = X beta_k + W_age u_{age,k} + W_pir u_{pir,k} + eps_k
## where W_age, W_pir are Demmler-Reinsch spline bases and u ~ N(0, sigma^2_u I).
## Returns a named list with all fitted quantities (see return statement).
fit_amm <- function(Z, covariates,
                    n_knots_age = 10, n_knots_pir = 10,
                    linear_covariates = c("male", "age", "pir"),
                    smooth_covariates = c("age", "pir")) {

  keep <- complete.cases(covariates[, c("male", "age", "pir")])
  Z    <- Z[keep, , drop = FALSE]
  cov_ <- covariates[keep, , drop = FALSE]
  N    <- nrow(Z)
  K    <- ncol(Z)

  has_age_lin <- "age"  %in% linear_covariates
  has_pir_lin <- "pir"  %in% linear_covariates
  has_male    <- "male" %in% linear_covariates
  has_age_sm  <- "age"  %in% smooth_covariates
  has_pir_sm  <- "pir"  %in% smooth_covariates

  X_cols <- list(`(Intercept)` = rep(1, N))
  if (has_male)    X_cols[["male"]] <- cov_$male
  if (has_age_lin) X_cols[["age"]]  <- cov_$age
  if (has_pir_lin) X_cols[["pir"]]  <- cov_$pir
  X_design <- do.call(cbind, X_cols)
  colnames(X_design) <- names(X_cols)
  A1 <- ncol(X_design)
  beta_names <- colnames(X_design)
  fixed_rhs <- if (length(beta_names) > 1) {
    paste(setdiff(beta_names, "(Intercept)"), collapse = " + ")
  } else {
    "1"
  }
  fixed_form <- as.formula(paste("z ~", fixed_rhs))

  W_age <- if (has_age_sm) .amm_build_dr_design(cov_$age, n_knots = n_knots_age) else NULL
  W_pir <- if (has_pir_sm) .amm_build_dr_design(cov_$pir, n_knots = n_knots_pir) else NULL
  M_a   <- if (!is.null(W_age)) ncol(W_age) else 0
  M_p   <- if (!is.null(W_pir)) ncol(W_pir) else 0

  phi_age <- if (M_a > 0) eigen(crossprod(W_age), symmetric = TRUE,
                                 only.values = TRUE)$values else numeric(0)
  phi_pir <- if (M_p > 0) eigen(crossprod(W_pir), symmetric = TRUE,
                                 only.values = TRUE)$values else numeric(0)

  beta_hat      <- matrix(NA_real_, nrow = A1, ncol = K,
                          dimnames = list(beta_names, NULL))
  U_age_hat     <- if (M_a > 0) matrix(NA_real_, nrow = M_a, ncol = K) else NULL
  U_pir_hat     <- if (M_p > 0) matrix(NA_real_, nrow = M_p, ncol = K) else NULL
  sigma2_U_age  <- if (M_a > 0) numeric(K) else NULL
  sigma2_U_pir  <- if (M_p > 0) numeric(K) else NULL
  sigma2_E_diag <- numeric(K)
  edf_age       <- if (M_a > 0) numeric(K) else NULL
  edf_pir       <- if (M_p > 0) numeric(K) else NULL
  edf           <- numeric(K)
  fits          <- vector("list", K)
  collapsed     <- vector("list", K)

  W_cols_age <- if (M_a > 0) paste0("Wa", seq_len(M_a)) else character(0)
  W_cols_pir <- if (M_p > 0) paste0("Wp", seq_len(M_p)) else character(0)
  form_age   <- if (M_a > 0) as.formula(paste0("~ -1 + ", paste(W_cols_age, collapse = " + "))) else NULL
  form_pir   <- if (M_p > 0) as.formula(paste0("~ -1 + ", paste(W_cols_pir, collapse = " + "))) else NULL
  group      <- factor(rep(1, N))

  df_base <- data.frame(
    male = cov_$male,
    age  = cov_$age,
    pir  = cov_$pir,
    grp  = group
  )
  for (j in seq_len(M_a)) df_base[[W_cols_age[j]]] <- W_age[, j]
  for (j in seq_len(M_p)) df_base[[W_cols_pir[j]]] <- W_pir[, j]

  random_struct <- NULL
  if (M_a > 0 && M_p > 0) {
    random_struct <- list(grp = pdBlocked(list(pdIdent(form_age), pdIdent(form_pir))))
  } else if (M_a > 0) {
    random_struct <- list(grp = pdIdent(form_age))
  } else if (M_p > 0) {
    random_struct <- list(grp = pdIdent(form_pir))
  }

  fit_lme <- function(random) {
    tryCatch(
      do.call(lme, list(
        fixed   = fixed_form,
        random  = random,
        data    = df_k,
        method  = "REML",
        control = lmeControl(opt = "optim")
      )),
      error = function(e) {
        message(str_glue("  lme error: {conditionMessage(e)}"))
        NULL
      }
    )
  }

  for (k in seq_len(K)) {
    message(str_glue("---- AMM fit: k = {k} of {K} ----"))
    df_k   <- df_base
    df_k$z <- Z[, k]

    if (is.null(random_struct)) {
      fit_k <- tryCatch(
        do.call(lm, list(formula = fixed_form, data = df_k)),
        error = function(e) {
          message(str_glue("  lm failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      if (is.null(fit_k)) next
      fits[[k]] <- fit_k
      beta_hat[, k]     <- coef(fit_k)[beta_names]
      sigma2_E_diag[k]  <- summary(fit_k)$sigma^2
      edf[k] <- A1
    } else {
      fit_k <- fit_lme(random_struct)
      dropped <- character(0)

      ## lme foreign-call errors are interpreted as variance-component boundary
      ## collapse (a random-effect variance -> 0, i.e., a smooth that is exactly
      ## linear). Refit at the boundary solution: drop each smooth in turn and keep
      ## the single-smooth fit with the highest REML logLik; if none converges (or
      ## only one smooth was present), fall back to fixed effects only (gls, REML,
      ## so logLik stays comparable with the lme fits).
      if (is.null(fit_k)) {
        message(str_glue("  lme failed at k = {k}; refitting at the variance-component boundary"))
        cands <- list()
        if (M_a > 0 && M_p > 0) {
          cands <- list(
            pir = fit_lme(list(grp = pdIdent(form_age))),
            age = fit_lme(list(grp = pdIdent(form_pir)))
          )
          cands <- Filter(Negate(is.null), cands)
        }
        if (length(cands) > 0) {
          best    <- names(cands)[which.max(sapply(cands, function(f) as.numeric(logLik(f))))]
          fit_k   <- cands[[best]]
          dropped <- best
        } else {
          fit_k <- tryCatch(
            do.call(gls, list(model = fixed_form, data = df_k, method = "REML")),
            error = function(e) {
              message(str_glue("  gls failed at k = {k}: {conditionMessage(e)}"))
              NULL
            }
          )
          if (is.null(fit_k)) {
            beta_hat[, k] <- 0
            if (M_a > 0) U_age_hat[, k] <- 0
            if (M_p > 0) U_pir_hat[, k] <- 0
            collapsed[[k]] <- c(if (M_a > 0) "age", if (M_p > 0) "pir")
            next
          }
          dropped <- c(if (M_a > 0) "age", if (M_p > 0) "pir")
        }
        message(str_glue(
          "    -> smooth(s) collapsed to linear: ",
          "{if (length(dropped)) paste(dropped, collapse = ', ') else 'none'}"
        ))
      }
      collapsed[[k]] <- dropped
      fits[[k]]      <- fit_k

      beta_hat[, k] <- (if (inherits(fit_k, "lme")) fixef(fit_k) else coef(fit_k))[beta_names]
      sigma2_E_diag[k] <- fit_k$sigma^2

      ## Random effects / variance components for the smooths that remain in fit_k
      ## (column order in ranef/VarCorr follows the order of the remaining smooths).
      keep_age <- M_a > 0 && !("age" %in% dropped)
      keep_pir <- M_p > 0 && !("pir" %in% dropped)
      if (M_a > 0) U_age_hat[, k] <- 0
      if (M_p > 0) U_pir_hat[, k] <- 0
      if (inherits(fit_k, "lme")) {
        re_vec <- as.numeric(ranef(fit_k)[1, ])
        vc     <- VarCorr(fit_k)
        pos    <- 0
        if (keep_age) {
          U_age_hat[, k]  <- re_vec[pos + seq_len(M_a)]
          sigma2_U_age[k] <- as.numeric(vc[pos + 1, "Variance"])
          pos <- pos + M_a
        }
        if (keep_pir) {
          U_pir_hat[, k]  <- re_vec[pos + seq_len(M_p)]
          sigma2_U_pir[k] <- as.numeric(vc[pos + 1, "Variance"])
        }
      }

      ## Collapsed smooths have sigma2_U = 0, i.e., lambda = Inf and zero penalized edf.
      edf_k_total <- A1
      if (M_a > 0) {
        lambda_a   <- sigma2_E_diag[k] / sigma2_U_age[k]
        edf_age[k] <- sum(phi_age / (phi_age + lambda_a))
        edf_k_total <- edf_k_total + edf_age[k]
      }
      if (M_p > 0) {
        lambda_p   <- sigma2_E_diag[k] / sigma2_U_pir[k]
        edf_pir[k] <- sum(phi_pir / (phi_pir + lambda_p))
        edf_k_total <- edf_k_total + edf_pir[k]
      }
      edf[k] <- edf_k_total
    }
  }

  R_mat <- Z - X_design %*% beta_hat
  if (M_a > 0) R_mat <- R_mat - W_age %*% U_age_hat
  if (M_p > 0) R_mat <- R_mat - W_pir %*% U_pir_hat

  edf_mean <- mean(edf)
  Sigma_E  <- crossprod(R_mat) / (N - edf_mean)

  list(
    beta_hat          = beta_hat,
    U_age_hat         = U_age_hat,
    U_pir_hat         = U_pir_hat,
    sigma2_U_age      = sigma2_U_age,
    sigma2_U_pir      = sigma2_U_pir,
    sigma2_E_diag     = sigma2_E_diag,
    Sigma_E           = Sigma_E,
    edf               = edf,
    edf_age           = edf_age,
    edf_pir           = edf_pir,
    lambda_age        = if (M_a > 0) sigma2_E_diag / sigma2_U_age else NULL,
    lambda_pir        = if (M_p > 0) sigma2_E_diag / sigma2_U_pir else NULL,
    X_design          = X_design,
    W_age             = W_age,
    W_pir             = W_pir,
    knots_age         = if (M_a > 0) attr(W_age, "knots_interior") else NULL,
    knots_pir         = if (M_p > 0) attr(W_pir, "knots_interior") else NULL,
    n_knots_age       = n_knots_age,
    n_knots_pir       = n_knots_pir,
    linear_covariates = linear_covariates,
    smooth_covariates = smooth_covariates,
    keep              = keep,
    fits              = fits,
    R_mat             = R_mat,
    collapsed         = collapsed
  )
}



## ==================== Sampling ====================

## Draw n samples of Z | (male, age, pir) from a fitted AMM: the conditional
## mean (amm_compute_conditional_mean, defined below under Interpretation)
## plus correlated residual noise drawn from the fitted Sigma_E. X is an
## n-row data frame with columns male, age, pir -- the sample_fun contract's
## covariate argument (see evaluate_generativity in src/utils.R). Mirrors
## sample_binary's contract in src/model/binary.R, generalized to the AMM's
## full covariate set instead of a single binary split.
sample_amm <- function(n, fit, X) {
  K <- ncol(fit$beta_hat)
  mu_hat <- t(sapply(seq_len(n), function(i) {
    amm_compute_conditional_mean(
      fit,
      list(male = X$male[i], age = X$age[i], pir = X$pir[i])
    )
  }))
  eps <- MASS::mvrnorm(n, mu = rep(0, K), Sigma = fit$Sigma_E)
  if (n == 1) eps <- matrix(eps, nrow = 1)
  draws <- mu_hat + eps
  asplit(draws, MARGIN = 1) |> lapply(as.numeric)
}



## ==================== Inference ====================

## Evaluate the smooth effect of covariate_name on a grid of values.
## Returns a (length(grid)) x K matrix of smooth (optionally total) effects.
## include_linear = TRUE adds the linear slope * grid to the smooth BLUP.
amm_evaluate_smooth <- function(fit_result, covariate_name, grid,
                                include_linear = TRUE) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  U_hat <- fit_result[[paste0("U_", covariate_name, "_hat")]]
  if (is.null(W_obj) || is.null(U_hat)) {
    stop(str_glue("Smooth for '{covariate_name}' not present in fit_result."))
  }
  all_knots <- attr(W_obj, "all_knots")
  Z_scaled  <- attr(W_obj, "Z_scaled")
  range_x   <- attr(W_obj, "range")

  grid_clipped <- pmax(range_x[1], pmin(range_x[2], grid))
  B_new <- splineDesign(knots = all_knots, x = grid_clipped, ord = 4,
                        outer.ok = TRUE)
  W_new <- B_new %*% Z_scaled
  out   <- W_new %*% U_hat
  if (include_linear && covariate_name %in% rownames(fit_result$beta_hat)) {
    beta_slope <- fit_result$beta_hat[covariate_name, ]
    out <- out + outer(grid, beta_slope)
  }
  out
}


## ==================== Diagnostics ====================

## Per-column summary of variance components, smoothing parameters, and EDF.
amm_summary <- function(fit_amm) {
  K_ <- ncol(fit_amm$beta_hat)
  data.frame(
    k            = seq_len(K_),
    sigma2_U_age = fit_amm$sigma2_U_age,
    sigma2_U_pir = fit_amm$sigma2_U_pir,
    sigma2_E     = fit_amm$sigma2_E_diag,
    lambda_age   = fit_amm$lambda_age,
    lambda_pir   = fit_amm$lambda_pir,
    edf          = fit_amm$edf
  )
}

## Shared arguments of the diagnostic plots below: path = NULL draws to the
## active device (so the caller can size it, e.g. with fig_png); a path opens
## a png at the original working size. main = "" omits a title and trims the
## top margin.

## Log10 smoothing-parameter plot (lambda by k for age and pir). The legend
## stays inside the axes, clear of both curves (see .legend_layout).
amm_plot_lambda <- function(fit_amm, path = NULL,
                            main = "AMM smoothing parameters by latent dimension",
                            lwd = 2, legend_cex = 1,
                            legend_loc = "topright") {
  K_amm <- ncol(fit_amm$beta_hat)
  l_age <- log10(fit_amm$lambda_age)
  l_pir <- log10(fit_amm$lambda_pir)
  if (!is.null(path)) {
    png(path, width = 960, height = 540, pointsize = 14)
    on.exit(dev.off(), add = TRUE)
    par(mar = c(4, 5, 3, 1))
  }
  xlab <- "k"
  ylab <- expression(log[10](lambda))
  xlim <- c(1, K_amm)
  ylim <- range(c(l_age, l_pir), finite = TRUE)
  leg_args <- list(legend = c("age", "pir"), col = c("darkblue", "darkorange"),
                   lwd = lwd, pch = c(19, 17), cex = legend_cex, bty = "n")
  ## Curves as objects for legend placement (densified between the k's)
  x_dense <- seq(1, K_amm, length.out = 200)
  dense   <- function(y) approx(seq_len(K_amm), y, xout = x_dense)$y
  objs <- list(list(x = x_dense, lo = dense(l_age), hi = dense(l_age)),
               list(x = x_dense, lo = dense(l_pir), hi = dense(l_pir)))

  old_mar <- par(mar = .trimmed_mar(par("mar"), main, xlab, ylab))
  on.exit(par(old_mar), add = TRUE, after = FALSE)
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc)
  .draw_frame(xlim, lay$ylim, main = main, xlab = xlab, ylab = ylab,
              ylim_data = ylim, x_at = seq_len(K_amm))
  lines(seq_len(K_amm), l_age, col = "darkblue",   lwd = lwd, type = "b", pch = 19)
  lines(seq_len(K_amm), l_pir, col = "darkorange", lwd = lwd, type = "b", pch = 17)
  do.call(legend, c(list(lay$pos), leg_args))
  invisible(NULL)
}

## Residual Q-Q panels (one panel per latent dimension k). ncol sets the
## panel columns (default: near-square grid); cex (if non-NULL) overrides the
## mfrow text shrink; point_cex sizes the points (pch 16, no outline, so
## translucent points show no halo).
amm_plot_resid_qq <- function(fit_amm, path = NULL,
                              col_train = rgb(0, 0, 0, 0.25),
                              ncol = NULL, cex = NULL,
                              mar = c(3, 3, 2, 1), point_cex = 0.25,
                              xlab = "Theoretical Quantiles",
                              ylab = "Sample Quantiles") {
  K_amm <- ncol(fit_amm$beta_hat)
  nc    <- ncol %||% ceiling(sqrt(K_amm))
  nr    <- ceiling(K_amm / nc)
  if (!is.null(path)) {
    png(path, width = 300 * nc, height = 300 * nr, pointsize = 12)
    on.exit(dev.off(), add = TRUE)
  }
  old_par <- par(mfrow = c(nr, nc))
  if (!is.null(cex)) old_par <- c(old_par, par(cex = cex))
  on.exit(par(old_par), add = TRUE, after = FALSE)
  for (k in seq_len(K_amm)) {
    main_k <- paste0("k = ", k)
    par(mar = .trimmed_mar(mar, main_k, xlab, ylab))
    qqnorm(fit_amm$R_mat[, k], main = main_k, xlab = xlab, ylab = ylab,
           pch = 16, cex = point_cex, col = col_train)
    qqline(fit_amm$R_mat[, k], col = "red")
  }
  invisible(NULL)
}

## Heatmap of the residual covariance matrix Sigma_E (lower triangle
## annotated). cell_cex sizes the cell values.
amm_plot_sigma_e <- function(fit_amm, path = NULL,
                             main = expression(hat(Sigma)[E]),
                             cell_cex = 0.6) {
  K_amm   <- ncol(fit_amm$beta_hat)
  zmax    <- max(abs(fit_amm$Sigma_E))
  breaks  <- seq(-zmax, zmax, length.out = 101)
  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF000000"))(100)

  if (!is.null(path)) {
    png(path, width = 800, height = 720, pointsize = 14)
    on.exit(dev.off(), add = TRUE)
    par(mar = c(4, 4, 3, 5))
  }
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "", ""))
  on.exit(par(old_mar), add = TRUE, after = FALSE)
  image(
    1:K_amm, 1:K_amm,
    t(fit_amm$Sigma_E)[, K_amm:1],
    zlim = c(-zmax, zmax),
    col  = col_pal, breaks = breaks,
    axes = FALSE, xlab = "", ylab = "",
    main = main
  )
  axis(1, at = 1:K_amm, labels = 1:K_amm, las = 1, tick = FALSE)
  axis(2, at = 1:K_amm, labels = K_amm:1, las = 1, tick = FALSE)
  for (i in 1:K_amm) for (j in 1:K_amm) if (i >= j) {
    text(j, K_amm - i + 1, sprintf("%.2f", fit_amm$Sigma_E[i, j]), cex = cell_cex)
  }
  invisible(NULL)
}

## Correlation heatmap of the random-effect BLUPs U_hat (M x K matrix) to
## diagnose the diagonal-covariance assumption on the random effects, beside
## a bar plot of the component variances. cell_cex sizes the cell values;
## bar_main titles the bar plot ("" omits it).
amm_plot_re_cov <- function(U_hat, K_, title_expr, path = NULL,
                            cell_cex = 0.6,
                            bar_main = "Component variances") {
  Sigma_U <- cov(U_hat)
  d       <- sqrt(diag(Sigma_U))
  Corr_U  <- Sigma_U / tcrossprod(d)
  Corr_U  <- pmin(pmax(Corr_U, -1), 1)   # rounding can push |r| past 1 (off the colour scale)

  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF0000"))(100)
  breaks  <- seq(-1, 1, length.out = 101)

  if (!is.null(path)) {
    png(path, width = 1100, height = 720, pointsize = 14)
    on.exit(dev.off(), add = TRUE)
  }
  old_par <- par(c("mar", "cex"))
  on.exit(par(old_par), add = TRUE, after = FALSE)
  cex_now <- par("cex")
  layout(matrix(c(1, 2), nrow = 1), widths = c(2, 1))
  par(cex = cex_now)   # layout() resets cex; keep the caller's base size

  base_mar <- if (is.null(path)) old_par$mar else c(4, 4, 3, 5)
  par(mar = .trimmed_mar(base_mar, title_expr, "", ""))
  image(1:K_, 1:K_, t(Corr_U)[, K_:1],
        zlim = c(-1, 1), col = col_pal, breaks = breaks,
        axes = FALSE, xlab = "", ylab = "", main = title_expr)
  axis(1, at = 1:K_, labels = 1:K_,    las = 1, tick = FALSE)
  axis(2, at = 1:K_, labels = K_:1,    las = 1, tick = FALSE)
  for (i in 1:K_) for (j in 1:K_) if (i >= j)
    text(j, K_ - i + 1, sprintf("%.2f", Corr_U[i, j]), cex = cell_cex)

  bar_ylab <- expression(log[10]~hat(sigma)[U[k]]^2)
  bar_base <- if (is.null(path)) old_par$mar else c(4, 5, 3, 2)
  if (is.null(path)) bar_base[2] <- bar_base[2] + 0.6   # room for the tall label
  par(mar = .trimmed_mar(bar_base, bar_main, "k", bar_ylab))
  barplot(log10(diag(Sigma_U)), names.arg = 1:K_, horiz = FALSE,
          xlab = "k", ylab = bar_ylab, main = bar_main)
  invisible(NULL)
}



## ==================== Interpretation ====================

## ----- Internal helpers -----

## Simulate one synthetic Z from the fitted AMM: mu_hat + E*, E* ~ N(0, Sigma_E).
.amm_simulate_one <- function(fit_amm) {
  N_kept <- sum(fit_amm$keep)
  K_     <- ncol(fit_amm$beta_hat)
  mu     <- fit_amm$X_design %*% fit_amm$beta_hat
  if (!is.null(fit_amm$U_age_hat)) mu <- mu + fit_amm$W_age %*% fit_amm$U_age_hat
  if (!is.null(fit_amm$U_pir_hat)) mu <- mu + fit_amm$W_pir %*% fit_amm$U_pir_hat
  L <- chol(fit_amm$Sigma_E)
  G <- matrix(rnorm(N_kept * K_), N_kept, K_)
  mu + G %*% L
}

## Wrap per-replicate coefficient triple into a fit-like object so that
## amm_evaluate_smooth and amm_compute_conditional_mean work unchanged on it.
## Design matrices and basis attributes are shared by reference from fit_amm.
.amm_make_replicate <- function(fit_amm, beta_hat_r, U_age_hat_r, U_pir_hat_r) {
  list(
    beta_hat          = beta_hat_r,
    U_age_hat         = U_age_hat_r,
    U_pir_hat         = U_pir_hat_r,
    W_age             = fit_amm$W_age,
    W_pir             = fit_amm$W_pir,
    smooth_covariates = fit_amm$smooth_covariates,
    linear_covariates = fit_amm$linear_covariates
  )
}

## Sequential blue-gray-red palette indexed by x_values (optionally centered
## at ref for contrast plots).
.amm_x_palette <- function(x_values, ref = NULL) {
  ramp <- colorRampPalette(c("blue", "gray60", "red"))(101)
  if (is.null(ref)) {
    r   <- rank(x_values, ties.method = "min")
    idx <- round((r - 1) / max(r - 1, 1) * 100) + 1
  } else {
    rng  <- max(abs(x_values - ref))
    if (rng == 0) rng <- 1
    norm <- (x_values - ref) / (2 * rng) + 0.5
    idx  <- pmin(pmax(round(norm * 100) + 1, 1), 101)
  }
  ramp[idx]
}

## Default covariate values for decoded-space effect plots.
.amm_default_x_values <- function(covariate_name, faceted = FALSE) {
  if (covariate_name == "age") {
    if (faceted) c(10, 30, 50, 70) else c(10, 20, 30, 40, 50, 60, 70, 80)
  } else if (covariate_name == "pir") {
    if (faceted) c(0.25, 0.5, 2, 4) else c(0.25, 0.5, 1, 2, 4, 5)
  } else {
    stop(str_glue("No default x_values for '{covariate_name}'."))
  }
}

## Build a y_grid from the range of a list of Qi vectors.
.amm_default_y_grid <- function(Qi_list) {
  rng <- range(unlist(Qi_list))
  sort(unique(c(rng[1], unlist(Qi_list), rng[2])))
}

## Legend for the vertical threshold lines (dashed) drawn by the CDF and
## survival plots. A no-op when threshold_lines is NULL.
.amm_legend_thresholds <- function(threshold_lines, where = "topright") {
  if (is.null(threshold_lines)) return(invisible(NULL))
  legend(where,
         legend = sapply(threshold_lines, `[[`, "label"),
         col    = sapply(threshold_lines, `[[`, "col"),
         lwd    = 1.5, lty = 2, bty = "n")
}

## ----- Latent-space conditional mean -----

## Conditional latent mean z = E[Z | covariate_values] from the fitted AMM.
## covariate_values is a named list with elements male, age, pir (scalars).
amm_compute_conditional_mean <- function(fit_result, covariate_values) {
  beta <- fit_result$beta_hat
  out  <- beta["(Intercept)", , drop = TRUE]

  if ("male" %in% rownames(beta) && !is.null(covariate_values$male)) {
    out <- out + covariate_values$male * beta["male", , drop = TRUE]
  }
  if ("age" %in% fit_result$smooth_covariates && !is.null(covariate_values$age)) {
    out <- out + as.numeric(amm_evaluate_smooth(fit_result, "age",
                                               covariate_values$age,
                                               include_linear = TRUE))
  } else if ("age" %in% rownames(beta) && !is.null(covariate_values$age)) {
    out <- out + covariate_values$age * beta["age", , drop = TRUE]
  }
  if ("pir" %in% fit_result$smooth_covariates && !is.null(covariate_values$pir)) {
    out <- out + as.numeric(amm_evaluate_smooth(fit_result, "pir",
                                               covariate_values$pir,
                                               include_linear = TRUE))
  } else if ("pir" %in% rownames(beta) && !is.null(covariate_values$pir)) {
    out <- out + covariate_values$pir * beta["pir", , drop = TRUE]
  }
  out
}

## ----- Parametric bootstrap (interpretation) -----

## B parametric-bootstrap replicates of the fitted AMM. Returns a length-B
## list of fit-like objects (via .amm_make_replicate). Failed replicates are
## filled with zero coefficients.
amm_boot_interpret <- function(amm, covariates_kept, B,
                                n_cores = 1, seed = 12345) {
  set.seed(seed)
  boot_one <- function(b) {
    Z_b   <- .amm_simulate_one(amm)
    fit_b <- tryCatch(
      fit_amm(Z_b, covariates_kept,
              n_knots_age = amm$n_knots_age,
              n_knots_pir = amm$n_knots_pir),
      error = function(e) NULL
    )
    if (is.null(fit_b)) return(NULL)
    list(
      beta_hat  = fit_b$beta_hat,
      U_age_hat = fit_b$U_age_hat,
      U_pir_hat = fit_b$U_pir_hat
    )
  }

  reps <- if (n_cores > 1) {
    old_plan <- future::plan(future::multisession, workers = n_cores)
    on.exit(future::plan(old_plan), add = TRUE)
    future.apply::future_lapply(seq_len(B), boot_one, future.seed = TRUE)
  } else {
    lapply(seq_len(B), boot_one)
  }

  zero_beta  <- matrix(0, nrow = nrow(amm$beta_hat),
                       ncol = ncol(amm$beta_hat),
                       dimnames = dimnames(amm$beta_hat))
  zero_U_age <- if (!is.null(amm$U_age_hat))
                  matrix(0, nrow = nrow(amm$U_age_hat),
                            ncol = ncol(amm$U_age_hat)) else NULL
  zero_U_pir <- if (!is.null(amm$U_pir_hat))
                  matrix(0, nrow = nrow(amm$U_pir_hat),
                            ncol = ncol(amm$U_pir_hat)) else NULL

  n_failed <- sum(vapply(reps, is.null, logical(1)))
  if (n_failed > 0) {
    message(str_glue("  amm_boot_interpret: {n_failed} of {B} replicates failed; ",
                     "filled with zero coefficients."))
  }

  lapply(reps, function(r) {
    if (is.null(r)) {
      .amm_make_replicate(amm, zero_beta, zero_U_age, zero_U_pir)
    } else {
      .amm_make_replicate(amm, r$beta_hat, r$U_age_hat, r$U_pir_hat)
    }
  })
}

## B x K matrix of bootstrap conditional latent means at covariate_values.
amm_replicate_z_at <- function(boot_reps, covariate_values) {
  do.call(rbind, lapply(boot_reps, function(rep_) {
    as.numeric(amm_compute_conditional_mean(rep_, covariate_values))
  }))
}

## J x K x B array of bootstrap smooth-curve evaluations on x_grid.
amm_replicate_smooth_at <- function(boot_reps, covariate_name, x_grid,
                                     include_linear = TRUE) {
  B_ <- length(boot_reps)
  J_ <- length(x_grid)
  K_ <- ncol(boot_reps[[1]]$beta_hat)
  out <- array(NA_real_, dim = c(J_, K_, B_))
  for (b in seq_len(B_)) {
    out[, , b] <- amm_evaluate_smooth(boot_reps[[b]], covariate_name,
                                      x_grid, include_linear = include_linear)
  }
  out
}

## ----- Decoded-space band computation -----

## Joint (sup-t over the grid) (1 - alpha) band around `center` from a G x B
## matrix of bootstrap replicates (rows = grid points, columns = replicates).
## Same construction as the group-comparison bands (.boot_band in
## src/model/utils.R).
.amm_joint_band <- function(mat_gb, center, alpha) {
  .boot_band(t(mat_gb), center, alpha, band_type = "joint")
}

## Ji x 3 matrix (mean_hat, lo, hi) of the decoded conditional QF at
## covariate_values, with a joint (over p) bootstrap band.
amm_compute_band_qf <- function(fit_result, boot_reps, covariate_values,
                                 alpha, Ji, pipeline) {
  z_hat  <- amm_compute_conditional_mean(fit_result, covariate_values)
  Qi_hat <- .decode_z_vec(pipeline, z_hat, Ji)

  Z_boot  <- amm_replicate_z_at(boot_reps, covariate_values)
  B_      <- nrow(Z_boot)
  Qi_boot <- vapply(seq_len(B_),
                    function(b) .decode_z_vec(pipeline, Z_boot[b, ], Ji),
                    numeric(Ji))

  bd <- .amm_joint_band(Qi_boot, Qi_hat, alpha)
  cbind(mean_hat = Qi_hat, lo = bd$lo, hi = bd$hi)
}

## Ji x 3 matrix (mean_hat, lo, hi) of the decoded contrast QF
## Qi(covariate_name = x_value) - Qi(reference), with a joint (over p)
## within-replicate band.
amm_compute_band_qf_contrast <- function(fit_result, boot_reps,
                                          covariate_name, x_value,
                                          alpha, Ji, pipeline,
                                          age_ref = 40, pir_ref = 1,
                                          male_ref = 0) {
  cov_x   <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cov_ref <- cov_x
  cov_x[[covariate_name]]   <- x_value
  cov_ref[[covariate_name]] <- switch(covariate_name,
                                      age = age_ref, pir = pir_ref)

  z_x   <- amm_compute_conditional_mean(fit_result, cov_x)
  z_ref <- amm_compute_conditional_mean(fit_result, cov_ref)
  Qi_x   <- .decode_z_vec(pipeline, z_x,   Ji)
  Qi_ref <- .decode_z_vec(pipeline, z_ref, Ji)
  Qi_hat_diff <- Qi_x - Qi_ref

  Zb_x   <- amm_replicate_z_at(boot_reps, cov_x)
  Zb_ref <- amm_replicate_z_at(boot_reps, cov_ref)
  B_     <- nrow(Zb_x)
  Qi_diff_boot <- vapply(seq_len(B_), function(b) {
    .decode_z_vec(pipeline, Zb_x[b, ], Ji) -
    .decode_z_vec(pipeline, Zb_ref[b, ], Ji)
  }, numeric(Ji))

  bd <- .amm_joint_band(Qi_diff_boot, Qi_hat_diff, alpha)
  cbind(mean_hat = Qi_hat_diff, lo = bd$lo, hi = bd$hi)
}

## ----- Latent-space effect curve plots -----

## Per-k smooth effect curves in z-space (one panel per latent dimension).
amm_plot_latent_effects <- function(fit_result, covariate_name,
                                     x_grid = NULL, n_grid = 100,
                                     band = FALSE, alpha = 0.05,
                                     boot_reps = NULL,
                                     covariates_df = NULL,
                                     ncol = NULL,
                                     panel_mains = NULL,
                                     xlab = covariate_name, ylab = "f(x)",
                                     cex = NULL, mar = c(3, 3, 2, 1),
                                     mgp = c(1.8, 0.6, 0),
                                     lwd = 2, main = "") {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  if (is.null(W_obj))
    stop(str_glue("No smooth for '{covariate_name}' in fit_result."))
  range_x <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(range_x[1], range_x[2], length.out = n_grid)

  K_    <- ncol(fit_result$beta_hat)
  f_mat <- amm_evaluate_smooth(fit_result, covariate_name, x_grid,
                                include_linear = TRUE)
  edf_k    <- fit_result[[paste0("edf_", covariate_name)]]
  beta_lin <- fit_result$beta_hat[covariate_name, ]

  band_lo <- band_hi <- NULL
  if (isTRUE(band) && !is.null(boot_reps)) {
    sm_arr <- amm_replicate_smooth_at(boot_reps, covariate_name, x_grid,
                                      include_linear = TRUE)
    band_lo <- band_hi <- matrix(NA_real_, nrow = length(x_grid), ncol = K_)
    for (k in seq_len(K_)) {
      bd <- .amm_joint_band(sm_arr[, k, ], f_mat[, k], alpha)
      band_lo[, k] <- bd$lo
      band_hi[, k] <- bd$hi
    }
  }

  nc <- ncol %||% ceiling(sqrt(K_))
  nr <- ceiling(K_ / nc)
  ## Display: mgp = NULL keeps the current axis spacing (e.g. set by fig_par);
  ## cex (if non-NULL) overrides the mfrow text shrink; margins are trimmed for
  ## omitted titles / axis labels; main is an outer title (none when blank)
  old_par <- par(mfrow = c(nr, nc), oma = c(0, 0, if (.is_blank(main)) 0 else 2, 0))
  if (!is.null(mgp)) old_par <- c(old_par, par(mgp = mgp))
  if (!is.null(cex)) old_par <- c(old_par, par(cex = cex))
  on.exit(par(old_par), add = TRUE)
  if (is.null(panel_mains)) {
    panel_mains <- sprintf("k = %d (edf = %.2f)", seq_len(K_), edf_k)
  }

  rug_x <- if (!is.null(covariates_df))
              covariates_df[[covariate_name]][fit_result$keep] else NULL

  for (k in seq_len(K_)) {
    y_vals <- c(f_mat[, k], beta_lin[k] * x_grid, band_lo[, k], band_hi[, k])
    y_lim  <- range(y_vals, finite = TRUE)
    par(mar = .trimmed_mar(mar, panel_mains[k], xlab, ylab))
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = xlab, ylab = ylab, main = panel_mains[k])
    if (!is.null(band_lo)) draw_band(x_grid, band_lo[, k], band_hi[, k])
    abline(h = 0, col = "gray85", lty = 1)
    lines(x_grid, beta_lin[k] * x_grid, col = "gray40", lwd = 0.75 * lwd, lty = 2)
    lines(x_grid, f_mat[, k], col = "black", lwd = lwd)
    if (!is.null(rug_x)) rug(rug_x, col = rgb(0, 0, 0, 0.2))
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}

## ----- Decoded QF family -----

## Draw overlaid curves (one per covariate value, coloured by `colors`) on a
## shared x grid, with optional vertical reference lines (quantile levels or
## thresholds; list(label=, value=, col=) entries, also listed in the legend)
## and an optional zero line. The legend is placed inside the axes clear of
## every curve and line (see .legend_layout in src/model/utils.R), and the
## margins drop the space for an omitted title. legend_title (e.g. "Age")
## lists just the covariate values under that title; NULL labels each entry
## "covariate = x".
.amm_draw_overlay <- function(x, curves, colors, x_values, covariate_name,
                              xlab, ylab, main, ylim,
                              vlines = NULL, zero_line = FALSE,
                              legend_loc = "topleft", legend_title = NULL,
                              legend_ncol = 1, legend_cex = 1,
                              lwd = 2, lwd_vline = 1.5,
                              vline_labels = c("legend", "top")) {
  vline_labels <- match.arg(vline_labels)
  v_vals <- if (!is.null(vlines)) sapply(vlines, `[[`, "value") else NULL
  v_cols <- if (!is.null(vlines)) sapply(vlines, `[[`, "col")   else NULL
  v_labs <- if (!is.null(vlines)) sapply(vlines, `[[`, "label") else NULL
  val_labels <- if (is.null(legend_title)) {
    sprintf("%s = %g", covariate_name, x_values)
  } else {
    sprintf("%g", x_values)
  }
  ## Reference lines are listed in the legend, or labelled above the frame
  v_in_leg <- vline_labels == "legend"
  leg_v     <- if (v_in_leg) v_labs else NULL
  leg_v_col <- if (v_in_leg) v_cols else NULL
  leg_args <- list(
    legend = c(val_labels, leg_v),
    col    = c(colors, leg_v_col),
    lwd    = c(rep(lwd, length(colors)), rep(lwd_vline, length(leg_v))),
    lty    = c(rep(1, length(colors)), rep(2, length(leg_v))),
    title  = legend_title,
    ncol   = legend_ncol,
    cex    = legend_cex,
    bty    = "n"
  )

  xlim <- range(x)
  v_y  <- ylim + c(-1, 1) * 0.04 * diff(ylim)   # full height before any extension
  objs <- c(
    lapply(curves, function(y) list(x = x, lo = y, hi = y)),
    if (zero_line) list(list(x = x, lo = 0 * x, hi = 0 * x)),
    .threshold_objs(v_vals, v_y[1], v_y[2])
  )

  old_mar <- par(mar = .trimmed_mar(par("mar"), main, xlab, ylab))
  on.exit(par(old_mar), add = TRUE)
  ## Labels above the frame: widen the top margin for them, and add any legend
  ## headroom at the bottom so the lines still reach their labels
  top_lab <- !v_in_leg && length(v_vals) > 0
  if (top_lab) {
    tl <- .threshold_label_layout(v_vals, v_labs, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))
  .draw_frame(xlim, lay$ylim, main = main, xlab = xlab, ylab = ylab,
              ylim_data = ylim,
              main_line = if (top_lab) tl$main_line else NULL)
  if (zero_line) abline(h = 0, col = "red", lty = 3)
  if (length(v_vals)) {
    .draw_thresholds(v_vals, v_cols, v_y[1], v_y[2], lwd = lwd_vline)
  }
  for (i in seq_along(curves)) lines(x, curves[[i]], col = colors[i], lwd = lwd)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(v_vals, v_labs, v_cols, tl)
  invisible(NULL)
}

## Shared arguments of the overlaid QF / CDF / survival plots below:
##   main        plot title ("" omits it and trims the top margin)
##   legend_loc  preferred legend corner (moved / given headroom if needed)
##   legend_title, legend_ncol, legend_cex  see .amm_draw_overlay
##   lwd         curve width (NULL: the original width)

## Overlaid conditional QFs at a range of x_values for covariate_name.
amm_plot_conditional_qf <- function(fit_result, covariate_name, pipeline,
                                     x_values = NULL,
                                     age_ref = 40, pir_ref = 1, male_ref = 0,
                                     Ji = NULL,
                                     quantile_lines = NULL,
                                     main = sprintf("Conditional QFs by %s",
                                                    covariate_name),
                                     legend_loc = "topleft",
                                     legend_title = NULL, legend_ncol = 1,
                                     legend_cex = 1, lwd = NULL,
                                     line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  pi_grid <- pi_grid_fun(Ji)

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  .amm_draw_overlay(pi_grid, Qi_list, .amm_x_palette(x_values), x_values,
                    covariate_name, xlab = "p", ylab = "Q(p)", main = main,
                    ylim = range(unlist(Qi_list), finite = TRUE),
                    vlines = quantile_lines, legend_loc = legend_loc,
                    legend_title = legend_title, legend_ncol = legend_ncol,
                    legend_cex = legend_cex, lwd = lwd %||% 4,
                    vline_labels = line_labels)
}

## Overlaid contrast QFs Qi(x) - Qi(ref) for a range of x_values.
amm_plot_contrast_qf <- function(fit_result, covariate_name, pipeline,
                                  x_values = NULL,
                                  age_ref = 40, pir_ref = 1, male_ref = 0,
                                  Ji = NULL,
                                  quantile_lines = NULL,
                                  main = NULL,
                                  legend_loc = "topleft",
                                  legend_title = NULL, legend_ncol = 1,
                                  legend_cex = 1, lwd = NULL,
                                  line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  pi_grid <- pi_grid_fun(Ji)
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  if (is.null(main)) main <- sprintf("QF contrasts vs %s = %g",
                                     covariate_name, ref_val)

  Qi_ref <- {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- ref_val
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  }
  contrasts <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji) - Qi_ref
  })
  .amm_draw_overlay(pi_grid, contrasts, .amm_x_palette(x_values, ref = ref_val),
                    x_values, covariate_name, xlab = "p",
                    ylab = "Q(p) contrast", main = main,
                    ylim = range(unlist(contrasts), 0, finite = TRUE),
                    vlines = quantile_lines, zero_line = TRUE,
                    legend_loc = legend_loc, legend_title = legend_title,
                    legend_ncol = legend_ncol, legend_cex = legend_cex,
                    lwd = lwd %||% 2,
                    vline_labels = line_labels)
}

## Overlaid conditional CDFs.
amm_plot_conditional_cdf <- function(fit_result, covariate_name, pipeline,
                                      x_values = NULL, y_grid = NULL,
                                      age_ref = 40, pir_ref = 1, male_ref = 0,
                                      Ji = NULL,
                                      threshold_lines = NULL,
                                      main = sprintf("Conditional CDFs by %s",
                                                     covariate_name),
                                      legend_loc = "topleft",
                                      legend_title = NULL, legend_ncol = 1,
                                      legend_cex = 1, lwd = NULL,
                                      line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .amm_default_y_grid(Qi_list)
  F_list <- lapply(Qi_list, qi_to_cdf, y_grid = y_grid)
  .amm_draw_overlay(y_grid, F_list, .amm_x_palette(x_values), x_values,
                    covariate_name, xlab = "y", ylab = "F(y)", main = main,
                    ylim = c(0, 1), vlines = threshold_lines,
                    legend_loc = legend_loc, legend_title = legend_title,
                    legend_ncol = legend_ncol, legend_cex = legend_cex,
                    lwd = lwd %||% 2,
                    vline_labels = line_labels)
}

## Overlaid contrast CDFs F(y|x) - F(y|ref).
amm_plot_contrast_cdf <- function(fit_result, covariate_name, pipeline,
                                   x_values = NULL, y_grid = NULL,
                                   age_ref = 40, pir_ref = 1, male_ref = 0,
                                   Ji = NULL,
                                   threshold_lines = NULL,
                                   main = NULL,
                                   legend_loc = "topright",
                                   legend_title = NULL, legend_ncol = 1,
                                   legend_cex = 1, lwd = NULL,
                                   line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  if (is.null(main)) main <- sprintf("CDF contrasts vs %s = %g",
                                     covariate_name, ref_val)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref <- .decode_z_vec(pipeline,
                          amm_compute_conditional_mean(fit_result, cv_ref), Ji)

  Qi_list <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .amm_default_y_grid(c(list(Qi_ref), Qi_list))
  F_ref  <- qi_to_cdf(Qi_ref, y_grid)
  F_diff <- lapply(Qi_list, function(Qi) qi_to_cdf(Qi, y_grid) - F_ref)
  .amm_draw_overlay(y_grid, F_diff, .amm_x_palette(x_values, ref = ref_val),
                    x_values, covariate_name, xlab = "y",
                    ylab = "F(y) contrast", main = main,
                    ylim = range(unlist(F_diff), 0, finite = TRUE),
                    vlines = threshold_lines, zero_line = TRUE,
                    legend_loc = legend_loc, legend_title = legend_title,
                    legend_ncol = legend_ncol, legend_cex = legend_cex,
                    lwd = lwd %||% 2,
                    vline_labels = line_labels)
}

## Overlaid conditional survival functions.
amm_plot_conditional_survival <- function(fit_result, covariate_name, pipeline,
                                           x_values = NULL, y_grid = NULL,
                                           age_ref = 40, pir_ref = 1, male_ref = 0,
                                           Ji = NULL,
                                           threshold_lines = NULL,
                                           main = sprintf(
                                             "Conditional survival functions by %s",
                                             covariate_name),
                                           legend_loc = "topright",
                                           legend_title = NULL, legend_ncol = 1,
                                           legend_cex = 1, lwd = NULL,
                                           line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .amm_default_y_grid(Qi_list)
  S_list <- lapply(Qi_list, .amm_qi_to_survival, y_grid = y_grid)
  .amm_draw_overlay(y_grid, S_list, .amm_x_palette(x_values), x_values,
                    covariate_name, xlab = "y", ylab = "1 - F(y)", main = main,
                    ylim = c(0, 1), vlines = threshold_lines,
                    legend_loc = legend_loc, legend_title = legend_title,
                    legend_ncol = legend_ncol, legend_cex = legend_cex,
                    lwd = lwd %||% 2,
                    vline_labels = line_labels)
}

## Overlaid contrast survival functions S(y|x) - S(y|ref).
amm_plot_contrast_survival <- function(fit_result, covariate_name, pipeline,
                                        x_values = NULL, y_grid = NULL,
                                        age_ref = 40, pir_ref = 1, male_ref = 0,
                                        Ji = NULL,
                                        threshold_lines = NULL,
                                        main = NULL,
                                        legend_loc = "topright",
                                        legend_title = NULL, legend_ncol = 1,
                                        legend_cex = 1, lwd = NULL,
                                        line_labels = c("legend", "top")) {
  line_labels <- match.arg(line_labels)
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  if (is.null(main)) main <- sprintf("Survival function contrasts vs %s = %g",
                                     covariate_name, ref_val)

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref <- .decode_z_vec(pipeline,
                          amm_compute_conditional_mean(fit_result, cv_ref), Ji)

  Qi_list <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .amm_default_y_grid(c(list(Qi_ref), Qi_list))
  S_ref  <- .amm_qi_to_survival(Qi_ref, y_grid)
  S_diff <- lapply(Qi_list, function(Qi) .amm_qi_to_survival(Qi, y_grid) - S_ref)
  .amm_draw_overlay(y_grid, S_diff, .amm_x_palette(x_values, ref = ref_val),
                    x_values, covariate_name, xlab = "y",
                    ylab = "1 - F(y) contrast", main = main,
                    ylim = range(unlist(S_diff), 0, finite = TRUE),
                    vlines = threshold_lines, zero_line = TRUE,
                    legend_loc = legend_loc, legend_title = legend_title,
                    legend_ncol = legend_ncol, legend_cex = legend_cex,
                    lwd = lwd %||% 2,
                    vline_labels = line_labels)
}


## Faceted conditional QFs with bootstrap bands (one panel per x_value).
amm_plot_conditional_qf_faceted <- function(fit_result, covariate_name, pipeline,
                                             x_values = NULL,
                                             age_ref = 40, pir_ref = 1, male_ref = 0,
                                             Ji = NULL, alpha = 0.05,
                                             boot_reps = NULL,
                                             quantile_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  pi_grid <- pi_grid_fun(Ji)

  q_vals <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL

  bands <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    amm_compute_band_qf(fit_result, boot_reps, cv, alpha, Ji, pipeline)
  })
  y_lim <- range(sapply(bands, function(b) range(b)))

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p", ylab = "Q(p)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(pi_grid, bands[[i]][, "lo"], bands[[i]][, "hi"])
    if (!is.null(quantile_lines)) {
      abline(v = q_vals, col = q_cols, lty = 2, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}


## Faceted contrast QFs with bootstrap bands (one panel per x_value).
amm_plot_contrast_qf_faceted <- function(fit_result, covariate_name, pipeline,
                                          x_values = NULL,
                                          age_ref = 40, pir_ref = 1, male_ref = 0,
                                          Ji = NULL, alpha = 0.05,
                                          boot_reps = NULL,
                                          quantile_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  pi_grid <- pi_grid_fun(Ji)

  q_vals <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL

  bands <- lapply(x_values, function(x) {
    amm_compute_band_qf_contrast(fit_result, boot_reps, covariate_name, x,
                                 alpha, Ji, pipeline,
                                 age_ref = age_ref, pir_ref = pir_ref,
                                 male_ref = male_ref)
  })
  y_lim <- range(sapply(bands, function(b) range(b)), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p",
         ylab = "Q(p) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(pi_grid, bands[[i]][, "lo"], bands[[i]][, "hi"])
    abline(h = 0, col = "red", lty = 3)
    if (!is.null(quantile_lines)) {
      abline(v = q_vals, col = q_cols, lty = 2, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}

## ----- Decoded CDF family -----


## Faceted conditional CDFs with bootstrap bands.
amm_plot_conditional_cdf_faceted <- function(fit_result, covariate_name, pipeline,
                                              x_values = NULL, y_grid = NULL,
                                              age_ref = 40, pir_ref = 1, male_ref = 0,
                                              Ji = NULL,
                                              threshold_lines = NULL,
                                              alpha = 0.05,
                                              boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  thr_vals <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL

  Qi_bands <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    amm_compute_band_qf(fit_result, boot_reps, cv, alpha, Ji, pipeline)
  })
  if (is.null(y_grid))
    y_grid <- .amm_default_y_grid(lapply(Qi_bands, function(b) b[, "mean_hat"]))

  F_means <- lapply(Qi_bands, function(b) qi_to_cdf(b[, "mean_hat"], y_grid))
  cv_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    cv
  })
  Z_boots <- lapply(cv_list, function(cv) amm_replicate_z_at(boot_reps, cv))
  F_bands <- lapply(seq_along(x_values), function(i) {
    Zb <- Z_boots[[i]]
    B_ <- nrow(Zb)
    Fb <- vapply(seq_len(B_),
                 function(b) qi_to_cdf(.decode_z_vec(pipeline, Zb[b, ], Ji),
                                       y_grid),
                 numeric(length(y_grid)))
    .amm_joint_band(Fb, F_means[[i]], alpha)
  })

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
         xlab = "y", ylab = "F(y)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(y_grid, F_bands[[i]]$lo, F_bands[[i]]$hi)
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 2, lwd = 1.5)
    }
    lines(y_grid, F_means[[i]],    col = "black", lwd = 4)
    if (i == 1) .amm_legend_thresholds(threshold_lines, "topleft")
  }
  invisible(NULL)
}


## Faceted contrast CDFs with bootstrap bands.
amm_plot_contrast_cdf_faceted <- function(fit_result, covariate_name, pipeline,
                                           x_values = NULL, y_grid = NULL,
                                           age_ref = 40, pir_ref = 1, male_ref = 0,
                                           Ji = NULL,
                                           threshold_lines = NULL,
                                           alpha = 0.05,
                                           boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  ref_val  <- switch(covariate_name, age = age_ref, pir = pir_ref)
  thr_vals <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref_pt <- .decode_z_vec(pipeline,
                             amm_compute_conditional_mean(fit_result, cv_ref), Ji)

  Qi_x_pt <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid))
    y_grid <- .amm_default_y_grid(c(list(Qi_ref_pt), Qi_x_pt))
  F_ref_pt     <- qi_to_cdf(Qi_ref_pt, y_grid)
  Z_ref_boot   <- amm_replicate_z_at(boot_reps, cv_ref)
  B_           <- nrow(Z_ref_boot)
  F_diff_means <- lapply(Qi_x_pt, function(Qi) qi_to_cdf(Qi, y_grid) - F_ref_pt)

  band_list <- lapply(seq_along(x_values), function(i) {
    cv <- cv_ref; cv[[covariate_name]] <- x_values[i]
    Zb_x <- amm_replicate_z_at(boot_reps, cv)
    Fd_b <- vapply(seq_len(B_), function(b) {
      qi_to_cdf(.decode_z_vec(pipeline, Zb_x[b, ],      Ji), y_grid) -
      qi_to_cdf(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji), y_grid)
    }, numeric(length(y_grid)))
    .amm_joint_band(Fd_b, F_diff_means[[i]], alpha)
  })
  y_lim <- range(unlist(band_list), unlist(F_diff_means), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = y_lim,
         xlab = "y", ylab = "F(y) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(y_grid, band_list[[i]]$lo, band_list[[i]]$hi)
    abline(h = 0, col = "red", lty = 3)
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 2, lwd = 1.5)
    }
    lines(y_grid, F_diff_means[[i]], col = "black", lwd = 4)
    if (i == 1) .amm_legend_thresholds(threshold_lines, "topright")
  }
  invisible(NULL)
}

## ----- Decoded survival function family -----

## Survival function 1 - F(y) = 1 - F(y) of the decoded distribution with quantile
## function Qi, evaluated on y_grid.
.amm_qi_to_survival <- function(Qi, y_grid) 1 - qi_to_cdf(Qi, y_grid)


## Faceted conditional survival functions with bootstrap bands.
amm_plot_conditional_survival_faceted <- function(fit_result, covariate_name, pipeline,
                                                   x_values = NULL, y_grid = NULL,
                                                   age_ref = 40, pir_ref = 1, male_ref = 0,
                                                   Ji = NULL,
                                                   threshold_lines = NULL,
                                                   alpha = 0.05,
                                                   boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  thr_vals <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL

  Qi_bands <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    amm_compute_band_qf(fit_result, boot_reps, cv, alpha, Ji, pipeline)
  })
  if (is.null(y_grid))
    y_grid <- .amm_default_y_grid(lapply(Qi_bands, function(b) b[, "mean_hat"]))

  S_means <- lapply(Qi_bands, function(b) .amm_qi_to_survival(b[, "mean_hat"], y_grid))
  cv_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    cv
  })
  Z_boots <- lapply(cv_list, function(cv) amm_replicate_z_at(boot_reps, cv))
  S_bands <- lapply(seq_along(x_values), function(i) {
    Zb <- Z_boots[[i]]
    B_ <- nrow(Zb)
    Sb <- vapply(seq_len(B_),
                 function(b) .amm_qi_to_survival(.decode_z_vec(pipeline, Zb[b, ], Ji),
                                                 y_grid),
                 numeric(length(y_grid)))
    .amm_joint_band(Sb, S_means[[i]], alpha)
  })

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
         xlab = "y", ylab = "1 - F(y)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(y_grid, S_bands[[i]]$lo, S_bands[[i]]$hi)
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 2, lwd = 1.5)
    }
    lines(y_grid, S_means[[i]],    col = "black", lwd = 4)
    if (i == 1) .amm_legend_thresholds(threshold_lines, "topright")
  }
  invisible(NULL)
}


## Faceted contrast survival functions with bootstrap bands.
amm_plot_contrast_survival_faceted <- function(fit_result, covariate_name, pipeline,
                                                x_values = NULL, y_grid = NULL,
                                                age_ref = 40, pir_ref = 1, male_ref = 0,
                                                Ji = NULL,
                                                threshold_lines = NULL,
                                                alpha = 0.05,
                                                boot_reps = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name, faceted = TRUE)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for faceted band plots.")
  ref_val  <- switch(covariate_name, age = age_ref, pir = pir_ref)
  thr_vals <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL

  cv_ref <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Qi_ref_pt <- .decode_z_vec(pipeline,
                             amm_compute_conditional_mean(fit_result, cv_ref), Ji)

  Qi_x_pt <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid))
    y_grid <- .amm_default_y_grid(c(list(Qi_ref_pt), Qi_x_pt))
  S_ref_pt     <- .amm_qi_to_survival(Qi_ref_pt, y_grid)
  Z_ref_boot   <- amm_replicate_z_at(boot_reps, cv_ref)
  B_           <- nrow(Z_ref_boot)
  S_diff_means <- lapply(Qi_x_pt, function(Qi) .amm_qi_to_survival(Qi, y_grid) - S_ref_pt)

  band_list <- lapply(seq_along(x_values), function(i) {
    cv <- cv_ref; cv[[covariate_name]] <- x_values[i]
    Zb_x <- amm_replicate_z_at(boot_reps, cv)
    Sd_b <- vapply(seq_len(B_), function(b) {
      .amm_qi_to_survival(.decode_z_vec(pipeline, Zb_x[b, ],      Ji), y_grid) -
      .amm_qi_to_survival(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji), y_grid)
    }, numeric(length(y_grid)))
    .amm_joint_band(Sd_b, S_diff_means[[i]], alpha)
  })
  y_lim <- range(unlist(band_list), unlist(S_diff_means), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = y_lim,
         xlab = "y", ylab = "1 - F(y) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    draw_band(y_grid, band_list[[i]]$lo, band_list[[i]]$hi)
    abline(h = 0, col = "red", lty = 3)
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 2, lwd = 1.5)
    }
    lines(y_grid, S_diff_means[[i]], col = "black", lwd = 4)
    if (i == 1) .amm_legend_thresholds(threshold_lines, "topright")
  }
  invisible(NULL)
}

## ----- Decoded moments family -----

.amm_compute_moment_curve <- function(fit_result, covariate_name, x_grid,
                                       age_ref, pir_ref, male_ref, Ji, pipeline) {
  M <- matrix(NA_real_, nrow = length(x_grid), ncol = 4,
              dimnames = list(NULL, c("mean", "variance", "skewness", "kurtosis")))
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- .decode_z_vec(pipeline,
                        amm_compute_conditional_mean(fit_result, cv), Ji)
    M[j, ] <- moments_from_Qi(Qi)
  }
  M
}

## Joint (over x, per moment) bands around `center` (the J x 4 point curve).
.amm_compute_moment_bands <- function(fit_result, boot_reps, covariate_name,
                                       x_grid, age_ref, pir_ref, male_ref,
                                       Ji, alpha, pipeline, center,
                                       contrast = FALSE) {
  J_ <- length(x_grid)
  B_ <- length(boot_reps)
  arr <- array(NA_real_, dim = c(J_, 4, B_))
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  cv_ref  <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  if (contrast) Z_ref_boot <- amm_replicate_z_at(boot_reps, cv_ref)

  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    Zb <- amm_replicate_z_at(boot_reps, cv)
    for (b in seq_len(B_)) {
      Qi_b <- .decode_z_vec(pipeline, Zb[b, ], Ji)
      m_b  <- moments_from_Qi(Qi_b)
      if (contrast) {
        Qi_ref_b <- .decode_z_vec(pipeline, Z_ref_boot[b, ], Ji)
        m_b <- m_b - moments_from_Qi(Qi_ref_b)
      }
      arr[j, , b] <- m_b
    }
  }
  lo <- hi <- matrix(NA_real_, nrow = J_, ncol = 4)
  for (m in seq_len(4)) {
    bd <- .amm_joint_band(matrix(arr[, m, ], nrow = J_), center[, m], alpha)
    lo[, m] <- bd$lo
    hi[, m] <- bd$hi
  }
  list(lo = lo, hi = hi)
}

.amm_plot_moment_panels <- function(x_grid, M_point, M_bands = NULL,
                                     moments, covariate_name, contrast = FALSE,
                                     panel_ylabs = NULL,
                                     xlab = covariate_name,
                                     cex = NULL, mar = c(4, 4.5, 3, 1),
                                     lwd = 3, main = "") {
  moment_labels <- c(mean = "Mean", variance = "Variance",
                     skewness = "Skewness", kurtosis = "Kurtosis")
  if (is.null(panel_ylabs)) {
    panel_ylabs <- paste0(moment_labels[moments], if (contrast) " contrast" else "")
  }
  old_par <- .panel_row_par(length(moments), mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(moments)) {
    m <- moments[i]
    j <- match(m, colnames(M_point))
    y_vals <- M_point[, j]
    if (!is.null(M_bands)) y_vals <- c(y_vals, M_bands$lo[, j], M_bands$hi[, j])
    y_lim <- range(y_vals, if (contrast) 0 else NULL, finite = TRUE)
    ## Trim the margins of an omitted panel title / axis label
    par(mar = .trimmed_mar(mar, moment_labels[m], xlab, panel_ylabs[i]))
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = xlab, ylab = panel_ylabs[i], main = moment_labels[m])
    if (!is.null(M_bands)) draw_band(x_grid, M_bands$lo[, j], M_bands$hi[, j])
    if (contrast) abline(h = 0, col = "red", lty = 3)
    lines(x_grid, M_point[, j], col = "black", lwd = lwd)
  }
  .panel_row_title(main, cex)
}

amm_plot_conditional_moments <- function(fit_result, covariate_name, pipeline,
                                          x_grid = NULL, n_grid = 50,
                                          age_ref = 40, pir_ref = 1, male_ref = 0,
                                          Ji = NULL,
                                          moments = c("mean", "variance",
                                                      "skewness", "kurtosis"),
                                          ...) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  M <- .amm_compute_moment_curve(fit_result, covariate_name, x_grid,
                                 age_ref, pir_ref, male_ref, Ji, pipeline)
  .amm_plot_moment_panels(x_grid, M, NULL, moments, covariate_name,
                          contrast = FALSE, ...)
  invisible(NULL)
}

amm_plot_conditional_moments_faceted <- function(fit_result, covariate_name, pipeline,
                                                  x_grid = NULL, n_grid = 50,
                                                  age_ref = 40, pir_ref = 1, male_ref = 0,
                                                  Ji = NULL,
                                                  moments = c("mean", "variance",
                                                              "skewness", "kurtosis"),
                                                  alpha = 0.05,
                                                  boot_reps = NULL,
                                                  ...) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for band plots.")
  M  <- .amm_compute_moment_curve(fit_result, covariate_name, x_grid,
                                  age_ref, pir_ref, male_ref, Ji, pipeline)
  Mb <- .amm_compute_moment_bands(fit_result, boot_reps, covariate_name, x_grid,
                                  age_ref, pir_ref, male_ref, Ji, alpha, pipeline,
                                  center = M, contrast = FALSE)
  .amm_plot_moment_panels(x_grid, M, Mb, moments, covariate_name, contrast = FALSE, ...)
  invisible(NULL)
}

amm_plot_contrast_moments <- function(fit_result, covariate_name, pipeline,
                                       x_grid = NULL, n_grid = 50,
                                       age_ref = 40, pir_ref = 1, male_ref = 0,
                                       Ji = NULL,
                                       moments = c("mean", "variance",
                                                   "skewness", "kurtosis"),
                                       ...) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  M_x   <- .amm_compute_moment_curve(fit_result, covariate_name, x_grid,
                                     age_ref, pir_ref, male_ref, Ji, pipeline)
  M_ref <- .amm_compute_moment_curve(fit_result, covariate_name, ref_val,
                                     age_ref, pir_ref, male_ref, Ji, pipeline)
  M_diff <- sweep(M_x, 2, M_ref, "-")
  .amm_plot_moment_panels(x_grid, M_diff, NULL, moments, covariate_name,
                          contrast = TRUE, ...)
  invisible(NULL)
}

amm_plot_contrast_moments_faceted <- function(fit_result, covariate_name, pipeline,
                                               x_grid = NULL, n_grid = 50,
                                               age_ref = 40, pir_ref = 1, male_ref = 0,
                                               Ji = NULL,
                                               moments = c("mean", "variance",
                                                           "skewness", "kurtosis"),
                                               alpha = 0.05,
                                               boot_reps = NULL,
                                               ...) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  if (is.null(boot_reps)) stop("boot_reps required for band plots.")
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  M_x   <- .amm_compute_moment_curve(fit_result, covariate_name, x_grid,
                                     age_ref, pir_ref, male_ref, Ji, pipeline)
  M_ref <- .amm_compute_moment_curve(fit_result, covariate_name, ref_val,
                                     age_ref, pir_ref, male_ref, Ji, pipeline)
  M_diff <- sweep(M_x, 2, M_ref, "-")
  Mb     <- .amm_compute_moment_bands(fit_result, boot_reps, covariate_name, x_grid,
                                      age_ref, pir_ref, male_ref, Ji, alpha, pipeline,
                                      center = M_diff, contrast = TRUE)
  .amm_plot_moment_panels(x_grid, M_diff, Mb, moments, covariate_name, contrast = TRUE, ...)
  invisible(NULL)
}

## ----- Threshold-crossing family -----

## Threshold summary of the decoded distribution with quantile function Qi:
## F(threshold_y) ("cdf": mass at or below) or 1 - F(threshold_y) ("survival":
## mass above).
.amm_threshold_summary <- function(Qi, threshold_y, eval_type) {
  F_y <- qi_to_cdf(Qi, threshold_y)
  if (eval_type == "survival") 1 - F_y else F_y
}

## F(threshold_y | x) (eval_type = "cdf", time below the threshold) or
## 1 - F(threshold_y | x) (eval_type = "survival", time above) as a function of
## x, with optional bootstrap bands. main / ylab override the default title and
## y-axis label (CDF terms, as in plot_decoded_conditional_thresholds).
amm_plot_threshold_crossing <- function(fit_result, covariate_name, threshold_y,
                                         pipeline,
                                         x_grid = NULL, n_grid = 50,
                                         age_ref = 40, pir_ref = 1, male_ref = 0,
                                         Ji = NULL, alpha = 0.05,
                                         boot_reps = NULL,
                                         pop_avg = NULL,
                                         eval_type = c("cdf", "survival"),
                                         main = NULL, ylab = NULL,
                                         xlab = covariate_name, lwd = 3) {
  eval_type <- match.arg(eval_type)
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])

  F_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- .decode_z_vec(pipeline,
                        amm_compute_conditional_mean(fit_result, cv), Ji)
    F_curve[j] <- .amm_threshold_summary(Qi, threshold_y, eval_type)
  }

  F_band <- NULL
  if (!is.null(boot_reps)) {
    B_     <- length(boot_reps)
    Fb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
      cv[[covariate_name]] <- x_grid[j]
      Zb <- amm_replicate_z_at(boot_reps, cv)
      Fb_arr[j, ] <- vapply(seq_len(B_),
                            function(b) .amm_threshold_summary(
                              .decode_z_vec(pipeline, Zb[b, ], Ji),
                              threshold_y, eval_type),
                            numeric(1))
    }
    F_band <- .amm_joint_band(Fb_arr, F_curve, alpha)
  }

  if (is.null(ylab)) ylab <- .default_threshold_ylabs(eval_type, threshold_y)
  if (is.null(main)) main <- sprintf("Fraction %s %g by %s",
                                     if (eval_type == "survival") "above" else "below",
                                     threshold_y, covariate_name)
  y_vals <- c(F_curve, F_band$lo, F_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = xlab, ylab = ylab, main = main)
  if (!is.null(F_band)) draw_band(x_grid, F_band$lo, F_band$hi)
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, F_curve, col = "black", lwd = lwd)
  invisible(NULL)
}

## Contrast in F(threshold_y | x) - F(threshold_y | ref) (eval_type = "cdf") or
## in 1 - F (eval_type = "survival"), with bands. main / ylab override the
## default title and y-axis label.
amm_plot_contrast_threshold_crossing <- function(fit_result, covariate_name,
                                                  threshold_y, pipeline,
                                                  x_grid = NULL, n_grid = 50,
                                                  age_ref = 40, pir_ref = 1,
                                                  male_ref = 0,
                                                  Ji = NULL, alpha = 0.05,
                                                  boot_reps = NULL,
                                                  eval_type = c("cdf", "survival"),
                                                  main = NULL, ylab = NULL,
                                                  xlab = covariate_name, lwd = 3) {
  eval_type <- match.arg(eval_type)
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  cv_ref  <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  F_ref <- .amm_threshold_summary(
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv_ref), Ji),
    threshold_y, eval_type)

  F_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    F_curve[j] <- .amm_threshold_summary(
      .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji),
      threshold_y, eval_type) - F_ref
  }

  F_band <- NULL
  if (!is.null(boot_reps)) {
    Z_ref_boot <- amm_replicate_z_at(boot_reps, cv_ref)
    B_         <- nrow(Z_ref_boot)
    Fb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
      Zb_x <- amm_replicate_z_at(boot_reps, cv)
      Fb_arr[j, ] <- vapply(seq_len(B_), function(b) {
        .amm_threshold_summary(.decode_z_vec(pipeline, Zb_x[b, ],       Ji),
                               threshold_y, eval_type) -
        .amm_threshold_summary(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji),
                               threshold_y, eval_type)
      }, numeric(1))
    }
    F_band <- .amm_joint_band(Fb_arr, F_curve, alpha)
  }

  if (is.null(ylab)) ylab <- .default_threshold_ylabs(eval_type, threshold_y,
                                                       contrast = TRUE)
  if (is.null(main)) main <- sprintf("Contrast in fraction %s %g by %s",
                                     if (eval_type == "survival") "above" else "below",
                                     threshold_y, covariate_name)
  y_vals <- c(F_curve, F_band$lo, F_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = xlab, ylab = ylab, main = main)
  if (!is.null(F_band)) draw_band(x_grid, F_band$lo, F_band$hi)
  abline(h = 0, col = "red", lty = 3)
  lines(x_grid, F_curve, col = "black", lwd = lwd)
  invisible(NULL)
}

## ----- Quantile-crossing family -----

## Q(p_star | x) as a function of x, with optional bootstrap bands. main /
## ylab override the default title and y-axis label.
amm_plot_quantile_crossing <- function(fit_result, covariate_name, p_star,
                                        pipeline,
                                        x_grid = NULL, n_grid = 50,
                                        age_ref = 40, pir_ref = 1, male_ref = 0,
                                        Ji = NULL, alpha = 0.05,
                                        boot_reps = NULL,
                                        pop_avg = NULL,
                                        main = NULL, ylab = NULL,
                                        xlab = covariate_name, lwd = 3) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])

  Q_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x_grid[j]
    Qi <- .decode_z_vec(pipeline,
                        amm_compute_conditional_mean(fit_result, cv), Ji)
    Q_curve[j] <- qi_at_p(Qi, p_star)
  }

  Q_band <- NULL
  if (!is.null(boot_reps)) {
    B_     <- length(boot_reps)
    Qb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
      cv[[covariate_name]] <- x_grid[j]
      Zb <- amm_replicate_z_at(boot_reps, cv)
      Qb_arr[j, ] <- vapply(seq_len(B_),
                            function(b) qi_at_p(
                              .decode_z_vec(pipeline, Zb[b, ], Ji), p_star),
                            numeric(1))
    }
    Q_band <- .amm_joint_band(Qb_arr, Q_curve, alpha)
  }

  if (is.null(ylab)) ylab <- sprintf("Q(%g)", p_star)
  if (is.null(main)) main <- sprintf("Quantile at p = %g by %s", p_star, covariate_name)
  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = xlab, ylab = ylab, main = main)
  if (!is.null(Q_band)) draw_band(x_grid, Q_band$lo, Q_band$hi)
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, Q_curve, col = "black", lwd = lwd)
  invisible(NULL)
}

## Contrast in Q(p_star | x) - Q(p_star | ref), with bands. main / ylab
## override the default title and y-axis label.
amm_plot_contrast_quantile_crossing <- function(fit_result, covariate_name,
                                                 p_star, pipeline,
                                                 x_grid = NULL, n_grid = 50,
                                                 age_ref = 40, pir_ref = 1,
                                                 male_ref = 0,
                                                 Ji = NULL, alpha = 0.05,
                                                 boot_reps = NULL,
                                                 main = NULL, ylab = NULL,
                                                 xlab = covariate_name, lwd = 3) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  cv_ref  <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  Q_ref <- qi_at_p(
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv_ref), Ji),
    p_star)

  Q_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    Q_curve[j] <- qi_at_p(
      .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji),
      p_star) - Q_ref
  }

  Q_band <- NULL
  if (!is.null(boot_reps)) {
    Z_ref_boot <- amm_replicate_z_at(boot_reps, cv_ref)
    B_         <- nrow(Z_ref_boot)
    Qb_arr <- matrix(NA_real_, nrow = length(x_grid), ncol = B_)
    for (j in seq_along(x_grid)) {
      cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
      Zb_x <- amm_replicate_z_at(boot_reps, cv)
      Qb_arr[j, ] <- vapply(seq_len(B_), function(b) {
        qi_at_p(.decode_z_vec(pipeline, Zb_x[b, ],       Ji), p_star) -
        qi_at_p(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji), p_star)
      }, numeric(1))
    }
    Q_band <- .amm_joint_band(Qb_arr, Q_curve, alpha)
  }

  if (is.null(ylab)) ylab <- sprintf("Q(%g) contrast", p_star)
  if (is.null(main)) main <- sprintf("Contrast in quantile at p = %g by %s",
                                     p_star, covariate_name)
  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = xlab, ylab = ylab, main = main)
  if (!is.null(Q_band)) draw_band(x_grid, Q_band$lo, Q_band$hi)
  abline(h = 0, col = "red", lty = 3)
  lines(x_grid, Q_curve, col = "black", lwd = lwd)
  invisible(NULL)
}

## ----- Paneled crossing plots -----

## One panel per threshold: F(threshold | x) or 1 - F(threshold | x) (per the
## entry's eval_type) as a function of x, or its contrast vs the reference
## value when contrast = TRUE. Each `thresholds` entry is a list(label=, value=,
## eval_type=, main=) as used by the group-comparison threshold plots; `...`
## is passed to the single-panel function (Ji, boot_reps, alpha, refs, ...).
amm_plot_threshold_crossing_panels <- function(fit_result, covariate_name,
                                                thresholds, pipeline,
                                                contrast = FALSE, main = "",
                                                panel_mains = NULL,
                                                panel_ylabs = NULL,
                                                xlab = covariate_name,
                                                cex = NULL,
                                                mar = c(4, 4.5, 3, 1),
                                                lwd = 3,
                                                ...) {
  plot_one <- if (contrast) amm_plot_contrast_threshold_crossing
              else amm_plot_threshold_crossing
  n_thr <- length(thresholds)
  if (is.null(panel_mains)) {
    panel_mains <- vapply(thresholds, function(thr) thr$main %||% thr$label,
                          character(1))
  }
  old_par <- .panel_row_par(n_thr, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_thr)) {
    thr  <- thresholds[[k]]
    ylab <- if (is.null(panel_ylabs)) NULL else panel_ylabs[k]
    ## Trim the margins of an omitted panel title / axis label
    par(mar = .trimmed_mar(mar, panel_mains[k], xlab, ylab %||% "y"))
    plot_one(fit_result, covariate_name, thr$value, pipeline,
             eval_type = thr$eval_type %||% "cdf",
             main      = panel_mains[k],
             ylab      = ylab,
             xlab      = xlab,
             lwd       = lwd,
             ...)
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}

## One panel per quantile level: Q(p | x) as a function of x, or its contrast
## vs the reference value when contrast = TRUE. Each `quantile_levels` entry is
## a list(label=, value=) as used by the group-comparison quantile-level plots;
## `...` is passed to the single-panel function.
amm_plot_quantile_crossing_panels <- function(fit_result, covariate_name,
                                               quantile_levels, pipeline,
                                               contrast = FALSE, main = "",
                                               panel_mains = NULL,
                                               panel_ylabs = NULL,
                                               xlab = covariate_name,
                                               cex = NULL,
                                               mar = c(4, 4.5, 3, 1),
                                               lwd = 3,
                                               ...) {
  plot_one <- if (contrast) amm_plot_contrast_quantile_crossing
              else amm_plot_quantile_crossing
  n_p <- length(quantile_levels)
  if (is.null(panel_mains)) {
    panel_mains <- vapply(quantile_levels, `[[`, character(1), "label")
  }
  old_par <- .panel_row_par(n_p, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_p)) {
    ylab <- if (is.null(panel_ylabs)) NULL else panel_ylabs[k]
    ## Trim the margins of an omitted panel title / axis label
    par(mar = .trimmed_mar(mar, panel_mains[k], xlab, ylab %||% "y"))
    plot_one(fit_result, covariate_name, quantile_levels[[k]]$value, pipeline,
             main = panel_mains[k], ylab = ylab, xlab = xlab, lwd = lwd, ...)
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}
