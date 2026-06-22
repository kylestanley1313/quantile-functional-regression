## ============================================================
## src/model/amm.R
## Additive mixed model for latent covariates: estimation,
## inference, diagnostics, and interpretation.
## Naming convention: amm_verb() / amm_verb_qualifier()
## Internal helpers: .amm_*()
## ============================================================

library(future)
library(future.apply)
library(mgcv)
library(nlme)
library(RLRsim)
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
amm_fit <- function(Z, covariates,
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
  boundary_diag <- vector("list", K)

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
      fit_k <- tryCatch(
        do.call(lme, list(
          fixed   = fixed_form,
          random  = random_struct,
          data    = df_k,
          method  = "REML",
          control = lmeControl(opt = "optim")
        )),
        error = function(e) {
          message(str_glue("  lme failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      ## lme foreign-call errors are interpreted as variance-component boundary
      ## collapse. Zero-fill encodes "this column contributes nothing."
      if (is.null(fit_k)) {
        beta_hat[, k] <- 0
        if (M_a > 0) U_age_hat[, k] <- 0
        if (M_p > 0) U_pir_hat[, k] <- 0
        gam_k <- tryCatch(
          mgcv::gam(
            z ~ male + s(age, bs = "ps") + s(pir, bs = "ps"),
            data = df_k, method = "REML"
          ),
          error = function(e) NULL
        )
        if (!is.null(gam_k)) {
          s_tbl <- summary(gam_k)$s.table
          boundary_diag[[k]] <- list(
            gam_sp     = gam_k$sp,
            gam_edf    = setNames(s_tbl[, "edf"], rownames(s_tbl)),
            gam_logLik = as.numeric(logLik(gam_k))
          )
          message(str_glue(
            "    -> mgcv refit: sp = ({paste(signif(gam_k$sp, 3), collapse=', ')}), ",
            "edf = ({paste(signif(s_tbl[, 'edf'], 3), collapse=', ')})"
          ))
        }
        next
      }
      fits[[k]] <- fit_k
      beta_hat[, k] <- fixef(fit_k)[beta_names]

      re_vec <- as.numeric(ranef(fit_k)[1, ])
      if (M_a > 0) U_age_hat[, k] <- re_vec[seq_len(M_a)]
      if (M_p > 0) U_pir_hat[, k] <- re_vec[(M_a + 1):(M_a + M_p)]

      sigma2_E_diag[k] <- fit_k$sigma^2
      vc <- VarCorr(fit_k)
      if (M_a > 0) sigma2_U_age[k] <- as.numeric(vc[1, "Variance"])
      if (M_p > 0) sigma2_U_pir[k] <- as.numeric(vc[max(M_a, 0) + 1, "Variance"])

      edf_k_total <- A1
      if (M_a > 0) {
        lambda_a  <- sigma2_E_diag[k] / sigma2_U_age[k]
        edf_age[k] <- sum(phi_age / (phi_age + lambda_a))
        edf_k_total <- edf_k_total + edf_age[k]
      }
      if (M_p > 0) {
        lambda_p  <- sigma2_E_diag[k] / sigma2_U_pir[k]
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
    linear_covariates = linear_covariates,
    smooth_covariates = smooth_covariates,
    keep              = keep,
    fits              = fits,
    R_mat             = R_mat,
    boundary_diag     = boundary_diag
  )
}

## Refit the AMM with one or more smooth covariates dropped (for likelihood
## ratio testing). drop is a character vector of covariate names to remove
## from smooth_covariates.
amm_fit_reduced <- function(Z, covariates, drop = NULL, ...) {
  amm_fit(Z, covariates,
          smooth_covariates = setdiff(c("age", "pir"), drop), ...)
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

## Sum of per-column log-likelihood differences between reduced and full fits,
## multiplied by -2. NULL fits (boundary collapses) contribute 0.
.amm_compute_lrt <- function(fits_full, fits_reduced) {
  K_ <- length(fits_full)
  stopifnot(length(fits_reduced) == K_)
  -2 * sum(sapply(seq_len(K_), function(k) {
    if (is.null(fits_full[[k]]) || is.null(fits_reduced[[k]])) return(0)
    as.numeric(logLik(fits_reduced[[k]])) - as.numeric(logLik(fits_full[[k]]))
  }))
}

## Marginal fitted values from the reduced model (X*beta from each per-k fit).
.amm_compute_fitted_reduced <- function(fit_result_reduced) {
  Z_hat <- sapply(fit_result_reduced$fits, fitted)
  if (!is.matrix(Z_hat))
    Z_hat <- matrix(Z_hat, ncol = length(fit_result_reduced$fits))
  Z_hat
}

## Adapted Hotelling T^2 test for the linear effect of covariate_name. Standard
## errors are adjusted for the mixed-model EDF relative to a plain OLS test.
amm_test_linear <- function(fit_result, covariate_name) {
  a <- which(rownames(fit_result$beta_hat) == covariate_name)
  if (length(a) != 1)
    stop(str_glue("Covariate '{covariate_name}' not found in beta_hat."))

  b      <- fit_result$beta_hat[a, ]
  K_     <- length(b)
  N_kept <- sum(fit_result$keep)
  nu     <- nrow(fit_result$beta_hat) +
            (if (!is.null(fit_result$edf_age)) mean(fit_result$edf_age) else 0) +
            (if (!is.null(fit_result$edf_pir)) mean(fit_result$edf_pir) else 0)

  c_aa <- mean(sapply(fit_result$fits, function(f) {
    if (inherits(f, "lme")) {
      vcov(f)[covariate_name, covariate_name] / f$sigma^2
    } else {
      vcov(f)[covariate_name, covariate_name] / summary(f)$sigma^2
    }
  }))

  T2      <- as.numeric(t(b) %*% solve(fit_result$Sigma_E, b) / c_aa)
  df1     <- K_
  df2     <- N_kept - nu - K_
  F_stat  <- (N_kept - nu - K_) / ((N_kept - nu - 1) * K_) * T2
  p_value <- pf(F_stat, df1, df2, lower.tail = FALSE)

  list(T2 = T2, F_stat = F_stat, df1 = df1, df2 = df2, p_value = p_value)
}

## Roy-Bose simultaneous confidence intervals for the linear coefficient
## beta_k (k = 1..K), adapted for the AMM (EDF-adjusted degrees of freedom).
amm_intervals_linear <- function(fit_result, covariate_name, alpha = 0.05) {
  a <- which(rownames(fit_result$beta_hat) == covariate_name)
  if (length(a) != 1)
    stop(str_glue("Covariate '{covariate_name}' not found in beta_hat."))

  b      <- fit_result$beta_hat[a, ]
  K_     <- length(b)
  N_kept <- sum(fit_result$keep)
  nu     <- nrow(fit_result$beta_hat) +
            (if (!is.null(fit_result$edf_age)) mean(fit_result$edf_age) else 0) +
            (if (!is.null(fit_result$edf_pir)) mean(fit_result$edf_pir) else 0)

  c_aa <- mean(sapply(fit_result$fits, function(f) {
    if (inherits(f, "lme")) {
      vcov(f)[covariate_name, covariate_name] / f$sigma^2
    } else {
      vcov(f)[covariate_name, covariate_name] / summary(f)$sigma^2
    }
  }))

  df1     <- K_
  df2     <- N_kept - nu - K_
  F_crit  <- qf(1 - alpha, df1, df2)
  T2_crit <- (N_kept - nu - 1) * K_ / (N_kept - nu - K_) * F_crit

  se_k <- sqrt(c_aa * diag(fit_result$Sigma_E))
  half <- sqrt(T2_crit) * se_k

  list(
    covariate_name = covariate_name,
    alpha          = alpha,
    beta_hat       = b,
    se             = se_k,
    T2_crit        = T2_crit,
    df1            = df1,
    df2            = df2,
    half_width     = half,
    lower          = b - half,
    upper          = b + half,
    method         = "Roy-Bose simultaneous (T^2 union-intersection)"
  )
}

## Crainiceanu-Ruppert exact RLRT for the smooth effect of covariate_name,
## aggregated across K latent dimensions via Bonferroni correction.
##
## Per-column null sampling uses RLRsim::exactRLRT. Short-circuited columns
## (observed RLRT = 0) fall back to RLRTSim with the original design,
## ignoring the nuisance variance component (conservative approximation).
amm_test_smooth <- function(fit_full, fit_reduced,
                             covariates,
                             covariate_name,
                             B = 10000,
                             seed_base = 1234) {
  K_           <- ncol(fit_full$beta_hat)
  Lambda_per_k <- numeric(K_)
  null_samples <- matrix(NA_real_, nrow = B, ncol = K_)
  p_per_k      <- numeric(K_)
  method_per_k <- character(K_)

  cov_kept <- covariates[fit_full$keep, , drop = FALSE]
  X_kept   <- cbind(
    `(Intercept)` = 1,
    male = cov_kept$male,
    age  = cov_kept$age,
    pir  = cov_kept$pir
  )
  W_tested <- fit_full[[paste0("W_", covariate_name)]]

  for (k in seq_len(K_)) {
    fit_k_full <- fit_full$fits[[k]]
    fit_k_red  <- fit_reduced$fits[[k]]
    if (is.null(fit_k_full) || is.null(fit_k_red)) {
      Lambda_per_k[k]   <- 0
      null_samples[, k] <- 0
      p_per_k[k]        <- 1
      method_per_k[k]   <- "collapsed"
      next
    }
    Lambda_per_k[k] <- max(
      0,
      -2 * (as.numeric(logLik(fit_k_red)) - as.numeric(logLik(fit_k_full)))
    )

    rlrt_k <- tryCatch(
      RLRsim::exactRLRT(m = fit_k_full, mA = fit_k_full, m0 = fit_k_red,
                        nsim = B, seed = seed_base + k),
      error = function(e) {
        message(str_glue("  exactRLRT failed at k = {k}: {conditionMessage(e)}"))
        NULL
      }
    )

    sample_k <- if (!is.null(rlrt_k)) as.numeric(rlrt_k$sample) else numeric(0)

    if (length(sample_k) >= B) {
      null_samples[, k] <- sample_k[1:B]
      method_per_k[k]   <- "exact"
    } else {
      sim_k <- tryCatch(
        RLRsim::RLRTSim(
          X = X_kept, Z = W_tested,
          sqrt.Sigma = diag(ncol(W_tested)),
          nsim = B, seed = seed_base + k + 100000L
        ),
        error = function(e) {
          message(str_glue("  RLRTSim fallback failed at k = {k}: {conditionMessage(e)}"))
          NULL
        }
      )
      if (is.null(sim_k)) {
        null_samples[, k] <- 0
        method_per_k[k]   <- "fallback_failed"
      } else {
        null_samples[, k] <- as.numeric(sim_k)[1:B]
        method_per_k[k]   <- "approx_no_nuisance"
      }
    }

    p_per_k[k] <- (1 + sum(null_samples[, k] >= Lambda_per_k[k])) / (1 + B)
  }

  Lambda_joint      <- sum(Lambda_per_k)
  joint_null_sample <- rowSums(null_samples)
  p_conv <- (1 + sum(joint_null_sample >= Lambda_joint)) / (1 + B)

  p_clean  <- pmin(pmax(p_per_k, .Machine$double.xmin), 1)
  X_Fisher <- -2 * sum(log(p_clean))
  p_Fisher <- pchisq(X_Fisher, df = 2 * K_, lower.tail = FALSE)
  p_Bonf   <- min(K_ * min(p_per_k), 1)

  n_approx <- sum(method_per_k == "approx_no_nuisance")
  if (n_approx > 0) {
    message(str_glue(
      "  Note: {n_approx} of {K_} columns used the RLRTSim fallback ",
      "(observed RLRT = 0). Null samples ignore the nuisance variance ",
      "component (approximate, conservative)."
    ))
  }

  p_per_k_Bonf <- pmin(K_ * p_per_k, 1)

  list(
    covariate_name    = covariate_name,
    Lambda_per_k      = Lambda_per_k,
    Lambda_joint      = Lambda_joint,
    p_per_k           = p_per_k,
    p_per_k_Bonf      = p_per_k_Bonf,
    p_conv            = p_conv,
    p_Fisher          = p_Fisher,
    p_Bonf            = p_Bonf,
    X_Fisher          = X_Fisher,
    null_samples      = null_samples,
    joint_null_sample = joint_null_sample,
    method_per_k      = method_per_k,
    B                 = B,
    K                 = K_
  )
}

## Parametric bootstrap test for the smooth effect of covariate_name.
## statistic = "iss" uses integrated sum-of-squares; statistic = "lrt" uses
## the REML log-likelihood ratio. Parallelism via future::multisession.
amm_boot_test <- function(fit_full, fit_reduced, covariates, covariate_name,
                           grid = NULL,
                           statistic = c("iss", "lrt"),
                           null_type = c("vc_only", "full"),
                           B = 500, n_cores = 4) {
  statistic <- match.arg(statistic)
  null_type <- match.arg(null_type)

  if (statistic == "lrt" && null_type != "vc_only") {
    stop("statistic='lrt' requires null_type='vc_only'.")
  }
  if (statistic == "iss" && is.null(grid)) {
    stop("statistic='iss' requires a non-NULL grid.")
  }

  T_obs <- if (statistic == "iss") {
    sm <- amm_evaluate_smooth(fit_full, covariate_name, grid,
                              include_linear = (null_type == "full"))
    sum(sm^2)
  } else {
    .amm_compute_lrt(fit_full$fits, fit_reduced$fits)
  }

  Z_hat_red <- .amm_compute_fitted_reduced(fit_reduced)
  L         <- chol(fit_reduced$Sigma_E)
  cov_kept  <- covariates[fit_reduced$keep, , drop = FALSE]
  N_kept    <- nrow(Z_hat_red)
  K_        <- ncol(Z_hat_red)

  boot_one <- function(b) {
    G_b   <- matrix(rnorm(N_kept * K_), N_kept, K_)
    Z_b   <- Z_hat_red + G_b %*% L
    fit_b <- tryCatch(amm_fit(Z_b, cov_kept), error = function(e) NULL)
    if (is.null(fit_b)) return(0)
    if (statistic == "iss") {
      sm_b <- amm_evaluate_smooth(fit_b, covariate_name, grid,
                                  include_linear = (null_type == "full"))
      sum(sm_b^2)
    } else {
      fit_b_red <- tryCatch(
        amm_fit_reduced(Z_b, cov_kept, drop = covariate_name),
        error = function(e) NULL
      )
      if (is.null(fit_b_red)) return(0)
      .amm_compute_lrt(fit_b$fits, fit_b_red$fits)
    }
  }

  T_boot <- if (n_cores > 1) {
    old_plan <- future::plan(future::multisession, workers = n_cores)
    on.exit(future::plan(old_plan), add = TRUE)
    unlist(future.apply::future_lapply(seq_len(B), boot_one, future.seed = TRUE))
  } else {
    sapply(seq_len(B), boot_one)
  }
  T_boot_ok <- T_boot[!is.na(T_boot)]
  p_value   <- (1 + sum(T_boot_ok >= T_obs)) / (1 + length(T_boot_ok))

  list(T_obs = T_obs, T_boot = T_boot, p_value = p_value,
       B = B, B_effective = length(T_boot_ok),
       statistic = statistic, null_type = null_type)
}


## ==================== Diagnostics ====================

## Per-column summary of variance components, smoothing parameters, and EDF.
amm_summary <- function(amm_fit) {
  K_ <- ncol(amm_fit$beta_hat)
  data.frame(
    k            = seq_len(K_),
    sigma2_U_age = amm_fit$sigma2_U_age,
    sigma2_U_pir = amm_fit$sigma2_U_pir,
    sigma2_E     = amm_fit$sigma2_E_diag,
    lambda_age   = amm_fit$lambda_age,
    lambda_pir   = amm_fit$lambda_pir,
    edf          = amm_fit$edf
  )
}

## Log10 smoothing-parameter plot (lambda by k for age and pir).
amm_plot_lambda <- function(amm_fit, path) {
  K_amm      <- ncol(amm_fit$beta_hat)
  lambda_log <- log10(c(amm_fit$lambda_age, amm_fit$lambda_pir))
  png(path, width = 960, height = 540, pointsize = 14)
  par(mfrow = c(1, 1), mar = c(4, 5, 3, 1))
  plot(
    NULL,
    xlim = c(1, K_amm),
    ylim = range(lambda_log, finite = TRUE),
    xlab = "k", ylab = expression(log[10](lambda)),
    main = "AMM smoothing parameters by latent dimension"
  )
  lines(1:K_amm, log10(amm_fit$lambda_age), col = "darkblue",   lwd = 2,
        type = "b", pch = 19)
  lines(1:K_amm, log10(amm_fit$lambda_pir), col = "darkorange", lwd = 2,
        type = "b", pch = 17)
  legend("topright",
    legend = c("age", "pir"),
    col    = c("darkblue", "darkorange"),
    lwd    = 2, pch = c(19, 17), bty = "n"
  )
  dev.off()
}

## Residual Q-Q panels (one panel per latent dimension k).
amm_plot_resid_qq <- function(amm_fit, path,
                               col_train = rgb(0, 0, 0, 0.25)) {
  K_amm <- ncol(amm_fit$beta_hat)
  nc    <- ceiling(sqrt(K_amm))
  nr    <- ceiling(K_amm / nc)
  pdf(path, width = 3 * nc, height = 3 * nr)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k in seq_len(K_amm)) {
    qqnorm(amm_fit$R_mat[, k], main = paste0("k = ", k),
           pch = 19, cex = 0.25, col = col_train)
    qqline(amm_fit$R_mat[, k], col = "red")
  }
  dev.off()
  par(mfrow = c(1, 1))
}

## Heatmap of the residual covariance matrix Sigma_E.
amm_plot_sigma_e <- function(amm_fit, path) {
  K_amm   <- ncol(amm_fit$beta_hat)
  zmax    <- max(abs(amm_fit$Sigma_E))
  breaks  <- seq(-zmax, zmax, length.out = 101)
  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF000000"))(100)

  png(path, width = 800, height = 720, pointsize = 14)
  par(mar = c(4, 4, 3, 5))
  image(
    1:K_amm, 1:K_amm,
    t(amm_fit$Sigma_E)[, K_amm:1],
    zlim = c(-zmax, zmax),
    col  = col_pal, breaks = breaks,
    axes = FALSE, xlab = "", ylab = "",
    main = expression(hat(Sigma)[E])
  )
  axis(1, at = 1:K_amm, labels = 1:K_amm, las = 1, tick = FALSE)
  axis(2, at = 1:K_amm, labels = K_amm:1, las = 1, tick = FALSE)
  for (i in 1:K_amm) for (j in 1:K_amm) if (i >= j) {
    text(j, K_amm - i + 1, sprintf("%.2f", amm_fit$Sigma_E[i, j]), cex = 0.6)
  }
  dev.off()
}

## Correlation heatmap of the random-effect BLUPs U_hat (M x K matrix) to
## diagnose the diagonal-covariance assumption on the random effects.
amm_plot_re_cov <- function(U_hat, K_, title_expr, path) {
  Sigma_U <- cov(U_hat)
  d       <- sqrt(diag(Sigma_U))
  Corr_U  <- Sigma_U / tcrossprod(d)

  col_pal <- colorRampPalette(c("#0000FF", "white", "#FF0000"))(100)
  breaks  <- seq(-1, 1, length.out = 101)

  png(path, width = 1100, height = 720, pointsize = 14)
  layout(matrix(c(1, 2), nrow = 1), widths = c(2, 1))

  par(mar = c(4, 4, 3, 5))
  image(1:K_, 1:K_, t(Corr_U)[, K_:1],
        zlim = c(-1, 1), col = col_pal, breaks = breaks,
        axes = FALSE, xlab = "", ylab = "", main = title_expr)
  axis(1, at = 1:K_, labels = 1:K_,    las = 1, tick = FALSE)
  axis(2, at = 1:K_, labels = K_:1,    las = 1, tick = FALSE)
  for (i in 1:K_) for (j in 1:K_) if (i >= j)
    text(j, K_ - i + 1, sprintf("%.2f", Corr_U[i, j]), cex = 0.6)

  par(mar = c(4, 5, 3, 2))
  barplot(log10(diag(Sigma_U)), names.arg = 1:K_, horiz = FALSE,
          xlab = "k", ylab = expression(log[10]~hat(sigma)[U[k]]^2),
          main = "Component variances")
  dev.off()
}

## Per-column CR null distribution histograms (one panel per k).
amm_plot_cr_per_k <- function(cr_res, title, path) {
  K_ <- cr_res$K
  nc <- ceiling(sqrt(K_))
  nr <- ceiling(K_ / nc)
  png(path, width = 200 * nc, height = 200 * nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1), oma = c(0, 0, 2, 0),
      mgp = c(1.8, 0.6, 0))
  for (k in seq_len(K_)) {
    xlim_k <- range(c(cr_res$null_samples[, k], cr_res$Lambda_per_k[k]),
                    finite = TRUE)
    hist(cr_res$null_samples[, k],
         breaks = 40, col = "gray85", border = "white",
         xlab = expression(Lambda[k]), main = paste0("k = ", k),
         xlim = xlim_k)
    abline(v = cr_res$Lambda_per_k[k], col = "red", lwd = 2)
    mtext(sprintf("p = %.3g", cr_res$p_per_k[k]), side = 3, line = 0.2, cex = 0.8)
  }
  mtext(title, outer = TRUE, cex = 1.1, font = 2)
  dev.off()
}

## Joint CR null distribution histogram.
amm_plot_cr_joint <- function(cr_res, title, path) {
  png(path, width = 720, height = 480, pointsize = 14)
  par(mar = c(4, 4, 3, 1))
  hist(cr_res$joint_null_sample, breaks = 40,
       col = "gray85", border = "white",
       xlab = expression(Lambda), main = title,
       xlim = range(c(cr_res$joint_null_sample, cr_res$Lambda_joint),
                    finite = TRUE))
  abline(v = cr_res$Lambda_joint, col = "red", lwd = 2)
  mtext(sprintf("Lambda = %.3g    p (conv) = %.3g    p (Fisher) = %.3g    p (Bonf) = %.3g",
                cr_res$Lambda_joint, cr_res$p_conv,
                cr_res$p_Fisher, cr_res$p_Bonf),
        side = 3, line = 0.2, cex = 0.85)
  dev.off()
}

## Bootstrap null distribution histogram for smooth tests.
amm_plot_boot_hist <- function(boot_res, title, path) {
  png(path, width = 720, height = 480, pointsize = 14)
  par(mar = c(4, 4, 3, 1))
  hist(boot_res$T_boot, breaks = 40, col = "gray85", border = "white",
       xlab = "T", main = title,
       xlim = range(c(boot_res$T_boot, boot_res$T_obs), na.rm = TRUE))
  abline(v = boot_res$T_obs, col = "red", lwd = 2)
  mtext(sprintf("T_obs = %.3g    p = %.3g    B_eff = %d",
                boot_res$T_obs, boot_res$p_value, boot_res$B_effective),
        side = 3, line = 0.2, cex = 0.9)
  dev.off()
}


## ==================== Interpretation ====================

## ----- Internal helpers -----

## Simulate one synthetic Z from the fitted AMM: mu_hat + E*, E* ~ N(0, Sigma_E).
.amm_simulate_one <- function(amm_fit) {
  N_kept <- sum(amm_fit$keep)
  K_     <- ncol(amm_fit$beta_hat)
  mu     <- amm_fit$X_design %*% amm_fit$beta_hat
  if (!is.null(amm_fit$U_age_hat)) mu <- mu + amm_fit$W_age %*% amm_fit$U_age_hat
  if (!is.null(amm_fit$U_pir_hat)) mu <- mu + amm_fit$W_pir %*% amm_fit$U_pir_hat
  L <- chol(amm_fit$Sigma_E)
  G <- matrix(rnorm(N_kept * K_), N_kept, K_)
  mu + G %*% L
}

## Wrap per-replicate coefficient triple into a fit-like object so that
## amm_evaluate_smooth and amm_compute_conditional_mean work unchanged on it.
## Design matrices and basis attributes are shared by reference from amm_fit.
.amm_make_replicate <- function(amm_fit, beta_hat_r, U_age_hat_r, U_pir_hat_r) {
  list(
    beta_hat          = beta_hat_r,
    U_age_hat         = U_age_hat_r,
    U_pir_hat         = U_pir_hat_r,
    W_age             = amm_fit$W_age,
    W_pir             = amm_fit$W_pir,
    smooth_covariates = amm_fit$smooth_covariates,
    linear_covariates = amm_fit$linear_covariates
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
    fit_b <- tryCatch(amm_fit(Z_b, covariates_kept), error = function(e) NULL)
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

## Ji x 3 matrix (mean_hat, lo, hi) of the decoded conditional QF at
## covariate_values, with pointwise bootstrap bands.
amm_compute_band_qf <- function(fit_result, boot_reps, covariate_values,
                                 alpha, Ji, pipeline) {
  z_hat  <- amm_compute_conditional_mean(fit_result, covariate_values)
  Qi_hat <- .decode_z_vec(pipeline, z_hat, Ji)

  Z_boot  <- amm_replicate_z_at(boot_reps, covariate_values)
  B_      <- nrow(Z_boot)
  Qi_boot <- vapply(seq_len(B_),
                    function(b) .decode_z_vec(pipeline, Z_boot[b, ], Ji),
                    numeric(Ji))

  band_lo <- apply(Qi_boot, 1, quantile, probs = alpha / 2)
  band_hi <- apply(Qi_boot, 1, quantile, probs = 1 - alpha / 2)
  cbind(mean_hat = Qi_hat, lo = band_lo, hi = band_hi)
}

## Ji x 3 matrix (mean_hat, lo, hi) of the decoded contrast QF
## Qi(covariate_name = x_value) - Qi(reference), with within-replicate bands.
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

  band_lo <- apply(Qi_diff_boot, 1, quantile, probs = alpha / 2)
  band_hi <- apply(Qi_diff_boot, 1, quantile, probs = 1 - alpha / 2)
  cbind(mean_hat = Qi_hat_diff, lo = band_lo, hi = band_hi)
}

## ----- Latent-space effect curve plots -----

## Per-k smooth effect curves in z-space (one panel per latent dimension).
amm_plot_latent_effects <- function(fit_result, covariate_name,
                                     x_grid = NULL, n_grid = 100,
                                     band = FALSE, alpha = 0.05,
                                     boot_reps = NULL,
                                     covariates_df = NULL,
                                     ncol = NULL) {
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
    band_lo <- apply(sm_arr, c(1, 2), quantile, probs = alpha / 2)
    band_hi <- apply(sm_arr, c(1, 2), quantile, probs = 1 - alpha / 2)
  }

  nc <- ncol %||% ceiling(sqrt(K_))
  nr <- ceiling(K_ / nc)
  old_par <- par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1), mgp = c(1.8, 0.6, 0))
  on.exit(par(old_par), add = TRUE)

  rug_x <- if (!is.null(covariates_df))
              covariates_df[[covariate_name]][fit_result$keep] else NULL

  for (k in seq_len(K_)) {
    y_vals <- c(f_mat[, k], beta_lin[k] * x_grid, band_lo[, k], band_hi[, k])
    y_lim  <- range(y_vals, finite = TRUE)
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = covariate_name, ylab = "f(x)",
         main = sprintf("k = %d (edf = %.2f)", k, edf_k[k]))
    if (!is.null(band_lo)) {
      lines(x_grid, band_lo[, k], col = "gray60", lwd = 0.5, lty = 3)
      lines(x_grid, band_hi[, k], col = "gray60", lwd = 0.5, lty = 3)
    }
    abline(h = 0, col = "gray85", lty = 1)
    lines(x_grid, beta_lin[k] * x_grid, col = "gray40", lwd = 1.5, lty = 2)
    lines(x_grid, f_mat[, k], col = "black", lwd = 2)
    if (!is.null(rug_x)) rug(rug_x, col = rgb(0, 0, 0, 0.2))
  }
  invisible(NULL)
}

## ----- Decoded QF family -----

## Overlaid conditional QFs at a range of x_values for covariate_name.
amm_plot_conditional_qf <- function(fit_result, covariate_name, pipeline,
                                     x_values = NULL,
                                     age_ref = 40, pir_ref = 1, male_ref = 0,
                                     Ji = NULL,
                                     quantile_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  pi_grid <- pi_grid_fun(Ji)

  q_vals   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL
  q_labels <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "label") else NULL

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  colors  <- .amm_x_palette(x_values)
  cwt_lbl <- data.frame(
    color = c(colors,
              if (!is.null(quantile_lines)) q_cols else NULL),
    width = c(rep(4, length(x_values)),
              if (!is.null(quantile_lines)) rep(1.5, length(quantile_lines)) else NULL),
    type  = c(rep(1, length(x_values)),
              if (!is.null(quantile_lines)) rep(4, length(quantile_lines)) else NULL),
    label = c(sprintf("%s = %g", covariate_name, x_values),
              if (!is.null(quantile_lines)) q_labels else NULL)
  )
  plot_funs(
    fun_list  = Qi_list,
    grid_list = rep(list(pi_grid), length(Qi_list)),
    colors    = colors,
    widths    = rep(4, length(x_values)),
    types     = rep(1, length(x_values)),
    ylab      = "Q(p)",
    color_width_type_labels = cwt_lbl,
    main = sprintf("Conditional QFs by %s", covariate_name)
  )
  if (!is.null(quantile_lines)) {
    abline(v = q_vals, col = q_cols, lty = 4, lwd = 1.5)
  }
  invisible(NULL)
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
    if (!is.null(quantile_lines)) {
      abline(v = q_vals, col = q_cols, lty = 4, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "lo"],       col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "hi"],       col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}

## Overlaid contrast QFs Qi(x) - Qi(ref) for a range of x_values.
amm_plot_contrast_qf <- function(fit_result, covariate_name, pipeline,
                                  x_values = NULL,
                                  age_ref = 40, pir_ref = 1, male_ref = 0,
                                  Ji = NULL,
                                  quantile_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  pi_grid <- pi_grid_fun(Ji)
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)

  q_vals   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL
  q_labels <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "label") else NULL

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
  colors <- .amm_x_palette(x_values, ref = ref_val)

  y_lim <- range(unlist(contrasts), 0)
  plot(NULL, xlim = c(0, 1), ylim = y_lim, xlab = "p", ylab = "Q(p) contrast",
       main = sprintf("QF contrasts vs %s = %g", covariate_name, ref_val))
  if (!is.null(quantile_lines)) {
    abline(v = q_vals, col = q_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(pi_grid, contrasts[[i]], col = colors[i], lwd = 2)
  }
  legend("topleft",
         legend = c(sprintf("%s = %g", covariate_name, x_values),
                    if (!is.null(quantile_lines)) q_labels else NULL),
         col    = c(colors,
                    if (!is.null(quantile_lines)) q_cols else NULL),
         lwd    = c(rep(2, length(colors)),
                    if (!is.null(quantile_lines)) rep(1.5, length(quantile_lines)) else NULL),
         lty    = c(rep(1, length(colors)),
                    if (!is.null(quantile_lines)) rep(4, length(quantile_lines)) else NULL),
         bty    = "n")
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
    abline(h = 0, col = "red", lty = 2)
    if (!is.null(quantile_lines)) {
      abline(v = q_vals, col = q_cols, lty = 4, lwd = 1.5)
    }
    lines(pi_grid, bands[[i]][, "lo"],       col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "hi"],       col = "black", lwd = 0.5, lty = 3)
    lines(pi_grid, bands[[i]][, "mean_hat"], col = "black", lwd = 4)
  }
  invisible(NULL)
}

## ----- Decoded CDF family -----

## Overlaid conditional CDFs.
amm_plot_conditional_cdf <- function(fit_result, covariate_name, pipeline,
                                      x_values = NULL, y_grid = NULL,
                                      age_ref = 40, pir_ref = 1, male_ref = 0,
                                      Ji = NULL,
                                      threshold_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])

  thr_vals   <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "label") else NULL

  Qi_list <- lapply(x_values, function(x) {
    cv <- list(male = male_ref, age = age_ref, pir = pir_ref)
    cv[[covariate_name]] <- x
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji)
  })
  if (is.null(y_grid)) y_grid <- .amm_default_y_grid(Qi_list)
  F_list <- lapply(Qi_list, qi_to_cdf, y_grid = y_grid)
  colors <- .amm_x_palette(x_values)

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)",
       main = sprintf("Conditional CDFs by %s", covariate_name))
  if (!is.null(threshold_lines)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(y_grid, F_list[[i]], col = colors[i], lwd = 2)
  }
  legend("topleft",
         legend = c(sprintf("%s = %g", covariate_name, x_values),
                    if (!is.null(threshold_lines)) thr_labels else NULL),
         col    = c(colors, thr_cols),
         lwd    = c(rep(2, length(colors)),
                    if (!is.null(threshold_lines)) rep(1.5, length(threshold_lines)) else NULL),
         lty    = c(rep(1, length(colors)),
                    if (!is.null(threshold_lines)) rep(4, length(threshold_lines)) else NULL),
         bty    = "n")
  invisible(NULL)
}

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
    list(lo = apply(Fb, 1, quantile, probs = alpha / 2),
         hi = apply(Fb, 1, quantile, probs = 1 - alpha / 2))
  })

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
         xlab = "y", ylab = "F(y)",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
    }
    lines(y_grid, F_bands[[i]]$lo, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_bands[[i]]$hi, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_means[[i]],    col = "black", lwd = 4)
  }
  invisible(NULL)
}

## Overlaid contrast CDFs F(y|x) - F(y|ref).
amm_plot_contrast_cdf <- function(fit_result, covariate_name, pipeline,
                                   x_values = NULL, y_grid = NULL,
                                   age_ref = 40, pir_ref = 1, male_ref = 0,
                                   Ji = NULL,
                                   threshold_lines = NULL) {
  if (is.null(x_values)) x_values <- .amm_default_x_values(covariate_name)
  if (is.null(Ji))       Ji <- length(decode_z_rot_to_Qi(pipeline,
                                list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val  <- switch(covariate_name, age = age_ref, pir = pir_ref)
  thr_vals <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "value") else NULL
  thr_cols <- if (!is.null(threshold_lines)) sapply(threshold_lines, `[[`, "col")   else NULL

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
  colors <- .amm_x_palette(x_values, ref = ref_val)

  y_lim <- range(unlist(F_diff), 0)
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "F(y) contrast",
       main = sprintf("CDF contrasts vs %s = %g", covariate_name, ref_val))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(threshold_lines)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
  }
  for (i in seq_along(x_values)) {
    lines(y_grid, F_diff[[i]], col = colors[i], lwd = 2)
  }
  legend("topright", legend = sprintf("%s = %g", covariate_name, x_values),
         col = colors, lwd = 2, bty = "n")
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

  band_list <- lapply(x_values, function(x) {
    cv <- cv_ref; cv[[covariate_name]] <- x
    Zb_x <- amm_replicate_z_at(boot_reps, cv)
    Fd_b <- vapply(seq_len(B_), function(b) {
      qi_to_cdf(.decode_z_vec(pipeline, Zb_x[b, ],      Ji), y_grid) -
      qi_to_cdf(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji), y_grid)
    }, numeric(length(y_grid)))
    list(lo = apply(Fd_b, 1, quantile, probs = alpha / 2),
         hi = apply(Fd_b, 1, quantile, probs = 1 - alpha / 2))
  })
  y_lim <- range(unlist(band_list), unlist(F_diff_means), 0)

  old_par <- par(mfrow = c(1, length(x_values)), mar = c(4, 4, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (i in seq_along(x_values)) {
    plot(NULL, xlim = range(y_grid), ylim = y_lim,
         xlab = "y", ylab = "F(y) contrast",
         main = sprintf("%s = %g", covariate_name, x_values[i]))
    abline(h = 0, col = "red", lty = 2)
    if (!is.null(threshold_lines)) {
      abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
    }
    lines(y_grid, band_list[[i]]$lo, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, band_list[[i]]$hi, col = "black", lwd = 0.5, lty = 3)
    lines(y_grid, F_diff_means[[i]], col = "black", lwd = 4)
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

.amm_compute_moment_bands <- function(fit_result, boot_reps, covariate_name,
                                       x_grid, age_ref, pir_ref, male_ref,
                                       Ji, alpha, pipeline, contrast = FALSE) {
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
  lo <- apply(arr, c(1, 2), quantile, probs = alpha / 2)
  hi <- apply(arr, c(1, 2), quantile, probs = 1 - alpha / 2)
  list(lo = lo, hi = hi)
}

.amm_plot_moment_panels <- function(x_grid, M_point, M_bands = NULL,
                                     moments, covariate_name, contrast = FALSE) {
  moment_labels <- c(mean = "Mean", variance = "Variance",
                     skewness = "Skewness", kurtosis = "Kurtosis")
  old_par <- par(mfrow = c(1, length(moments)), mar = c(4, 4.5, 3, 1))
  on.exit(par(old_par), add = TRUE)
  for (m in moments) {
    j <- match(m, colnames(M_point))
    y_vals <- M_point[, j]
    if (!is.null(M_bands)) y_vals <- c(y_vals, M_bands$lo[, j], M_bands$hi[, j])
    y_lim <- range(y_vals, if (contrast) 0 else NULL, finite = TRUE)
    plot(NULL, xlim = range(x_grid), ylim = y_lim,
         xlab = covariate_name,
         ylab = paste0(moment_labels[m], if (contrast) " contrast" else ""),
         main = moment_labels[m])
    if (contrast) abline(h = 0, col = "red", lty = 2)
    if (!is.null(M_bands)) {
      lines(x_grid, M_bands$lo[, j], col = "black", lwd = 0.5, lty = 3)
      lines(x_grid, M_bands$hi[, j], col = "black", lwd = 0.5, lty = 3)
    }
    lines(x_grid, M_point[, j], col = "black", lwd = 3)
  }
}

amm_plot_conditional_moments <- function(fit_result, covariate_name, pipeline,
                                          x_grid = NULL, n_grid = 50,
                                          age_ref = 40, pir_ref = 1, male_ref = 0,
                                          Ji = NULL,
                                          moments = c("mean", "variance",
                                                      "skewness", "kurtosis")) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  M <- .amm_compute_moment_curve(fit_result, covariate_name, x_grid,
                                 age_ref, pir_ref, male_ref, Ji, pipeline)
  .amm_plot_moment_panels(x_grid, M, NULL, moments, covariate_name,
                          contrast = FALSE)
  invisible(NULL)
}

amm_plot_conditional_moments_faceted <- function(fit_result, covariate_name, pipeline,
                                                  x_grid = NULL, n_grid = 50,
                                                  age_ref = 40, pir_ref = 1, male_ref = 0,
                                                  Ji = NULL,
                                                  moments = c("mean", "variance",
                                                              "skewness", "kurtosis"),
                                                  alpha = 0.05,
                                                  boot_reps = NULL) {
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
                                  contrast = FALSE)
  .amm_plot_moment_panels(x_grid, M, Mb, moments, covariate_name, contrast = FALSE)
  invisible(NULL)
}

amm_plot_contrast_moments <- function(fit_result, covariate_name, pipeline,
                                       x_grid = NULL, n_grid = 50,
                                       age_ref = 40, pir_ref = 1, male_ref = 0,
                                       Ji = NULL,
                                       moments = c("mean", "variance",
                                                   "skewness", "kurtosis")) {
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
                          contrast = TRUE)
  invisible(NULL)
}

amm_plot_contrast_moments_faceted <- function(fit_result, covariate_name, pipeline,
                                               x_grid = NULL, n_grid = 50,
                                               age_ref = 40, pir_ref = 1, male_ref = 0,
                                               Ji = NULL,
                                               moments = c("mean", "variance",
                                                           "skewness", "kurtosis"),
                                               alpha = 0.05,
                                               boot_reps = NULL) {
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
                                      contrast = TRUE)
  .amm_plot_moment_panels(x_grid, M_diff, Mb, moments, covariate_name, contrast = TRUE)
  invisible(NULL)
}

## ----- Threshold-crossing family -----

## F(threshold_y | x) as a function of x, with optional bootstrap bands.
amm_plot_threshold_crossing <- function(fit_result, covariate_name, threshold_y,
                                         pipeline,
                                         x_grid = NULL, n_grid = 50,
                                         age_ref = 40, pir_ref = 1, male_ref = 0,
                                         Ji = NULL, alpha = 0.05,
                                         boot_reps = NULL,
                                         pop_avg = NULL) {
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
    F_curve[j] <- qi_to_cdf(Qi, threshold_y)
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
                            function(b) qi_to_cdf(
                              .decode_z_vec(pipeline, Zb[b, ], Ji), threshold_y),
                            numeric(1))
    }
    F_band <- list(lo = apply(Fb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Fb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(F_curve, F_band$lo, F_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, 0, 1, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("F(%g)", threshold_y),
       main = sprintf("Fraction below %g by %s", threshold_y, covariate_name))
  if (!is.null(F_band)) {
    lines(x_grid, F_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, F_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, F_curve, col = "black", lwd = 3)
  invisible(NULL)
}

## Contrast in F(threshold_y | x) - F(threshold_y | ref), with bands.
amm_plot_contrast_threshold_crossing <- function(fit_result, covariate_name,
                                                  threshold_y, pipeline,
                                                  x_grid = NULL, n_grid = 50,
                                                  age_ref = 40, pir_ref = 1,
                                                  male_ref = 0,
                                                  Ji = NULL, alpha = 0.05,
                                                  boot_reps = NULL) {
  W_obj <- fit_result[[paste0("W_", covariate_name)]]
  rng   <- attr(W_obj, "range")
  if (is.null(x_grid)) x_grid <- seq(rng[1], rng[2], length.out = n_grid)
  if (is.null(Ji))     Ji     <- length(decode_z_rot_to_Qi(pipeline,
                                  list(rep(0, ncol(fit_result$beta_hat))), 1)[[1]])
  ref_val <- switch(covariate_name, age = age_ref, pir = pir_ref)
  cv_ref  <- list(male = male_ref, age = age_ref, pir = pir_ref)
  cv_ref[[covariate_name]] <- ref_val
  F_ref <- qi_to_cdf(
    .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv_ref), Ji),
    threshold_y)

  F_curve <- numeric(length(x_grid))
  for (j in seq_along(x_grid)) {
    cv <- cv_ref; cv[[covariate_name]] <- x_grid[j]
    F_curve[j] <- qi_to_cdf(
      .decode_z_vec(pipeline, amm_compute_conditional_mean(fit_result, cv), Ji),
      threshold_y) - F_ref
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
        qi_to_cdf(.decode_z_vec(pipeline, Zb_x[b, ],       Ji), threshold_y) -
        qi_to_cdf(.decode_z_vec(pipeline, Z_ref_boot[b, ], Ji), threshold_y)
      }, numeric(1))
    }
    F_band <- list(lo = apply(Fb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Fb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(F_curve, F_band$lo, F_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("F(%g) contrast", threshold_y),
       main = sprintf("Contrast in fraction below %g by %s",
                      threshold_y, covariate_name))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(F_band)) {
    lines(x_grid, F_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, F_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  lines(x_grid, F_curve, col = "black", lwd = 3)
  invisible(NULL)
}

## ----- Quantile-crossing family -----

## Q(p_star | x) as a function of x, with optional bootstrap bands.
amm_plot_quantile_crossing <- function(fit_result, covariate_name, p_star,
                                        pipeline,
                                        x_grid = NULL, n_grid = 50,
                                        age_ref = 40, pir_ref = 1, male_ref = 0,
                                        Ji = NULL, alpha = 0.05,
                                        boot_reps = NULL,
                                        pop_avg = NULL) {
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
    Q_band <- list(lo = apply(Qb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Qb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, pop_avg)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("Q(%g)", p_star),
       main = sprintf("Quantile at p = %g by %s", p_star, covariate_name))
  if (!is.null(Q_band)) {
    lines(x_grid, Q_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, Q_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  if (!is.null(pop_avg)) abline(h = pop_avg, col = "gray40", lty = 2)
  lines(x_grid, Q_curve, col = "black", lwd = 3)
  invisible(NULL)
}

## Contrast in Q(p_star | x) - Q(p_star | ref), with bands.
amm_plot_contrast_quantile_crossing <- function(fit_result, covariate_name,
                                                 p_star, pipeline,
                                                 x_grid = NULL, n_grid = 50,
                                                 age_ref = 40, pir_ref = 1,
                                                 male_ref = 0,
                                                 Ji = NULL, alpha = 0.05,
                                                 boot_reps = NULL) {
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
    Q_band <- list(lo = apply(Qb_arr, 1, quantile, probs = alpha / 2),
                   hi = apply(Qb_arr, 1, quantile, probs = 1 - alpha / 2))
  }

  y_vals <- c(Q_curve, Q_band$lo, Q_band$hi, 0)
  plot(NULL, xlim = range(x_grid), ylim = range(y_vals, finite = TRUE),
       xlab = covariate_name, ylab = sprintf("Q(%g) contrast", p_star),
       main = sprintf("Contrast in quantile at p = %g by %s", p_star, covariate_name))
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(Q_band)) {
    lines(x_grid, Q_band$lo, col = "black", lwd = 0.5, lty = 3)
    lines(x_grid, Q_band$hi, col = "black", lwd = 0.5, lty = 3)
  }
  lines(x_grid, Q_curve, col = "black", lwd = 3)
  invisible(NULL)
}