## ============================================================
## src/model/binary.R
## Group comparison model: estimation, inference, diagnostics,
## and interpretation.
## Naming convention: binary_verb() / binary_verb_qualifier()
## ============================================================

library(biotools)
library(MASS)
library(MVN)


## ==================== Estimation ====================

## Fit a multivariate linear model Z ~ x where Z is N x K and x is a binary
## scalar predictor. Rows where x is NA are excluded.
binary_fit <- function(Z, x) {
  lm(Z ~ x)
}


## ==================== Inference ====================

## Hotelling-Lawley T^2 test for a group difference in latent means.
## Returns the manova summary object.
binary_test <- function(model) {
  res_manova <- manova(model)
  summary(res_manova, test = "Hotelling-Lawley")
}

## Roy-Bose simultaneous confidence intervals for each component beta_k,
## derived from the same Hotelling/F machinery as binary_test. The
## union-intersection bound gives joint coverage >= 1 - alpha across all
## linear combinations a^T beta; specializing a = e_k gives per-k intervals.
binary_intervals <- function(model, alpha = 0.05) {
  beta_hat <- coef(model)["x", ]
  N <- nrow(model$model)
  K <- length(beta_hat)

  n0    <- sum(model$model[, "x"] == 0)
  n1    <- sum(model$model[, "x"] == 1)
  n_eff <- n0 * n1 / (n0 + n1)

  S_p <- crossprod(residuals(model)) / (N - 2)
  se  <- sqrt(diag(S_p) / n_eff)

  t_stat <- abs(beta_hat) / se

  F_crit  <- qf(1 - alpha, df1 = K, df2 = N - K - 1)
  rb_crit <- sqrt((N - 2) * K / (N - K - 1) * F_crit)

  lower <- beta_hat - rb_crit * se
  upper <- beta_hat + rb_crit * se

  ## p_k = P(F_{K, N-K-1} > (N-K-1)/(K*(N-2)) * t_k^2)
  F_stat <- (N - K - 1) / (K * (N - 2)) * t_stat^2
  p_adj  <- pf(F_stat, df1 = K, df2 = N - K - 1, lower.tail = FALSE)

  data.frame(
    component   = seq_len(K),
    estimate    = beta_hat,
    se          = se,
    lower       = lower,
    upper       = upper,
    t_stat      = t_stat,
    p_adj       = p_adj,
    significant = (lower > 0) | (upper < 0)
  )
}

## Forest plot of Roy-Bose simultaneous confidence intervals.
## rb_df is the data frame returned by binary_intervals().
binary_plot_intervals <- function(rb_df, xlim = NULL, main = "") {
  K       <- nrow(rb_df)
  bright_red  <- "#FF0000"
  medium_red  <- "#C54E57"
  dark_red    <- "#847777"
  dark_blue   <- "#5E819D"
  bright_blue <- "#0000FF"

  p_to_color <- function(p) {
    sapply(p, function(pp) {
      if (pp <= 0.001) {
        t <- pp / 0.001
        v <- colorRamp(c(bright_red, medium_red))(t)
      } else if (pp <= 0.05) {
        t <- (pp - 0.001) / (0.05 - 0.001)
        v <- colorRamp(c(medium_red, dark_red))(t)
      } else {
        t <- (pp - 0.05) / (1 - 0.05)
        v <- colorRamp(c(dark_blue, bright_blue))(t)
      }
      rgb(v[1], v[2], v[3], maxColorValue = 255)
    })
  }

  if (is.null(xlim)) {
    rng <- range(c(rb_df$lower, rb_df$upper, 0))
    pad <- 0.08 * diff(rng)
    xlim <- c(rng[1] - pad, rng[2] + pad)
  }

  y_pos <- K:1
  old_par <- par(mar = c(4.5, 4, 3, 2))
  on.exit(par(old_par), add = TRUE)

  plot(NULL, xlim = xlim, ylim = c(0.5, K + 0.5),
       xlab = expression(hat(beta)[k]),
       ylab = "k",
       yaxt = "n", main = main)
  axis(2, at = y_pos, labels = seq_len(K), las = 1)
  abline(v = 0, col = "gray80", lwd = 1)

  cap_h <- 0.18
  for (i in seq_len(K)) {
    yi <- y_pos[i]
    segments(rb_df$lower[i], yi, rb_df$upper[i], yi, col = "black", lwd = 1.5)
    segments(rb_df$lower[i], yi - cap_h, rb_df$lower[i], yi + cap_h,
             col = "black", lwd = 1.5)
    segments(rb_df$upper[i], yi - cap_h, rb_df$upper[i], yi + cap_h,
             col = "black", lwd = 1.5)
    points(rb_df$estimate[i], yi, pch = 19, cex = 1.0)
  }

  usr <- par("usr")
  xr  <- diff(usr[1:2])
  yr  <- diff(usr[3:4])
  box_w <- 0.022 * xr
  box_h <- 0.35
  box_x0 <- usr[1] + 0.006 * xr
  box_x1 <- box_x0 + box_w
  cols_p <- p_to_color(rb_df$p_adj)
  for (i in seq_len(K)) {
    yi <- y_pos[i]
    rect(box_x0, yi - box_h / 2, box_x1, yi + box_h / 2,
         col = cols_p[i], border = "black", lwd = 0.5)
  }

  lx0 <- usr[1] + 0.89 * xr
  lx1 <- usr[1] + 0.93 * xr
  ly0 <- usr[3] + 0.75 * yr
  ly1 <- usr[3] + 0.92 * yr
  third <- (ly1 - ly0) / 3

  draw_seg <- function(yb, yt, col_lo, col_hi, n = 60) {
    ramp <- colorRamp(c(col_lo, col_hi))
    ys <- seq(yb, yt, length.out = n + 1)
    for (s in seq_len(n)) {
      v <- ramp((s - 0.5) / n)
      rect(lx0, ys[s], lx1, ys[s + 1],
           col = rgb(v[1], v[2], v[3], maxColorValue = 255), border = NA)
    }
  }
  draw_seg(ly0,             ly0 + third,     bright_red,  medium_red)
  draw_seg(ly0 + third,     ly0 + 2 * third, medium_red,  dark_red)
  draw_seg(ly0 + 2 * third, ly1,             dark_blue,   bright_blue)
  rect(lx0, ly0, lx1, ly1, col = NA, border = "black")

  lab_x <- lx1 + 0.008 * xr
  text(lab_x, ly0,             labels = "0",     adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + third,     labels = "0.001", adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + 2 * third, labels = "0.05",  adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly1,             labels = "1",     adj = c(0, 0.5), cex = 0.75)
  text((lx0 + lx1) / 2, ly1 + 0.025 * yr,
       labels = expression(p[adj]), adj = c(0.5, 0))

  invisible(NULL)
}


## ==================== Diagnostics ====================

## Save a battery of residual-diagnostic plots for binary_fit() output:
##   (1) Marginal Q-Q plots
##   (2) Marginal histograms
##   (3) Pairwise residual scatterplots
##   (4) Box's M test (printed to console)
##   (5) Per-group covariance heatmaps
## path_dir must already exist.
binary_diagnostics <- function(model, path_dir,
                                col_train = rgb(0, 0, 0, 0.25)) {
  x     <- model$model[, "x"]
  Z     <- model$model$Z
  resid <- residuals(model)
  K     <- ncol(resid)

  nc <- ceiling(sqrt(K))
  nr <- ceiling(K / nc)

  ## (1) Q-Q plots
  path <- file.path(path_dir, 'group-comp_norm-marg-qq.png')
  png(path, width = 250 * nc, height = 250 * nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k in seq_len(K)) {
    qqnorm(resid[, k], main = paste0("k = ", k), pch = 19, cex = 0.25,
           col = col_train)
    qqline(resid[, k], col = "red")
  }
  dev.off()
  par(mfrow = c(1, 1))

  ## (2) Histograms
  path <- file.path(path_dir, 'group-comp_norm-marg-hist.png')
  png(path, width = 250 * nc, height = 250 * nr, pointsize = 12)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (k in seq_len(K)) {
    hist(resid[, k], main = paste0("k = ", k))
  }
  dev.off()
  par(mfrow = c(1, 1))

  ## (3) Pairwise scatterplots
  n_plots <- choose(K, 2)
  nc2 <- ceiling(sqrt(n_plots))
  nr2 <- ceiling(n_plots / nc2)
  path <- file.path(path_dir, 'group-comp_norm-mv-pw-plots.png')
  png(path, width = 250 * nc2, height = 250 * nr2, pointsize = 12)
  par(mfrow = c(nr2, nc2), mar = c(3, 3, 2, 1))
  for (k1 in 1:(K - 1)) {
    for (k2 in (k1 + 1):K) {
      plot(resid[, k1], resid[, k2],
           main = str_glue("k = {k1} vs. k = {k2}"),
           pch = 19, cex = 0.25, col = col_train)
    }
  }
  dev.off()
  par(mfrow = c(1, 1))

  ## (4) Formal MVN tests
  result <- MVN::mvn(resid, mvn_test = "mardia", univariate_test = "AD")
  print(result$multivariate_normality)
  print(result$univariate_normality)

  ## (5) Box's M
  boxm_result <- boxM(Z, x)
  print(boxm_result)

  ## (6) Per-group covariance heatmaps
  cov0  <- cov(Z[x == 0,])
  cov1  <- cov(Z[x == 1,])
  K_dim <- nrow(cov0)
  zmax  <- max(abs(c(cov0, cov1)))
  breaks   <- seq(-zmax, zmax, length.out = 101)
  col_pal  <- colorRampPalette(c("#2166ac", "white", "#b2182b"))(100)

  plot_cov_heatmap <- function(M, title) {
    image(
      1:K_dim, 1:K_dim, t(M)[, K_dim:1],
      zlim = c(-zmax, zmax), col = col_pal, breaks = breaks,
      axes = FALSE, xlab = "", ylab = "", main = title
    )
    axis(1, at = 1:K_dim, labels = 1:K_dim, las = 1, tick = FALSE)
    axis(2, at = 1:K_dim, labels = K_dim:1, las = 1, tick = FALSE)
    for (i in 1:K_dim) for (j in 1:K_dim) if (i >= j) {
      text(j, K_dim - i + 1, sprintf("%.2f", M[i, j]), cex = 0.7)
    }
  }

  path <- file.path(path_dir, 'group-comp_cov-heatmaps.png')
  png(path, width = 1200, height = 540, pointsize = 14)
  layout(matrix(c(1, 2, 3), nrow = 1), widths = c(4, 4, 1))
  par(mar = c(4, 4, 3, 1))
  plot_cov_heatmap(cov0, "Group 0 covariance")
  plot_cov_heatmap(cov1, "Group 1 covariance")
  par(mar = c(4, 0.5, 3, 4))
  legend_levels <- seq(-zmax, zmax, length.out = 100)
  image(
    1, legend_levels, t(matrix(legend_levels, ncol = 1)),
    col = col_pal, breaks = breaks, axes = FALSE, xlab = "", ylab = ""
  )
  axis(4, las = 1)
  dev.off()

  invisible(NULL)
}


## ==================== Interpretation ====================

## Parametric bootstrap of conditional latent means under the fitted linear
## model. Simulates R synthetic datasets from the fitted model, refits lm on
## each, and extracts per-replicate group-0 and group-1 conditional means in
## z-space.
##
## Returns a list with:
##   z_hat_0   length-K point estimate for group 0
##   z_hat_1   length-K point estimate for group 1
##   z_boot_0  length-R list of K-vectors (bootstrap group-0 means)
##   z_boot_1  length-R list of K-vectors (bootstrap group-1 means)
##
## Decoding z → Qi is left to the script (pipeline-dependent).
binary_boot_interpret <- function(model, R = 1000, seed = 12345) {
  X         <- model.matrix(model)
  B_hat     <- coef(model)
  E_hat     <- residuals(model)
  N_fit     <- nrow(E_hat)
  Sigma_hat <- crossprod(E_hat) / (N_fit - 2)
  mu_hat    <- X %*% B_hat
  x         <- model$model[, "x"]

  z_hat_0 <- as.numeric(B_hat["(Intercept)", ])
  z_hat_1 <- as.numeric(B_hat["(Intercept)", ] + B_hat["x", ])

  z_boot_0 <- vector('list', R)
  z_boot_1 <- vector('list', R)
  set.seed(seed)
  for (r in seq_len(R)) {
    E_star  <- MASS::mvrnorm(N_fit, mu = rep(0, ncol(B_hat)), Sigma = Sigma_hat)
    Z_star  <- mu_hat + E_star
    model_r <- lm(Z_star ~ x)
    B_r     <- coef(model_r)
    z_boot_0[[r]] <- as.numeric(B_r["(Intercept)", ])
    z_boot_1[[r]] <- as.numeric(B_r["(Intercept)", ] + B_r["x", ])
  }

  list(
    z_hat_0  = z_hat_0,
    z_hat_1  = z_hat_1,
    z_boot_0 = z_boot_0,
    z_boot_1 = z_boot_1
  )
}
