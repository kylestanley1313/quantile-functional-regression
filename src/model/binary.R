## ============================================================
## src/model/binary.R
## Group comparison model: estimation, inference, diagnostics,
## and interpretation.
## Naming convention: binary_verb() / binary_verb_qualifier()
## ============================================================

library(MASS)


## ==================== Estimation ====================

fit_binary <- function(Z, X) {
  x <- if (is.null(dim(X))) as.numeric(X) else as.numeric(X[, 1])
  lm(Z ~ x)
}


## ==================== Sampling ====================

sample_binary <- function(n, fit, X) {
  x <- if (is.null(dim(X))) as.numeric(X) else as.numeric(X[, 1])
  stopifnot(length(x) == n, all(x %in% c(0, 1)))

  B_hat     <- coef(fit)
  E_hat     <- residuals(fit)
  N_fit     <- nrow(E_hat)
  Sigma_hat <- crossprod(E_hat) / (N_fit - 2)
  K         <- ncol(B_hat)

  mu_hat <- matrix(B_hat["(Intercept)", ], nrow = n, ncol = K, byrow = TRUE) +
    outer(x, B_hat["x", ])
  eps <- MASS::mvrnorm(n, mu = rep(0, K), Sigma = Sigma_hat)
  if (n == 1) eps <- matrix(eps, nrow = 1)

  draws <- mu_hat + eps
  asplit(draws, MARGIN = 1) |> lapply(as.numeric)
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

## Margin p-value legend geometry (inches): gap from plot box, ramp width, gap
## from ramp to tick labels
binary_lgd_gap_in <- 0.12
binary_lgd_w_in   <- 0.15
binary_lgd_lab_in <- 0.05

## Width (inches) of right margin needed for the margin p-value legend on the
## current device (depends on its font metrics, so call with the device open)
binary_legend_margin_in <- function() {
  lab_w_in <- max(strwidth(c("0", "0.001", "0.05", "1"), units = "inches",
                           cex = 0.75))
  binary_lgd_gap_in + binary_lgd_w_in + binary_lgd_lab_in + lab_w_in + 0.1
}

## In-box top p-value legend geometry (inches below the top of the plot box)
## for the horizontal layout: a horizontal ramp with tick labels beneath it.
## band is the height kept clear of intervals. Call with the device open.
binary_legend_top_geom <- function() {
  lab_h    <- strheight("0.001", units = "inches", cex = 0.75)
  ramp_top <- 0.08
  ramp_bot <- ramp_top + 0.13
  lab_top  <- ramp_bot + 0.04
  list(ramp_top = ramp_top, ramp_bot = ramp_bot, lab_top = lab_top,
       band = lab_top + lab_h + 0.1)
}

## Forest plot of Roy-Bose simultaneous confidence intervals.
## rb_df is the data frame returned by binary_intervals() (or any data frame
## with estimate, lower, upper, p_adj columns). labels, xlab, ylab, p_lab and
## mar override the per-row axis labels, axis titles, p-value legend title and
## plot margins. legend_in_margin = TRUE draws the p-value legend in the right
## margin (widened as needed) so it never covers an interval. show_legend =
## FALSE omits the p-value legend (the per-row p-value boxes are still drawn).
## horizontal = TRUE lays the coefficients out left to right with vertical
## intervals (estimates on the y-axis, rotated labels on the x-axis, p-value
## boxes along the bottom); xlim then gives the range of the estimate (y) axis,
## and xlab / ylab remain the literal x / y axis titles. In the horizontal
## layout, legend_in_margin instead puts the legend as a horizontal ramp at
## the top middle of the plot box, extending the estimate axis upward so the
## legend sits in a band clear of all intervals.
binary_plot_intervals <- function(rb_df, xlim = NULL, main = "",
                                  labels = seq_len(nrow(rb_df)),
                                  xlab = expression(hat(beta)[k]),
                                  ylab = "k",
                                  p_lab = expression(p[adj]),
                                  mar = c(4.5, 4, 3, 2),
                                  legend_in_margin = FALSE,
                                  show_legend = TRUE,
                                  horizontal = FALSE) {
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

  ## Right-margin legend: widen the right margin to fit ramp + labels.
  ## Horizontal layout: the legend goes in a band at the top of the box instead.
  top_legend <- show_legend && legend_in_margin && horizontal
  if (show_legend && legend_in_margin && !horizontal) {
    mar[4] <- max(mar[4], binary_legend_margin_in() / par("csi"))
  }

  ## Coefficient positions: top to bottom (vertical) or left to right
  ## (horizontal). Drawing below is written in (value, position) coordinates
  ## and mapped to (x, y) or (y, x) by orientation.
  pos <- if (horizontal) seq_len(K) else K:1
  seg_vp <- function(v0, p0, v1, p1, ...) {
    if (horizontal) segments(p0, v0, p1, v1, ...) else segments(v0, p0, v1, p1, ...)
  }
  rect_vp <- function(v0, p0, v1, p1, ...) {
    if (horizontal) rect(p0, v0, p1, v1, ...) else rect(v0, p0, v1, p1, ...)
  }

  old_par <- par(mar = mar)
  on.exit(par(old_par), add = TRUE)

  ## Top legend band: raise the top of the estimate axis so the band (a
  ## fraction f of the plot height) sits above the highest interval. With the
  ## default 4% axis padding, usr = lim +/- 0.04 * diff(lim).
  if (top_legend) {
    top_geom <- binary_legend_top_geom()
    f <- top_geom$band / par("pin")[2]
    if (f >= 0.9) stop("Plot too short for the top p-value legend")
    L <- xlim[1]
    m <- max(rb_df$upper, rb_df$estimate)
    U_need <- (m + 0.04 * L - 1.08 * f * L) / (1.04 - 1.08 * f)
    xlim[2] <- max(xlim[2], U_need)
  }

  if (horizontal) {
    plot(NULL, xlim = c(0.5, K + 0.5), ylim = xlim,
         xlab = xlab, ylab = ylab, xaxt = "n", main = main)
    axis(1, at = pos, labels = labels, las = 2)
    abline(h = 0, col = "gray80", lwd = 1)
  } else {
    plot(NULL, xlim = xlim, ylim = c(0.5, K + 0.5),
         xlab = xlab, ylab = ylab, yaxt = "n", main = main)
    axis(2, at = pos, labels = labels, las = 1)
    abline(v = 0, col = "gray80", lwd = 1)
  }

  cap_h <- 0.18
  for (i in seq_len(K)) {
    pi_ <- pos[i]
    seg_vp(rb_df$lower[i], pi_, rb_df$upper[i], pi_, col = "black", lwd = 1.5)
    seg_vp(rb_df$lower[i], pi_ - cap_h, rb_df$lower[i], pi_ + cap_h,
           col = "black", lwd = 1.5)
    seg_vp(rb_df$upper[i], pi_ - cap_h, rb_df$upper[i], pi_ + cap_h,
           col = "black", lwd = 1.5)
    if (horizontal) {
      points(pi_, rb_df$estimate[i], pch = 19, cex = 1.0)
    } else {
      points(rb_df$estimate[i], pi_, pch = 19, cex = 1.0)
    }
  }

  usr <- par("usr")
  xr  <- diff(usr[1:2])
  yr  <- diff(usr[3:4])
  ## p-value boxes along the low end of the value axis (left or bottom edge)
  vr <- if (horizontal) yr else xr
  v_min <- if (horizontal) usr[3] else usr[1]
  box_w <- 0.022 * vr
  box_h <- 0.35
  box_v0 <- v_min + 0.006 * vr
  box_v1 <- box_v0 + box_w
  cols_p <- p_to_color(rb_df$p_adj)
  for (i in seq_len(K)) {
    pi_ <- pos[i]
    rect_vp(box_v0, pi_ - box_h / 2, box_v1, pi_ + box_h / 2,
            col = cols_p[i], border = "black", lwd = 0.5)
  }

  if (!show_legend) return(invisible(NULL))

  ## Horizontal ramp at the top middle of the plot box (in the band kept clear
  ## above the intervals): p = 0 (red) on the left to p = 1 (blue) on the
  ## right, tick labels below, p_lab to the left
  if (top_legend) {
    x_per_in <- xr / par("pin")[1]
    y_per_in <- yr / par("pin")[2]
    ramp_w <- min(2, 0.5 * par("pin")[1]) * x_per_in
    lx0 <- mean(usr[1:2]) - ramp_w / 2
    lx1 <- lx0 + ramp_w
    ry0 <- usr[4] - top_geom$ramp_bot * y_per_in
    ry1 <- usr[4] - top_geom$ramp_top * y_per_in
    third <- (lx1 - lx0) / 3
    draw_hseg <- function(xl, xr_, col_lo, col_hi, n = 60) {
      ramp <- colorRamp(c(col_lo, col_hi))
      xs <- seq(xl, xr_, length.out = n + 1)
      for (s in seq_len(n)) {
        v <- ramp((s - 0.5) / n)
        rect(xs[s], ry0, xs[s + 1], ry1,
             col = rgb(v[1], v[2], v[3], maxColorValue = 255), border = NA)
      }
    }
    draw_hseg(lx0,             lx0 + third,     bright_red, medium_red)
    draw_hseg(lx0 + third,     lx0 + 2 * third, medium_red, dark_red)
    draw_hseg(lx0 + 2 * third, lx1,             dark_blue,  bright_blue)
    rect(lx0, ry0, lx1, ry1, col = NA, border = "black")
    lab_y <- usr[4] - top_geom$lab_top * y_per_in
    text(lx0 + (0:3) * third, lab_y, labels = c("0", "0.001", "0.05", "1"),
         adj = c(0.5, 1), cex = 0.75)
    text(lx0 - 0.08 * x_per_in, (ry0 + ry1) / 2, labels = p_lab, adj = c(1, 0.5))
    return(invisible(NULL))
  }

  if (legend_in_margin) {
    x_per_in <- xr / par("pin")[1]
    lx0 <- usr[2] + binary_lgd_gap_in * x_per_in
    lx1 <- lx0 + binary_lgd_w_in * x_per_in
    old_xpd <- par(xpd = NA)
    on.exit(par(old_xpd), add = TRUE)
  } else {
    lx0 <- usr[1] + 0.89 * xr
    lx1 <- usr[1] + 0.93 * xr
  }
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

  lab_x <- if (legend_in_margin) lx1 + binary_lgd_lab_in * x_per_in else lx1 + 0.008 * xr
  text(lab_x, ly0,             labels = "0",     adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + third,     labels = "0.001", adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly0 + 2 * third, labels = "0.05",  adj = c(0, 0.5), cex = 0.75)
  text(lab_x, ly1,             labels = "1",     adj = c(0, 0.5), cex = 0.75)
  text((lx0 + lx1) / 2, ly1 + 0.025 * yr,
       labels = p_lab, adj = c(0.5, 0))

  invisible(NULL)
}


## ==================== Diagnostics ====================

## Save a battery of residual-diagnostic plots for fit_binary() output:
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
  path <- file.path(path_dir, 'gc_male_diag_norm-marg-qq.png')
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
  path <- file.path(path_dir, 'gc_male_diag_norm-marg-hist.png')
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
  path <- file.path(path_dir, 'gc_male_diag_norm-mv-pw-plots.png')
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
  boxm_result <- biotools::boxM(Z, x)
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

  path <- file.path(path_dir, 'gc_male_diag_cov-heatmaps.png')
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
