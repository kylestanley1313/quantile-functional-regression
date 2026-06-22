## ============================================================
## src/model/utils.R
## Shared model utilities used by both binary.R and amm.R.
## ============================================================


## ---------- Scalar summaries from a quantile function ----------

## CDF value at y_grid points, obtained by inverting a discrete Qi.
qi_to_cdf <- function(Qi, y_grid) {
  Ji        <- length(Qi)
  Qi_sorted <- sort(Qi)
  pi_local  <- (1:Ji) / (Ji + 1)
  approx(
    x = Qi_sorted, y = pi_local,
    xout = y_grid, rule = 2, ties = "ordered"
  )$y
}

## Quantile value at probability p_star, interpolated from a discrete Qi.
qi_at_p <- function(Qi, p_star) {
  pi_local <- seq_along(Qi) / (length(Qi) + 1)
  approx(
    x = pi_local, y = Qi,
    xout = p_star, rule = 2, ties = "ordered"
  )$y
}

## Mean, variance, skewness, and kurtosis of the distribution represented by Qi.
## Qi is a length-Ji vector on grid (1:Ji)/(Ji+1) ~ uniform(0,1), so the k-th
## moment is approximated by mean(Qi^k).
moments_from_Qi <- function(Qi) {
  mu <- mean(Qi)
  s2 <- mean((Qi - mu)^2)
  sd <- sqrt(s2)
  c(
    mean     = mu,
    variance = s2,
    skewness = mean((Qi - mu)^3) / sd^3,
    kurtosis = mean((Qi - mu)^4) / sd^4
  )
}


## ---------- Bootstrap helpers ----------

## Coerce bootstrap input to an R x Ji matrix (accepts list of length-Ji
## vectors or an R x Ji matrix already).
.as_qi_boot_matrix <- function(Qi_boot) {
  if (is.list(Qi_boot)) do.call(rbind, Qi_boot) else Qi_boot
}

## Build a y_grid spanning both groups' point estimates and bootstrap replicates.
default_y_grid_two_group <- function(Qi_hat_0, Qi_hat_1,
                                      Qi_boot_0, Qi_boot_1) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  y_pool <- c(Qi_hat_0, Qi_hat_1,
              as.vector(Qi_boot_0), as.vector(Qi_boot_1))
  sort(unique(c(min(y_pool), Qi_hat_0, Qi_hat_1, max(y_pool))))
}


## ---------- Decoding helper ----------

## Decode a single z vector to a length-Ji Qi vector via the pipeline.
.decode_z_vec <- function(pipeline, z, Ji) {
  decode_z_rot_to_Qi(pipeline, list(as.numeric(z)), Ji)[[1]]
}


## ---------- Two-group decoded-space plot primitives ----------
##
## These primitives are consumed by both the Group Comparison section (binary
## predictor x) and the AMM male-effect interpretation section. Each function
## takes Qi point estimates and Qi bootstrap replicate matrices (or lists) for
## the two groups, computes pointwise quantile bands internally, and draws to
## the active device. The caller manages png/dev.off.

## Two-group conditional QFs on the (0,1) p-grid with pointwise bands.
plot_decoded_conditional_qfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  band_lo_0 <- apply(Qi_boot_0, 2, quantile, probs = alpha)
  band_hi_0 <- apply(Qi_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1 <- apply(Qi_boot_1, 2, quantile, probs = alpha)
  band_hi_1 <- apply(Qi_boot_1, 2, quantile, probs = 1 - alpha)

  fun_list  <- list(Qi_hat_0, band_lo_0, band_hi_0,
                    Qi_hat_1, band_lo_1, band_hi_1)
  grid_list <- rep(list(pi_grid), length(fun_list))
  colors    <- rep(group_colors, each = 3)
  widths    <- rep(c(1, 0.5, 0.5), 2)
  types     <- rep(c(1, 3, 3), 2)
  cwt_lbl   <- data.frame(
    color = group_colors, width = c(4, 4),
    type  = c(1, 1), label = group_labels
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = colors, widths = widths, types = types,
    ylab      = "Q(p)",
    color_width_type_labels = cwt_lbl, main = main
  )
  invisible(NULL)
}

## Group-difference QF (group 1 - group 0) with pointwise bands using
## within-replicate subtraction so bands pinch at any common point.
plot_decoded_conditional_qf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  Qi_hat_diff   <- Qi_hat_1 - Qi_hat_0
  mat_boot_diff <- Qi_boot_1 - Qi_boot_0
  band_lo_diff  <- apply(mat_boot_diff, 2, quantile, probs = alpha)
  band_hi_diff  <- apply(mat_boot_diff, 2, quantile, probs = 1 - alpha)

  zero_line <- rep(0, length(pi_grid))
  fun_list  <- list(Qi_hat_diff, band_lo_diff, band_hi_diff, zero_line)
  grid_list <- rep(list(pi_grid), length(fun_list))
  colors    <- c("black", "black", "black", "red")
  widths    <- c(4, 1, 1, 1)
  types     <- c(1, 3, 3, 2)
  cwt_lbl   <- data.frame(
    color = "black", width = 4, type = 1, label = diff_label
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = colors, widths = widths, types = types,
    ylab      = "Q(p) difference",
    color_width_type_labels = cwt_lbl, main = main
  )
  invisible(NULL)
}

## Two-group conditional CDFs on a shared y_grid with pointwise bands and
## optional vertical threshold lines.
plot_decoded_conditional_cdfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL,
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    thresholds = NULL
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  F_hat_0  <- qi_to_cdf(Qi_hat_0, y_grid)
  F_hat_1  <- qi_to_cdf(Qi_hat_1, y_grid)
  F_boot_0 <- t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  F_boot_1 <- t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  band_lo_0 <- apply(F_boot_0, 2, quantile, probs = alpha)
  band_hi_0 <- apply(F_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1 <- apply(F_boot_1, 2, quantile, probs = alpha)
  band_hi_1 <- apply(F_boot_1, 2, quantile, probs = 1 - alpha)

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)", main = main)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
  }
  lines(y_grid, F_hat_0,   col = group_colors[1], lwd = 4)
  lines(y_grid, band_lo_0, col = group_colors[1], lwd = 0.5, lty = 3)
  lines(y_grid, band_hi_0, col = group_colors[1], lwd = 0.5, lty = 3)
  lines(y_grid, F_hat_1,   col = group_colors[2], lwd = 4)
  lines(y_grid, band_lo_1, col = group_colors[2], lwd = 0.5, lty = 3)
  lines(y_grid, band_hi_1, col = group_colors[2], lwd = 0.5, lty = 3)

  legend("topleft",
    legend = c(group_labels, thr_labels),
    col    = c(group_colors, thr_cols),
    lwd    = c(4, 4, rep(1.5, length(thr_labels))),
    lty    = c(1, 1, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Group-difference CDF (group 1 - group 0) on a shared y_grid with
## pointwise bands and optional threshold lines.
plot_decoded_conditional_cdf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL, diff_label = "1 - 0",
    thresholds = NULL
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  F_hat_0  <- qi_to_cdf(Qi_hat_0, y_grid)
  F_hat_1  <- qi_to_cdf(Qi_hat_1, y_grid)
  F_boot_0 <- t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  F_boot_1 <- t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  F_hat_diff      <- F_hat_1 - F_hat_0
  F_boot_diff_mat <- F_boot_1 - F_boot_0
  band_lo_diff    <- apply(F_boot_diff_mat, 2, quantile, probs = alpha)
  band_hi_diff    <- apply(F_boot_diff_mat, 2, quantile, probs = 1 - alpha)

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  y_lim <- range(c(F_hat_diff, band_lo_diff, band_hi_diff, 0))
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "F(y) difference", main = main)
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 1.5)
  }
  lines(y_grid, F_hat_diff,   col = "black", lwd = 4)
  lines(y_grid, band_lo_diff, col = "black", lwd = 1, lty = 3)
  lines(y_grid, band_hi_diff, col = "black", lwd = 1, lty = 3)

  legend("topleft",
    legend = c(diff_label, "y = 0", thr_labels),
    col    = c("black", "red", thr_cols),
    lwd    = c(4, 1, rep(1.5, length(thr_labels))),
    lty    = c(1, 2, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Two-group moments (4 panels: mean, variance, skewness, kurtosis) with
## interval bars from the bootstrap.
plot_decoded_conditional_moments <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  mom_hat_0  <- moments_from_Qi(Qi_hat_0)
  mom_hat_1  <- moments_from_Qi(Qi_hat_1)
  mom_boot_0 <- t(apply(Qi_boot_0, 1, moments_from_Qi))
  mom_boot_1 <- t(apply(Qi_boot_1, 1, moments_from_Qi))
  band_lo_0  <- apply(mom_boot_0, 2, quantile, probs = alpha)
  band_hi_0  <- apply(mom_boot_0, 2, quantile, probs = 1 - alpha)
  band_lo_1  <- apply(mom_boot_1, 2, quantile, probs = alpha)
  band_hi_1  <- apply(mom_boot_1, 2, quantile, probs = 1 - alpha)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_0[k], band_hi_0[k], mom_hat_0[k],
                band_lo_1[k], band_hi_1[k], mom_hat_1[k])
    y_pad <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "group", ylab = moment_names[k],
         main = moment_names[k])
    axis(1, at = c(1, 2), labels = group_labels)
    arrows(1, band_lo_0[k], 1, band_hi_0[k],
           angle = 90, code = 3, length = 0.08,
           col = group_colors[1], lwd = 1.5)
    points(1, mom_hat_0[k], col = group_colors[1], pch = 19, cex = 1.6)
    arrows(2, band_lo_1[k], 2, band_hi_1[k],
           angle = 90, code = 3, length = 0.08,
           col = group_colors[2], lwd = 1.5)
    points(2, mom_hat_1[k], col = group_colors[2], pch = 19, cex = 1.6)
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}

## Group-difference moments (4 panels) with interval bars from the
## within-replicate bootstrap contrast.
plot_decoded_conditional_moment_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  mom_hat_0     <- moments_from_Qi(Qi_hat_0)
  mom_hat_1     <- moments_from_Qi(Qi_hat_1)
  mom_boot_0    <- t(apply(Qi_boot_0, 1, moments_from_Qi))
  mom_boot_1    <- t(apply(Qi_boot_1, 1, moments_from_Qi))
  mom_hat_diff  <- mom_hat_1 - mom_hat_0
  mom_boot_diff <- mom_boot_1 - mom_boot_0
  band_lo_diff  <- apply(mom_boot_diff, 2, quantile, probs = alpha)
  band_hi_diff  <- apply(mom_boot_diff, 2, quantile, probs = 1 - alpha)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_diff[k], band_hi_diff[k], mom_hat_diff[k], 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = str_glue("{moment_names[k]} difference"),
         main = moment_names[k])
    axis(1, at = 1, labels = diff_label)
    abline(h = 0, col = "red", lty = 2)
    arrows(1, band_lo_diff[k], 1, band_hi_diff[k],
           angle = 90, code = 3, length = 0.08,
           col = "black", lwd = 1.5)
    points(1, mom_hat_diff[k], col = "black", pch = 19, cex = 1.6)
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}