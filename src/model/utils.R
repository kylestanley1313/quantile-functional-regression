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


## Pointwise or joint bootstrap band around `center` from an R x G matrix of
## bootstrap replicates (rows = replicates, columns = grid points).
## Both types have nominal (1 - alpha) coverage (two-sided, as in amm.R).
##   band_type = "pointwise": per-column alpha/2 / (1 - alpha/2) percentiles of
##     the replicates (`center` is ignored).
##   band_type = "joint": a sup-t band obtained by inflating the pointwise
##     bootstrap SD by the (1 - alpha) quantile of the replicate-wise max
##     standardized deviation from `center`. Columns with zero SD (e.g. a CDF
##     pinned at 0 or 1) do not constrain the band.
.boot_band <- function(mat_boot, center, alpha = 0.05,
                       band_type = c("joint", "pointwise")) {
  band_type <- match.arg(band_type)
  if (band_type == "pointwise") {
    return(list(
      lo = apply(mat_boot, 2, quantile, probs = alpha / 2),
      hi = apply(mat_boot, 2, quantile, probs = 1 - alpha / 2)
    ))
  }
  sd_pt <- apply(mat_boot, 2, sd)
  pos   <- sd_pt > 0
  crit  <- .joint_crit(mat_boot, center, alpha)
  lo <- hi <- center
  lo[pos] <- center[pos] - crit * sd_pt[pos]
  hi[pos] <- center[pos] + crit * sd_pt[pos]
  list(lo = lo, hi = hi)
}

## Sup-t critical value of the joint band in `.boot_band`: the (1 - alpha)
## quantile of the replicate-wise max standardized deviation from `center`
## across the grid (zero-SD columns excluded).
.joint_crit <- function(mat_boot, center, alpha = 0.05) {
  sd_pt <- apply(mat_boot, 2, sd)
  pos   <- sd_pt > 0
  z_dev <- apply(mat_boot, 1, function(row) {
    if (!any(pos)) return(0)
    max(abs((row[pos] - center[pos]) / sd_pt[pos]))
  })
  as.numeric(quantile(z_dev, probs = 1 - alpha))
}

## Pointwise and joint intervals of a summary at a few points, as vertical
## cross sections of the `.boot_band` bands. v_boot is R x n (replicates at the
## points), v_hat the length-n point estimate, and crit the sup-t critical value
## computed over the full grid the band is drawn on, so the joint interval
## matches the joint band rather than a band over the n points alone.
.boot_intervals_at <- function(v_boot, v_hat, crit, alpha = 0.05) {
  sd_v <- apply(v_boot, 2, sd)
  list(
    pw_lo = apply(v_boot, 2, quantile, probs = alpha / 2),
    pw_hi = apply(v_boot, 2, quantile, probs = 1 - alpha / 2),
    jt_lo = v_hat - crit * sd_v,
    jt_hi = v_hat + crit * sd_v
  )
}

## Draw a point estimate with nested intervals at x: a thick bar for the
## pointwise interval and a thin capped whisker for the joint interval.
.draw_interval_pair <- function(x, v_hat, iv, k, col) {
  arrows(x, iv$jt_lo[k], x, iv$jt_hi[k], angle = 90, code = 3, length = 0.08,
         col = col, lwd = 1.5)
  segments(x, iv$pw_lo[k], x, iv$pw_hi[k], col = adjustcolor(col, 0.5),
           lwd = 8, lend = "butt")
  points(x, v_hat[k], col = col, pch = 19, cex = 1.6)
}


## ---------- Decoding helper ----------

## Decode a single latent vector to a length-Ji Qi vector via the pipeline.
## Which stage output `z` is taken to live in follows pipeline$latent ('z_rot'
## by default, or 'c'); see the "Latent-space selection" block in src/utils.R.
## This is the single choke point for every decode in src/model/amm.R, so the
## whole AMM interpretation section follows the switch without further changes.
.decode_z_vec <- function(pipeline, z, Ji) {
  decode_latent_to_Qi(pipeline, list(as.numeric(z)), Ji)[[1]]
}


## ---------- Two-group decoded-space plot primitives ----------
##
## These primitives are consumed by both the Group Comparison section (binary
## predictor x) and the AMM male-effect interpretation section. Each function
## takes Qi point estimates and Qi bootstrap replicate matrices (or lists) for
## the two groups, computes pointwise quantile bands internally, and draws to
## the active device. The caller manages png/dev.off.

## Two-group conditional QFs on the (0,1) p-grid with pointwise or joint
## (sup-t across p) bootstrap bands.
plot_decoded_conditional_qfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft'
) {
  band_type <- match.arg(band_type)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  b0 <- .boot_band(Qi_boot_0, Qi_hat_0, alpha, band_type)
  b1 <- .boot_band(Qi_boot_1, Qi_hat_1, alpha, band_type)

  fun_list  <- list(Qi_hat_0, Qi_hat_1)
  grid_list <- rep(list(pi_grid), length(fun_list))
  cwt_lbl   <- data.frame(
    color = group_colors, width = c(4, 4),
    type  = c(1, 1), label = group_labels
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = group_colors, widths = c(1, 1), types = c(1, 1),
    ylab      = "Q(p)",
    color_width_type_labels = cwt_lbl, main = main,
    legend_loc = legend_loc,
    bands     = list(c(list(x = pi_grid), b0), c(list(x = pi_grid), b1))
  )
  invisible(NULL)
}

## Group-difference QF (group 1 - group 0) with pointwise or joint (sup-t
## across p) bands using within-replicate subtraction so bands pinch at any
## common point.
plot_decoded_conditional_qf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0",
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft'
) {
  band_type <- match.arg(band_type)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  Qi_hat_diff   <- Qi_hat_1 - Qi_hat_0
  mat_boot_diff <- Qi_boot_1 - Qi_boot_0
  bd <- .boot_band(mat_boot_diff, Qi_hat_diff, alpha, band_type)

  zero_line <- rep(0, length(pi_grid))
  fun_list  <- list(Qi_hat_diff, zero_line)
  grid_list <- rep(list(pi_grid), length(fun_list))
  colors    <- c("black", "red")
  widths    <- c(4, 1)
  types     <- c(1, 2)
  cwt_lbl   <- data.frame(
    color = "black", width = 4, type = 1, label = diff_label
  )
  plot_funs(
    fun_list  = fun_list, grid_list = grid_list,
    colors    = colors, widths = widths, types = types,
    ylab      = "Q(p) Contrast",
    color_width_type_labels = cwt_lbl, main = main,
    legend_loc = legend_loc,
    bands     = list(c(list(x = pi_grid), bd))
  )
  invisible(NULL)
}

## Two-group conditional CDFs on a shared y_grid with pointwise or joint
## (sup-t across y) bootstrap bands and optional vertical threshold lines.
plot_decoded_conditional_cdfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL,
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    thresholds = NULL,
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft'
) {
  band_type <- match.arg(band_type)
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
  b0 <- .boot_band(F_boot_0, F_hat_0, alpha, band_type)
  b1 <- .boot_band(F_boot_1, F_hat_1, alpha, band_type)

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "F(y)", main = main)
  draw_band(y_grid, b0$lo, b0$hi)
  draw_band(y_grid, b1$lo, b1$hi)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 2)
  }
  lines(y_grid, F_hat_0, col = group_colors[1], lwd = 4)
  lines(y_grid, F_hat_1, col = group_colors[2], lwd = 4)

  legend(legend_loc,
    legend = c(group_labels, thr_labels),
    col    = c(group_colors, thr_cols),
    lwd    = c(4, 4, rep(2, length(thr_labels))),
    lty    = c(1, 1, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Group-difference CDF (group 1 - group 0) on a shared y_grid with pointwise
## or joint (sup-t across y) bands and optional threshold lines.
plot_decoded_conditional_cdf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL, diff_label = "1 - 0",
    thresholds = NULL,
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft'
) {
  band_type <- match.arg(band_type)
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
  bd <- .boot_band(F_boot_diff_mat, F_hat_diff, alpha, band_type)
  band_lo_diff    <- bd$lo
  band_hi_diff    <- bd$hi

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  y_lim <- range(c(F_hat_diff, band_lo_diff, band_hi_diff, 0))
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "F(y) Contrast", main = main)
  draw_band(y_grid, band_lo_diff, band_hi_diff)
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 2)
  }
  lines(y_grid, F_hat_diff, col = "black", lwd = 4)

  legend(legend_loc,
    legend = c(diff_label, thr_labels),
    col    = c("black", thr_cols),
    lwd    = c(4, rep(1.5, length(thr_labels))),
    lty    = c(1, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Two-group conditional survival curves 1 - F(y) on a shared y_grid with
## pointwise or joint (sup-t across y) bootstrap bands and optional vertical
## threshold lines. Axis labels are written in CDF terms (1 - F(y)), not
## survival-function notation.
plot_decoded_conditional_survivals <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL,
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    thresholds = NULL,
    band_type = c("joint", "pointwise"),
    legend_loc = 'topright'
) {
  band_type <- match.arg(band_type)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  S_hat_0  <- 1 - qi_to_cdf(Qi_hat_0, y_grid)
  S_hat_1  <- 1 - qi_to_cdf(Qi_hat_1, y_grid)
  S_boot_0 <- 1 - t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  S_boot_1 <- 1 - t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  b0 <- .boot_band(S_boot_0, S_hat_0, alpha, band_type)
  b1 <- .boot_band(S_boot_1, S_hat_1, alpha, band_type)

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  plot(NULL, xlim = range(y_grid), ylim = c(0, 1),
       xlab = "y", ylab = "1 - F(y)", main = main)
  draw_band(y_grid, b0$lo, b0$hi)
  draw_band(y_grid, b1$lo, b1$hi)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 2)
  }
  lines(y_grid, S_hat_0, col = group_colors[1], lwd = 4)
  lines(y_grid, S_hat_1, col = group_colors[2], lwd = 4)

  legend(legend_loc,
    legend = c(group_labels, thr_labels),
    col    = c(group_colors, thr_cols),
    lwd    = c(4, 4, rep(2, length(thr_labels))),
    lty    = c(1, 1, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Group-difference survival curve (group 1 - group 0) of 1 - F(y) on a shared
## y_grid with pointwise or joint (sup-t across y) bands and optional threshold
## lines. The survival contrast is the negative of the CDF contrast; the axis
## label is kept in CDF terms.
plot_decoded_conditional_survival_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    y_grid = NULL, diff_label = "1 - 0",
    thresholds = NULL,
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft'
) {
  band_type <- match.arg(band_type)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  S_hat_0  <- 1 - qi_to_cdf(Qi_hat_0, y_grid)
  S_hat_1  <- 1 - qi_to_cdf(Qi_hat_1, y_grid)
  S_boot_0 <- 1 - t(apply(Qi_boot_0, 1, qi_to_cdf, y_grid = y_grid))
  S_boot_1 <- 1 - t(apply(Qi_boot_1, 1, qi_to_cdf, y_grid = y_grid))
  S_hat_diff      <- S_hat_1 - S_hat_0
  S_boot_diff_mat <- S_boot_1 - S_boot_0
  bd <- .boot_band(S_boot_diff_mat, S_hat_diff, alpha, band_type)

  thr_vals   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "value") else NULL
  thr_cols   <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "col")   else NULL
  thr_labels <- if (!is.null(thresholds)) sapply(thresholds, `[[`, "label") else NULL

  y_lim <- range(c(S_hat_diff, bd$lo, bd$hi, 0))
  plot(NULL, xlim = range(y_grid), ylim = y_lim,
       xlab = "y", ylab = "(1 - F(y)) Contrast", main = main)
  draw_band(y_grid, bd$lo, bd$hi)
  abline(h = 0, col = "red", lty = 2)
  if (!is.null(thresholds)) {
    abline(v = thr_vals, col = thr_cols, lty = 4, lwd = 2)
  }
  lines(y_grid, S_hat_diff, col = "black", lwd = 4)

  legend(legend_loc,
    legend = c(diff_label, thr_labels),
    col    = c("black", thr_cols),
    lwd    = c(4, rep(1.5, length(thr_labels))),
    lty    = c(1, rep(4, length(thr_labels))),
    bty    = "n"
  )
  invisible(NULL)
}

## Vet a per-threshold `eval_type` vector: NULL -> all "cdf"; a single value is
## recycled; every entry must be "cdf" or "survival".
.check_threshold_eval_type <- function(eval_type, n_thr) {
  if (is.null(eval_type)) eval_type <- rep("cdf", n_thr)
  if (length(eval_type) == 1L) eval_type <- rep(eval_type, n_thr)
  if (length(eval_type) != n_thr) {
    stop("`eval_type` must have length 1 or length(thresholds).")
  }
  if (!all(eval_type %in% c("cdf", "survival"))) {
    stop('`eval_type` entries must be "cdf" or "survival".')
  }
  eval_type
}

## Default per-panel y-axis labels for the threshold plots, in CDF terms:
## "F(y = v)" for a cdf panel, "1 - F(y = v)" for a survival panel, with a
## " Contrast" suffix when `contrast = TRUE`.
.default_threshold_ylabs <- function(eval_type, thr_vals, contrast = FALSE) {
  suffix <- if (contrast) " Contrast" else ""
  vapply(seq_along(eval_type), function(k) {
    base <- if (eval_type[k] == "survival") {
      sprintf("1 - F(y = %g)", thr_vals[k])
    } else {
      sprintf("F(y = %g)", thr_vals[k])
    }
    paste0(base, suffix)
  }, character(1))
}

## ---------- Cross-section (threshold / quantile-level) plots ----------
##
## These plots show a functional summary at a few fixed points (intensity
## thresholds on the CDF / survival curve, or quantile levels on the QF), with
## both a pointwise and a joint interval per point. Each interval is a vertical
## cross section of the corresponding band in the CDF / survival / QF plots:
## the joint interval uses the sup-t critical value computed over the full grid
## that band is drawn on (y_grid or the p-grid), not over the few points shown.

## Apply f(row, pts) to each bootstrap row and return an R x length(pts) matrix
## (also when length(pts) == 1).
.eval_boot_rows <- function(mat_boot, f, pts) {
  matrix(t(apply(mat_boot, 1, f, pts)), ncol = length(pts))
}

## Flip selected columns (or entries) from F to 1 - F.
.flip_survival <- function(v, flip) {
  if (is.matrix(v)) v[, flip] <- 1 - v[, flip] else v[flip] <- 1 - v[flip]
  v
}

## Two-group panels: one panel per point, each group's estimate with nested
## pointwise / joint intervals.
.plot_two_group_cross_sections <- function(v_hat_0, v_hat_1, iv_0, iv_1,
                                           panel_mains, panel_ylabs,
                                           main, group_labels, group_colors) {
  n_pts   <- length(v_hat_0)
  old_par <- par(mfrow = c(1, n_pts), mar = c(2.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_pts)) {
    y_vals <- c(v_hat_0[k], v_hat_1[k],
                sapply(iv_0, `[`, k), sapply(iv_1, `[`, k))
    y_pad  <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = panel_ylabs[k],
         main = panel_mains[k])
    axis(1, at = c(1, 2), labels = group_labels)
    .draw_interval_pair(1, v_hat_0, iv_0, k, group_colors[1])
    .draw_interval_pair(2, v_hat_1, iv_1, k, group_colors[2])
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}

## Contrast panels: one panel per point, the group 1 - group 0 estimate with
## nested pointwise / joint intervals and a zero reference line.
.plot_contrast_cross_sections <- function(v_hat_diff, iv_diff,
                                          panel_mains, panel_ylabs,
                                          main, diff_label) {
  n_pts   <- length(v_hat_diff)
  old_par <- par(mfrow = c(1, n_pts), mar = c(2.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_pts)) {
    y_vals <- c(v_hat_diff[k], sapply(iv_diff, `[`, k), 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = panel_ylabs[k],
         main = panel_mains[k])
    axis(1, at = 1, labels = diff_label)
    abline(h = 0, col = "red", lty = 2)
    .draw_interval_pair(1, v_hat_diff, iv_diff, k, "black")
  }
  mtext(main, outer = TRUE, cex = 1.1, font = 2)
  invisible(NULL)
}

## Two-group probability mass relative to fixed activity-intensity thresholds
## (one panel per threshold) with pointwise and joint bootstrap intervals. Each
## `thresholds` entry is a list(label=, value=, col=) as used by the CDF plots.
## For panel k, `eval_type[k]` selects the vertical-axis summary: "cdf" plots
## F(threshold) (mass at or below the intensity) and "survival" plots
## 1 - F(threshold) (mass above it); `eval_type` may be NULL (all "cdf"), a
## single value, or a vector matching `thresholds`. The intervals are cross
## sections of the CDF / survival bands on `y_grid` (default: the same
## default_y_grid_two_group used by the CDF plots); the survival joint band has
## the same critical value as the CDF one. `panel_mains` / `panel_ylabs`
## override the per-panel title and y-axis label (defaults: threshold label,
## and a CDF-terms label).
plot_decoded_conditional_thresholds <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    thresholds,
    eval_type = NULL,
    panel_mains = NULL,
    panel_ylabs = NULL,
    y_grid = NULL,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  thr_vals   <- sapply(thresholds, `[[`, "value")
  thr_labels <- sapply(thresholds, `[[`, "label")
  n_thr      <- length(thr_vals)

  eval_type <- .check_threshold_eval_type(eval_type, n_thr)
  if (is.null(panel_mains)) panel_mains <- thr_labels
  if (is.null(panel_ylabs)) {
    panel_ylabs <- .default_threshold_ylabs(eval_type, thr_vals, contrast = FALSE)
  }
  stopifnot(length(panel_mains) == n_thr, length(panel_ylabs) == n_thr)
  flip <- eval_type == "survival"

  ## Joint critical values over the full CDF grid
  crit_0 <- .joint_crit(.eval_boot_rows(Qi_boot_0, qi_to_cdf, y_grid),
                        qi_to_cdf(Qi_hat_0, y_grid), alpha)
  crit_1 <- .joint_crit(.eval_boot_rows(Qi_boot_1, qi_to_cdf, y_grid),
                        qi_to_cdf(Qi_hat_1, y_grid), alpha)

  ## Summaries at the thresholds
  v_hat_0  <- .flip_survival(qi_to_cdf(Qi_hat_0, thr_vals), flip)
  v_hat_1  <- .flip_survival(qi_to_cdf(Qi_hat_1, thr_vals), flip)
  v_boot_0 <- .flip_survival(.eval_boot_rows(Qi_boot_0, qi_to_cdf, thr_vals), flip)
  v_boot_1 <- .flip_survival(.eval_boot_rows(Qi_boot_1, qi_to_cdf, thr_vals), flip)

  .plot_two_group_cross_sections(
    v_hat_0, v_hat_1,
    .boot_intervals_at(v_boot_0, v_hat_0, crit_0, alpha),
    .boot_intervals_at(v_boot_1, v_hat_1, crit_1, alpha),
    panel_mains, panel_ylabs, main, group_labels, group_colors
  )
}

## Group-difference probability mass at fixed thresholds (group 1 - group 0, one
## panel per threshold) with pointwise and joint intervals from the
## within-replicate bootstrap contrast, as cross sections of the CDF / survival
## contrast bands on `y_grid`. `eval_type[k]` picks F(threshold) ("cdf") or
## 1 - F(threshold) ("survival") for panel k; the survival contrast is the
## negative of the CDF contrast. `panel_mains` / `panel_ylabs` override the
## per-panel title and y-axis label.
plot_decoded_conditional_threshold_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    thresholds,
    eval_type = NULL,
    panel_mains = NULL,
    panel_ylabs = NULL,
    y_grid = NULL,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  if (is.null(y_grid)) {
    y_grid <- default_y_grid_two_group(Qi_hat_0, Qi_hat_1,
                                        Qi_boot_0, Qi_boot_1)
  }

  thr_vals   <- sapply(thresholds, `[[`, "value")
  thr_labels <- sapply(thresholds, `[[`, "label")
  n_thr      <- length(thr_vals)

  eval_type <- .check_threshold_eval_type(eval_type, n_thr)
  if (is.null(panel_mains)) panel_mains <- thr_labels
  if (is.null(panel_ylabs)) {
    panel_ylabs <- .default_threshold_ylabs(eval_type, thr_vals, contrast = TRUE)
  }
  stopifnot(length(panel_mains) == n_thr, length(panel_ylabs) == n_thr)
  flip <- eval_type == "survival"

  ## Joint critical value over the full CDF-contrast grid
  crit_diff <- .joint_crit(
    .eval_boot_rows(Qi_boot_1, qi_to_cdf, y_grid) -
      .eval_boot_rows(Qi_boot_0, qi_to_cdf, y_grid),
    qi_to_cdf(Qi_hat_1, y_grid) - qi_to_cdf(Qi_hat_0, y_grid),
    alpha
  )

  ## Contrasts at the thresholds
  v_hat_diff <-
    .flip_survival(qi_to_cdf(Qi_hat_1, thr_vals), flip) -
    .flip_survival(qi_to_cdf(Qi_hat_0, thr_vals), flip)
  v_boot_diff <-
    .flip_survival(.eval_boot_rows(Qi_boot_1, qi_to_cdf, thr_vals), flip) -
    .flip_survival(.eval_boot_rows(Qi_boot_0, qi_to_cdf, thr_vals), flip)

  .plot_contrast_cross_sections(
    v_hat_diff,
    .boot_intervals_at(v_boot_diff, v_hat_diff, crit_diff, alpha),
    panel_mains, panel_ylabs, main, diff_label
  )
}

## Two-group quantile values Q(p) at fixed quantile levels (one panel per level)
## with pointwise and joint bootstrap intervals, as cross sections of the bands
## in plot_decoded_conditional_qfs. Each `quantile_levels` entry is a
## list(label=, value=, col=) as used by the QF plots. `panel_mains` /
## `panel_ylabs` override the per-panel title and y-axis label (defaults: level
## label, and "Q(p)").
plot_decoded_conditional_quantile_levels <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    quantile_levels,
    panel_mains = NULL,
    panel_ylabs = NULL,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red")
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  p_vals   <- sapply(quantile_levels, `[[`, "value")
  p_labels <- sapply(quantile_levels, `[[`, "label")
  n_p      <- length(p_vals)

  if (is.null(panel_mains)) panel_mains <- p_labels
  if (is.null(panel_ylabs)) panel_ylabs <- sprintf("Q(%g)", p_vals)
  stopifnot(length(panel_mains) == n_p, length(panel_ylabs) == n_p)

  ## Joint critical values over the full p-grid
  crit_0 <- .joint_crit(Qi_boot_0, Qi_hat_0, alpha)
  crit_1 <- .joint_crit(Qi_boot_1, Qi_hat_1, alpha)

  ## Quantiles at the levels
  v_hat_0  <- qi_at_p(Qi_hat_0, p_vals)
  v_hat_1  <- qi_at_p(Qi_hat_1, p_vals)
  v_boot_0 <- .eval_boot_rows(Qi_boot_0, qi_at_p, p_vals)
  v_boot_1 <- .eval_boot_rows(Qi_boot_1, qi_at_p, p_vals)

  .plot_two_group_cross_sections(
    v_hat_0, v_hat_1,
    .boot_intervals_at(v_boot_0, v_hat_0, crit_0, alpha),
    .boot_intervals_at(v_boot_1, v_hat_1, crit_1, alpha),
    panel_mains, panel_ylabs, main, group_labels, group_colors
  )
}

## Group-difference quantile values Q(p) at fixed quantile levels (group 1 -
## group 0, one panel per level) with pointwise and joint intervals from the
## within-replicate bootstrap contrast, as cross sections of the bands in
## plot_decoded_conditional_qf_diff.
plot_decoded_conditional_quantile_level_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    quantile_levels,
    panel_mains = NULL,
    panel_ylabs = NULL,
    alpha = 0.05, main = "", diff_label = "1 - 0"
) {
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)

  p_vals   <- sapply(quantile_levels, `[[`, "value")
  p_labels <- sapply(quantile_levels, `[[`, "label")
  n_p      <- length(p_vals)

  if (is.null(panel_mains)) panel_mains <- p_labels
  if (is.null(panel_ylabs)) panel_ylabs <- sprintf("Q(%g) Contrast", p_vals)
  stopifnot(length(panel_mains) == n_p, length(panel_ylabs) == n_p)

  ## Joint critical value over the full p-grid
  crit_diff <- .joint_crit(Qi_boot_1 - Qi_boot_0, Qi_hat_1 - Qi_hat_0, alpha)

  ## Contrasts at the levels
  v_hat_diff  <- qi_at_p(Qi_hat_1, p_vals) - qi_at_p(Qi_hat_0, p_vals)
  v_boot_diff <- .eval_boot_rows(Qi_boot_1, qi_at_p, p_vals) -
                 .eval_boot_rows(Qi_boot_0, qi_at_p, p_vals)

  .plot_contrast_cross_sections(
    v_hat_diff,
    .boot_intervals_at(v_boot_diff, v_hat_diff, crit_diff, alpha),
    panel_mains, panel_ylabs, main, diff_label
  )
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
  band_lo_0  <- apply(mom_boot_0, 2, quantile, probs = alpha / 2)
  band_hi_0  <- apply(mom_boot_0, 2, quantile, probs = 1 - alpha / 2)
  band_lo_1  <- apply(mom_boot_1, 2, quantile, probs = alpha / 2)
  band_hi_1  <- apply(mom_boot_1, 2, quantile, probs = 1 - alpha / 2)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(2.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_0[k], band_hi_0[k], mom_hat_0[k],
                band_lo_1[k], band_hi_1[k], mom_hat_1[k])
    y_pad <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "", ylab = moment_names[k],
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
  band_lo_diff  <- apply(mom_boot_diff, 2, quantile, probs = alpha / 2)
  band_hi_diff  <- apply(mom_boot_diff, 2, quantile, probs = 1 - alpha / 2)

  moment_names <- c("Mean", "Variance", "Skewness", "Kurtosis")
  old_par <- par(mfrow = c(1, 4), mar = c(2.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_diff[k], band_hi_diff[k], mom_hat_diff[k], 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    plot(NULL,
         xlim = c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = str_glue("{moment_names[k]} Contrast"),
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
