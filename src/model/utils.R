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
## pointwise interval and a thin capped whisker for the joint interval. size
## scales the line widths, cap length and point size together.
.draw_interval_pair <- function(x, v_hat, iv, k, col, size = 1) {
  arrows(x, iv$jt_lo[k], x, iv$jt_hi[k], angle = 90, code = 3,
         length = 0.08 * size, col = col, lwd = 1.5 * size)
  segments(x, iv$pw_lo[k], x, iv$pw_hi[k], col = adjustcolor(col, 0.5),
           lwd = 8 * size, lend = "butt")
  points(x, v_hat[k], col = col, pch = 19, cex = 1.6 * size)
}

## par() for a row of panels: mfrow, margins and outer title room (none when
## main is blank); cex (if non-NULL) then overrides mfrow's automatic text
## shrink. Returns old par.
.panel_row_par <- function(n_panels, mar, cex = NULL, main = "") {
  oma_top <- if (.is_blank(main)) 0 else 2
  old_par <- par(mfrow = c(1, n_panels), mar = mar, oma = c(0, 0, oma_top, 0))
  if (!is.null(cex)) old_par <- c(old_par, par(cex = cex))
  old_par
}

## Outer (figure) title for a row of panels: the original 1.1 when cex is NULL,
## otherwise the current title size.
.panel_row_title <- function(main, cex = NULL) {
  mtext(main, outer = TRUE, font = 2,
        cex = if (is.null(cex)) 1.1 else cex * par("cex.main"))
}


## ---------- Legend placement ----------

## Choose where a legend goes so it sits inside the plot box without covering
## any drawn object, raising (or lowering) one end of ylim if no corner is
## clear. Starts a new plot on the current device (plot.new + a provisional
## plot.window); the caller then draws with .draw_frame(xlim, out$ylim, ...)
## and places the legend at out$pos. objs is a list of list(x, lo, hi), one
## per curve (lo = hi = y), band (lo/hi edges) or vertical line (x = v,
## lo/hi = its extent). pref is tried first; pad is the clearance between
## legend and objects as a fraction of the plot height.
.legend_layout <- function(leg_args, xlim, ylim, objs, pref = "topleft",
                           pad = 0.02, extend = c("top", "bottom")) {
  plot.new()
  plot.window(xlim = xlim, ylim = ylim)
  usr  <- par("usr")
  span <- diff(usr[3:4])
  ext  <- 0.04   # yaxs = "r" pads usr by 4% of the ylim range on each side
  corners <- unique(c(pref, "topleft", "topright", "bottomleft", "bottomright"))
  ## Corners on a side that may not be extended are only used if already clear
  can_extend <- function(pos) {
    any(vapply(extend, function(s) grepl(paste0("^", s), pos), logical(1)))
  }

  ## Objects' y-extent under the legend's x-span
  y_under <- function(r) {
    vals <- unlist(lapply(objs, function(o) {
      k <- o$x >= r$left & o$x <= r$left + r$w
      c(o$lo[k], o$hi[k])
    }))
    if (length(vals)) range(vals) else NULL
  }

  best <- NULL
  for (pos in corners) {
    r  <- do.call(legend, c(list(pos), leg_args, plot = FALSE))$rect
    yr <- y_under(r)
    y0 <- r$top - r$h - pad * span
    y1 <- r$top + pad * span
    if (is.null(yr) || yr[2] < y0 || yr[1] > y1) {
      return(list(ylim = ylim, pos = pos))
    }
    if (!can_extend(pos)) next
    ## Smallest ylim end-shift that clears the objects at this corner
    f  <- r$h / span
    c_ <- ext - (1 + 2 * ext) * (f + pad)
    if (grepl("^top", pos)) {
      new_ylim <- c(ylim[1], max(ylim[2], (yr[2] + c_ * ylim[1]) / (1 + c_)))
    } else {
      new_ylim <- c(min(ylim[1], (yr[1] + c_ * ylim[2]) / (1 + c_)), ylim[2])
    }
    if (is.null(best) || diff(new_ylim) < diff(best$ylim)) {
      best <- list(ylim = new_ylim, pos = pos)
    }
  }
  best
}

## TRUE for a title / axis label that draws nothing ("" or NULL); expressions
## count as non-blank.
.is_blank <- function(x) {
  is.null(x) || (is.character(x) && !any(nzchar(x)))
}

## Margins with the whitespace for an omitted title or axis label removed: the
## top shrinks to a small gap without a title, and a side without an axis label
## keeps only the room its tick labels need. Returns the margins (does not set
## them).
.trimmed_mar <- function(mar, main = "", xlab = "", ylab = "") {
  tick_room <- par("mgp")[2] + par("cex.axis") * 1.2 + 0.3
  if (.is_blank(main)) mar[3] <- 0.5
  if (.is_blank(xlab)) mar[1] <- min(mar[1], tick_room)
  if (.is_blank(ylab)) mar[2] <- min(mar[2], tick_room)
  mar
}

## Axes, box and titles for a plot whose plot.new() has already been called
## (e.g. by .legend_layout). y ticks are labelled only within ylim_data (the
## data range before any legend headroom was added), so headroom stays blank.
## main_line (if non-NULL) sets the title's margin line, e.g. to sit above
## threshold labels.
.draw_frame <- function(xlim, ylim, main = "", xlab = "", ylab = "",
                        ylim_data = ylim, main_line = NULL, x_at = NULL) {
  plot.window(xlim = xlim, ylim = ylim)
  if (is.null(x_at)) axis(1) else axis(1, at = x_at)   # x_at: explicit x ticks
  ## Label ticks within the data range plus the usual 4% axis padding
  y_at  <- axTicks(2)
  y_pad <- 0.04 * diff(range(ylim_data)) + 1e-8 * diff(ylim)
  axis(2, at = y_at[y_at >= min(ylim_data) - y_pad &
                    y_at <= max(ylim_data) + y_pad])
  box()
  title(xlab = xlab, ylab = ylab)
  if (is.null(main_line)) title(main = main) else title(main = main, line = main_line)
}

## Bottom margin (lines) for the panel plots: the larger of mar_bottom and the
## room the group labels need (the 0.2-line gap .axis_group_labels leaves,
## their line count at tick-label size and 1.2 line spacing, and a small gap),
## so multi-line labels fit without extra whitespace.
.group_label_mar <- function(mar_bottom, labels) {
  ## No labels, no axis: keep room for the lowest y tick label, which runs
  ## parallel to the axis and can extend below the frame
  if (.is_blank(labels)) return(max(mar_bottom, 1))
  n_lines <- max(lengths(strsplit(as.character(labels), "\n")))
  max(mar_bottom, 0.2 + n_lines * 1.2 * par("cex.axis") + 0.5)
}

## x-axis group labels for the panel plots; multi-line labels hang below the
## axis instead of being centred on it. Blank labels draw no axis at all.
.axis_group_labels <- function(at, labels) {
  if (.is_blank(labels)) return(invisible(NULL))
  if (any(grepl("\n", labels))) {
    axis(1, at = at, labels = labels, padj = 1, mgp = c(3, 0.2, 0))
  } else {
    axis(1, at = at, labels = labels)
  }
}

## Vertical threshold lines drawn only over [y0, y1] (the data range), so a
## legend placed in headroom added above or below them stays clear.
.draw_thresholds <- function(thr_vals, thr_cols, y0, y1, lwd = 2) {
  segments(thr_vals, y0, thr_vals, y1, col = thr_cols, lty = 2, lwd = lwd)
}

## Objects (for .legend_layout) for vertical lines at thr_vals over [y0, y1]
.threshold_objs <- function(thr_vals, y0, y1) {
  lapply(thr_vals, function(v) list(x = v, lo = y0, hi = y1))
}

## Labels for vertical reference lines (thresholds / quantile levels) in one
## row of the top margin, at tick-label size. Each label is centred over its
## line unless that would overlap a neighbour: overlapping labels are merged
## into a cluster laid side by side (gap_in apart) and centred on the mean of
## their lines, repeating until no clusters overlap, then kept within the plot
## width. Labels may so sit off-centre, but their colours tie them to their
## lines. Call before plot.new() (it widens the top margin, which the caller
## restores via the returned old_par). Returns the label positions (x, user
## units) and the title line that clears them.
.threshold_label_layout <- function(thr_vals, thr_labels, xlim,
                                    row_h = 0.9, gap_in = 0.06) {
  cex_lab <- par("cex") * par("cex.axis")
  w_in    <- strwidth(thr_labels, units = "inches", cex = cex_lab)
  pin_w   <- par("pin")[1]
  usr_l   <- xlim[1] - 0.04 * diff(xlim)                # xaxs = "r" padding
  in_per_unit <- pin_w / (1.08 * diff(xlim))
  ctr <- (thr_vals - usr_l) * in_per_unit               # desired centres (in)

  ## Clusters of labels (in x order), each laid out contiguously and centred
  ## on the mean desired centre of its members
  ord <- order(ctr)
  clusters <- lapply(ord, function(k) k)
  place <- function(cl) {
    widths <- w_in[cl] + gap_in
    left   <- mean(ctr[cl]) - sum(widths) / 2
    left + cumsum(widths) - widths / 2                  # centres of members
  }
  repeat {
    merged <- FALSE
    if (length(clusters) > 1) {
      for (i in seq_len(length(clusters) - 1)) {
        a <- clusters[[i]]; b <- clusters[[i + 1]]
        a_right <- max(place(a) + (w_in[a] + gap_in) / 2)
        b_left  <- min(place(b) - (w_in[b] + gap_in) / 2)
        if (a_right > b_left) {
          clusters[[i]] <- c(a, b)
          clusters[[i + 1]] <- NULL
          merged <- TRUE
          break
        }
      }
    }
    if (!merged) break
  }
  pos_in <- numeric(length(thr_vals))
  for (cl in clusters) pos_in[cl] <- place(cl)

  ## Keep the row within the plot width (shift it as a whole if needed)
  half <- (w_in + gap_in) / 2
  shift <- max(0, -min(pos_in - half)) - max(0, max(pos_in + half) - pin_w)
  pos_in <- pos_in + shift

  old_par <- par(mar = par("mar") + c(0, 0, row_h, 0))
  list(at = usr_l + pos_in / in_per_unit, cex = cex_lab, old_par = old_par,
       main_line = 0.5 + row_h + 0.2)
}

## Draw the labels laid out by .threshold_label_layout
.draw_threshold_labels <- function(thr_vals, thr_labels, thr_cols, lay) {
  mtext(thr_labels, side = 3, at = lay$at, col = thr_cols, cex = lay$cex,
        line = 0.2)
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

## Shared arguments of the single-panel plots below:
##   legend_loc  preferred legend corner; if any curve, band or reference line
##               would sit under the legend there, the other corners are tried
##               and, failing those, the y-axis is extended to make room (see
##               .legend_layout), so the legend always stays inside the axes.
##   threshold_labels  (CDF / survival plots) "legend" lists the thresholds in
##               the legend; "top" labels each threshold line above the plot
##               frame instead (keeping the legend small), and any legend
##               headroom is then added at the bottom.
##   legend_cex  legend text size.
##   lwd         line width of the estimates, legend keys and threshold lines;
##               NULL keeps the original widths.

## Two-group conditional QFs on the (0,1) p-grid with pointwise or joint
## (sup-t across p) bootstrap bands.
plot_decoded_conditional_qfs <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft',
    legend_cex = 1,
    lwd = NULL,
    quantile_lines = NULL,
    threshold_labels = c("legend", "top")
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  b0 <- .boot_band(Qi_boot_0, Qi_hat_0, alpha, band_type)
  b1 <- .boot_band(Qi_boot_1, Qi_hat_1, alpha, band_type)

  lwd_est <- lwd %||% 1
  lwd_leg <- lwd %||% 4
  xlim <- range(pi_grid)
  q_vals   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL
  q_labels <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "label") else NULL
  q_in_leg <- threshold_labels == "legend"
  leg_q     <- if (q_in_leg) q_labels else NULL
  leg_q_col <- if (q_in_leg) q_cols   else NULL
  lwd_q     <- lwd %||% 1.5
  ylim <- range(c(Qi_hat_0, Qi_hat_1, b0$lo, b0$hi, b1$lo, b1$hi), finite = TRUE)
  q_y  <- ylim + c(-1, 1) * 0.04 * diff(ylim)   # full height before any extension
  leg_args <- list(legend = c(group_labels, leg_q),
                   col = c(group_colors, leg_q_col),
                   lwd = c(lwd_leg, lwd_leg, rep(lwd_q, length(leg_q))),
                   lty = c(1, 1, rep(2, length(leg_q))),
                   cex = legend_cex, bty = "n")
  objs <- list(
    list(x = pi_grid, lo = b0$lo, hi = b0$hi),
    list(x = pi_grid, lo = b1$lo, hi = b1$hi),
    list(x = pi_grid, lo = Qi_hat_0, hi = Qi_hat_0),
    list(x = pi_grid, lo = Qi_hat_1, hi = Qi_hat_1)
  )
  objs <- c(objs, .threshold_objs(q_vals, q_y[1], q_y[2]))
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "p", "Q(p)"))
  on.exit(par(old_mar), add = TRUE)
  ## Quantile levels labelled above the frame: widen the top margin for them,
  ## and add any legend headroom at the bottom so the lines reach their labels
  top_lab <- !q_in_leg && !is.null(quantile_lines)
  if (top_lab) {
    tl <- .threshold_label_layout(q_vals, q_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "p", ylab = "Q(p)",
              ylim_data = ylim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(pi_grid, b0$lo, b0$hi)
  draw_band(pi_grid, b1$lo, b1$hi)
  if (!is.null(quantile_lines)) {
    .draw_thresholds(q_vals, q_cols, q_y[1], q_y[2], lwd = lwd_q)
  }
  lines(pi_grid, Qi_hat_0, col = group_colors[1], lwd = lwd_est)
  lines(pi_grid, Qi_hat_1, col = group_colors[2], lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(q_vals, q_labels, q_cols, tl)
  invisible(NULL)
}

## Group-difference QF (group 1 - group 0) with pointwise or joint (sup-t
## across p) bands using within-replicate subtraction so bands pinch at any
## common point.
plot_decoded_conditional_qf_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0",
    band_type = c("joint", "pointwise"),
    legend_loc = 'topleft',
    legend_cex = 1,
    lwd = NULL,
    quantile_lines = NULL,
    threshold_labels = c("legend", "top")
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
  Qi_boot_0 <- .as_qi_boot_matrix(Qi_boot_0)
  Qi_boot_1 <- .as_qi_boot_matrix(Qi_boot_1)
  Ji        <- length(Qi_hat_0)
  pi_grid   <- pi_grid_fun(Ji)

  Qi_hat_diff   <- Qi_hat_1 - Qi_hat_0
  mat_boot_diff <- Qi_boot_1 - Qi_boot_0
  bd <- .boot_band(mat_boot_diff, Qi_hat_diff, alpha, band_type)

  lwd_est <- lwd %||% 4
  xlim <- range(pi_grid)
  q_vals   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "value") else NULL
  q_cols   <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "col")   else NULL
  q_labels <- if (!is.null(quantile_lines)) sapply(quantile_lines, `[[`, "label") else NULL
  q_in_leg <- threshold_labels == "legend"
  leg_q     <- if (q_in_leg) q_labels else NULL
  leg_q_col <- if (q_in_leg) q_cols   else NULL
  lwd_q     <- lwd %||% 1.5
  ylim <- range(c(Qi_hat_diff, bd$lo, bd$hi, 0), finite = TRUE)
  q_y  <- ylim + c(-1, 1) * 0.04 * diff(ylim)   # full height before any extension
  leg_args <- list(legend = c(diff_label, leg_q),
                   col = c("black", leg_q_col),
                   lwd = c(lwd_est, rep(lwd_q, length(leg_q))),
                   lty = c(1, rep(2, length(leg_q))),
                   cex = legend_cex, bty = "n")
  objs <- list(
    list(x = pi_grid, lo = bd$lo, hi = bd$hi),
    list(x = pi_grid, lo = Qi_hat_diff, hi = Qi_hat_diff),
    list(x = pi_grid, lo = 0 * pi_grid, hi = 0 * pi_grid)   # zero reference line
  )
  objs <- c(objs, .threshold_objs(q_vals, q_y[1], q_y[2]))
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "p", "Q(p) Contrast"))
  on.exit(par(old_mar), add = TRUE)
  ## Quantile levels labelled above the frame: widen the top margin for them,
  ## and add any legend headroom at the bottom so the lines reach their labels
  top_lab <- !q_in_leg && !is.null(quantile_lines)
  if (top_lab) {
    tl <- .threshold_label_layout(q_vals, q_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "p", ylab = "Q(p) Contrast",
              ylim_data = ylim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(pi_grid, bd$lo, bd$hi)
  abline(h = 0, col = "red", lty = 3)
  if (!is.null(quantile_lines)) {
    .draw_thresholds(q_vals, q_cols, q_y[1], q_y[2], lwd = lwd_q)
  }
  lines(pi_grid, Qi_hat_diff, col = "black", lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(q_vals, q_labels, q_cols, tl)
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
    legend_loc = 'topleft',
    threshold_labels = c("legend", "top"),
    legend_cex = 1,
    lwd = NULL
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
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
  thr_in_leg <- threshold_labels == "legend"
  leg_thr     <- if (thr_in_leg) thr_labels else NULL
  leg_thr_col <- if (thr_in_leg) thr_cols   else NULL

  lwd_est <- lwd %||% 4
  lwd_thr <- lwd %||% 2
  xlim <- range(y_grid)
  ylim <- c(0, 1)
  thr_y <- ylim + c(-1, 1) * 0.04 * diff(ylim)   # full height before any extension
  leg_args <- list(
    legend = c(group_labels, leg_thr),
    col    = c(group_colors, leg_thr_col),
    lwd    = c(lwd_est, lwd_est, rep(lwd_thr, length(leg_thr))),
    lty    = c(1, 1, rep(2, length(leg_thr))),
    cex    = legend_cex,
    bty    = "n"
  )
  objs <- c(
    list(
      list(x = y_grid, lo = b0$lo, hi = b0$hi),
      list(x = y_grid, lo = b1$lo, hi = b1$hi),
      list(x = y_grid, lo = F_hat_0, hi = F_hat_0),
      list(x = y_grid, lo = F_hat_1, hi = F_hat_1)
    ),
    .threshold_objs(thr_vals, thr_y[1], thr_y[2])
  )
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "y", "F(y)"))
  on.exit(par(old_mar), add = TRUE)
  ## Thresholds labelled above the frame: widen the top margin for them, and
  ## add any legend headroom at the bottom so the lines still reach their labels
  top_lab <- !thr_in_leg && !is.null(thresholds)
  if (top_lab) {
    tl <- .threshold_label_layout(thr_vals, thr_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "y", ylab = "F(y)",
              ylim_data = ylim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(y_grid, b0$lo, b0$hi)
  draw_band(y_grid, b1$lo, b1$hi)
  if (!is.null(thresholds)) {
    .draw_thresholds(thr_vals, thr_cols, thr_y[1], thr_y[2], lwd = lwd_thr)
  }
  lines(y_grid, F_hat_0, col = group_colors[1], lwd = lwd_est)
  lines(y_grid, F_hat_1, col = group_colors[2], lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(thr_vals, thr_labels, thr_cols, tl)
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
    legend_loc = 'topleft',
    threshold_labels = c("legend", "top"),
    legend_cex = 1,
    lwd = NULL
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
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
  thr_in_leg <- threshold_labels == "legend"
  leg_thr     <- if (thr_in_leg) thr_labels else NULL
  leg_thr_col <- if (thr_in_leg) thr_cols   else NULL

  lwd_est     <- lwd %||% 4
  lwd_thr     <- lwd %||% 2
  lwd_thr_leg <- lwd %||% 1.5
  xlim  <- range(y_grid)
  y_lim <- range(c(F_hat_diff, band_lo_diff, band_hi_diff, 0))
  thr_y <- y_lim + c(-1, 1) * 0.04 * diff(y_lim)
  leg_args <- list(
    legend = c(diff_label, leg_thr),
    col    = c("black", leg_thr_col),
    lwd    = c(lwd_est, rep(lwd_thr_leg, length(leg_thr))),
    lty    = c(1, rep(2, length(leg_thr))),
    cex    = legend_cex,
    bty    = "n"
  )
  objs <- c(
    list(
      list(x = y_grid, lo = band_lo_diff, hi = band_hi_diff),
      list(x = y_grid, lo = F_hat_diff, hi = F_hat_diff),
      list(x = y_grid, lo = 0 * y_grid, hi = 0 * y_grid)   # zero reference line
    ),
    .threshold_objs(thr_vals, thr_y[1], thr_y[2])
  )
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "y", "F(y) Contrast"))
  on.exit(par(old_mar), add = TRUE)
  ## Thresholds labelled above the frame: widen the top margin for them, and
  ## add any legend headroom at the bottom so the lines still reach their labels
  top_lab <- !thr_in_leg && !is.null(thresholds)
  if (top_lab) {
    tl <- .threshold_label_layout(thr_vals, thr_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, y_lim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "y", ylab = "F(y) Contrast",
              ylim_data = y_lim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(y_grid, band_lo_diff, band_hi_diff)
  abline(h = 0, col = "red", lty = 3)
  if (!is.null(thresholds)) {
    .draw_thresholds(thr_vals, thr_cols, thr_y[1], thr_y[2], lwd = lwd_thr)
  }
  lines(y_grid, F_hat_diff, col = "black", lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(thr_vals, thr_labels, thr_cols, tl)
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
    legend_loc = 'topright',
    threshold_labels = c("legend", "top"),
    legend_cex = 1,
    lwd = NULL
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
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
  thr_in_leg <- threshold_labels == "legend"
  leg_thr     <- if (thr_in_leg) thr_labels else NULL
  leg_thr_col <- if (thr_in_leg) thr_cols   else NULL

  lwd_est <- lwd %||% 4
  lwd_thr <- lwd %||% 2
  xlim <- range(y_grid)
  ylim <- c(0, 1)
  thr_y <- ylim + c(-1, 1) * 0.04 * diff(ylim)
  leg_args <- list(
    legend = c(group_labels, leg_thr),
    col    = c(group_colors, leg_thr_col),
    lwd    = c(lwd_est, lwd_est, rep(lwd_thr, length(leg_thr))),
    lty    = c(1, 1, rep(2, length(leg_thr))),
    cex    = legend_cex,
    bty    = "n"
  )
  objs <- c(
    list(
      list(x = y_grid, lo = b0$lo, hi = b0$hi),
      list(x = y_grid, lo = b1$lo, hi = b1$hi),
      list(x = y_grid, lo = S_hat_0, hi = S_hat_0),
      list(x = y_grid, lo = S_hat_1, hi = S_hat_1)
    ),
    .threshold_objs(thr_vals, thr_y[1], thr_y[2])
  )
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "y", "1 - F(y)"))
  on.exit(par(old_mar), add = TRUE)
  ## Thresholds labelled above the frame: widen the top margin for them, and
  ## add any legend headroom at the bottom so the lines still reach their labels
  top_lab <- !thr_in_leg && !is.null(thresholds)
  if (top_lab) {
    tl <- .threshold_label_layout(thr_vals, thr_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, ylim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "y", ylab = "1 - F(y)",
              ylim_data = ylim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(y_grid, b0$lo, b0$hi)
  draw_band(y_grid, b1$lo, b1$hi)
  if (!is.null(thresholds)) {
    .draw_thresholds(thr_vals, thr_cols, thr_y[1], thr_y[2], lwd = lwd_thr)
  }
  lines(y_grid, S_hat_0, col = group_colors[1], lwd = lwd_est)
  lines(y_grid, S_hat_1, col = group_colors[2], lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(thr_vals, thr_labels, thr_cols, tl)
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
    legend_loc = 'topleft',
    threshold_labels = c("legend", "top"),
    legend_cex = 1,
    lwd = NULL
) {
  band_type <- match.arg(band_type)
  threshold_labels <- match.arg(threshold_labels)
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
  thr_in_leg <- threshold_labels == "legend"
  leg_thr     <- if (thr_in_leg) thr_labels else NULL
  leg_thr_col <- if (thr_in_leg) thr_cols   else NULL

  lwd_est     <- lwd %||% 4
  lwd_thr     <- lwd %||% 2
  lwd_thr_leg <- lwd %||% 1.5
  xlim  <- range(y_grid)
  y_lim <- range(c(S_hat_diff, bd$lo, bd$hi, 0))
  thr_y <- y_lim + c(-1, 1) * 0.04 * diff(y_lim)
  leg_args <- list(
    legend = c(diff_label, leg_thr),
    col    = c("black", leg_thr_col),
    lwd    = c(lwd_est, rep(lwd_thr_leg, length(leg_thr))),
    lty    = c(1, rep(2, length(leg_thr))),
    cex    = legend_cex,
    bty    = "n"
  )
  objs <- c(
    list(
      list(x = y_grid, lo = bd$lo, hi = bd$hi),
      list(x = y_grid, lo = S_hat_diff, hi = S_hat_diff),
      list(x = y_grid, lo = 0 * y_grid, hi = 0 * y_grid)   # zero reference line
    ),
    .threshold_objs(thr_vals, thr_y[1], thr_y[2])
  )
  ## Drop margin space for an omitted title
  old_mar <- par(mar = .trimmed_mar(par("mar"), main, "y", "(1 - F(y)) Contrast"))
  on.exit(par(old_mar), add = TRUE)
  ## Thresholds labelled above the frame: widen the top margin for them, and
  ## add any legend headroom at the bottom so the lines still reach their labels
  top_lab <- !thr_in_leg && !is.null(thresholds)
  if (top_lab) {
    tl <- .threshold_label_layout(thr_vals, thr_labels, xlim)
    on.exit(par(tl$old_par), add = TRUE, after = FALSE)   # undo before old_mar
  }
  lay <- .legend_layout(leg_args, xlim, y_lim, objs, pref = legend_loc,
                        extend = if (top_lab) "bottom" else c("top", "bottom"))

  .draw_frame(xlim, lay$ylim, main = main, xlab = "y",
              ylab = "(1 - F(y)) Contrast", ylim_data = y_lim,
              main_line = if (top_lab) tl$main_line else NULL)
  draw_band(y_grid, bd$lo, bd$hi)
  abline(h = 0, col = "red", lty = 3)
  if (!is.null(thresholds)) {
    .draw_thresholds(thr_vals, thr_cols, thr_y[1], thr_y[2], lwd = lwd_thr)
  }
  lines(y_grid, S_hat_diff, col = "black", lwd = lwd_est)
  do.call(legend, c(list(lay$pos), leg_args))
  if (top_lab) .draw_threshold_labels(thr_vals, thr_labels, thr_cols, tl)
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
                                           main, group_labels, group_colors,
                                           cex = NULL,
                                           mar = c(2.5, 4.5, 3, 1),
                                           size = 1,
                                           xlim = NULL) {
  n_pts   <- length(v_hat_0)
  old_par <- .panel_row_par(n_pts, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_pts)) {
    y_vals <- c(v_hat_0[k], v_hat_1[k],
                sapply(iv_0, `[`, k), sapply(iv_1, `[`, k))
    y_pad  <- 0.08 * diff(range(y_vals))
    ## Trim the top / left margin for an omitted panel title / y label (the
    ## bottom keeps its room: the group labels sit there)
    mar_k <- .trimmed_mar(mar, panel_mains[k], "x", panel_ylabs[k])
    mar_k[1] <- .group_label_mar(mar[1], group_labels)
    par(mar = mar_k)
    plot(NULL,
         xlim = xlim %||% c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = panel_ylabs[k],
         main = panel_mains[k])
    .axis_group_labels(c(1, 2), group_labels)
    .draw_interval_pair(1, v_hat_0, iv_0, k, group_colors[1], size)
    .draw_interval_pair(2, v_hat_1, iv_1, k, group_colors[2], size)
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}

## Contrast panels: one panel per point, the group 1 - group 0 estimate with
## nested pointwise / joint intervals and a zero reference line.
.plot_contrast_cross_sections <- function(v_hat_diff, iv_diff,
                                          panel_mains, panel_ylabs,
                                          main, diff_label,
                                          cex = NULL,
                                          mar = c(2.5, 4.5, 3, 1),
                                          size = 1,
                                          xlim = NULL) {
  n_pts   <- length(v_hat_diff)
  old_par <- .panel_row_par(n_pts, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_len(n_pts)) {
    y_vals <- c(v_hat_diff[k], sapply(iv_diff, `[`, k), 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    ## Trim the top / left margin for an omitted panel title / y label (the
    ## bottom keeps its room: the group labels sit there)
    mar_k <- .trimmed_mar(mar, panel_mains[k], "x", panel_ylabs[k])
    mar_k[1] <- .group_label_mar(mar[1], diff_label)
    par(mar = mar_k)
    plot(NULL,
         xlim = xlim %||% c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = panel_ylabs[k],
         main = panel_mains[k])
    .axis_group_labels(1, diff_label)
    abline(h = 0, col = "red", lty = 3)
    .draw_interval_pair(1, v_hat_diff, iv_diff, k, "black", size)
  }
  .panel_row_title(main, cex)
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
    group_colors = c("black", "red"),
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    xlim = NULL
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
    panel_mains, panel_ylabs, main, group_labels, group_colors,
    cex = cex, mar = mar, size = size,
    xlim = xlim
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
    alpha = 0.05, main = "", diff_label = "1 - 0",
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    xlim = NULL
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
    panel_mains, panel_ylabs, main, diff_label,
    cex = cex, mar = mar, size = size,
    xlim = xlim
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
    group_colors = c("black", "red"),
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    xlim = NULL
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
    panel_mains, panel_ylabs, main, group_labels, group_colors,
    cex = cex, mar = mar, size = size,
    xlim = xlim
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
    alpha = 0.05, main = "", diff_label = "1 - 0",
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    xlim = NULL
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
    panel_mains, panel_ylabs, main, diff_label,
    cex = cex, mar = mar, size = size,
    xlim = xlim
  )
}

## Two-group moments (4 panels: mean, variance, skewness, kurtosis) with
## interval bars from the bootstrap.
plot_decoded_conditional_moments <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "",
    group_labels = c("0", "1"),
    group_colors = c("black", "red"),
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    panel_ylabs = NULL,
    xlim = NULL
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
  if (is.null(panel_ylabs)) panel_ylabs <- moment_names
  mar[1]  <- .group_label_mar(mar[1], group_labels)
  old_par <- .panel_row_par(4, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_0[k], band_hi_0[k], mom_hat_0[k],
                band_lo_1[k], band_hi_1[k], mom_hat_1[k])
    y_pad <- 0.08 * diff(range(y_vals))
    par(mar = .trimmed_mar(mar, moment_names[k], "x", panel_ylabs[k]))
    plot(NULL,
         xlim = xlim %||% c(0.5, 2.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "", ylab = panel_ylabs[k],
         main = moment_names[k])
    .axis_group_labels(c(1, 2), group_labels)
    arrows(1, band_lo_0[k], 1, band_hi_0[k],
           angle = 90, code = 3, length = 0.08 * size,
           col = group_colors[1], lwd = 1.5 * size)
    points(1, mom_hat_0[k], col = group_colors[1], pch = 19, cex = 1.6 * size)
    arrows(2, band_lo_1[k], 2, band_hi_1[k],
           angle = 90, code = 3, length = 0.08 * size,
           col = group_colors[2], lwd = 1.5 * size)
    points(2, mom_hat_1[k], col = group_colors[2], pch = 19, cex = 1.6 * size)
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}

## Group-difference moments (4 panels) with interval bars from the
## within-replicate bootstrap contrast.
plot_decoded_conditional_moment_diff <- function(
    Qi_hat_0, Qi_hat_1, Qi_boot_0, Qi_boot_1,
    alpha = 0.05, main = "", diff_label = "1 - 0",
    cex = NULL,
    mar = c(2.5, 4.5, 3, 1),
    size = 1,
    panel_ylabs = NULL,
    xlim = NULL
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
  if (is.null(panel_ylabs)) panel_ylabs <- paste(moment_names, "Contrast")
  mar[1]  <- .group_label_mar(mar[1], diff_label)
  old_par <- .panel_row_par(4, mar, cex, main)
  on.exit(par(old_par), add = TRUE)
  for (k in seq_along(moment_names)) {
    y_vals <- c(band_lo_diff[k], band_hi_diff[k], mom_hat_diff[k], 0)
    y_pad  <- 0.08 * diff(range(y_vals))
    par(mar = .trimmed_mar(mar, moment_names[k], "x", panel_ylabs[k]))
    plot(NULL,
         xlim = xlim %||% c(0.5, 1.5),
         ylim = c(min(y_vals) - y_pad, max(y_vals) + y_pad),
         xaxt = "n", xlab = "",
         ylab = panel_ylabs[k],
         main = moment_names[k])
    .axis_group_labels(1, diff_label)
    abline(h = 0, col = "red", lty = 3)
    arrows(1, band_lo_diff[k], 1, band_hi_diff[k],
           angle = 90, code = 3, length = 0.08 * size,
           col = "black", lwd = 1.5 * size)
    points(1, mom_hat_diff[k], col = "black", pch = 19, cex = 1.6 * size)
  }
  .panel_row_title(main, cex)
  invisible(NULL)
}
