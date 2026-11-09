## Supplementary NHANES analyses that are not part of the main analyses in
## demo_nhanes.R: an inspection of the main pipeline's generativity
## (unrealistic synthetic QFs and where they sit in the latent Gaussian), a
## G-PCA representation of the quantile functions and hierarchical clustering
## of the latent embeddings.

library(dplyr)
library(mgcv)
library(purrr)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')




## ========== Setup ========== ##

## Mirrors the top of demo_nhanes.R (data, pipeline, covariates, figure spec).
## Keep the two in sync -- in particular dir_art and the sign flips, which
## determine the orientation of every latent coordinate plotted below.

## Globals
dir_art <- 'demo_nhanes_v11'

## Load data
path <- file.path('data', 'processed', 'nhanes-fda-mims_v2.rds')
y_list <- readRDS(path)
N <- length(y_list)

## Define grid
p_grid <- p_grid_fun(
  breaks = c(1/(7350 + 1), 0.95, 7350/(7350 + 1)),
  interval_counts = c(51, 50)
)

## Load the main pipeline fit by demo_nhanes.R. It is never refit here:
## stage seeds come from gen_seed(), so a refit would train a different flow
## and the embeddings would no longer match the main analyses.
path <- str_glue('artifacts/{dir_art}/pipe_nhanes.rds')
if (!file.exists(path)) {
  stop(str_glue("{path} not found; run demo_nhanes.R first to fit the pipeline."))
}
pipeline <- readRDS(path)
pipeline$latent <- 'z_rot'
pipeline <- set_rotation_signs(pipeline, flip = c(1, 2, 3, 4, 5))
pipeline <- set_c_signs(pipeline, flip = c(1))

## Encode
y_ctx <- new_context(
  payload = y_list,
  cache = pipeline$training$cache,
  meta = list()
)
Qi_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Q_ctx <- encode(pipeline, Qi_ctx, from = 1, to = 2)
c_ctx <- encode(pipeline, Q_ctx, from = 2, to = 3)
z_ctx <- encode(pipeline, c_ctx, from = 3, to = 4)
z_rot_ctx <- encode(pipeline, z_ctx, from = 4, to = 5)

## Load covariates
path <- file.path('data', 'processed', 'nhanes_cov_cdc.rds')
df_cov <- readRDS(path)
df_cov <- df_cov %>%
  filter(sub_id %in% names(y_list)) %>%
  arrange(match(sub_id, names(y_list)))
df_cov$male <- as.numeric(df_cov$gender == 1)

## Age- and Sex- adjusted covariates
df_cov$bmd_adj <- residuals(
  gam(bmd_l_min ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$hdl_adj <- residuals(
  gam(hdl ~ male + s(age), data = df_cov, na.action = na.exclude)
)
df_cov$ghb_adj <- residuals(
  gam(ghb ~ male + s(age), data = df_cov, na.action = na.exclude)
)


## ----- Paper figure spec

## Same spec as demo_nhanes.R: render at final printed size so fig_pointsize
## is the printed text size.
fig_res       <- 600
fig_pointsize <- 7
fig_cex       <- 1
fig_cex_lab   <- 1.0
fig_cex_main  <- 1.2 * fig_cex_lab
fig_cex_axis  <- 0.8 * fig_cex_lab
fig_cex_legend <- fig_cex_lab
fig_png <- function(path, width_in, height_in, pointsize = fig_pointsize) {
  png(path, width = width_in, height = height_in, units = "in",
      res = fig_res, pointsize = pointsize)
}
fig_par_paper <- function() {
  fig_par(cex_main = fig_cex_main, cex_lab = fig_cex_lab, cex = fig_cex,
          cex_axis = fig_cex_axis)
}
fig_half_w <- 3.25
fig_half_h <- 2.6
col_real    <- "gray40"
col_outlier <- "#D95F02"   # unrealistic synthetic draws (as in demo_nhanes.R)




## ========== Generativity Inspection ========== ##

## Inspects the generativity run of demo_nhanes.R ("Check generativity"; S x R
## replicates, stored with keep_draws = TRUE). Per replicate, every stored
## synthetic latent is decoded and given
##   - w2: W2 distance of its QF to the mean QF of that split's training half
##     (unrealistic if w2 >= gen_outlier_w2, as in demo_nhanes.R), and
##   - the percentile of its latent Mahalanobis radius under the split's
##     fitted latent model (mean-cov of the training half), via chi^2_K;
## and the generativity score is recomputed with the unrealistic draws
## dropped: OT is re-solved between the remaining synthetic QFs and the
## validation QFs (T_sv_trim; T_tv unchanged).
## The stored draws live in the sign-flipped z_rot space of demo_nhanes.R, so
## the sign flips in Setup must match demo_nhanes.R.

gen_outlier_w2 <- 30      # as in demo_nhanes.R
gen_n_cores    <- 3       # each worker holds a pipeline copy (~500 MB) + an OT solve

dir_gen     <- file.path('artifacts', dir_art, 'plots', 'representation', 'generativity')
path_gen    <- file.path(dir_gen, 'gen_res.rds')
path_inspec <- file.path(dir_gen, 'gen_inspect.rds')
if (!file.exists(path_gen)) stop(str_glue("{path_gen} not found; run demo_nhanes.R first."))
gen_res <- readRDS(path_gen)
stopifnot(isTRUE(gen_res$meta$keep_draws), .latent_of(pipeline) == gen_res$meta$latent)

## One pass over all (split, replicate) pairs, cached (~10-15 min)
inspect_generativity <- function(pipeline, gen_res, y_list, outlier_w2, n_cores) {
  force(outlier_w2)   # evaluate before export: a promise would be resolved on the worker
  p_grid     <- pipeline$training$cache$p_grid
  p_grid_aug <- sort(unique(c(p_grid, pi_grid_fun(gen_res$meta$J_aug - length(p_grid)))))
  w_aug      <- get_quadrature_weights(p_grid_aug)
  cost_type  <- gen_res$meta$cost_type
  .repo_root <- getwd()

  if (n_cores > 1) {
    old_plan <- future::plan(future::multisession, workers = n_cores)
    on.exit(future::plan(old_plan), add = TRUE)
  }

  draws <- list(); reps <- list()
  for (s in seq_along(gen_res$splits)) {
    sp      <- gen_res$splits[[s]]
    Z_tr    <- encode_y_to_latent(pipeline, y_list[sp$idx_tr_grp])
    fit_lat <- fit_mean_cov(Z_tr)
    Q_val   <- encode_y_to_Qi_aug(pipeline, y_list[sp$idx_val_grp], p_grid_aug)
    Q_tr_mean <- colMeans(do.call(rbind, encode_y_to_Qi_aug(pipeline, y_list[sp$idx_tr_grp], p_grid_aug)))

    one_rep <- function(r) {
      if (!exists("py_pkg", envir = .GlobalEnv, inherits = FALSE)) {
        setwd(.repo_root)
        source(file.path(.repo_root, "src", "cot.R"))
      }
      z  <- sp$z_draws[[r]]
      Qs <- decode_latent_to_Qi_aug(pipeline, lapply(seq_len(nrow(z)), function(i) z[i, ]),
                                    p_grid_aug, p_grid)
      w2 <- sqrt(as.numeric(sweep(do.call(rbind, Qs), 2, Q_tr_mean)^2 %*% w_aug))
      w2[!is.finite(w2)] <- Inf
      m2 <- mahalanobis(z, fit_lat$mu, fit_lat$Sigma)
      keep <- w2 < outlier_w2
      T_sv_trim <- compute_generativity_score(Qs[keep], Q_val, w_aug, cost_type)
      ## Sanity check on the first replicate: recomputing the untrimmed score
      ## reproduces the stored one
      T_sv_full <- if (r == 1) compute_generativity_score(Qs, Q_val, w_aug, cost_type) else NA_real_
      list(draws = data.frame(m2 = m2, tail = pchisq(m2, ncol(z), lower.tail = FALSE), w2 = w2),
           T_sv_trim = T_sv_trim, T_sv_full = T_sv_full)
    }

    R <- length(sp$z_draws)
    out <- if (n_cores > 1) {
      future.apply::future_lapply(
        seq_len(R), one_rep, future.seed = NULL,
        future.globals = c("sp", "pipeline", "fit_lat", "Q_val", "Q_tr_mean", "w_aug",
                           "p_grid_aug", "p_grid", "outlier_w2", "cost_type", ".repo_root")
      )
    } else {
      lapply(seq_len(R), one_rep)
    }

    if (!isTRUE(all.equal(out[[1]]$T_sv_full, sp$T_sv[1], tolerance = 1e-6))) {
      warning(sprintf('Split %d: recomputed T_sv (%.6f) differs from the stored one (%.6f).',
                      s, out[[1]]$T_sv_full, sp$T_sv[1]))
    }
    for (r in seq_len(R)) {
      draws[[length(draws) + 1]] <- cbind(split = s, rep = r, out[[r]]$draws)
    }
    reps[[s]] <- data.frame(
      split     = s, rep = seq_len(R),
      n_outlier = vapply(out, function(o) sum(o$draws$w2 >= outlier_w2), 0),
      log_norm  = sp$log_norm,
      log_norm_trim = log(vapply(out, `[[`, 0, 'T_sv_trim') / sp$T_tv)
    )
    message(sprintf('  generativity inspection: split %d of %d done', s, length(gen_res$splits)))
  }
  list(draws = do.call(rbind, draws), reps = do.call(rbind, reps), outlier_w2 = outlier_w2)
}

if (file.exists(path_inspec) && isTRUE(readRDS(path_inspec)$outlier_w2 == gen_outlier_w2)) {
  gen_insp <- readRDS(path_inspec)
} else {
  start <- Sys.time()
  gen_insp <- inspect_generativity(pipeline, gen_res, y_list, gen_outlier_w2, gen_n_cores)
  print(Sys.time() - start)
  saveRDS(gen_insp, path_inspec)
}
dd   <- gen_insp$draws
reps <- gen_insp$reps
dd$unreal <- dd$w2 >= gen_outlier_w2

cat(sprintf('\n%d of %d synthetic draws (%.3f%%) are unrealistic (W2 >= %g): %.2f per replicate\n',
            sum(dd$unreal), nrow(dd), 100 * mean(dd$unreal), gen_outlier_w2, mean(reps$n_outlier)))
cat(sprintf('Unrealistic draws beyond the 90th / 99th / 99.9th percentile of the latent Gaussian: %.1f%% / %.1f%% / %.1f%%\n',
            100 * mean(dd$tail[dd$unreal] < 0.1), 100 * mean(dd$tail[dd$unreal] < 0.01),
            100 * mean(dd$tail[dd$unreal] < 0.001)))
cat(sprintf('Log norm. generativity, all draws: median %.3f, 90th pct %.3f; unrealistic draws removed: median %.3f, 90th pct %.3f\n',
            median(reps$log_norm), quantile(reps$log_norm, 0.9),
            median(reps$log_norm_trim), quantile(reps$log_norm_trim, 0.9)))

## Figure 1: generativity scores with and without the unrealistic draws
## (same replicates; T_tv unchanged)
fig_png(file.path(dir_gen, 'gen_scores_no_outliers.png'), fig_half_w, fig_half_h)
fig_par_paper()
plot_generativity_boxes(
  groups   = c("All draws", "Unrealistic removed"),
  scores   = list(split(reps$log_norm, reps$split), split(reps$log_norm_trim, reps$split)),
  hline    = 0,
  ylab     = "Log norm. generativity",
  collapse = TRUE
)
dev.off()

## Figure 2: W2 to the mean QF vs percentile of the latent Mahalanobis radius,
## per draw (a random subsample of the realistic draws, all unrealistic draws
## on top)
set.seed(1234)
idx_ok  <- which(!dd$unreal)
idx_ok  <- idx_ok[sample.int(length(idx_ok), min(20000, length(idx_ok)))]
idx_bad <- which(dd$unreal)
tail_x  <- function(tail) -log10(pmax(tail, 1e-12))
x_ticks <- c(0.5, 0.9, 0.99, 0.999, 0.9999)
w2_cap  <- max(dd$w2[is.finite(dd$w2)])
fig_png(file.path(dir_gen, 'wasserstein_vs_mahalanobis.png'), fig_half_w, fig_half_h)
fig_par_paper()
plot(tail_x(dd$tail[idx_ok]), pmin(dd$w2[idx_ok], w2_cap), log = 'y', pch = 16, cex = 0.25,
     col = adjustcolor(col_real, 0.25),
     xlim = c(0, max(tail_x(dd$tail), -log10(1 - max(x_ticks)))),
     ylim = range(pmin(dd$w2, w2_cap)), xaxt = 'n',
     xlab = "Percentile of latent Mahalanobis radius", ylab = "W2 to mean QF (log)")
axis(1, at = -log10(1 - x_ticks), labels = 100 * x_ticks)
points(tail_x(dd$tail[idx_bad]), pmin(dd$w2[idx_bad], w2_cap), pch = 16, cex = 0.4,
       col = col_outlier)
abline(h = gen_outlier_w2, col = "red", lty = 3, lwd = 1.2)
dev.off()




## ========== G-PCA Embeddings ========== ##

## Functional PCA of the LQD-transformed quantile functions (G-PCA): a linear
## alternative to the main pipeline's WAME + flow representation. Embeddings
## are plotted in the same style as data_c.png in demo_nhanes.R.

dir_gpca <- file.path('artifacts', dir_art, 'plots', 'representation')
if (!dir.exists(dir_gpca)) dir.create(dir_gpca, recursive = TRUE)

## K = 5 matches the main pipeline and the near-lossless K for g_pca from
## assess_losslessness.R
K_gpca <- 5

## Construct pipeline (same grid, support, and anchoring as the main pipeline)
pipe_gpca <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100),
    stage_lqd(),
    stage_g_pca(K = K_gpca)
  ),
  supp_Y = c(0, seq(0.006, 500, by = 0.001)),
  p_star = 0,
  y_star = 0,
  y_min = 0,
  seed = 12345
)

## Fitting (cached per K)
path <- str_glue('artifacts/{dir_art}/pipe_nhanes_g-pca_K-{K_gpca}.rds')
if (file.exists(path)) {
  pipe_gpca <- readRDS(path)
} else {
  pipe_gpca <- fit(pipe_gpca, y_list)
  saveRDS(pipe_gpca, path)
}

## Encode to the G-PCA scores (the g_pca payload also carries Q_star_list)
y_gpca_ctx <- new_context(
  payload = y_list,
  cache = pipe_gpca$training$cache,
  meta = list()
)
c_gpca_ctx <- encode(pipe_gpca, y_gpca_ctx, from = 0, to = pipe_gpca$n_stages)
C_gpca <- do.call(rbind, c_gpca_ctx$payload$c_list)

## Plot G-PCA embeddings
k_from <- 1
k_to <- min(2, K_gpca)
col_train <- rgb(0, 0, 0, alpha = 0.25)
path_plot <- file.path(dir_gpca, str_glue('data_g-pca_K-{K_gpca}.png'))
fig_png(path_plot, 4.5, 4.5)   # TODO: final printed size
plot_embeddings(
  C_gpca[, k_from:k_to],
  colors = rep(col_train, N),
  shapes = rep(16, N),
  sizes = rep(0.5, N),
  color_shape_size_labels = NULL,
  stats = FALSE,
  cex_main = fig_cex_main,
  cex_lab = fig_cex_lab,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend
)
dev.off()




## ========== Q-PCA Embeddings ========== ##

## Functional PCA of the quantile functions themselves (Q-PCA): the G-PCA
## comparison above without the LQD transform. Embeddings are plotted in the
## same style as data_c.png in demo_nhanes.R.

dir_qpca <- file.path('artifacts', dir_art, 'plots', 'representation')
if (!dir.exists(dir_qpca)) dir.create(dir_qpca, recursive = TRUE)

## K = 5 matches the main pipeline and the G-PCA fit above
K_qpca <- 5

## Construct pipeline (same grid, support, and anchoring as the main pipeline)
pipe_qpca <- construct_pipeline(
  stages = list(
    stage_eqf_sgrid(),
    stage_eqf_cgrid(p_grid = p_grid, Ji_min = 100),
    stage_q_pca(K = K_qpca)
  ),
  supp_Y = c(0, seq(0.006, 500, by = 0.001)),
  p_star = 0,
  y_star = 0,
  y_min = 0,
  seed = 12345
)

## Fitting (cached per K)
path <- str_glue('artifacts/{dir_art}/pipe_nhanes_q-pca_K-{K_qpca}.rds')
if (file.exists(path)) {
  pipe_qpca <- readRDS(path)
} else {
  pipe_qpca <- fit(pipe_qpca, y_list)
  saveRDS(pipe_qpca, path)
}

## Encode to the Q-PCA scores (the q_pca payload is the score list itself)
y_qpca_ctx <- new_context(
  payload = y_list,
  cache = pipe_qpca$training$cache,
  meta = list()
)
c_qpca_ctx <- encode(pipe_qpca, y_qpca_ctx, from = 0, to = pipe_qpca$n_stages)
C_qpca <- do.call(rbind, c_qpca_ctx$payload)

## Plot Q-PCA embeddings
k_from <- 1
k_to <- min(2, K_qpca)
col_train <- rgb(0, 0, 0, alpha = 0.25)
path_plot <- file.path(dir_qpca, str_glue('data_q-pca_K-{K_qpca}.png'))
fig_png(path_plot, 4.5, 4.5)   # TODO: final printed size
plot_embeddings(
  C_qpca[, k_from:k_to],
  colors = rep(col_train, N),
  shapes = rep(16, N),
  sizes = rep(0.5, N),
  color_shape_size_labels = NULL,
  stats = FALSE,
  cex_main = fig_cex_main,
  cex_lab = fig_cex_lab,
  cex = fig_cex,
  cex_axis = fig_cex_axis,
  legend_cex = fig_cex_legend
)
dev.off()




## ========== Hierarchical Clustering ========== ##

## Ward clustering of the latent embeddings, characterized by decoding
## embedding-space group means back to QF space, with a QF-space clustering as
## a baseline. Outputs land in plots/hca/<latent> so that the c and z_rot runs
## sit side by side rather than overwriting each other.

## ---------- Helpers ---------- ##

## Place every Qi on the shared augmented p-grid, as a J x N matrix. Subjects
## have different Ji, so this is what makes them comparable pointwise.
hca_qf_aug <- function(Qi_list, p_grid_aug, supp_Y) {
  vapply(
    Qi_list,
    function(Qi) inv_eqf_cgrid(Qi, pi_grid_fun(length(Qi)), p_grid_aug, supp_Y),
    numeric(length(p_grid_aug))
  )
}

## Pairwise Wasserstein distances between QFs, as a `dist`, from the augmented
## matrix above. The metric is a w-weighted L2, so scaling coordinates by
## sqrt(w) reduces it to plain Euclidean distance and dist() does the O(N^2)
## work in C -- the R-level loop in compute_pwds() is hopeless at N ~ 8k.
hca_qf_dist <- function(Q_aug, p_grid_aug) {
  w_aug <- get_quadrature_weights(p_grid_aug)
  dist(sweep(t(Q_aug), 2, sqrt(w_aug), `*`))
}

## Wasserstein barycenter of the columns in `idx`. In one dimension the W2
## barycenter has a closed form -- it is the pointwise mean of the quantile
## functions -- so this is exact, not an iterative solve.
hca_qf_barycenter <- function(Q_aug, idx) {
  rowMeans(Q_aug[, idx, drop = FALSE])
}

## Relabel clusters 1..k manually: cutree() cluster i becomes cluster
## `new_ids[i]`, so `new_ids` must be a permutation of the cutree() ids.
hca_relabel_manual <- function(cl, new_ids) {
  stopifnot(setequal(new_ids, cl), !anyDuplicated(new_ids))
  new_ids[cl]
}

## Overlaid per-cluster densities of a covariate.
hca_plot_density <- function(v, cl, cols, xlab, main, path) {
  ok    <- !is.na(v)
  dens  <- lapply(seq_along(cols), function(g) density(v[ok & cl == g]))
  png(path, width = 960, height = 720, pointsize = 18)
  plot(
    NULL,
    xlim = range(sapply(dens, function(d) range(d$x))),
    ylim = c(0, max(sapply(dens, function(d) max(d$y)))),
    xlab = xlab, ylab = 'density', main = main
  )
  for (g in seq_along(dens)) lines(dens[[g]], col = cols[g], lwd = 3)
  legend(
    'topright',
    legend = str_glue('cluster {seq_along(cols)}'),
    col = cols, lwd = 3, bty = 'n'
  )
  dev.off()
}

## Cluster-wise horizontal boxplots stacked vertically, with the pooled median
## marked, drawn into the current device. Groups are passed in reverse because
## boxplot() draws its first group at the bottom, and cluster 1 reads better at
## the top.
##
## Args:
##   show_names      label the cluster axis; turn off for all but the first of
##                   several panels sharing that axis
##   faint_outliers  draw outliers as small translucent dots, so that the
##                   boxes stay readable when N is large
##   show_n          print each cluster's non-missing count in the right margin
##   hide_outliers   draw no outliers at all, which also limits the value
##                   axis to the span of the whiskers so that a few extreme
##                   points do not compress the boxes. Overrides
##                   faint_outliers.
##   mean_ci         overlay each cluster's mean (white diamond) and its 95%
##                   t-interval (black bar). The reference line then marks the
##                   pooled mean rather than the pooled median, so that it is
##                   the quantity the intervals are compared against.
##   labels          group names for the cluster axis, indexed by group id;
##                   defaults to the ids themselves
hca_draw_boxes <- function(v, cl, cols, xlab, main, ylab = 'cluster',
                           show_names = TRUE, faint_outliers = FALSE,
                           show_n = FALSE, hide_outliers = FALSE,
                           mean_ci = FALSE, labels = seq_along(cols)) {
  idx    <- rev(seq_along(cols))
  groups <- lapply(idx, function(g) v[cl == g & !is.na(v)])
  out_par <- if (faint_outliers) {
    list(outpch = 16, outcex = 0.4, outcol = adjustcolor('black', alpha.f = 0.25))
  }
  do.call(boxplot, c(
    list(
      groups,
      horizontal = TRUE,
      names      = if (show_names) labels[idx] else rep('', length(idx)),
      col        = cols[idx],
      las        = 1,
      xlab       = xlab,
      ylab       = ylab,
      main       = main,
      outline    = !hide_outliers
    ),
    out_par
  ))
  if (mean_ci) {
    abline(v = mean(v, na.rm = TRUE), lty = 3, lwd = 2)
    m    <- vapply(groups, mean, numeric(1))
    ## NA for a cluster of fewer than two, which then draws no bar
    half <- vapply(
      groups,
      function(x) qt(0.975, max(length(x) - 1, 1)) * sd(x) / sqrt(length(x)),
      numeric(1)
    )
    at <- seq_along(groups)
    ## White casing keeps the bar visible on top of the box and median line
    segments(m - half, at, m + half, at, lwd = 5, col = 'white', lend = 'butt')
    segments(m - half, at, m + half, at, lwd = 2.5, lend = 'butt')
    points(m, at, pch = 23, bg = 'white', cex = 0.9)
  } else {
    abline(v = median(v, na.rm = TRUE), lty = 3, lwd = 2)
  }
  if (show_n) hca_margin_n(lengths(groups))
}

## Counts in the right margin of a cluster-axis panel, one per row.
hca_margin_n <- function(n) {
  mtext(n, side = 4, at = seq_along(n), las = 1, line = 0.3,
        cex = 0.7 * par('cex'), col = 'grey40')
}

## hca_draw_boxes() as a standalone png. `width` and `height` are in pixels at
## 72 dpi; the file written is res / 72 times larger in each direction.
hca_plot_boxes <- function(v, cl, cols, xlab, main, path,
                           width = 960, height = 720, pointsize = 18,
                           res = 72, mar = c(5, 5, 4, 2), ...) {
  ## An individual empty cluster is fine (boxplot just leaves a gap), but an
  ## all-missing covariate has no finite xlim and would error.
  if (all(is.na(v))) {
    warning(str_glue("hca_plot_boxes: '{xlab}' is entirely missing; skipping."))
    return(invisible(NULL))
  }
  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit(dev.off(), add = TRUE)
  par(mar = mar)
  hca_draw_boxes(v, cl, cols, xlab = xlab, main = main, ...)
}

## One figure summarizing every covariate by cluster: a row of panels sharing
## the cluster axis (cluster 1 at the top). A binary covariate gets a bar per
## cluster for its proportion; a continuous one gets cluster-wise boxplots
## without outliers, overlaid with each cluster's mean and 95% CI. The dotted line is the pooled proportion / mean, and the grey numbers
## in each panel's right margin are the non-missing counts behind each row,
## since missingness differs by covariate and by cluster.
##
## `cl` may hold any grouping coded 1..length(cols), not just clusters: pass
## the group names as `labels` and the axis title as `group_lab`, and the left
## margin widens to fit the names.
hca_plot_cov_summary <- function(covs, cl, cols, path,
                                 width = 1200, height = 520, pointsize = 16,
                                 res = 288, labels = NULL,
                                 group_lab = 'cluster') {
  covs <- as.data.frame(covs)
  k    <- length(cols)
  idx  <- rev(seq_len(k))
  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit(dev.off(), add = TRUE)
  ## Cluster labels live in the outer margin so that all panels are equally
  ## wide. mfrow shrinks cex for a row this long, so set it back explicitly.
  par(mfrow = c(1, ncol(covs)), mar = c(4, 0.8, 2.5, 2.6),
      mgp = c(2.4, 0.7, 0), cex = 0.8)
  if (is.null(labels)) {
    labels   <- seq_len(k)
    oma_left <- 3.5
  } else {
    ## Widest name, in margin lines, plus room for the tick and axis title
    oma_left <- max(strwidth(labels, units = 'inches')) /
      (par('mai')[1] / par('mar')[1]) + 2.5
  }
  par(oma = c(0, oma_left, 0, 0))
  for (j in seq_len(ncol(covs))) {
    v  <- covs[[j]]
    nm <- names(covs)[j]
    if (all(is.na(v))) {
      warning(str_glue("hca_plot_cov_summary: '{nm}' is entirely missing."))
      plot.new()
      title(main = nm)
      next
    }
    if (all(v[!is.na(v)] %in% c(0, 1))) {
      prop <- vapply(idx, function(g) mean(v[cl == g], na.rm = TRUE), numeric(1))
      ## Same row positions and bar width as boxplot(), so rows line up
      plot(NULL, xlim = c(0, 1), ylim = c(0.5, k + 0.5), yaxt = 'n',
           xlab = 'proportion', ylab = '', main = nm)
      rect(0, seq_len(k) - 0.4, prop, seq_len(k) + 0.4, col = cols[idx])
      axis(2, at = seq_len(k), labels = if (j == 1) labels[idx] else FALSE,
           las = 1)
      abline(v = mean(v, na.rm = TRUE), lty = 3, lwd = 2)
      hca_margin_n(vapply(idx, function(g) sum(cl == g & !is.na(v)), numeric(1)))
    } else {
      hca_draw_boxes(v, cl, cols, xlab = '', main = nm, ylab = '',
                     show_names = j == 1, show_n = TRUE,
                     hide_outliers = TRUE, mean_ci = TRUE, labels = labels)
    }
  }
  mtext(group_lab, side = 2, outer = TRUE, line = oma_left - 1.3, cex = 0.8)
}

## Move labels at heights `y` apart until neighbours are at least `gap` apart,
## keeping each run of colliding labels centred on its original positions.
hca_spread_labels <- function(y, gap) {
  o   <- order(y)
  ys  <- y[o]
  grp <- seq_along(ys)
  repeat {
    pos <- ys
    for (g in unique(grp)) {
      i <- which(grp == g)
      pos[i] <- mean(ys[i]) + (seq_along(i) - (length(i) + 1) / 2) * gap
    }
    clash <- which(diff(pos) < gap - 1e-9 & diff(grp) != 0)
    if (length(clash) == 0) break
    grp[grp == grp[clash[1] + 1]] <- grp[clash[1]]
  }
  y[o] <- pos
  y
}

## --- Clustered-heatmap colour scales

## Missing covariate values get their own colour rather than falling back on
## white: in a binary column white is already 0, so NA would otherwise be
## indistinguishable from it (low_income, say, inherits every pir NA). Orange
## sits outside both covariate ramps and outside the green cluster rules.
hca_na_col <- 'orange'

## Black/white for a binary covariate.
hca_cols_bw <- function(v, na_col = hca_na_col) {
  out <- rep('white', length(v))
  out[!is.na(v) & v == 1] <- 'black'
  out[is.na(v)] <- na_col
  out
}

## Greys for a continuous covariate.
hca_grey_pal <- function(n = 256) gray.colors(n, start = 0.92, end = 0)
hca_cols_grey <- function(v, n = 256, na_col = hca_na_col) {
  pal <- hca_grey_pal(n)
  rng <- range(v, na.rm = TRUE)
  out <- rep(na_col, length(v))
  ok  <- !is.na(v)
  out[ok] <- if (diff(rng) > 0) {
    pal[round((v[ok] - rng[1]) / diff(rng) * (n - 1)) + 1]
  } else {
    pal[1]
  }
  out
}

## Blue (negative) - white (zero) - red (positive) over [-lim, lim].
hca_div_pal <- function(n = 255) colorRampPalette(c('blue', 'white', 'red'))(n)
hca_cols_div <- function(v, lim, n = 255) {
  if (!is.finite(lim) || lim <= 0) lim <- 1
  vc <- pmax(pmin(v, lim), -lim)
  hca_div_pal(n)[round((vc + lim) / (2 * lim) * (n - 1)) + 1]
}

## One labelled colour ramp in the key panel.
hca_legend_ramp <- function(pal, y0, y1, title, labels) {
  rasterImage(as.raster(matrix(rev(pal), ncol = 1)), 0.05, y0, 0.30, y1)
  text(0.05, y1 + 0.035, title, adj = 0, font = 2)
  text(0.34, seq(y1, y0, length.out = length(labels)), labels, adj = 0)
}

## Color each dendrogram branch that lies wholly inside one cluster with that
## cluster's color; branches above the cut keep the default. `cl` is indexed by
## observation, as returned by cutree(), and `cols` by cluster id.
hca_color_dendrogram <- function(dend, cl, cols) {
  rec <- function(node) {
    if (is.leaf(node)) {
      g <- cl[as.integer(node)]
    } else {
      gs <- integer(length(node))
      for (i in seq_along(node)) {
        child     <- rec(node[[i]])
        node[[i]] <- child$node
        gs[i]     <- child$g
      }
      g <- if (!anyNA(gs) && all(gs == gs[1])) gs[1] else NA_integer_
    }
    if (!is.na(g)) attr(node, 'edgePar') <- list(col = cols[g])
    list(node = node, g = g)
  }
  rec(dend)$node
}

## Clustered heatmap: an N x (A + K) raster of covariate and latent values,
## rows in dendrogram order with the tree drawn alongside (one leaf per row).
## Columns are never re-ordered, so the latent block stays in component order.
##
## Args:
##   hc         hclust whose leaves index the rows of `covs` and `Z`
##   Z          N x K latent matrix. Each column gets its own symmetric scale,
##              +/- max|Z[, k]|, so zero is white everywhere and every
##              component is legible regardless of its variance.
##   covs       data.frame of covariates, one per heatmap column, drawn left of
##              the latent block in the order given
##   cov_types  optional 'binary'/'continuous' per covariate; the default
##              infers 'binary' when every non-NA value of a column is 0 or 1
##   n_clusts   positive integer -> cutree(hc, n_clusts) to partition the rows;
##              NULL (default) leaves the rows unmarked, the dendrogram
##              uncolored, and makes the two flags below no-ops
##   cl         optional cluster label per row, used in place of the raw
##              cutree() ids so relabelled clusters print consistently with
##              the other figures. Must be a relabelling of that same cut.
##   cl_cols    color per cluster id, for the dendrogram branches; defaults
##              to the 'Dark 3' palette used by the other figures
##   cluster_lines   draw rules (in `line_col`) between the row blocks
##   label_clusters  print each cluster's number in the right margin, centred
##                   on its block
##   na_col     colour for missing covariate values, in both column types
##   z_labs     column labels for the Z block; defaults to 'k = 1..K'. Pass a
##              mostly-'' vector when Z is wide (Qi on the augmented p-grid)
##              so the axis stays readable.
##   cov_frac   share of the plot width given to the covariate block; the
##              default A/(A + K) makes every column equally wide, which is
##              right for a K-column latent but squeezes the covariates to a
##              sliver when K is in the hundreds
##   path       output png
##   width, height   figure size in pixels at 72 dpi; the file written is
##                   res / 72 times larger in each direction
##   res        resolution in dpi. Text, lines and layout scale with it, so
##              raising it sharpens the figure without changing its look.
hca_plot_heatmap <- function(hc, Z, covs, path,
                            cov_types = NULL, n_clusts = NULL, cl = NULL,
                            cl_cols = NULL,
                            cluster_lines = TRUE, label_clusters = TRUE,
                            line_col = 'green3',
                            na_col = hca_na_col, z_labs = NULL,
                            cov_frac = NULL,
                            width = 1000, height = 1200, pointsize = 16,
                            res = 288) {
  covs <- as.data.frame(covs)
  nr   <- nrow(Z)
  A    <- ncol(covs)
  K    <- ncol(Z)
  stopifnot(nrow(covs) == nr, length(hc$order) == nr, A >= 1)

  if (is.null(cov_types)) {
    cov_types <- vapply(
      covs,
      function(v) if (all(v[!is.na(v)] %in% c(0, 1))) 'binary' else 'continuous',
      character(1)
    )
  }
  stopifnot(length(cov_types) == A,
            cov_types %in% c('binary', 'continuous'))

  ## vapply (not sapply/mapply) so a single covariate still yields a matrix
  cols_cov <- vapply(
    seq_len(A),
    function(j) {
      v <- covs[[j]]
      if (cov_types[j] == 'binary') {
        hca_cols_bw(v, na_col = na_col)
      } else {
        hca_cols_grey(v, na_col = na_col)
      }
    },
    character(nr)
  )
  ## One symmetric scale per component: +/- max|Z[, k]|, so each column uses
  ## the full blue-to-red range regardless of how much variance it carries.
  lims <- apply(abs(Z), 2, max, na.rm = TRUE)
  cols_lat <- vapply(
    seq_len(K),
    function(k) hca_cols_div(Z[, k], lims[k]),
    character(nr)
  )

  ord      <- hc$order
  cols_cov <- cols_cov[ord, , drop = FALSE]
  cols_lat <- cols_lat[ord, , drop = FALSE]
  if (is.null(z_labs)) z_labs <- paste('k =', seq_len(K))
  stopifnot(length(z_labs) == K)
  ## The two blocks are drawn as separate rasters on a [0, 1] axis so their
  ## widths are independent of their column counts -- otherwise a wide Z (Qi
  ## on the augmented p-grid, K ~ 500) squeezes the covariates to a sliver.
  if (is.null(cov_frac)) cov_frac <- A / (A + K)
  stopifnot(cov_frac > 0, cov_frac < 1)

  dend <- as.dendrogram(hc)
  if (!is.null(n_clusts)) {
    stopifnot(length(n_clusts) == 1, n_clusts == round(n_clusts), n_clusts >= 1)
    if (is.null(cl)) cl <- cutree(hc, k = n_clusts)
    if (is.null(cl_cols)) cl_cols <- hcl.colors(n_clusts, palette = 'Dark 3')
    stopifnot(length(cl) == nr, length(cl_cols) == n_clusts)
    dend <- hca_color_dendrogram(dend, cl, cl_cols)
  }

  png(path, width = width * res / 72, height = height * res / 72,
      pointsize = pointsize, res = res)
  on.exit({dev.off(); layout(1)}, add = TRUE)
  layout(matrix(1:3, nrow = 1), widths = c(1.1, 2.4, 0.9))

  ## Dendrogram: yaxs = 'i' makes leaf i sit at y = i, matching the raster
  par(mar = c(5, 2, 6, 0))
  plot(dend, horiz = TRUE, axes = FALSE, yaxs = 'i', leaflab = 'none')

  ## Heatmap. as.raster() draws its first row at the top, so flip to put
  ## dendrogram-order row 1 at the bottom where the first leaf is.
  ## Right margin holds the cluster numbers, between heatmap and key
  par(mar = c(5, 0, 6, if (label_clusters) 2.5 else 1))
  plot(NULL, xlim = c(0, 1), ylim = 0.5 + c(0, nr),
      xaxs = 'i', yaxs = 'i', axes = FALSE, xlab = '', ylab = '')
  rasterImage(as.raster(cols_cov[nr:1, , drop = FALSE]),
              xleft = 0, ybottom = 0.5, xright = cov_frac, ytop = nr + 0.5,
              interpolate = FALSE)
  rasterImage(as.raster(cols_lat[nr:1, , drop = FALSE]),
              xleft = cov_frac, ybottom = 0.5, xright = 1, ytop = nr + 0.5,
              interpolate = FALSE)
  ## Column centres in axis units, per block
  at_cov <- (seq_len(A) - 0.5) / A * cov_frac
  at_z   <- cov_frac + (seq_len(K) - 0.5) / K * (1 - cov_frac)
  ## Two axis() calls over only the labelled positions: one call spanning both
  ## blocks lets axis() thin out labels it judges too dense, which silently
  ## drops the sparse p-marks used for a wide Z.
  axis(3, at = at_cov, labels = names(covs), las = 2, tick = FALSE, line = -0.5)
  keep <- nzchar(z_labs)
  if (any(keep)) {
    axis(3, at = at_z[keep], labels = z_labs[keep], las = 2,
         tick = FALSE, line = -0.5)
  }
  abline(v = cov_frac, lwd = 2)
  if (!is.null(n_clusts)) {
    ## Clusters are contiguous in dendrogram order, so run-lengths give both
    ## the cut points and each block's extent in one pass
    runs <- rle(unname(cl[ord]))
    stopifnot(length(runs$values) == n_clusts)
    ends <- cumsum(runs$lengths)
    if (cluster_lines) {
      abline(h = head(ends, -1) + 0.5, col = line_col, lwd = 2)
    }
    if (label_clusters) {
      ## mtext rather than axis() so no label is dropped when a block is thin
      mids <- ends - (runs$lengths - 1) / 2
      for (i in seq_along(mids)) {
        mtext(runs$values[i], side = 4, at = mids[i], line = 0.6, las = 1)
      }
    }
  }
  box()

  ## Key
  par(mar = c(5, 1, 6, 1))
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = '', ylab = '')
  rect(0.05, c(0.90, 0.83), 0.30, c(0.96, 0.89),
      col = c('white', 'black'), border = 'grey50')
  text(0.05, 0.995, 'binary', adj = 0, font = 2)
  text(0.34, c(0.93, 0.86), c('0', '1'), adj = 0)
  hca_legend_ramp(hca_grey_pal(), 0.52, 0.74, 'continuous', c('max', 'min'))
  ## Scale is per column, so the key shows sign rather than a shared bound
  hca_legend_ramp(hca_div_pal(), 0.14, 0.36, 'latent', c('+', '0', '-'))
  ## NA applies to the covariate columns only; latent values are never missing
  rect(0.05, 0.035, 0.30, 0.085, col = na_col, border = 'grey50')
  text(0.05, 0.115, 'missing', adj = 0, font = 2)
  text(0.34, 0.06, 'NA', adj = 0)
}


## ---------- Settings ---------- ##

# hca_latent   <- 'z_rot'      # 'c' or 'z_rot'
# hca_k        <- 20        # set after eyeballing hca_dendrogram.png, then re-run

hca_J_aug <- 500      # augmented p-grid size for the Qi representation

## Shared by the four figures that make up the activity panel in LaTeX
## (cluster-qfs-contrast.png over the three box-q*.png), so text sizes match.
hca_pointsize <- 14
hca_res       <- 288

## Cluster Relabels, by representation and then k. Entry i is the new label
## for cutree() cluster i, so 1:k keeps the default labeling. A representation
## or k with no entry also keeps the default.
cluster_relabels <- list(
  'c' = list(
    '11' = c(2, 3, 1, 8, 9, 7, 10, 4, 6, 5, 11)
  )
)

## Phenotypes defined manually post-hoc
phenotypes_to_clusters <- list(
  'c' = list(
    '11' = list(
      'very active' = c(1, 2),
      'moderately active' = c(3, 4, 5),
      'weekend warrior' = c(6),
      'frequent mover' = c(7),
      'inactive' = c(8, 9, 10),
      'very inactive' = c(11)
    )
  )
)

## Subject-level activity quantiles, boxplotted per cluster alongside the
## covariates. These are a property of the data, not of the clustering, so
## they are evaluated once rather than per representation or per k.
hca_q_levels <- c(0.5, 0.75, 0.99)
Q_lvl_hca <- t(hca_qf_aug(Qi_ctx$payload, hca_q_levels,
                          pipeline$training$cache$supp_Y))

## 'c' and 'z_rot' are latent embeddings; 'qi' clusters the quantile functions
## themselves, on the augmented p-grid under the Wasserstein metric. Qi is not
## a latent representation, so it skips the pairwise-embedding scatter plot --
## every other figure is produced for it exactly as for the embeddings.
for (hca_latent in c('c')) {

  hca_is_qi     <- hca_latent == 'qi'
  hca_rep_label <- switch(
    hca_latent,
    z_rot = 'z_rot-embeddings',
    c     = 'c-embeddings',
    qi    = 'Qi',
    stop(str_glue("Unknown hca_latent: '{hca_latent}'"))
  )

  dir_hca <- file.path('artifacts', dir_art, 'plots', 'hca', hca_latent)
  if (!dir.exists(dir_hca)) dir.create(dir_hca, recursive = TRUE)

  ## NOTE: N, K, Ji, mask, x, y_grid, ... are all clobbered by earlier
  ## sections, so every name introduced below carries an _hca suffix.
  Ji_hca         <- length(Qi_ctx$payload[[1]])
  pi_grid_hca    <- pi_grid_fun(Ji_hca)
  p_grid_aug_hca <- sort(unique(c(
    p_grid, pi_grid_fun(hca_J_aug - length(p_grid))
  )))

  ## Representation, metric, and cluster representative. The tree does not
  ## depend on k, so it is built once per representation rather than once per
  ## k -- the Qi distance matrix alone is ~8k x 8k.
  start <- Sys.time()
  if (hca_is_qi) {
    ## Qi on the shared augmented grid; rows are subjects, as for a latent
    Q_aug_hca <- hca_qf_aug(Qi_ctx$payload, p_grid_aug_hca,
                            pipeline$training$cache$supp_Y)
    Z_hca     <- t(Q_aug_hca)
    d_hca     <- hca_qf_dist(Q_aug_hca, p_grid_aug_hca)
    grid_hca  <- p_grid_aug_hca
    ## In 1-D the W2 barycenter is the pointwise mean of the QFs, so it is the
    ## exact analogue of decoding the mean of a cluster's embeddings.
    hca_rep_mean <- function(idx) hca_qf_barycenter(Q_aug_hca, idx)
  } else {
    Z_hca <- switch(
      hca_latent,
      z_rot = do.call(rbind, z_rot_ctx$payload),
      c     = c_ctx_to_matrix(pipeline, c_ctx)
    )
    d_hca    <- dist(Z_hca)
    grid_hca <- pi_grid_hca
    pipe_hca <- pipeline
    pipe_hca$latent <- hca_latent
    hca_rep_mean <- function(idx) {
      z_bar <- colMeans(Z_hca[idx, , drop = FALSE])
      decode_latent_to_Qi(pipe_hca, list(z_bar), Ji_hca)[[1]]
    }
  }
  N_hca <- nrow(Z_hca)
  K_hca <- ncol(Z_hca)
  stopifnot(nrow(df_cov) == N_hca)

  hc_hca <- hclust(d_hca, method = 'ward.D2')
  ## dist objects are ~N^2/2 doubles; drop once the tree is built
  rm(d_hca)
  gc()
  print(Sys.time() - start)

  png(file.path(dir_hca, 'dendrogram.png'),
      width = 1280, height = 720, pointsize = 14)
  plot(hc_hca, labels = FALSE, hang = -1, xlab = '', sub = '',
       main = str_glue("Ward linkage on {hca_rep_label}"))
  dev.off()

  ## Heatmap Z block: for Qi the natural analogue of the latent columns is the
  ## QF itself, centred on the overall barycenter so that the diverging scale
  ## has a meaningful zero. 500 columns, so label only a few p levels.
  if (hca_is_qi) {
    Z_heat_hca <- sweep(Z_hca, 2, hca_qf_barycenter(Q_aug_hca, seq_len(N_hca)))
    z_labs_hca <- rep('', K_hca)
    p_marks    <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    z_labs_hca[sapply(p_marks, function(p) which.min(abs(p_grid_aug_hca - p)))] <-
      sprintf('p = %g', p_marks)
    ## ~500 Z columns vs a handful of covariates, so fix the split by width
    cov_frac_hca <- 0.3
  } else {
    Z_heat_hca   <- Z_hca
    z_labs_hca   <- NULL
    cov_frac_hca <- NULL   # equal-width columns, as before
  }

  for (hca_k in c(11)) {

    ## One directory per k: the figure names below carry no k suffix,
    ## so the path alone identifies a plot.
    dir_hca_k <- file.path(dir_hca, str_glue('k-{hca_k}'))
    if (!dir.exists(dir_hca_k)) dir.create(dir_hca_k, recursive = TRUE)

    ## Number clusters manually via cluster_relabels, so that ids, and hence
    ## colors, are interpretable.
    relabel_hca <- cluster_relabels[[hca_latent]][[as.character(hca_k)]]
    if (is.null(relabel_hca)) relabel_hca <- seq_len(hca_k)
    cl_hca   <- hca_relabel_manual(cutree(hc_hca, k = hca_k), relabel_hca)
    cols_hca <- hcl.colors(hca_k, palette = 'Dark 3')

    ## rect.hclust() fills `border` left-to-right along hc$order, not by cluster id,
    ## so the palette has to be permuted to keep box colors consistent with every
    ## other plot in this section.
    hca_rect_cols <- function(hc, cl, cols) cols[unique(cl[hc$order])]

    png(file.path(dir_hca_k, 'dendrogram.png'),
        width = 1280, height = 720, pointsize = 14)
    plot(hc_hca, labels = FALSE, hang = -1, xlab = '', sub = '',
        main = str_glue("Ward linkage on {hca_rep_label} (k = {hca_k})"))
    rect.hclust(hc_hca, k = hca_k,
                border = hca_rect_cols(hc_hca, cl_hca, cols_hca))
    dev.off()

    print(table(cluster = cl_hca))


    ## ---------- Clustered Heatmap ---------- ##

    hca_plot_heatmap(
      hc             = hc_hca,
      Z              = Z_heat_hca,
      covs           = df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
      n_clusts       = hca_k,
      cl             = cl_hca,
      cl_cols        = cols_hca,
      cluster_lines  = TRUE,
      label_clusters = TRUE,
      z_labs         = z_labs_hca,
      cov_frac       = cov_frac_hca,
      path           = file.path(dir_hca_k, 'heatmap.png')
    )


    ## ---------- Embedding space ---------- ##

    ## Qi has no embedding coordinates to scatter, so this is the one figure
    ## the Qi representation skips.
    if (!hca_is_qi) {
      k_from <- 1
      k_to <- 3
      png(file.path(dir_hca_k, 'embeddings.png'),
          width = 960, height = 960, pointsize = 18)
      plot_embeddings(
        Z_hca[,k_from:k_to],
        colors = adjustcolor(cols_hca[cl_hca], alpha.f = 0.3),
        shapes = rep(19, N_hca),
        sizes  = rep(0.5, N_hca),
        color_shape_size_labels = data.frame(
          color = cols_hca,
          shape = 19,
          size  = 1,
          label = str_glue('cluster {seq_len(hca_k)}')
        ),
        stats = FALSE,
        ## Small, and in two columns for a large k, so the legend stays in
        ## the corner of the first Q-Q panel clear of its points
        legend_cex  = 0.65,
        legend_ncol = if (hca_k > 12) 2 else 1
      )
      dev.off()
    }


    ## ---------- Distribution space ---------- ##

    ## Cluster QF := decode(mean of the cluster's embeddings) for a latent,
    ## or the cluster's Wasserstein barycenter for Qi.
    Qi_cluster <- lapply(
      seq_len(hca_k),
      function(g) hca_rep_mean(which(cl_hca == g))
    )

    png(file.path(dir_hca_k, 'cluster-qfs.png'),
        width = 960, height = 960, pointsize = 18)
    plot_funs(
      fun_list  = Qi_cluster,
      grid_list = rep(list(grid_hca), hca_k),
      colors    = cols_hca,
      widths    = rep(3, hca_k),
      types     = rep(1, hca_k),
      color_width_type_labels = data.frame(
        color = cols_hca,
        width = 3,
        type  = 1,
        label = str_glue('cluster {seq_len(hca_k)}')
      ),
      main = str_glue("Cluster QFs ({hca_rep_label}, k = {hca_k})")
    )
    dev.off()

    ## Same curves against the root QF -- the representative of the whole
    ## sample -- so each cluster reads as a departure from the overall
    ## distribution.
    Qi_root <- hca_rep_mean(seq_len(N_hca))

    ## Sized as the top row of a LaTeX panel: 2:1 at full width, above the
    ## three box-q*.png figures (2:3 each, a third of the width). All four
    ## share pointsize and res so their text matches once placed.
    Qi_contrast <- lapply(Qi_cluster, function(q) q - Qi_root)

    png(file.path(dir_hca_k, 'cluster-qfs-contrast.png'),
        width = 960 * hca_res / 72, height = 480 * hca_res / 72,
        pointsize = hca_pointsize, res = hca_res)
    par(mar = c(4, 4, 2.5, 3), mgp = c(2.4, 0.7, 0))
    plot_funs(
      fun_list  = Qi_contrast,
      grid_list = rep(list(grid_hca), hca_k),
      colors    = cols_hca,
      widths    = rep(2, hca_k),
      types     = rep(1, hca_k),
      ylab = 'Q(p) Contrast',
      main = str_glue("Cluster QF Contrasts")
    )
    abline(h = 0, lty = 2, col = 'gray50')
    ## The quantile levels boxplotted in box-q*.png
    abline(v = hca_q_levels, lty = 3, col = 'gray30')
    mtext(hca_q_levels, side = 3, at = hca_q_levels, line = 0.1, cex = 0.7,
          col = 'gray30')
    ## Cluster numbers in the right margin in place of a legend, at the height
    ## where each curve ends, spread apart where curves end close together.
    usr   <- par('usr')
    y_end <- vapply(Qi_contrast, function(q) q[length(q)], numeric(1))
    y_lab <- hca_spread_labels(y_end, gap = 1.15 * strheight('0', cex = 0.8))
    x_lab <- usr[2] + 0.02 * diff(usr[1:2])
    segments(max(grid_hca), y_end, x_lab, y_lab, col = cols_hca, xpd = NA)
    text(x_lab, y_lab, seq_len(hca_k), col = cols_hca, pos = 4, offset = 0.2,
         cex = 0.8, font = 2, xpd = NA)
    dev.off()




    ## ---------- Covariate summaries ---------- ##

    tbl_hca_male <- data.frame(cluster = cl_hca, male = df_cov$male) %>%
      group_by(cluster) %>%
      summarise(prop_male = mean(male, na.rm = TRUE)) %>%
      as.data.frame()
    print(tbl_hca_male)
    write.csv(
      tbl_hca_male,
      file.path(dir_hca_k, 'cov-male.csv'),
      row.names = FALSE
    )

    ## Each continuous covariate gets a density overlay and a stack of
    ## cluster-wise boxplots. File stems match the previous explicit calls.
    hca_cov_plots <- c('age', 'pir', 'bmd_adj', 'hdl_adj', 'ghb_adj')
    for (cv in hca_cov_plots) {
      slug <- gsub('_', '-', cv)
      hca_plot_density(
        df_cov[[cv]], cl_hca, cols_hca,
        xlab = cv, main = str_glue("{cv} by cluster (k = {hca_k})"),
        path = file.path(dir_hca_k, str_glue('cov-{slug}.png'))
      )
      hca_plot_boxes(
        df_cov[[cv]], cl_hca, cols_hca,
        xlab = cv, main = str_glue("{cv} by cluster (k = {hca_k})"),
        path = file.path(dir_hca_k, str_glue('box-{slug}.png'))
      )
    }

    ## All covariates in one figure
    hca_plot_cov_summary(
      df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
      cl_hca, cols_hca,
      path = file.path(dir_hca_k, 'cov-summary.png')
    )

    ## Same figure grouped on the manual phenotypes, where this
    ## representation and k have any. Phenotypes keep their listed order and
    ## are not color-coded, since the palette belongs to the clusters.
    pheno_hca <- phenotypes_to_clusters[[hca_latent]][[as.character(hca_k)]]
    if (!is.null(pheno_hca)) {
      ## Every cluster must belong to exactly one phenotype
      stopifnot(setequal(unlist(pheno_hca), seq_len(hca_k)),
                !anyDuplicated(unlist(pheno_hca)))
      ph_hca <- rep(seq_along(pheno_hca), lengths(pheno_hca))[
        match(cl_hca, unlist(pheno_hca))
      ]
      print(table(phenotype = factor(ph_hca, labels = names(pheno_hca))))
      hca_plot_cov_summary(
        df_cov[, c('male', 'age', 'pir', 'bmd_adj', 'hdl_adj')],
        ph_hca, rep('grey80', length(pheno_hca)),
        path      = file.path(dir_hca_k, 'cov-summary-phenotype.png'),
        labels    = names(pheno_hca),
        group_lab = 'phenotype'
      )
    }

    ## Activity quantiles on the original scale. Sized as the bottom row of
    ## the LaTeX panel under cluster-qfs-contrast.png: 2:3, a third as wide.
    for (j in seq_along(hca_q_levels)) {
      q_pct <- round(100 * hca_q_levels[j])
      hca_plot_boxes(
        Q_lvl_hca[, j], cl_hca, cols_hca,
        xlab = 'MIMS',
        main = str_glue("Q({hca_q_levels[j]})"),
        path = file.path(dir_hca_k, str_glue('box-q{q_pct}.png')),
        width = 320, height = 480, pointsize = hca_pointsize, res = hca_res,
        mar = c(4, 4, 2.5, 0.8),
        faint_outliers = TRUE
      )
    }


  }
}

