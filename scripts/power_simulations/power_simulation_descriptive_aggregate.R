# Power simulation (Setting 4), descriptive arm -- aggregate stage ---------
# Scans every <ART_DIR>/replicates/b-*/ row directory written by
# power_simulation_descriptive_compute.R, combines their per-row Hotelling
# p-value and S*R-vector log-normalized conditional generativity scores
# (manuscript Section 2.1.2; the directed statistic is whichever of
# skewness/mean matches this run's REGIME -- compute.R only ever fills in one
# of them) into one long table, and produces three figures vs. the relabeled
# fraction psi:
#   - hotelling_power.png      Hotelling rejection rate (ribbon: Wilson band).
#   - generativity_omnibus.png Omnibus log-normalized generativity score: ONE
#                               boxplot per psi, pooling all
#                               S * R * NSIM_PER_SETTING scores at that psi
#                               (this deliberately deviates from the
#                               split-wise-boxplot convention of
#                               plot_generativity_boxes -- splits are pooled,
#                               not shown side by side).
#   - generativity_directed.png  Same, for the REGIME-matching directed
#                                 statistic.
# Reads only ART_DIR -- row parameters come from each row's settings.yml.
library(ggplot2)
library(data.table)
library(yaml)
library(stringr)
source("scripts/power_simulations/utils.R")   # wilson_interval

dir_art <- Sys.getenv("ART_DIR", unset = NA)
if (is.na(dir_art)) stop("Missing required env var: ART_DIR")
alpha <- as.numeric(Sys.getenv("ALPHA", unset = "0.05"))
dir_plots <- file.path(dir_art, "plots")
dir.create(dir_plots, showWarnings = FALSE, recursive = TRUE)

# Collect rows ------------------------------------------------------------
dir_replicates <- file.path(dir_art, "replicates")
row_dirs <- list.dirs(dir_replicates, recursive = FALSE)
row_dirs <- row_dirs[grepl("^b-[0-9]+$", basename(row_dirs))]
if (length(row_dirs) == 0) stop("No row directories found in ", dir_replicates)

# One task = one simulation. Each row's three log_norm_* fields are length
# S * R (split-major: split 1's R values, then split 2's, ...); p_hotelling is
# a scalar, recycled across the S * R lines here and de-duplicated by b before
# the Hotelling panel.
read_row <- function(rd) {
  settings <- read_yaml(file.path(rd, "settings.yml"))
  res <- readRDS(file.path(rd, "results.rds"))
  S <- res$S
  R <- res$R
  data.table(
    b             = settings$b,
    sim_rep       = settings$sim_rep,
    regime        = settings$regime,
    effect        = settings$effect,
    psi           = settings$psi,
    split         = rep(seq_len(S), each = R),
    replicate     = rep(seq_len(R), times = S),
    p_hotelling   = res$p_hotelling,
    log_norm_wass = res$log_norm_wass,
    log_norm_skew = res$log_norm_skew,
    log_norm_mean = res$log_norm_mean
  )
}
results <- rbindlist(lapply(row_dirs, read_row))

# Directed statistic matching this row's regime -- compute.R only computes the
# one matching REGIME and leaves the other NA, so this just picks it out.
results[, log_norm_directed := fifelse(regime == "mean", log_norm_mean, log_norm_skew)]

# Warn (don't block) on gaps -- a missing row shouldn't stop plotting the rest.
b_found <- sort(unique(results$b))
b_expected <- seq_len(max(b_found))
if (!setequal(b_found, b_expected)) {
  warning(str_glue(
    "Missing rows: b = {paste(setdiff(b_expected, b_found), collapse = ', ')}"
  ))
}

# Hotelling rejection rate + Wilson band -----------------------------------
# One p-value per simulation (b), not per (split, replicate) -- de-duplicate
# first, then the rate over the NSIM_PER_SETTING simulations at each setting.
hotelling_tbl <- unique(results[, .(b, sim_rep, regime, effect, psi, p_hotelling)], by = "b")
rej_hotelling <- hotelling_tbl[, {
  k <- sum(p_hotelling < alpha, na.rm = TRUE)
  n <- sum(!is.na(p_hotelling))
  if (n == 0) {
    .(rate = NA_real_, lower = NA_real_, upper = NA_real_, n = 0L)
  } else {
    ci <- wilson_interval(k, n)
    .(rate = k / n, lower = ci$lower, upper = ci$upper, n = n)
  }
}, by = .(regime, effect, psi)]

# Plotting --------------------------------------------------------------
house_theme <- theme(
  legend.position = 'bottom',
  text = element_text(size = 15, hjust = 0.5),
  plot.title = element_text(size = 15, hjust = 0),
  axis.title = element_text(size = 15)
)

p_hotelling_plot <- ggplot(rej_hotelling, aes(x = psi, y = rate)) +
  geom_hline(yintercept = alpha, linetype = 3, colour = "red") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, fill = "steelblue", colour = NA) +
  geom_line(colour = "steelblue") +
  geom_point(colour = "steelblue") +
  facet_grid(regime ~ effect, labeller = label_both) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = expression(psi ~ "(relabeled fraction)"),
       y = "Rejection rate",
       title = "Hotelling power",
       subtitle = expression("Dashed red:" ~ alpha ~ "; band: Wilson 95% CI")) +
  house_theme
ggsave(file.path(dir_plots, "hotelling_power.png"), p_hotelling_plot, width = 10, height = 7)

# Pooled boxplot: every (split, replicate, sim) score at a psi goes into one
# box -- no per-split dodging (cf. this file's header note).
box_plot <- function(dt, y_col, title_str) {
  ggplot(dt, aes(x = factor(psi), y = .data[[y_col]])) +
    geom_hline(yintercept = 0, linetype = 3, colour = "red") +
    geom_boxplot(outlier.size = 0.6) +
    facet_grid(regime ~ effect, labeller = label_both) +
    labs(x = expression(psi ~ "(relabeled fraction)"),
         y = "Log-normalized score",
         title = title_str,
         subtitle = "One box per psi, pooling all splits x replicates x sims; dashed red: 0") +
    house_theme
}

p_omnibus <- box_plot(results, "log_norm_wass", "Omnibus conditional generativity")
ggsave(file.path(dir_plots, "generativity_omnibus.png"), p_omnibus, width = 10, height = 7)

p_directed <- box_plot(results, "log_norm_directed", "Directed conditional generativity")
ggsave(file.path(dir_plots, "generativity_directed.png"), p_directed, width = 10, height = 7)
