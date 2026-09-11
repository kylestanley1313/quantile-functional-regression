# Power simulation (Setting 4), inferential arm -- aggregate stage ---------
# Scans every <ART_DIR>/replicates/b-*/ row directory written by
# power_simulation_inferential_compute.R, combines their per-row Hotelling and
# conditional-generativity p-values into one long table, and produces the
# Hotelling vs. omnibus vs. directed generativity figures (the directed
# statistic is whichever of skewness/mean matches this run's REGIME --
# compute.R only ever fills in one of them):
#   - pvalue_vs_psi.png     median p-value (ribbon: Q1-Q3) vs. relabeled
#                           fraction.
#   - rejection_vs_psi.png  rejection rate (ribbon: Wilson band) vs. relabeled
#                           fraction.
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

read_row <- function(rd) {
  settings <- read_yaml(file.path(rd, "settings.yml"))
  res <- readRDS(file.path(rd, "results.rds"))
  data.table(
    b           = settings$b,
    sim_rep     = settings$sim_rep,
    regime      = settings$regime,
    effect      = settings$effect,
    psi         = settings$psi,
    p_hotelling = res$p_hotelling,
    p_gen_wass  = res$p_gen_wass,
    p_gen_skew  = res$p_gen_skew,
    p_gen_mean  = res$p_gen_mean
  )
}
results <- rbindlist(lapply(row_dirs, read_row))

# Directed statistic matching this row's regime -- compute.R only computes the
# one matching REGIME and leaves the other NA, so this just picks it out.
results[, p_directed := fifelse(regime == "mean", p_gen_mean, p_gen_skew)]

# Warn (don't block) on gaps -- a missing row shouldn't stop plotting the rest.
b_found <- sort(unique(results$b))
b_expected <- seq_len(max(b_found))
if (!setequal(b_found, b_expected)) {
  warning(str_glue(
    "Missing rows: b = {paste(setdiff(b_expected, b_found), collapse = ', ')}"
  ))
}

# Long table: Hotelling, omnibus generativity, and this run's directed
# generativity statistic, all three together on one axis of "method".
method_labels <- c(
  p_hotelling = "Hotelling",
  p_gen_wass  = "Generativity (omnibus)",
  p_directed  = "Generativity (directed)"
)
long <- melt(
  results,
  id.vars      = c("regime", "effect", "psi", "sim_rep"),
  measure.vars = names(method_labels),
  variable.name = "method", value.name = "p"
)
long[, method := factor(method_labels[as.character(method)], levels = method_labels)]

# Rejection rates + Wilson bands -----------------------------------------
rej <- long[, {
  k <- sum(p < alpha, na.rm = TRUE)
  n <- sum(!is.na(p))
  if (n == 0) {
    .(rate = NA_real_, lower = NA_real_, upper = NA_real_, n = 0L)
  } else {
    ci <- wilson_interval(k, n)
    .(rate = k / n, lower = ci$lower, upper = ci$upper, n = n)
  }
}, by = .(regime, effect, psi, method)]

# Median p-values + Q1/Q3 bands -------------------------------------------
pval_summ <- long[, {
  qs <- quantile(p, probs = c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE)
  .(q1 = qs[1], median = qs[2], q3 = qs[3])
}, by = .(regime, effect, psi, method)]

# Plotting --------------------------------------------------------------
house_theme <- theme(
  legend.position = 'bottom',
  text = element_text(size = 15, hjust = 0.5),
  plot.title = element_text(size = 15, hjust = 0),
  axis.title = element_text(size = 15)
)

p_rejection <- ggplot(rej, aes(x = psi, y = rate, colour = method, fill = method)) +
  geom_hline(yintercept = alpha, linetype = 3, colour = "red") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, colour = NA) +
  geom_line() +
  geom_point() +
  facet_grid(regime ~ effect, labeller = label_both) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = expression(psi ~ "(relabeled fraction)"),
       y = "Rejection rate",
       colour = NULL, fill = NULL,
       title = "Rejection rate vs. relabeled fraction",
       subtitle = expression("Dashed red:" ~ alpha ~ "; band: Wilson 95% CI")) +
  house_theme
ggsave(file.path(dir_plots, "rejection_vs_psi.png"), p_rejection, width = 10, height = 7)

p_pval <- ggplot(pval_summ, aes(x = psi, y = median, colour = method, fill = method)) +
  geom_hline(yintercept = alpha, linetype = 3, colour = "red") +
  geom_ribbon(aes(ymin = q1, ymax = q3), alpha = 0.15, colour = NA) +
  geom_line() +
  geom_point() +
  facet_grid(regime ~ effect, labeller = label_both) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = expression(psi ~ "(relabeled fraction)"),
       y = "Median p-value",
       colour = NULL, fill = NULL,
       title = "Median p-value vs. relabeled fraction",
       subtitle = expression("Dashed red:" ~ alpha ~ "; band: Q1-Q3")) +
  house_theme
ggsave(file.path(dir_plots, "pvalue_vs_psi.png"), p_pval, width = 10, height = 7)
