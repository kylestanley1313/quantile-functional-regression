# Sensitivity simulation -- aggregate stage -----------------------------------
# Scans every <ART_DIR>/b-*/ row directory written by
# scripts/sensitivity_simulation_compute.R, combines their per-K results into
# one long table, and produces the summary plots. Reads only ART_DIR -- row
# parameters (sim_rep, mu_alpha_2, n_2) come from each row's settings.yml, so
# the full sweep is never re-enumerated here.
library(ggplot2)
library(data.table)
library(yaml)
library(stringr)

dir_art <- Sys.getenv("ART_DIR", unset = NA)
if (is.na(dir_art)) stop("Missing required env var: ART_DIR")
dir_plots <- file.path(dir_art, "plots")
dir.create(dir_plots, showWarnings = FALSE, recursive = TRUE)

# Collect rows ------------------------------------------------------------
row_dirs <- list.dirs(dir_art, recursive = FALSE)
row_dirs <- row_dirs[grepl("^b-[0-9]+$", basename(row_dirs))]
if (length(row_dirs) == 0) stop("No row directories found in ", dir_art)

read_row <- function(rd) {
  settings <- read_yaml(file.path(rd, "settings.yml"))
  res <- readRDS(file.path(rd, "results.rds"))
  data.table(
    b          = settings$b,
    sim_rep    = settings$sim_rep,
    mu_alpha_2 = settings$mu_alpha_2,
    n_2        = settings$n_2,
    K          = res$K_levels,
    p_vals     = res$p_vals_by_k,
    p_vals_pca = res$p_vals_by_k_pca,
    gen_uncond = res$gen_uncond_by_k,
    gen_cond1  = res$gen_cond1_by_k,
    gen_cond2  = res$gen_cond2_by_k
  )
}
results <- rbindlist(lapply(row_dirs, read_row))

# Warn (don't block) on gaps -- a missing row shouldn't stop plotting the rest.
b_found <- sort(unique(results$b))
b_expected <- seq_len(max(b_found))
if (!setequal(b_found, b_expected)) {
  warning(str_glue(
    "Missing rows: b = {paste(setdiff(b_expected, b_found), collapse = ', ')}"
  ))
}

# Plotting ------------------------------------------------------------------
summarize_by_k <- function(col, transform = identity) {
  results[, .(value = mean(transform(get(col)), na.rm = TRUE)), by = .(mu_alpha_2, n_2, K)]
}

plot_metric_vs_k <- function(summ, y_lab, title_str) {
  ggplot(data = summ) +
    aes(x = K, y = value, colour = factor(mu_alpha_2, ordered = TRUE),
        pch = factor(n_2)) +
    geom_point() +
    geom_line() +
    labs(colour = expression(mu[alpha]~":"),
         linetype = expression(n[2]~":"),
         pch = expression(n[2]~":"),
         y = y_lab,
         x = "Retained K",
         title = title_str,
         subtitle = expression("Across swept"~K)) +
    theme(legend.position = 'bottom',
          text = element_text(size = 15, hjust = 0.5),
          plot.title = element_text(size = 15, hjust = 0),
          axis.title = element_text(size = 15))
}

save_metric_plot <- function(col, y_lab, title_str, file_name, transform = identity) {
  p <- plot_metric_vs_k(summarize_by_k(col, transform), y_lab, title_str)
  ggsave(file.path(dir_plots, file_name), p, width = 8, height = 6)
}

is_sig <- function(x) x < 0.05

save_metric_plot("p_vals", "Power", "Power using encoder-decoder Z-embeddings",
                  "power.png", is_sig)
save_metric_plot("p_vals", "P-Value", "P-Value using encoder-decoder Z-embeddings",
                  "pvalue.png")
save_metric_plot("p_vals_pca", "Power", "Power using standard PCA on quantile functions",
                  "power_pca.png", is_sig)
save_metric_plot("p_vals_pca", "P-Value", "P-Value using standard PCA on quantile functions",
                  "pvalue_pca.png")
save_metric_plot("gen_uncond", "Log normalized generativity", "Unconditional generativity vs K",
                  "generativity_unconditional.png")
save_metric_plot("gen_cond1", "Log normalized generativity", "Conditional generativity (group 1) vs K",
                  "generativity_cond_group1.png")
save_metric_plot("gen_cond2", "Log normalized generativity", "Conditional generativity (group 2) vs K",
                  "generativity_cond_group2.png")
