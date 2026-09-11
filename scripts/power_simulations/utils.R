## ---------- Power-Simulation Generativity Utilities ---------- ##
##
## Shared by both power-simulation arms (power_simulation_inferential_*.R and
## power_simulation_descriptive_*.R): the discrepancy primitive, the
## conditional latent fit / decoder sampler, the inferential arm's
## frozen-reference null test (Section 5.1 of generativity_framing.md), the
## descriptive arm's log-normalized score (manuscript Section 2.1.2), and
## wilson_interval for the rejection-rate curves. Assumes src/cot.R (which
## sources src/utils.R and src/model.R) is already sourced -- reuses
## q_moments, compute_generativity_score, fit_mean_cov, sample_mean_cov, and
## decode_z_to_Qi_aug / encode_y_to_Qi_aug from there. future / future.apply
## are attached transitively via src/model.R and referenced with `::`.


## ---- Discrepancy primitive ---- ##

## Dataset-level discrepancy G(A, B) between two collections of quantile
## functions discretized on a shared grid (Section 5.2).
##
## statistic = "wasserstein" is the omnibus score: the 2-Wasserstein distance
## between the empirical measures of A and B, delegated to
## compute_generativity_score. Its uniform marginals rep(1/|A|, |A|) and
## rep(1/|B|, |B|) already realize the semidiscrete transport when |A| != |B|,
## so an unbalanced comparison needs no special handling.
##
## statistic = "skewness" / "mean" / "variance" is a directed functional
## discrepancy |phi(A) - phi(B)|, where phi(.) averages a per-QF moment over the
## collection. This is a functional contrast, not an optimal-transport cost.
generativity_discrepancy <- function(
    Q_a_list,
    Q_b_list,
    w,
    statistic = c("wasserstein", "skewness", "mean", "variance")
) {
  statistic <- match.arg(statistic)
  if (statistic == "wasserstein") {
    return(compute_generativity_score(Q_a_list, Q_b_list, w, "wasserstein"))
  }
  moment <- switch(statistic,
    skewness = "skew",
    mean     = "mean",
    variance = "var"
  )
  phi_a <- mean(q_moments(Q_a_list, w)[[moment]])
  phi_b <- mean(q_moments(Q_b_list, w)[[moment]])
  abs(phi_a - phi_b)
}


## Conditional latent model theta_hat^(v) = (mu_hat^(v), Sigma_hat). With x or
## v NULL this is the unconditional fit_mean_cov. Otherwise the mean is the
## group-v mean and, by default (shared_cov = TRUE), the covariance is pooled
## across all groups -- the lever for small per-group samples. The return
## shape matches fit_mean_cov, so sample_mean_cov consumes it unchanged.
fit_latent_conditional <- function(Z, x = NULL, v = NULL, ridge = 0,
                                   shared_cov = TRUE) {
  if (is.null(x) || is.null(v)) {
    return(fit_mean_cov(Z, ridge = ridge))
  }
  in_v <- x == v
  stopifnot(sum(in_v) >= 2)
  K     <- ncol(Z)
  Sigma <- if (shared_cov) stats::cov(Z) else stats::cov(Z[in_v, , drop = FALSE])
  list(
    mu    = colMeans(Z[in_v, , drop = FALSE]),
    Sigma = Sigma + ridge * diag(K)
  )
}


## Build the draw_ref_qf(n) closure: draw n latent vectors from latent_fit (a
## list(mu, Sigma)) and decode them to quantile functions on p_grid_aug.
make_decoder_sampler <- function(pipeline, latent_fit, p_grid_aug, p_grid) {
  function(n) {
    decode_z_to_Qi_aug(pipeline, sample_mean_cov(n, latent_fit), p_grid_aug, p_grid)
  }
}


## ---- Inferential arm: frozen-reference hypothesis test ---- ##

## Frozen-reference model-based null test (Section 5.1). Tests whether
## obs_qf_list -- a held-out anchor of N_v quantile functions on the common grid
## -- is consistent with the fitted latent model whose decoded draws are
## produced by draw_ref_qf(n), a caller-supplied closure returning a length-n
## list of QFs on that same grid.
##
## A single size-M reference is drawn once and frozen: it is the common
## right-hand argument of every discrepancy, so its finite-sample optimal
## transport floor and any center/spread it inherits from the fit cancel in the
## ranking. The null is B discrepancies between fresh size-N_v model draws and
## that reference; the observed statistic instead anchors on obs_qf_list. Size
## matching -- null and observed anchors both N_v, only the reference larger --
## keeps the empirical-OT bias common across the observed and null statistics.
generativity_frozen_ref_test <- function(
    obs_qf_list,
    draw_ref_qf,
    w,
    statistic = c("wasserstein", "skewness", "mean", "variance"),
    M,
    B       = 999,
    alpha   = 0.05,
    n_cores = 1,
    seed    = 12345
) {
  statistic <- match.arg(statistic)
  N_v <- length(obs_qf_list)
  stopifnot(M > N_v)

  set.seed(seed)
  ref_seed   <- sample.int(.Machine$integer.max, 1)
  null_seeds <- sample.int(.Machine$integer.max, B)

  ## Step 1: frozen reference, drawn once and reused as the common RHS.
  set.seed(ref_seed)
  ref_qf_list <- draw_ref_qf(M)

  ## Step 3: observed statistic, anchored on the held-out obs_qf_list.
  stat_obs <- generativity_discrepancy(obs_qf_list, ref_qf_list, w, statistic)

  ## Step 2: null distribution from fresh size-N_v model draws.
  null_one <- function(b) {
    # A fresh multisession worker has no py_pkg (src/cot.R creates it only in
    # the launching session); decode_z_to_Qi_aug inside draw_ref_qf needs a
    # worker-local one, and this file (not re-sourced by src/cot.R) supplies
    # generativity_discrepancy. Guarded so a reused worker re-inits once.
    if (!exists("py_pkg", envir = .GlobalEnv, inherits = FALSE)) {
      setwd(.repo_root)
      source(file.path(.repo_root, "src", "cot.R"))
      source(file.path(.repo_root, "scripts", "power_simulations", "utils.R"))
    }
    set.seed(null_seeds[b])
    anchor_b <- draw_ref_qf(N_v)
    generativity_discrepancy(anchor_b, ref_qf_list, w, statistic)
  }

  .repo_root <- getwd()

  if (n_cores > 1) {
    # Reuse a caller-established worker pool if one is active; otherwise
    # self-manage a scoped pool for this call only.
    caller_has_pool <- !inherits(future::plan(), "sequential")
    if (!caller_has_pool) {
      old_plan <- future::plan(future::multisession, workers = n_cores)
      on.exit(future::plan(old_plan), add = TRUE)
    }
    # future.seed = NULL: each task self-seeds via null_seeds[b], so future's
    # own RNG-stream management is neither needed nor safe to enable here.
    # future.globals is explicit so future does not walk draw_ref_qf into a
    # live py_pkg handle; generativity_discrepancy rides along because
    # src/cot.R does not define it.
    stat_null <- unlist(future.apply::future_lapply(
      seq_len(B), null_one,
      future.seed    = NULL,
      future.globals = c("null_seeds", "draw_ref_qf", "ref_qf_list", "w",
                          "statistic", "N_v", ".repo_root",
                          "generativity_discrepancy")
    ))
  } else {
    stat_null <- unlist(lapply(seq_len(B), null_one))
  }

  ## Step 4: empirical p-value and (1 - alpha) critical value.
  stat_null_ok <- stat_null[!is.na(stat_null)]
  p_value <- (1 + sum(stat_null_ok >= stat_obs)) / (1 + length(stat_null_ok))
  q_crit  <- as.numeric(stats::quantile(stat_null_ok, probs = 1 - alpha,
                                        names = FALSE))

  list(
    stat_obs    = stat_obs,
    stat_null   = stat_null,
    p_value     = p_value,
    q_crit      = q_crit,
    reject      = stat_obs > q_crit,
    statistic   = statistic,
    M           = M,
    N_v         = N_v,
    B           = B,
    B_effective = length(stat_null_ok),
    alpha       = alpha
  )
}


## ---- Descriptive arm: log-normalized conditional generativity score ---- ##

## S-split log-normalized conditional generativity score (manuscript Section
## 2.1.2, applied to the conditional/per-group model rather than the
## unconditional one -- see evaluate_pipeline_generativity in src/utils.R for
## the unconditional sibling whose structure this mirrors). Descriptive only,
## no null distribution or p-value.
##
## For each of S independent train/val splits of the full N = length(y_list)
## dataset: fit the conditional latent model on the training half, then for R
## synthetic replicates return log(T_sv / T_tv), where T_tv is that split's real
## train-vs-val discrepancy (fixed within the split) and T_sv[r] is the
## synthetic-vs-val discrepancy for replicate r. The S*R values are returned
## flat, split-major (split 1's R values, then split 2's, ...).
##
## Splits are STRATIFIED by the true label x_true: each stratum (x_true == 0,
## x_true == 1) is halved by frac_train on its own, so the train and val halves
## each hold exactly floor(frac_train * n_0) group-0 and floor(frac_train * n_1)
## group-1 observations for every split and every relabeling proportion.
##
## Label convention (asymmetric on purpose): the TRAIN side selects group v by
## the psi-relabeled `label` -- the conditional latent model is fit on the
## psi-corrupted group -- while the VAL side selects group v by the true
## `x_true`. The score therefore measures how well the corrupted conditional
## model reproduces the *true* group-v marginal, which is what makes it move
## with the relabeling proportion.
##
## Synthetic draws are sized to match the train side (the true group-v subset
## of the split's training half) so T_sv and T_tv share the same finite-sample
## OT bias and log(T_sv / T_tv) is centered near 0 under a well-specified model.
evaluate_conditional_generativity <- function(
    pipeline, Z_mat, y_list, label, x_true, v,
    statistic = c("wasserstein", "skewness", "mean", "variance"),
    S = 5, R = 20, frac_train = 0.5, ridge = 0, shared_cov = TRUE,
    p_grid_aug, p_grid, w_aug,
    n_cores = 1, seed = 12345
) {
  statistic <- match.arg(statistic)
  n <- nrow(Z_mat)
  stopifnot(
    length(y_list) == n,
    length(label)  == n,
    length(x_true) == n,
    v %in% c(0L, 1L)
  )

  .repo_root <- getwd()

  set.seed(seed)
  split_seeds <- sample.int(.Machine$integer.max, S)

  idx0   <- which(x_true == 0)
  idx1   <- which(x_true == 1)
  n_tr_0 <- floor(frac_train * length(idx0))
  n_tr_1 <- floor(frac_train * length(idx1))
  stopifnot(
    n_tr_0 >= 1, n_tr_0 < length(idx0),
    n_tr_1 >= 1, n_tr_1 < length(idx1)
  )

  splits <- vector("list", S)
  for (s in seq_len(S)) {
    set.seed(split_seeds[s])

    ## Stratified train/val partition: halve each true-label stratum on its own
    ## so both halves keep identical group-0 / group-1 counts for every split
    ## (an unstratified sample(perm) split would let those counts drift).
    perm0   <- sample(idx0)
    perm1   <- sample(idx1)
    idx_tr  <- c(perm0[seq_len(n_tr_0)],
                 perm1[seq_len(n_tr_1)])
    idx_val <- c(perm0[(n_tr_0 + 1L):length(idx0)],
                 perm1[(n_tr_1 + 1L):length(idx1)])

    ## Conditional latent model on the training half; group v by RELABELED label.
    fit_v <- fit_latent_conditional(
      Z_mat[idx_tr, , drop = FALSE],
      x = label[idx_tr], v = v, ridge = ridge, shared_cov = shared_cov
    )
    draw_ref_qf <- make_decoder_sampler(pipeline, fit_v, p_grid_aug, p_grid)

    ## Train side: true group v within the training half.
    y_tr_v     <- y_list[idx_tr][x_true[idx_tr] == v]
    Q_tr_v_aug <- encode_y_to_Qi_aug(pipeline, y_tr_v, p_grid_aug)
    n_tr_v     <- length(y_tr_v)

    ## Val side: TRUE group v within the validation half (the held-out reference).
    y_val_v     <- y_list[idx_val][x_true[idx_val] == v]
    Q_val_v_aug <- encode_y_to_Qi_aug(pipeline, y_val_v, p_grid_aug)

    T_tv <- generativity_discrepancy(Q_tr_v_aug, Q_val_v_aug, w_aug, statistic)

    rep_seeds <- sample.int(.Machine$integer.max, R)

    gen_one_rep <- function(r) {
      # See generativity_frozen_ref_test's identical guard: a fresh multisession
      # worker has no py_pkg, and this file (not re-sourced by src/cot.R)
      # supplies generativity_discrepancy.
      if (!exists("py_pkg", envir = .GlobalEnv, inherits = FALSE)) {
        setwd(.repo_root)
        source(file.path(.repo_root, "src", "cot.R"))
        source(file.path(.repo_root, "scripts", "power_simulations", "utils.R"))
      }
      set.seed(rep_seeds[r])
      Q_synth_aug <- draw_ref_qf(n_tr_v)
      generativity_discrepancy(Q_synth_aug, Q_val_v_aug, w_aug, statistic)
    }

    if (n_cores > 1) {
      # Reuse a caller-established worker pool if one is active; otherwise
      # self-manage a scoped pool for this call only (created once, on the
      # first split -- later splits see the non-sequential plan and skip).
      caller_has_pool <- !inherits(future::plan(), "sequential")
      if (!caller_has_pool) {
        old_plan <- future::plan(future::multisession, workers = n_cores)
        on.exit(future::plan(old_plan), add = TRUE)
      }
      T_sv <- unlist(future.apply::future_lapply(
        seq_len(R), gen_one_rep,
        future.seed    = NULL,
        future.globals = c("rep_seeds", "draw_ref_qf", "n_tr_v", "Q_val_v_aug",
                            "w_aug", "statistic", ".repo_root",
                            "generativity_discrepancy")
      ))
    } else {
      T_sv <- unlist(lapply(seq_len(R), gen_one_rep))
    }

    splits[[s]] <- list(T_tv = T_tv, T_sv = T_sv, log_norm = log(T_sv / T_tv))
  }

  list(
    log_norm  = unlist(lapply(splits, `[[`, "log_norm")),
    T_tv      = vapply(splits, `[[`, numeric(1), "T_tv"),
    T_sv      = unlist(lapply(splits, `[[`, "T_sv")),
    split     = rep(seq_len(S), each = R),
    replicate = rep(seq_len(R), times = S),
    statistic = statistic,
    S         = S,
    R         = R
  )
}


## ---- Shared plotting helper ---- ##

## Wilson score interval for a binomial proportion k / n, vectorized over k and
## n. Used for rejection-rate bands on the power-simulation curves.
wilson_interval <- function(k, n, conf = 0.95) {
  z      <- stats::qnorm(1 - (1 - conf) / 2)
  phat   <- k / n
  denom  <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / denom
  halfwd <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / denom
  list(
    estimate = phat,
    lower    = pmax(0, centre - halfwd),
    upper    = pmin(1, centre + halfwd)
  )
}
