# Power simulations (Setting 4)

Two experiment arms that both ask the same question -- *if a fitted SGM
encoder-decoder is given progressively corrupted group labels, does that
corruption show up in the generativity diagnostics the same way it shows up in
a group-difference test?* -- but answer it differently: one as a hypothesis
test, one as a descriptive score.

## Shared setup

Both arms sweep the same simulation grid, one LSF task per grid point:

- **`regime`** (`mean` or `skewness`, one full sweep per batch submission):
  group 1 differs from group 0 either by a mean shift or by a skew-normal
  slant shift of the given `effect` size.
- **`effect`**: the size of that group-1 shift.
- **`psi`** (0 to 1): the fraction of observations whose group label is
  reassigned uniformly at random before fitting. `psi = 0` is the intact
  labeling; `psi = 1` destroys all group structure (relabeling, not flipping
  -- flipping at `psi = 1` would just invert the labels, not destroy them).

Each task simulates its own two-group dataset, fits a **fresh** SGM encoder
(`stage_eqf_sgrid` + `stage_eqf_cgrid` + `stage_wame` + `stage_flow`), applies
the `psi`-relabeling, and runs a two-sample Hotelling test on the relabeled
embeddings -- this is the shared power curve both arms report. What differs is
how each arm then diagnoses group `V_TEST`'s conditional generativity.

## Inferential arm

`power_simulation_inferential_compute.R` / `_aggregate.R`, submitted via
`batch/submit_power_simulation_inferential.sh`.

Fits the conditional latent model to the **relabeled** group-`V_TEST`
embeddings, then runs `generativity_frozen_ref_test` (a frozen-reference null
test: a single large reference draw from the fitted model is frozen as the
common right-hand side, and a held-out true-label anchor's discrepancy against
it is compared to `B_NULL` null discrepancies) for the omnibus (Wasserstein)
statistic and the directed statistic matching `regime`. Reports a p-value per
task.

**Output** (`artifacts/<ART_DIR_NAME>/plots/`):
- `rejection_vs_psi.png` -- rejection rate (Wilson 95% band) vs. `psi`, for
  Hotelling and both generativity statistics.
- `pvalue_vs_psi.png` -- median p-value (Q1-Q3 band) vs. `psi`.

## Descriptive arm

`power_simulation_descriptive_compute.R` / `_aggregate.R`, submitted via
`batch/submit_power_simulation_descriptive.sh`.

No null distribution or p-value. Instead, `evaluate_conditional_generativity`
runs `S` independent stratified train/val splits of the task's full dataset;
per split it fits the conditional latent model to the **relabeled**
group-`V_TEST` subset of the training half, and scores
`log(T_syn-val / T_train-val)` (manuscript Section 2.1.2) against the
**true**-label group-`V_TEST` subset of the held-out validation half, for `R`
synthetic replicates. That asymmetry -- fit on relabeled data, scored against
true labels -- is what makes the score move with `psi`. Each task therefore
produces `S * R` scores per statistic.

**Output** (`artifacts/<ART_DIR_NAME>/plots/`):
- `hotelling_power.png` -- Hotelling rejection rate vs. `psi`.
- `generativity_omnibus.png` -- one boxplot per `psi`, pooling every
  `S * R * NSIM_PER_SETTING` omnibus score at that setting (deliberately not
  split-wise -- splits are pooled, not shown side by side).
- `generativity_directed.png` -- same, for the `regime`-matching directed
  statistic.

## Running

Each `batch/submit_power_simulation_*.sh` is self-contained: sweep and LSF
resource parameters are set at the top of the file (not passed as args). Set
`REGIME` and submit; submit again with the other value for its sweep.
`COMPUTE_ONLY=true` runs just the compute array (useful for a pilot row before
committing to the full sweep); `AGGREGATE_ONLY=true` re-runs just the plots
against an existing `artifacts/<ART_DIR_NAME>/replicates/` directory.

Both R scripts in a pair must run with the working directory set to the repo
root (they `source("src/cot.R")` / `source("scripts/power_simulations/utils.R")`
by relative path) -- the batch scripts already `cd` there before invoking
`Rscript`.

## Files

| file | what it does |
|---|---|
| `power_simulation_inferential_compute.R` | one row of the inferential sweep |
| `power_simulation_inferential_aggregate.R` | inferential plots |
| `power_simulation_descriptive_compute.R` | one row of the descriptive sweep |
| `power_simulation_descriptive_aggregate.R` | descriptive plots |
| `utils.R` | shared generativity primitives (both arms) -- discrepancy statistic, conditional latent fit, decoder sampler, the frozen-reference test, the log-normalized score, `wilson_interval` |
