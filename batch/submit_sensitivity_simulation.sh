#!/usr/bin/env bash
# Submits scripts/sensitivity_simulation_compute.R as an LSF job array: one
# task per (sim_rep, mu_alpha_2, n_2) row of the sweep -- no row-splitting or
# row-looping happens in R, each task computes exactly one row and writes it
# to artifacts/<ART_DIR_NAME>/b-<LSB_JOBINDEX>/. A dependent aggregation task
# then runs scripts/sensitivity_simulation_aggregate.R once every row is done.
#
# Usage: batch/submit_sensitivity_simulation.sh
# All sweep and parallelism parameters are set below, not passed as args.
set -euo pipefail

# --- 0. Run mode --------------------------------------------------------------
# If both are false (default), compute then aggregate run in sequence. Set
# exactly one to true to (re)run just that stage -- e.g. AGGREGATE_ONLY=true
# to re-plot from already-computed rows without resubmitting the sweep.
COMPUTE_ONLY=false
AGGREGATE_ONLY=true
if [[ "$COMPUTE_ONLY" == true && "$AGGREGATE_ONLY" == true ]]; then
  echo "COMPUTE_ONLY and AGGREGATE_ONLY can't both be true." >&2
  exit 1
fi
RUN_COMPUTE=true
RUN_AGGREGATE=true
if [[ "$AGGREGATE_ONLY" == true ]]; then RUN_COMPUTE=false; fi
if [[ "$COMPUTE_ONLY" == true ]]; then RUN_AGGREGATE=false; fi

# --- 1. Simulation parameters ------------------------------------------------
ART_DIR_NAME="sensitivity-K-2"     # subdir under artifacts/
NSIM_PER_SETTING=50
MU_MU_1=0
SIGMA_MU_1=5
MU_LOG_SIGMA_1=0
SIGMA_LOG_SIGMA_1=1
MU_ALPHA_2_LEVELS=(1.0 1.25 1.5 1.75 2.0)
SIGMA_ALPHA_2=3
NOBS=500
N_1=100
N_2_LEVELS=(100)
K_LEVELS="1:25"       # contiguous K sweep, "min:max"
K_MAX=50

# --- 2. Parallelism / LSF resource parameters --------------------------------
N_CONCURRENT=50      # max array tasks running at once (LSF "%N" throttle)
WALLTIME_TASK="04:00"
MEM_TASK="rusage[mem=10MB]"          # units/format are site-specific
WALLTIME_AGGREGATE="01:00"
MEM_AGGREGATE="rusage[mem=10MB]"

QUEUE="cceb_normal"

# Absolute path to Rscript, rather than `module load R/4.5; Rscript` -- LSF's
# job-execution shell isn't guaranteed to inherit the `module` bash function.
# This install already has every R package these scripts need; conda's `sgm`
# env is used only via reticulate for the Python/torch side (src/config.yml's
# python.path), independent of which R binary is running.
RSCRIPT_BIN="/appl/R-4.5/bin/Rscript"

# --- 3. Derived counts and paths ---------------------------------------------
N_MU=${#MU_ALPHA_2_LEVELS[@]}
N_N2=${#N_2_LEVELS[@]}
NSIM=$(( NSIM_PER_SETTING * N_MU * N_N2 ))
if (( NSIM == 0 )); then
  echo "NSIM is 0 -- check NSIM_PER_SETTING/MU_ALPHA_2_LEVELS/N_2_LEVELS." >&2
  exit 1
fi
# ";"-joined (not ","): these strings travel inside bsub's own comma-delimited
# -env list below, so a "," delimiter here would collide with it.
MU_ALPHA_2_LEVELS_STR=$(IFS=';'; echo "${MU_ALPHA_2_LEVELS[*]}")
N_2_LEVELS_STR=$(IFS=';'; echo "${N_2_LEVELS[*]}")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="$REPO_ROOT/artifacts/$ART_DIR_NAME"
LOG_DIR="$ART_DIR/logs"

# LSF requires -o/-e directories to pre-exist; the R scripts create their own
# row/plot dirs at runtime but won't have run yet when bsub captures this
# submission's output.
mkdir -p "$LOG_DIR"

# Clears row output from any prior run so aggregation can't silently mix in
# stale results. Safe no-op if none exist yet. Skipped for AGGREGATE_ONLY,
# which reads rows a prior run already left behind.
if [[ "$RUN_COMPUTE" == true ]]; then
  rm -rf "$ART_DIR"/b-*
fi

# --- 4. Compute array: one task per row --------------------------------------
COMPUTE_JOB_ID=""
if [[ "$RUN_COMPUTE" == true ]]; then
  # Pins BLAS/OMP threading to 1 (torch could otherwise claim every core on
  # whatever node it lands on). LD_PRELOAD fixes module R loading the system's
  # older libstdc++ before reticulate ever imports torch, whose bundled
  # libtorch_python.so needs a newer GLIBCXX. LD_LIBRARY_PATH fixes torch's
  # MKL backend failing to dlopen its own libmkl_*.so plugins on some compute
  # nodes -- we never `conda activate` (reticulate just points at sgm's python
  # binary directly), and MKL's loader leans on LD_LIBRARY_PATH rather than
  # RPATH, unlike most of the rest of the env.
  ENV_GUARD='export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_PRELOAD=~/conda_envs/sgm/lib/libstdc++.so.6 LD_LIBRARY_PATH=~/conda_envs/sgm/lib:$LD_LIBRARY_PATH;'

  # Maps this task's LSB_JOBINDEX (1..NSIM) to its (sim_rep, mu_alpha_2, n_2)
  # triple -- inverse of column-major expand.grid(sim_rep, mu_alpha_2, n_2)
  # order (sim_rep fastest, n_2 slowest). Bash arrays can't cross bsub's exec
  # boundary, so the levels travel as the joined strings above and are
  # rebuilt here, on the compute node, from env vars set by -env below.
  DECODE='
idx0=$((LSB_JOBINDEX - 1))
sim_rep=$(( idx0 % NSIM_PER_SETTING + 1 ))
idx1=$(( idx0 / NSIM_PER_SETTING ))
mu_idx=$(( idx1 % N_MU ))
idx2=$(( idx1 / N_MU ))
n2_idx=$(( idx2 % N_N2 ))
IFS=";" read -ra MU_ARR <<< "$MU_ALPHA_2_LEVELS_STR"
IFS=";" read -ra N2_ARR <<< "$N_2_LEVELS_STR"
export MU_ALPHA_2="${MU_ARR[$mu_idx]}"
export N_2="${N2_ARR[$n2_idx]}"
export SIM_REP="$sim_rep"
'

  # --no-init-file: this repo's .Rprofile sources renv/activate.R, which
  # doesn't exist in this checkout -- skipping it lets Rscript start against
  # R/4.5's default library.
  RUN_CMD="$ENV_GUARD $DECODE $RSCRIPT_BIN --no-init-file scripts/sensitivity_simulation_compute.R"

  ENV_VARS="all, NSIM_PER_SETTING=${NSIM_PER_SETTING}, N_MU=${N_MU}, N_N2=${N_N2}, MU_ALPHA_2_LEVELS_STR=${MU_ALPHA_2_LEVELS_STR}, N_2_LEVELS_STR=${N_2_LEVELS_STR}, MU_MU_1=${MU_MU_1}, SIGMA_MU_1=${SIGMA_MU_1}, MU_LOG_SIGMA_1=${MU_LOG_SIGMA_1}, SIGMA_LOG_SIGMA_1=${SIGMA_LOG_SIGMA_1}, SIGMA_ALPHA_2=${SIGMA_ALPHA_2}, NOBS=${NOBS}, N_1=${N_1}, K_LEVELS=${K_LEVELS}, K_MAX=${K_MAX}, ART_DIR=${ART_DIR}"

  COMPUTE_SUBMIT=$(bsub \
    -J "sens_sim[1-${NSIM}]%${N_CONCURRENT}" \
    -q "$QUEUE" \
    -n 1 -R "span[hosts=1] $MEM_TASK" \
    -W "$WALLTIME_TASK" \
    -o "$LOG_DIR/compute_%J_%I.out" \
    -e "$LOG_DIR/compute_%J_%I.err" \
    -env "$ENV_VARS" \
    "cd '$REPO_ROOT' && $RUN_CMD")

  echo "$COMPUTE_SUBMIT"
  COMPUTE_JOB_ID=$(echo "$COMPUTE_SUBMIT" | grep -oE '[0-9]+' | head -1)
  if [[ -z "$COMPUTE_JOB_ID" ]]; then
    echo "Could not parse a job ID from bsub's output above -- aborting before submitting the aggregation job." >&2
    exit 1
  fi
fi

# --- 5. Aggregation: depends on the whole array finishing (if it ran) --------
if [[ "$RUN_AGGREGATE" == true ]]; then
  # done(): every array element must reach DONE before this fires. If a task
  # crashes instead, this job stays PEND until you notice/fix and resubmit --
  # safer than silently aggregating an incomplete sweep. No dependency when
  # this run didn't also submit the compute array (COMPUTE_ONLY wasn't run
  # first) -- AGGREGATE_ONLY just reads rows a prior run already left behind.
  WAIT_ARGS=()
  if [[ "$RUN_COMPUTE" == true ]]; then
    WAIT_ARGS=(-w "done(${COMPUTE_JOB_ID})")
  fi
  bsub \
    -J "sens_sim_aggregate" \
    -q "$QUEUE" \
    -n 1 -R "$MEM_AGGREGATE" \
    -W "$WALLTIME_AGGREGATE" \
    "${WAIT_ARGS[@]}" \
    -o "$LOG_DIR/aggregate_%J.out" \
    -e "$LOG_DIR/aggregate_%J.err" \
    -env "all, ART_DIR=${ART_DIR}" \
    "cd '$REPO_ROOT' && $RSCRIPT_BIN --no-init-file scripts/sensitivity_simulation_aggregate.R"

  if [[ "$RUN_COMPUTE" == true ]]; then
    echo "Submitted aggregation job, dependent on done(${COMPUTE_JOB_ID})"
  else
    echo "Submitted aggregation job (no compute dependency -- AGGREGATE_ONLY)"
  fi
fi
