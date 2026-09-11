#!/usr/bin/env bash
# Submits power_simulation_descriptive_compute.R (scripts/power_simulations/)
# as an LSF job array: one task per (sim_rep, effect, psi) row of the
# Setting 4 sweep for a single REGIME -- one task per SIMULATION (not one task
# per split), like batch/submit_power_simulation_inferential.sh. Each task
# runs the Hotelling test on the full N_0 + N_1 dataset and computes S*R
# omnibus + S*R directed log-normalized conditional generativity scores
# (S stratified train/val splits of that dataset, R synthetic replicates per
# split), writing them to artifacts/<ART_DIR_NAME>/replicates/b-<LSB_JOBINDEX>/.
# A dependent aggregation task then runs power_simulation_descriptive_aggregate.R
# once every row is done. Submit again with REGIME set to the other value to
# get its sweep.
#
# This is the descriptive counterpart to the inferential arm -- keeps
# Hotelling power but replaces the generativity hypothesis test with the
# log-normalized conditional generativity score (manuscript Section 2.1.2),
# pooled into one boxplot per relabel fraction rather than a rejection rate.
# See scripts/power_simulations/README.md for both arms.
#
# Usage: batch/submit_power_simulation_descriptive.sh
# All sweep and parallelism parameters are set below, not passed as args.
set -euo pipefail

# --- 0. Run mode ----------------------------------------------------------
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
REGIME="mean"                  # "mean" or "skewness" -- one regime per run;
                                # submit again with the other value for its sweep
if [[ "$REGIME" != "mean" && "$REGIME" != "skewness" ]]; then
  echo "REGIME must be 'mean' or 'skewness' (got '$REGIME')." >&2
  exit 1
fi
ART_DIR_NAME="power-sim-descriptive-5-${REGIME}"     # subdir under artifacts/
NSIM_PER_SETTING=50             # simulations per (effect, psi) setting = LSF rows

# Swept axis. EFFECT_LEVELS is interpreted according to REGIME: a group-mean
# gap added to MU_MU for group 1 in the "mean" regime, or a group-1 skew-normal
# slant shift added to MU_ALPHA in the "skewness" regime.
EFFECT_LEVELS=(5.0) #(1.0 1.5 2.0)
PSI_LEVELS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Scalar knobs (fixed across the sweep).
NOBS=500
N_0=500
N_1=500
K=2                           # retained latent dimension (fixed, no K sweep)
K_MAX=10
S=5                             # stratified train/val splits, per simulation
R=10                            # synthetic replicates per split, per statistic
ALPHA=0.05                      # only used for the Hotelling rejection-rate curve
V_TEST=1                        # covariate value the generativity score conditions on
RIDGE=0                        # latent-covariance ridge (small-n lever)
MU_MU=0
SIGMA_MU=5
MU_LOG_SIGMA=0
SIGMA_LOG_SIGMA=1
MU_ALPHA=0
SIGMA_ALPHA=3

# --- 2. Parallelism / LSF resource parameters --------------------------------
N_CONCURRENT=50
WALLTIME_TASK="12:00"           # per statistic: S*R = 100 decode calls plus S
                                 # conditional refits / re-encodes (vs. R=20 in
                                 # the split-per-row predecessor) -- still well
                                 # under the hypothesis-test variants' B_NULL=499.
                                 # Provisional: recalibrate from a pilot run's
                                 # actual wall-time.
MEM_TASK="rusage[mem=4000MB]"        # torch + reticulate headroom
WALLTIME_AGGREGATE="01:00"
MEM_AGGREGATE="rusage[mem=2000MB]"

QUEUE="cceb_normal"
RSCRIPT_BIN="/appl/R-4.5/bin/Rscript"

# --- 3. Derived counts and paths ---------------------------------------------
N_EFF=${#EFFECT_LEVELS[@]}
N_PSI=${#PSI_LEVELS[@]}
NSIM=$(( NSIM_PER_SETTING * N_EFF * N_PSI ))
if (( NSIM == 0 )); then
  echo "NSIM is 0 -- check NSIM_PER_SETTING / *_LEVELS." >&2
  exit 1
fi
# ";"-joined (not ","): these strings travel inside bsub's own comma-delimited
# -env list below.
EFFECT_LEVELS_STR=$(IFS=';'; echo "${EFFECT_LEVELS[*]}")
PSI_LEVELS_STR=$(IFS=';'; echo "${PSI_LEVELS[*]}")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="$REPO_ROOT/artifacts/$ART_DIR_NAME"
LOG_DIR="$ART_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ "$RUN_COMPUTE" == true ]]; then
  rm -rf "$ART_DIR/replicates"
fi

# --- 4. Compute array: one task per row --------------------------------------
COMPUTE_JOB_ID=""
if [[ "$RUN_COMPUTE" == true ]]; then
  # Pins BLAS/OMP threading to 1 and fixes libstdc++ / MKL loading for the sgm
  # conda Python that reticulate drives (see submit_sensitivity_simulation.sh).
  ENV_GUARD='export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_PRELOAD=/project/jsmorris-lab/stanley5/conda_envs/sgm/lib/libstdc++.so.6 LD_LIBRARY_PATH=/project/jsmorris-lab/stanley5/conda_envs/sgm/lib:$LD_LIBRARY_PATH;'

  # Maps this task's LSB_JOBINDEX (1..NSIM) to its (sim_rep, effect, psi) pair
  # -- inverse of column-major expand.grid(sim_rep, effect, psi) order (sim_rep
  # fastest, psi slowest). sim_rep here indexes one of the NSIM_PER_SETTING
  # simulations at this setting. REGIME is fixed for the whole run (see above),
  # so it isn't decoded here -- it travels straight through in ENV_VARS below.
  DECODE='
idx0=$((LSB_JOBINDEX - 1))
sim_rep=$(( idx0 % NSIM_PER_SETTING + 1 ))
idx1=$(( idx0 / NSIM_PER_SETTING ))
eff_idx=$(( idx1 % N_EFF ))
idx2=$(( idx1 / N_EFF ))
psi_idx=$(( idx2 % N_PSI ))
IFS=";" read -ra EFFECT_ARR <<< "$EFFECT_LEVELS_STR"
IFS=";" read -ra PSI_ARR <<< "$PSI_LEVELS_STR"
export SIM_REP="$sim_rep"
export EFFECT="${EFFECT_ARR[$eff_idx]}"
export PSI="${PSI_ARR[$psi_idx]}"
'

  RUN_CMD="$ENV_GUARD $DECODE $RSCRIPT_BIN --no-init-file scripts/power_simulations/power_simulation_descriptive_compute.R"

  ENV_VARS="all, NSIM_PER_SETTING=${NSIM_PER_SETTING}, S=${S}, N_EFF=${N_EFF}, N_PSI=${N_PSI}, EFFECT_LEVELS_STR=${EFFECT_LEVELS_STR}, PSI_LEVELS_STR=${PSI_LEVELS_STR}, REGIME=${REGIME}, NOBS=${NOBS}, N_0=${N_0}, N_1=${N_1}, K=${K}, K_MAX=${K_MAX}, R=${R}, ALPHA=${ALPHA}, V_TEST=${V_TEST}, RIDGE=${RIDGE}, MU_MU=${MU_MU}, SIGMA_MU=${SIGMA_MU}, MU_LOG_SIGMA=${MU_LOG_SIGMA}, SIGMA_LOG_SIGMA=${SIGMA_LOG_SIGMA}, MU_ALPHA=${MU_ALPHA}, SIGMA_ALPHA=${SIGMA_ALPHA}, ART_DIR=${ART_DIR}"

  COMPUTE_SUBMIT=$(bsub \
    -J "power_sim_descriptive[1-${NSIM}]%${N_CONCURRENT}" \
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
  WAIT_ARGS=()
  if [[ "$RUN_COMPUTE" == true ]]; then
    WAIT_ARGS=(-w "done(${COMPUTE_JOB_ID})")
  fi
  bsub \
    -J "power_sim_descriptive_aggregate" \
    -q "$QUEUE" \
    -n 1 -R "$MEM_AGGREGATE" \
    -W "$WALLTIME_AGGREGATE" \
    "${WAIT_ARGS[@]}" \
    -o "$LOG_DIR/aggregate_%J.out" \
    -e "$LOG_DIR/aggregate_%J.err" \
    -env "all, ART_DIR=${ART_DIR}, ALPHA=${ALPHA}" \
    "cd '$REPO_ROOT' && $RSCRIPT_BIN --no-init-file scripts/power_simulations/power_simulation_descriptive_aggregate.R"

  if [[ "$RUN_COMPUTE" == true ]]; then
    echo "Submitted aggregation job, dependent on done(${COMPUTE_JOB_ID})"
  else
    echo "Submitted aggregation job (no compute dependency -- AGGREGATE_ONLY)"
  fi
fi
