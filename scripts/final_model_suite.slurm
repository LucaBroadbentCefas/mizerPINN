#!/bin/bash
#SBATCH --job-name=pinn_final_suite
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-69%5
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

# ============================================================================
# Final PINN model suite
#
# 70 tasks total:
#   0-2   : single-species main runs (sp_3, sp_7, sp_11)
#   3-11  : single-species state/architecture ablations
#   12-13 : multispecies no-data / perfect-data baselines
#   14-38 : known-noise sweep, discrepancy gate ON
#   39-43 : known-noise gate-OFF controls
#   44-49 : deterministic data-removal experiments
#   50-52 : lambda_PDE = 0 comparisons
#   53-56 : Rmax recovery from four starting factors
#   57-61 : noisy Rmax recovery, five noise realisations
#   62-65 : global CV recovery from four starting values
#   66-69 : fishing-effort/F recovery from four scalar effort starts
#
# All tasks are independent. Noise is generated per task into unique files.
# Set DRY_RUN=1 to print the selected task command without training.
# ============================================================================

PROJECT_DIR="/gpfs/home/sfc26usu/mizerPINN"
SINGLE_BASE_DIR="validation/fixtures/pde_single_species"
MS_BASE_DIR="validation/fixtures/pde_multispecies"

# Deterministic data variants created for the final suite.
DATA_VARIANTS_DIR="final_runs/observations/final_runs"

# Task-isolated generated files. Nothing in validation/fixtures is overwritten.
FINAL_INPUT_ROOT="final_runs/final_suite_inputs"
NOISE_OUTPUT_ROOT="final_runs/generated_observations/final_suite_noise"
MANIFEST_ROOT="final_runs/final_suite_manifest"

DRY_RUN="${DRY_RUN:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
JOB_ID="${SLURM_JOB_ID:-manual}"

if (( TASK_ID < 0 || TASK_ID > 69 )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-69." >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p slurm_logs
mkdir -p "${FINAL_INPUT_ROOT}/job_${ARRAY_JOB_ID}"
mkdir -p "${NOISE_OUTPUT_ROOT}/job_${ARRAY_JOB_ID}"
mkdir -p "${MANIFEST_ROOT}/job_${ARRAY_JOB_ID}"

if [[ "${DRY_RUN}" != "1" ]]; then
    module purge
    module add mamba/25.3.1-0
    eval "$(conda shell.bash hook)"
    conda activate mizer-torch
fi

require_file() {
    local path="$1"
    if [[ "${DRY_RUN}" != "1" && ! -f "${path}" ]]; then
        echo "Required file does not exist: ${path}" >&2
        exit 3
    fi
}

require_dir() {
    local path="$1"
    if [[ "${DRY_RUN}" != "1" && ! -d "${path}" ]]; then
        echo "Required directory does not exist: ${path}" >&2
        exit 3
    fi
}

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Canonical final-fit hyperparameters
# ---------------------------------------------------------------------------

N_STEPS=15000
N_TIME=128
N_EVAL=60
LR="1e-3"
LR_MIN="1e-5"

STATE_PARAMETERIZATION="log-u"
RESIDUAL_FORM="scaled"
BOUNDARY_LOSS_FORM="relative"

MODEL_ARCH="fourier"
FOURIER_NUM_FEATURES=32
FOURIER_SCALE="0.5"
FOURIER_SEED=123

NETWORK_SEED=123

# Five independent observation-noise realisations.
# The same five latent noise seeds are reused at every CV, giving a paired CV sweep.
NOISE_SEEDS=(41001 41002 41003 41004 41005)
NOISE_CVS=("0.1" "0.2" "0.3" "0.4" "0.5")

RMAX_FACTORS=("0.25" "0.5" "2.0" "4.0")
CV_STARTS=("0.1" "0.3" "0.6" "1.0")

# --effort-init is a scalar absolute effort applied to every time x gear cell.
# These starts span the scale of the current reference effort vector.
EFFORT_STARTS=("0.05" "0.5" "1.0" "3.0")

# Globals set by noise/Rmax preparation helpers.
DATA_CSV=""
INPUT_DIR="${MS_BASE_DIR}"
DATA_DEFAULT_CV="0.3"
RMAX_TRUE_FILE=""

prepare_noisy_data() {
    local cv="$1"
    local noise_seed="$2"
    local tag="$3"
    local source_csv="${DATA_VARIANTS_DIR}/perfect.csv"
    local out_dir="${NOISE_OUTPUT_ROOT}/job_${ARRAY_JOB_ID}/task_${TASK_ID}"
    local out_csv="${out_dir}/${tag}_cv_${cv//./p}_noise_${noise_seed}.csv"

    require_file "${source_csv}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: generate noisy data -> ${out_csv}"
    else
        mkdir -p "${out_dir}"
        python scripts/make_noisy_observations.py \
            --input "${source_csv}" \
            --output "${out_csv}" \
            --cv "${cv}" \
            --seed "${noise_seed}" \
            --task-id "${TASK_ID}" \
            --overwrite
        [[ -s "${out_csv}" ]] || {
            echo "Generated noise file is missing/empty: ${out_csv}" >&2
            exit 4
        }
    fi

    DATA_CSV="${out_csv}"
    DATA_DEFAULT_CV="${cv}"
}

prepare_rmax_input() {
    local factor="$1"
    local dest="${FINAL_INPUT_ROOT}/job_${ARRAY_JOB_ID}/task_${TASK_ID}/rmax_factor_${factor//./p}"

    require_dir "${MS_BASE_DIR}"
    require_file "${MS_BASE_DIR}/r_max.csv"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: copy ${MS_BASE_DIR} -> ${dest} and multiply r_max.csv by ${factor}"
    else
        rm -rf "${dest}"
        mkdir -p "${dest}"
        cp -a "${MS_BASE_DIR}/." "${dest}/"
        cp "${MS_BASE_DIR}/r_max.csv" "${dest}/r_max_true.csv"

        python - "${dest}/r_max.csv" "${factor}" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
factor = float(sys.argv[2])

if not math.isfinite(factor) or factor <= 0:
    raise ValueError(f"Rmax factor must be finite and positive; got {factor}")

with path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))

if len(rows) < 2:
    raise ValueError(f"No r_max values found in {path}")
if len(rows[0]) != 1:
    raise ValueError(f"Expected one column in {path}; found {len(rows[0])}")

updated = [rows[0]]
for row in rows[1:]:
    if not row:
        continue
    value = float(row[0]) * factor
    if not math.isfinite(value) or value <= 0:
        raise ValueError("All modified r_max values must be finite and positive.")
    updated.append([f"{value:.17g}"])

with path.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(updated)
PY
    fi

    INPUT_DIR="${dest}"
    RMAX_TRUE_FILE="${dest}/r_max_true.csv"
}

record_run() {
    local run_root="$1"
    local run_label="$2"
    local manifest_dir="${MANIFEST_ROOT}/job_${ARRAY_JOB_ID}"
    local manifest_file="${manifest_dir}/task_$(printf '%02d' "${TASK_ID}")_${run_label}.txt"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: record ${run_label} under ${run_root}"
        return
    fi

    local run_dir
    run_dir="$(
        find "${run_root}" -maxdepth 1 -mindepth 1 -type d \
            -name "*_job${JOB_ID}_task${TASK_ID}" \
            -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"

    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "Could not locate completed run directory for task ${TASK_ID}." >&2
        return 1
    fi

    printf '%s\n' "${run_label}" > "${run_dir}/final_suite_label.txt"

    if [[ -n "${RMAX_TRUE_FILE}" && -f "${RMAX_TRUE_FILE}" ]]; then
        cp "${RMAX_TRUE_FILE}" "${run_dir}/r_max_true.csv"
    fi

    {
        echo "task_id=${TASK_ID}"
        echo "run_label=${run_label}"
        echo "run_dir=${run_dir}"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "slurm_job_id=${JOB_ID}"
        echo "slurm_array_job_id=${ARRAY_JOB_ID}"
    } > "${manifest_file}"

    echo "Recorded run: ${manifest_file}"
}

run_single_species() {
    local species_folder="$1"
    local state_parameterization="$2"
    local model_arch="$3"
    local run_label="$4"
    local input_dir="${SINGLE_BASE_DIR}/${species_folder}"

    require_dir "${input_dir}"

    local args=(
        --input-dir "${input_dir}"
        --n-steps "${N_STEPS}"
        --n-time "${N_TIME}"
        --n-eval "${N_EVAL}"
        --lr "${LR}"
        --lr-scheduler cosine
        --lr-min "${LR_MIN}"

        --state-parameterization "${state_parameterization}"
        --state-scale-eps 1e-30
        --residual-form "${RESIDUAL_FORM}"

        --pde-penalty squared
        --pde-pseudo-huber-delta 1.0

        --boundary-loss-form "${BOUNDARY_LOSS_FORM}"
        --bc-penalty squared
        --bc-pseudo-huber-delta 1.0
        --bc-g-min 1e-12

        --lambda-pde 1.0
        --lambda-ic 1.0
        --lambda-bc 1.0
        --initial-w-pde 1.0
        --initial-w-ic 1.0
        --initial-w-bc 0.1

        --lambda-timestep 0.0
        --collocation-strategy uniform
        --time-sampling stratified

        --causal-loss expert
        --causal-curriculum linear
        --causal-start-fraction 0.05
        --causal-ramp-steps 20000
        --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
        --causal-n-chunks 64
        --causal-epsilon 1.0

        --loss-weighting expert-grad-norm
        --expert-weight-update-every 2000
        --expert-weight-alpha 0.9
        --expert-weight-batch fixed
        --weight-min 1e-3
        --weight-max 1e2
        --expert-weight-min 1e-3
        --expert-weight-max 1e2
        --no-hard-set-first-weight-update

        --model-arch "${model_arch}"

        --weight-factorization rwf
        --rwf-mu 1.0
        --rwf-sigma 0.1
        --rwf-apply-to all
        --rwf-base-init xavier_uniform

        --hidden-width 384
        --hidden-layers 5

        --diag-final-n-time 301
        --diag-final-n-eval 100

        --seed "${NETWORK_SEED}"
        --device cpu
        --print-every 500
        --checkpoint-every 5000
        --hpc
    )

    if [[ "${model_arch}" == "fourier" ]]; then
        args+=(
            --fourier-num-features "${FOURIER_NUM_FEATURES}"
            --fourier-scale "${FOURIER_SCALE}"
            --fourier-include-raw-input
            --fourier-seed "${FOURIER_SEED}"
        )
    fi

    echo "============================================================"
    echo "task=${TASK_ID}"
    echo "run_label=${run_label}"
    echo "mode=single_species"
    echo "input_dir=${input_dir}"
    echo "state_parameterization=${state_parameterization}"
    echo "model_arch=${model_arch}"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "============================================================"

    run_cmd python -m scripts.train_pde_only_single_species "${args[@]}"

    if [[ "${DRY_RUN}" != "1" ]]; then
        record_run "runs/pde_only_single_species" "${run_label}"
    fi
}

run_multispecies() {
    local run_label="$1"
    local lambda_pde="$2"
    local lambda_data="$3"
    local use_gate="$4"
    local estimate_rmax="$5"
    local estimate_cv="$6"
    local cv_init="$7"
    local estimate_effort="$8"
    local effort_init="$9"

    require_dir "${INPUT_DIR}"
    if [[ -n "${DATA_CSV}" ]]; then
        require_file "${DATA_CSV}"
    fi

    local args=(
        --input-dir "${INPUT_DIR}"
        --species-mode all

        --n-steps "${N_STEPS}"
        --n-time "${N_TIME}"
        --n-eval "${N_EVAL}"

        --lr "${LR}"
        --lr-scheduler cosine
        --lr-min "${LR_MIN}"

        --state-parameterization "${STATE_PARAMETERIZATION}"
        --state-scale-eps 1e-30
        --residual-form "${RESIDUAL_FORM}"

        --pde-penalty squared
        --pde-pseudo-huber-delta 1.0

        --boundary-loss-form "${BOUNDARY_LOSS_FORM}"
        --bc-penalty squared
        --bc-pseudo-huber-delta 1.0
        --bc-g-min 1e-12

        --lambda-pde "${lambda_pde}"
        --lambda-ic 1.0
        --lambda-bc 1.0

        --initial-w-pde 1.0
        --initial-w-ic 1.0
        --initial-w-bc 0.1
        --initial-w-data 1.0

        --lambda-data "${lambda_data}"
        --data-default-cv "${DATA_DEFAULT_CV}"
        --data-loss-eps 1e-30
        --data-time-quadrature-points 3

        --lambda-timestep 0.0
        --collocation-strategy uniform
        --time-sampling stratified

        --causal-loss expert
        --causal-curriculum linear
        --causal-start-fraction 0.05
        --causal-ramp-steps 20000
        --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
        --causal-n-chunks 64
        --causal-epsilon 1.0

        --loss-weighting expert-grad-norm
        --expert-weight-update-every 2000
        --expert-weight-alpha 0.9
        --expert-weight-batch fixed
        --weight-min 1e-3
        --weight-max 1e2
        --expert-weight-min 1e-3
        --expert-weight-max 1e2
        --no-hard-set-first-weight-update

        --model-arch fourier
        --fourier-num-features "${FOURIER_NUM_FEATURES}"
        --fourier-scale "${FOURIER_SCALE}"
        --fourier-include-raw-input
        --fourier-seed "${FOURIER_SEED}"

        --weight-factorization rwf
        --rwf-mu 1.0
        --rwf-sigma 0.1
        --rwf-apply-to all
        --rwf-base-init xavier_uniform

        --hidden-width 384
        --hidden-layers 5

        --diag-final-n-time 301
        --diag-final-n-eval 100

        --seed "${NETWORK_SEED}"
        --device cpu
        --print-every 500
        --checkpoint-every 5000
        --hpc
    )

    if [[ -n "${DATA_CSV}" ]]; then
        args+=(--data-csv "${DATA_CSV}")
    fi

    if [[ "${use_gate}" == "1" ]]; then
        args+=(--data-discrepancy-gate)
    fi

    if [[ "${estimate_rmax}" == "1" ]]; then
        args+=(
            --estimate-rmax
            --rmax-lr 1e-3
            --rmax-log-lower 0
            --rmax-log-upper 50
        )
    fi

    if [[ "${estimate_cv}" == "1" ]]; then
        args+=(
            --estimate-data-cv
            --data-cv-scope global
            --data-cv-init "${cv_init}"
            --data-cv-lower 0.02
            --data-cv-upper 1.5
            --data-cv-lr 1e-3
        )
    fi

    if [[ "${estimate_effort}" == "1" ]]; then
        args+=(
            --estimate-effort
            --effort-lr 1e-2
            --effort-init "${effort_init}"
        )
    fi

    echo "============================================================"
    echo "task=${TASK_ID}"
    echo "run_label=${run_label}"
    echo "mode=multispecies"
    echo "input_dir=${INPUT_DIR}"
    echo "data_csv=${DATA_CSV:-none}"
    echo "lambda_pde=${lambda_pde}"
    echo "lambda_data=${lambda_data}"
    echo "data_discrepancy_gate=${use_gate}"
    echo "estimate_rmax=${estimate_rmax}"
    echo "estimate_data_cv=${estimate_cv}"
    echo "estimate_effort=${estimate_effort}"
    echo "data_default_cv=${DATA_DEFAULT_CV}"
    echo "start_time=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "============================================================"

    run_cmd python -m scripts.train_pde_multispecies "${args[@]}"

    if [[ "${DRY_RUN}" != "1" ]]; then
        record_run "runs/pde_multispecies" "${run_label}"
    fi
}

# ============================================================================
# Task selection
# ============================================================================

if (( TASK_ID <= 2 )); then
    species=("sp_3" "sp_7" "sp_11")
    sp="${species[$TASK_ID]}"
    run_single_species "${sp}" "log-u" "fourier" "single_${sp}_logu_fourier"
    exit 0
fi

if (( TASK_ID >= 3 && TASK_ID <= 11 )); then
    species=("sp_3" "sp_7" "sp_11")
    ablation_states=("log-u" "log-n" "log-n")
    ablation_archs=("mlp" "fourier" "mlp")
    ablation_tags=("logu_mlp" "logn_fourier" "logn_mlp")

    idx=$((TASK_ID - 3))
    sp_idx=$((idx / 3))
    ab_idx=$((idx % 3))

    sp="${species[$sp_idx]}"
    state="${ablation_states[$ab_idx]}"
    arch="${ablation_archs[$ab_idx]}"
    tag="${ablation_tags[$ab_idx]}"

    run_single_species "${sp}" "${state}" "${arch}" "single_${sp}_${tag}"
    exit 0
fi

# Defaults for multispecies tasks.
INPUT_DIR="${MS_BASE_DIR}"
DATA_CSV=""
DATA_DEFAULT_CV="0.3"
RMAX_TRUE_FILE=""

# 12: no data
if (( TASK_ID == 12 )); then
    run_multispecies "ms_no_data" "1.0" "0.0" "0" "0" "0" "" "0" ""
    exit 0
fi

# 13: perfect data
if (( TASK_ID == 13 )); then
    DATA_CSV="${DATA_VARIANTS_DIR}/perfect.csv"
    run_multispecies "ms_perfect" "1.0" "1.0" "0" "0" "0" "" "0" ""
    exit 0
fi

# 14-38: five CVs x five independent noise realisations, gate ON
if (( TASK_ID >= 14 && TASK_ID <= 38 )); then
    idx=$((TASK_ID - 14))
    cv_idx=$((idx / 5))
    rep_idx=$((idx % 5))
    cv="${NOISE_CVS[$cv_idx]}"
    noise_seed="${NOISE_SEEDS[$rep_idx]}"
    rep=$((rep_idx + 1))

    prepare_noisy_data "${cv}" "${noise_seed}" "noise_gate_on_rep${rep}"
    run_multispecies \
        "noise_cv${cv//./p}_rep${rep}_gate_on" \
        "1.0" "1.0" "1" "0" "0" "" "0" ""
    exit 0
fi

# 39-43: one gate-OFF control per CV, paired with noise realisation 1
if (( TASK_ID >= 39 && TASK_ID <= 43 )); then
    cv_idx=$((TASK_ID - 39))
    cv="${NOISE_CVS[$cv_idx]}"
    noise_seed="${NOISE_SEEDS[0]}"

    prepare_noisy_data "${cv}" "${noise_seed}" "noise_gate_off"
    run_multispecies \
        "noise_cv${cv//./p}_rep1_gate_off" \
        "1.0" "1.0" "0" "0" "0" "" "0" ""
    exit 0
fi

# 44-49: deterministic sparse/missing-data experiments
if (( TASK_ID >= 44 && TASK_ID <= 49 )); then
    data_files=(
        "every_3yr.csv"
        "gap_10_20.csv"
        "gap_30_40.csv"
        "missing_species_3.csv"
        "missing_species_7.csv"
        "missing_species_11.csv"
    )
    data_tags=(
        "every_3yr"
        "gap_10_20"
        "gap_30_40"
        "missing_species_3"
        "missing_species_7"
        "missing_species_11"
    )

    idx=$((TASK_ID - 44))
    DATA_CSV="${DATA_VARIANTS_DIR}/${data_files[$idx]}"
    run_multispecies \
        "data_${data_tags[$idx]}" \
        "1.0" "1.0" "0" "0" "0" "" "0" ""
    exit 0
fi

# 50-52: lambda_PDE = 0 comparisons.
# Only lambda_PDE changes relative to the corresponding primary experiment.
if (( TASK_ID == 50 )); then
    DATA_CSV="${DATA_VARIANTS_DIR}/perfect.csv"
    run_multispecies "lambdaPDE0_perfect" "0.0" "1.0" "0" "0" "0" "" "0" ""
    exit 0
fi

if (( TASK_ID == 51 )); then
    prepare_noisy_data "0.3" "${NOISE_SEEDS[0]}" "lambdaPDE0_noise"
    run_multispecies "lambdaPDE0_noise_cv0p3_rep1" "0.0" "1.0" "1" "0" "0" "" "0" ""
    exit 0
fi

if (( TASK_ID == 52 )); then
    DATA_CSV="${DATA_VARIANTS_DIR}/gap_30_40.csv"
    run_multispecies "lambdaPDE0_gap_30_40" "0.0" "1.0" "0" "0" "0" "" "0" ""
    exit 0
fi

# 53-56: Rmax recovery from four deterministic starting factors, perfect data
if (( TASK_ID >= 53 && TASK_ID <= 56 )); then
    idx=$((TASK_ID - 53))
    factor="${RMAX_FACTORS[$idx]}"
    prepare_rmax_input "${factor}"
    DATA_CSV="${DATA_VARIANTS_DIR}/perfect.csv"

    run_multispecies \
        "rmax_start_factor_${factor//./p}" \
        "1.0" "1.0" "0" "1" "0" "" "0" ""
    exit 0
fi

# 57-61: noisy Rmax recovery at CV 0.3, five noise realisations.
# Rmax starts at 0.5 x the exported true value.
if (( TASK_ID >= 57 && TASK_ID <= 61 )); then
    rep_idx=$((TASK_ID - 57))
    rep=$((rep_idx + 1))
    noise_seed="${NOISE_SEEDS[$rep_idx]}"

    prepare_rmax_input "0.5"
    prepare_noisy_data "0.3" "${noise_seed}" "rmax_noise_rep${rep}"

    run_multispecies \
        "rmax_noise_cv0p3_rep${rep}_start0p5" \
        "1.0" "1.0" "1" "1" "0" "" "0" ""
    exit 0
fi

# 62-65: recover one global CV from the same CV=0.3 noisy dataset,
# starting from four different values. Discrepancy gate must be OFF because
# the training code forbids --estimate-data-cv + --data-discrepancy-gate.
if (( TASK_ID >= 62 && TASK_ID <= 65 )); then
    idx=$((TASK_ID - 62))
    cv_init="${CV_STARTS[$idx]}"

    prepare_noisy_data "0.3" "${NOISE_SEEDS[0]}" "cv_recovery_start_${cv_init//./p}"

    run_multispecies \
        "cv_recovery_true0p3_start${cv_init//./p}" \
        "1.0" "1.0" "0" "0" "1" "${cv_init}" "0" ""
    exit 0
fi

# 66-69: recover the annual time x gear fishing-effort matrix.
# The implementation derives fishing mortality F from estimated effort,
# fixed catchability and fixed selectivity.
if (( TASK_ID >= 66 && TASK_ID <= 69 )); then
    idx=$((TASK_ID - 66))
    effort_init="${EFFORT_STARTS[$idx]}"
    DATA_CSV="${DATA_VARIANTS_DIR}/perfect.csv"

    run_multispecies \
        "effort_recovery_start${effort_init//./p}" \
        "1.0" "1.0" "0" "0" "0" "" "1" "${effort_init}"
    exit 0
fi

echo "No task mapping found for TASK_ID=${TASK_ID}" >&2
exit 99
