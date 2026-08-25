#!/bin/bash
#SBATCH --job-name=pinn_cv_noise
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-7%8
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN
mkdir -p slurm_logs

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_NUMBER=$((TASK_ID + 1))

# Eight strictly positive fixed CVs. CV=0 is excluded because sd_log=0 is
# invalid for the current lognormal likelihood.
CVS=(
  "0.05"
  "0.10"
  "0.20"
  "0.30"
  "0.40"
  "0.50"
  "0.75"
  "1.00"
)

if (( TASK_ID < 0 || TASK_ID >= ${#CVS[@]} )); then
  echo "Invalid array task ID: ${TASK_ID}" >&2
  exit 2
fi

CV="${CVS[$TASK_ID]}"
CV_TAG="${CV//./p}"

INPUT_DIR="validation/fixtures/pde_multispecies"
TRUE_OBS_CSV="validation/observations.csv"
GENERATED_OBS_DIR="runs/generated_observations/cv_sweep_job_${SLURM_ARRAY_JOB_ID:-manual}"
OBS_CSV="${GENERATED_OBS_DIR}/obs_task${TASK_NUMBER}_cv${CV_TAG}.csv"

# Use the same latent standard-normal draws for every CV. This isolates the
# effect of increasing noise magnitude. For independent noise realisations,
# change this to: NOISE_SEED=$((20260729 + TASK_ID))
NOISE_SEED=20260729

mkdir -p "${GENERATED_OBS_DIR}"

python scripts/make_noisy_observations.py \
  --input "${TRUE_OBS_CSV}" \
  --output "${OBS_CSV}" \
  --cv "${CV}" \
  --seed "${NOISE_SEED}" \
  --task-id "${TASK_NUMBER}"

if [[ ! -s "${OBS_CSV}" ]]; then
  echo "Generated observation file is missing or empty: ${OBS_CSV}" >&2
  exit 3
fi

echo "Starting CV sweep task"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=${TASK_ID}"
echo "task_number=${TASK_NUMBER}"
echo "cv=${CV}"
echo "noise_seed=${NOISE_SEED}"
echo "observation_csv=${OBS_CSV}"
echo "start_time=$(date --iso-8601=seconds)"

# Stagger startup because the current training code creates run directories
# using timestamps with one-second resolution.
sleep $((TASK_ID * 2))

python -m scripts.train_pde_multispecies \
  --input-dir "${INPUT_DIR}" \
  --species-mode all \
  --n-steps 30000 \
  --n-time 128 \
  --n-eval 60 \
  --lr 1e-3 \
  --state-parameterization log-u \
  --state-scale-eps 1e-30 \
  --residual-form scaled \
  --pde-penalty squared \
  --pde-pseudo-huber-delta 1.0 \
  --boundary-loss-form relative \
  --bc-penalty squared \
  --bc-pseudo-huber-delta 1.0 \
  --lambda-pde 1.0 \
  --lambda-ic 1.0 \
  --lambda-bc 1.0 \
  --initial-w-bc 0.1 \
  --data-csv "${OBS_CSV}" \
  --lambda-data 1.0 \
  --initial-w-data 1.0 \
  --data-default-cv "${CV}" \
  --data-loss-eps 1e-30 \
  --data-time-quadrature-points 3 \
  --lambda-timestep 0.0 \
  --collocation-strategy uniform \
  --time-sampling stratified \
  --causal-loss expert \
  --causal-curriculum linear \
  --causal-start-fraction 0.05 \
  --causal-ramp-steps 20000 \
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0" \
  --causal-n-chunks 64 \
  --causal-epsilon 1.0 \
  --loss-weighting expert-grad-norm \
  --expert-weight-update-every 2000 \
  --expert-weight-alpha 0.9 \
  --expert-weight-batch fixed \
  --weight-min 1e-3 \
  --weight-max 1e2 \
  --expert-weight-min 1e-3 \
  --expert-weight-max 1e2 \
  --no-hard-set-first-weight-update \
  --model-arch fourier \
  --fourier-num-features 32 \
  --fourier-scale 0.5 \
  --fourier-include-raw-input \
  --fourier-seed 123 \
  --weight-factorization rwf \
  --rwf-mu 1.0 \
  --rwf-sigma 0.1 \
  --rwf-apply-to all \
  --rwf-base-init xavier_uniform \
  --hidden-width 384 \
  --hidden-layers 5 \
  --lr-scheduler cosine \
  --lr-min 1e-5 \
  --rmax-lr 1e-3 \
  --rmax-log-lower 0 \
  --rmax-log-upper 50 \
  --diag-final-n-time 301 \
  --diag-final-n-eval 100 \
  --seed 123 \
  --device cpu \
  --print-every 500 \
  --checkpoint-every 5000 \
  --hpc

echo "Finished CV sweep task"
echo "cv=${CV}"
echo "observation_csv=${OBS_CSV}"
echo "end_time=$(date --iso-8601=seconds)"
