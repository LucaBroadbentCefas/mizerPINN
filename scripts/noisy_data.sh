#!/bin/bash
#SBATCH --job-name=pinn_data_loss
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-1%2
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN
mkdir -p slurm_logs

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

SCHEME_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCHEME_NAMES=(
  "data_no_adaptive"
  "data_adaptive"
)

LOSS_WEIGHTINGS=(
  "none"
  "expert-grad-norm"
)

DISABLE_ADAPTIVE=(
  "true"
  "false"
)

SCHEME_NAME="${SCHEME_NAMES[$SCHEME_ID]}"
LOSS_WEIGHTING="${LOSS_WEIGHTINGS[$SCHEME_ID]}"
USE_DISABLE_ADAPTIVE="${DISABLE_ADAPTIVE[$SCHEME_ID]}"

N_TIME="128"
N_EVAL="60"
LR="1e-3"
LR_SCHEDULER="cosine"
LR_MIN="1e-5"
WEIGHT_MIN="1e-3"
WEIGHT_MAX="1e2"
DATA_CSV="validation/fixtures/pde_multispecies/observations.csv"

echo "Starting ${SCHEME_NAME}"
echo "Start time: $(date)"
echo "scheme_id=${SCHEME_ID}"
echo "scheme_name=${SCHEME_NAME}"
echo "loss_weighting=${LOSS_WEIGHTING}"
echo "disable_adaptive=${USE_DISABLE_ADAPTIVE}"

COMMON_ARGS=(
  --input-dir validation/fixtures/pde_multispecies
  --species-mode all
  --n-steps 40000
  --n-time "${N_TIME}"
  --n-eval "${N_EVAL}"
  --lr "${LR}"

  --residual-form log
  --boundary-loss-form relative
  --lambda-pde 1.0
  --lambda-ic 1.0
  --lambda-bc 1.0
  --initial-w-bc 0.1

  --data-csv "${DATA_CSV}"
  --lambda-data 1.0
  --initial-w-data 1.0
  --data-default-cv 0.3
  --data-loss-eps 1e-30

  --lambda-timestep 0.0
  --collocation-strategy uniform

  --time-sampling stratified
  --causal-loss expert
  --causal-curriculum linear
  --causal-start-fraction 0.05
  --causal-ramp-steps 40000
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
  --causal-n-chunks 64
  --causal-epsilon 1.0

  --loss-weighting "${LOSS_WEIGHTING}"
  --expert-weight-update-every 2000
  --expert-weight-alpha 0.9
  --expert-weight-batch fixed
  --weight-min "${WEIGHT_MIN}"
  --weight-max "${WEIGHT_MAX}"
  --expert-weight-min "${WEIGHT_MIN}"
  --expert-weight-max "${WEIGHT_MAX}"
  --no-hard-set-first-weight-update

  --model-arch fourier
  --fourier-num-features 32
  --fourier-scale 1
  --fourier-include-raw-input
  --fourier-seed 123
  --weight-factorization rwf
  --rwf-mu 1.0
  --rwf-sigma 0.1
  --rwf-apply-to all
  --rwf-base-init xavier_uniform
  --hidden-width 384
  --hidden-layers 5

  --lr-scheduler "${LR_SCHEDULER}"
  --lr-min "${LR_MIN}"

  --seed 123
  --device cpu
  --print-every 500
  --checkpoint-every 5000
)

if [[ "${USE_DISABLE_ADAPTIVE}" == "true" ]]; then
  COMMON_ARGS+=(--disable-wang-weights)
fi

python -m scripts.train_pde_multispecies "${COMMON_ARGS[@]}"

echo "Finished ${SCHEME_NAME}"
echo "End time: $(date)"
