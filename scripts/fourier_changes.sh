#!/bin/bash
#SBATCH --job-name=pinn_state_u_huber_tests
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-4%5
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN
mkdir -p slurm_logs

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"

RUN_NAMES=(
  "state_u_squared_scale_1"
  "state_u_huber_scale_1"
  "state_u_huber_scale_05"
  "state_u_huber_scale_01"
  "state_u_huber_no_adaptive"
)

PDE_PENALTIES=(
  "squared"
  "pseudo-huber"
  "pseudo-huber"
  "pseudo-huber"
  "pseudo-huber"
)

BC_PENALTIES=(
  "squared"
  "pseudo-huber"
  "pseudo-huber"
  "pseudo-huber"
  "pseudo-huber"
)

FOURIER_SCALES=(
  "1.0"
  "1.0"
  "0.5"
  "0.1"
  "1.0"
)

LOSS_WEIGHTINGS=(
  "expert-grad-norm"
  "expert-grad-norm"
  "expert-grad-norm"
  "expert-grad-norm"
  "none"
)

RUN_NAME="${RUN_NAMES[$RUN_ID]}"
PDE_PENALTY="${PDE_PENALTIES[$RUN_ID]}"
BC_PENALTY="${BC_PENALTIES[$RUN_ID]}"
FOURIER_SCALE="${FOURIER_SCALES[$RUN_ID]}"
LOSS_WEIGHTING="${LOSS_WEIGHTINGS[$RUN_ID]}"

N_STEPS="25000"
N_TIME="128"
N_EVAL="60"
LR="1e-3"
LR_MIN="1e-5"
DATA_CSV="validation/fixtures/pde_multispecies/observations.csv"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "run_id=${RUN_ID}"
echo "pde_penalty=${PDE_PENALTY}"
echo "bc_penalty=${BC_PENALTY}"
echo "fourier_scale=${FOURIER_SCALE}"
echo "loss_weighting=${LOSS_WEIGHTING}"

sleep $((RUN_ID * 5))

ARGS=(
  --input-dir validation/fixtures/pde_multispecies
  --species-mode all
  --n-steps "${N_STEPS}"
  --n-time "${N_TIME}"
  --n-eval "${N_EVAL}"
  --lr "${LR}"

  --state-parameterization log-u
  --state-scale-eps 1e-30
  --residual-form scaled

  --pde-penalty "${PDE_PENALTY}"
  --pde-pseudo-huber-delta 1.0

  --boundary-loss-form relative
  --bc-penalty "${BC_PENALTY}"
  --bc-pseudo-huber-delta 1.0

  --lambda-pde 1.0
  --lambda-ic 1.0
  --lambda-bc 1.0
  --initial-w-bc 0.1

  --data-csv "${DATA_CSV}"
  --lambda-data 1.0
  --initial-w-data 1.0
  --data-default-cv 0.3
  --data-loss-eps 1e-30
  --data-time-quadrature-points 1

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

  --model-arch fourier
  --fourier-num-features 32
  --fourier-scale "${FOURIER_SCALE}"
  --fourier-include-raw-input
  --fourier-seed 123

  --weight-factorization rwf
  --rwf-mu 1.0
  --rwf-sigma 0.1
  --rwf-apply-to all
  --rwf-base-init xavier_uniform

  --hidden-width 384
  --hidden-layers 5

  --lr-scheduler cosine
  --lr-min "${LR_MIN}"

  --seed 123
  --device cpu
  --print-every 500
  --checkpoint-every 5000
)

if [[ "${LOSS_WEIGHTING}" == "expert-grad-norm" ]]; then
  ARGS+=(
    --expert-weight-update-every 2000
    --expert-weight-alpha 0.9
    --expert-weight-batch fixed
    --weight-min 1e-3
    --weight-max 1e2
    --expert-weight-min 1e-3
    --expert-weight-max 1e2
    --no-hard-set-first-weight-update
  )
fi

python -m scripts.train_pde_multispecies "${ARGS[@]}"

echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
