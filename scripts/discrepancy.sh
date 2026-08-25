#!/bin/bash
#SBATCH --job-name=pinn_discrepancy_test
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-3%4
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

# -------------------------------------------------------------------------
# Four discrepancy-loss experiments
#
# 0 = A: existing data loss, no discrepancy gate
# 1 = B: discrepancy gate, lambda_data = 1
# 2 = C: discrepancy gate, lambda_data = 0.3
# 3 = D: discrepancy gate, lambda_data = 3
# -------------------------------------------------------------------------

RUN_NAMES=(
  "control_no_gate_lambda1"
  "discrepancy_lambda1"
  "discrepancy_lambda0p3"
  "discrepancy_lambda3"
)

LAMBDA_DATA_VALUES=(
  "1.0"
  "1.0"
  "0.3"
  "3.0"
)

USE_DISCREPANCY_GATE=(
  "0"
  "1"
  "1"
  "1"
)

RUN_NAME="${RUN_NAMES[$RUN_ID]}"
LAMBDA_DATA="${LAMBDA_DATA_VALUES[$RUN_ID]}"
DISCREPANCY_GATE="${USE_DISCREPANCY_GATE[$RUN_ID]}"

# -------------------------------------------------------------------------
# Common configuration
# -------------------------------------------------------------------------

N_STEPS="10000"
N_TIME="128"
N_EVAL="60"

LR="1e-3"
LR_MIN="1e-5"

INPUT_DIR="validation/fixtures/pde_multispecies"
DATA_CSV="${INPUT_DIR}/observations.csv"

echo "============================================================"
echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "git_commit=$(git rev-parse HEAD)"
echo "run_id=${RUN_ID}"
echo "lambda_data=${LAMBDA_DATA}"
echo "discrepancy_gate=${DISCREPANCY_GATE}"
echo "n_steps=${N_STEPS}"
echo "============================================================"

# Prevent simultaneous runs receiving effectively identical timestamped
# run-directory names.
sleep $((RUN_ID * 5))

# -------------------------------------------------------------------------
# Common training arguments
# -------------------------------------------------------------------------

ARGS=(
  --input-dir "${INPUT_DIR}"
  --species-mode all

  --n-steps "${N_STEPS}"
  --n-time "${N_TIME}"
  --n-eval "${N_EVAL}"

  --lr "${LR}"
  --lr-scheduler cosine
  --lr-min "${LR_MIN}"

  --state-parameterization log-u
  --state-scale-eps 1e-30
  --residual-form scaled

  --pde-penalty squared
  --pde-pseudo-huber-delta 1.0

  --boundary-loss-form log
  --bc-penalty squared
  --bc-pseudo-huber-delta 1.0
  --bc-g-min 1e-12

  --lambda-pde 1.0
  --lambda-ic 1.0
  --lambda-bc 0.1

  --initial-w-pde 1.0
  --initial-w-ic 1.0
  --initial-w-bc 1.0
  --initial-w-data 1.0

  --data-csv "${DATA_CSV}"
  --lambda-data "${LAMBDA_DATA}"
  --data-default-cv 0.3
  --data-loss-eps 1e-30
  --data-time-quadrature-points 3

  --lambda-timestep 0.0

  --collocation-strategy uniform
  --time-sampling stratified

  --causal-loss expert
  --causal-curriculum linear
  --causal-start-fraction 0.05
  --causal-ramp-steps 5000
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
  --causal-n-chunks 64
  --causal-epsilon 1.0

  --loss-weighting none

  --model-arch fourier
  --fourier-num-features 32
  --fourier-scale 0.5
  --fourier-include-raw-input
  --fourier-seed 123

  --weight-factorization rwf
  --rwf-mu 1.0
  --rwf-sigma 0.1
  --rwf-apply-to all
  --rwf-base-init xavier_uniform

  --hidden-width 384
  --hidden-layers 5

  --diag-final-n-time 301
  --diag-final-n-eval 100

  --seed 123
  --device cpu

  --print-every 250
  --checkpoint-every 5000
)

# Only runs B-D get the new discrepancy gate.
if [[ "${DISCREPANCY_GATE}" == "1" ]]; then
  ARGS+=(--data-discrepancy-gate)
fi

python -m scripts.train_pde_multispecies "${ARGS[@]}"

echo "============================================================"
echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
echo "============================================================"
