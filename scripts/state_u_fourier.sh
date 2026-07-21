#!/bin/bash
#SBATCH --job-name=pinn_state_u_fourier_tests
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-7%5
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
  "u_mlp_q3"
  "u_fourier_s025_q3"
  "u_fourier_s05_q3"
  "u_fourier_s10_q3"
  "u_fourier_s05_q1"
  "u_fourier_s05_q5"
  "u_fourier_s05_q3_nt256"
  "u_fourier_s05_q5_long"
)

MODEL_ARCHS=(
  "mlp"
  "fourier"
  "fourier"
  "fourier"
  "fourier"
  "fourier"
  "fourier"
  "fourier"
)

FOURIER_SCALES=(
  "0.5"
  "0.25"
  "0.5"
  "1.0"
  "0.5"
  "0.5"
  "0.5"
  "0.5"
)

DATA_QUADRATURE_POINTS=(
  "3"
  "3"
  "3"
  "3"
  "1"
  "5"
  "3"
  "5"
)

N_TIMES=(
  "128"
  "128"
  "128"
  "128"
  "128"
  "128"
  "256"
  "128"
)

N_EVALS=(
  "60"
  "60"
  "60"
  "60"
  "60"
  "60"
  "60"
  "60"
)

N_STEPS_VALUES=(
  "30000"
  "30000"
  "30000"
  "30000"
  "30000"
  "30000"
  "30000"
  "60000"
)

RUN_NAME="${RUN_NAMES[$RUN_ID]}"
MODEL_ARCH="${MODEL_ARCHS[$RUN_ID]}"
FOURIER_SCALE="${FOURIER_SCALES[$RUN_ID]}"
DATA_TIME_QUADRATURE_POINTS="${DATA_QUADRATURE_POINTS[$RUN_ID]}"
N_TIME="${N_TIMES[$RUN_ID]}"
N_EVAL="${N_EVALS[$RUN_ID]}"
N_STEPS="${N_STEPS_VALUES[$RUN_ID]}"

PDE_PENALTY="squared"
BC_PENALTY="squared"
LOSS_WEIGHTING="expert-grad-norm"

CAUSAL_RAMP_STEPS="20000"
LR="1e-3"
LR_MIN="1e-5"

BASE_INPUT_DIR="validation/fixtures/pde_multispecies"
INPUT_DIR="validation/fixtures/pde_multispecies_tmax30_${SLURM_ARRAY_JOB_ID:-local}_${RUN_ID}"

rm -rf "${INPUT_DIR}"
cp -a "${BASE_INPUT_DIR}" "${INPUT_DIR}"

printf '"value"\n30\n' > "${INPUT_DIR}/t_max.csv"

awk -F',' \
  'NR == 1 || ($5 + 0) <= 30' \
  "${BASE_INPUT_DIR}/observations.csv" \
  > "${INPUT_DIR}/observations.csv"

DATA_CSV="${INPUT_DIR}/observations.csv"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "run_id=${RUN_ID}"
echo "model_arch=${MODEL_ARCH}"
echo "fourier_scale=${FOURIER_SCALE}"
echo "data_time_quadrature_points=${DATA_TIME_QUADRATURE_POINTS}"
echo "n_time=${N_TIME}"
echo "n_eval=${N_EVAL}"
echo "n_steps=${N_STEPS}"
echo "causal_ramp_steps=${CAUSAL_RAMP_STEPS}"
echo "input_dir=${INPUT_DIR}"

sleep $((RUN_ID * 5))

ARGS=(
  --input-dir "${INPUT_DIR}"
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
  --data-time-quadrature-points "${DATA_TIME_QUADRATURE_POINTS}"

  --lambda-timestep 0.0
  --collocation-strategy uniform

  --time-sampling stratified
  --causal-loss expert
  --causal-curriculum linear
  --causal-start-fraction 0.05
  --causal-ramp-steps "${CAUSAL_RAMP_STEPS}"
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
  --causal-n-chunks 64
  --causal-epsilon 1.0

  --loss-weighting "${LOSS_WEIGHTING}"

  --model-arch "${MODEL_ARCH}"

  --weight-factorization rwf
  --rwf-mu 1.0
  --rwf-sigma 0.1
  --rwf-apply-to all
  --rwf-base-init xavier_uniform

  --hidden-width 384
  --hidden-layers 5

  --lr-scheduler cosine
  --lr-min "${LR_MIN}"

  --diag-final-n-time 301
  --diag-final-n-eval 100

  --seed 123
  --device cpu
  --print-every 500
  --checkpoint-every 5000
)

if [[ "${MODEL_ARCH}" == "fourier" ]]; then
  ARGS+=(
    --fourier-num-features 32
    --fourier-scale "${FOURIER_SCALE}"
    --fourier-include-raw-input
    --fourier-seed 123
  )
fi

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
