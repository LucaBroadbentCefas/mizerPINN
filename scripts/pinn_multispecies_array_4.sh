#!/bin/bash
#SBATCH --job-name=pinn_multispecies
#SBATCH --cpus-per-task=16
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

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

SCHEME_ID="${SLURM_ARRAY_TASK_ID:-0}"

SCHEME_NAMES=(
  "expert_64_nohard_cap100"
  "no_expert_weighting_64"
  "expert_32_nohard_cap100"
  "no_causal_chunks_nohard_cap100"
  "causal_epsilon_decreased"
)

LOSS_WEIGHTINGS=(
  "expert-grad-norm"
  "none"
  "expert-grad-norm"
  "expert-grad-norm"
  "expert-grad-norm"
)

TIME_SAMPLINGS=(
  "stratified"
  "stratified"
  "stratified"
  "stratified"
  "stratified"
)

CAUSAL_LOSSES=(
  "expert"
  "expert"
  "expert"
  "off"
  "expert"
)

CAUSAL_CURRICULA=(
  "linear"
  "linear"
  "linear"
  "off"
  "linear"
)

CAUSAL_N_CHUNKS=(
  "64"
  "64"
  "32"
  "64"
  "64"
)

CAUSAL_EPSILONS=(
  "10.0"
  "10.0"
  "10.0"
  "10.0"
  "1"
)

SCHEME_NAME="${SCHEME_NAMES[$SCHEME_ID]}"
LOSS_WEIGHTING="${LOSS_WEIGHTINGS[$SCHEME_ID]}"
TIME_SAMPLING="${TIME_SAMPLINGS[$SCHEME_ID]}"
CAUSAL_LOSS="${CAUSAL_LOSSES[$SCHEME_ID]}"
CAUSAL_CURRICULUM="${CAUSAL_CURRICULA[$SCHEME_ID]}"
CAUSAL_N_CHUNK="${CAUSAL_N_CHUNKS[$SCHEME_ID]}"
CAUSAL_EPSILON="${CAUSAL_EPSILONS[$SCHEME_ID]}"

N_TIME="128"
N_EVAL="60"
LR="1e-3"
LR_SCHEDULER="cosine"
LR_MIN="1e-5"
WEIGHT_MIN="1e-3"
WEIGHT_MAX="1e2"
RUN_NAME="${SCHEME_NAME}"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "scheme_id=${SCHEME_ID}"
echo "scheme_name=${SCHEME_NAME}"
echo "loss_weighting=${LOSS_WEIGHTING}"
echo "time_sampling=${TIME_SAMPLING}"
echo "causal_loss=${CAUSAL_LOSS}"
echo "causal_curriculum=${CAUSAL_CURRICULUM}"
echo "causal_n_chunks=${CAUSAL_N_CHUNK}"
echo "causal_epsilon=${CAUSAL_EPSILON}"
echo "weight_min=${WEIGHT_MIN}"
echo "weight_max=${WEIGHT_MAX}"
echo "hard_set_first_weight_update=false"
echo "n_time=${N_TIME}"
echo "n_eval=${N_EVAL}"
echo "lr=${LR}"
echo "lr_scheduler=${LR_SCHEDULER}"
echo "fourier_num_features=32"

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
  --lambda-timestep 0.0
  --collocation-strategy uniform
  --time-sampling "${TIME_SAMPLING}"
  --causal-loss "${CAUSAL_LOSS}"
  --causal-curriculum "${CAUSAL_CURRICULUM}"
  --causal-start-fraction 0.05
  --causal-ramp-steps 40000
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0"
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
  --hpc
)

if [[ "${CAUSAL_LOSS}" == "expert" ]]; then
  COMMON_ARGS+=(
    --causal-n-chunks "${CAUSAL_N_CHUNK}"
    --causal-epsilon "${CAUSAL_EPSILON}"
  )
fi

python -m scripts.train_pde_multispecies "${COMMON_ARGS[@]}"

echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
