#!/bin/bash
#SBATCH --job-name=pinn_state_u_huber_25k
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

RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"

RUN_NAMES=(
  "state_u_squared"
  "state_u_pseudo_huber"
)

PDE_PENALTIES=(
  "squared"
  "pseudo-huber"
)

BC_PENALTIES=(
  "squared"
  "pseudo-huber"
)

RUN_NAME="${RUN_NAMES[$RUN_ID]}"
PDE_PENALTY="${PDE_PENALTIES[$RUN_ID]}"
BC_PENALTY="${BC_PENALTIES[$RUN_ID]}"

N_STEPS="25000"
N_TIME="128"
N_EVAL="60"
LR="1e-3"
LR_SCHEDULER="cosine"
LR_MIN="1e-5"
WEIGHT_MIN="1e-3"
WEIGHT_MAX="1e2"
DATA_CSV="validation/fixtures/pde_multispecies/observations.csv"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
sleep $((RUN_ID * 5))
echo "run_id=${RUN_ID}"
echo "run_name=${RUN_NAME}"
echo "state_parameterization=log-u"
echo "residual_form=scaled"
echo "pde_penalty=${PDE_PENALTY}"
echo "pde_pseudo_huber_delta=1.0"
echo "bc_penalty=${BC_PENALTY}"
echo "bc_pseudo_huber_delta=1.0"
echo "n_steps=${N_STEPS}"
echo "n_time=${N_TIME}"
echo "n_eval=${N_EVAL}"
echo "lr=${LR}"
echo "lr_scheduler=${LR_SCHEDULER}"
echo "lr_min=${LR_MIN}"
echo "loss_weighting=expert-grad-norm"
echo "time_sampling=stratified"
echo "causal_loss=expert"
echo "causal_curriculum=linear"
echo "causal_n_chunks=64"
echo "causal_epsilon=1.0"
echo "weight_min=${WEIGHT_MIN}"
echo "weight_max=${WEIGHT_MAX}"
echo "expert_weight_update_every=2000"
echo "expert_weight_alpha=0.9"
echo "data_csv=${DATA_CSV}"
echo "fourier_num_features=32"
echo "hidden_width=384"
echo "hidden_layers=5"

COMMON_ARGS=(
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

  --loss-weighting expert-grad-norm
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

# Use the base trainer rather than scripts.train_pde_multispecies.
# The scripts wrapper forces HPC history output every 2000 steps, which can
# conceal a short-lived loss spike. The base trainer writes the full history.
python -m PINNmizer.training.train_pde_multispecies "${COMMON_ARGS[@]}"

echo "Finished ${RUN_NAME}"
echo "End time: $(date)"