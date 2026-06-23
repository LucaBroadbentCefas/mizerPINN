#!/bin/bash
#SBATCH --job-name=pinn_wave1_single_ff32
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=15:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

TIME_SAMPLING="stratified"
CAUSAL_LOSS="expert"
CAUSAL_CURRICULUM="linear"
CAUSAL_N_CHUNKS="64"
CAUSAL_EPSILON="10.0"
N_TIME="128"
N_EVAL="60"
LR="3e-4"
LR_SCHEDULER="cosine"
LR_MIN="1e-6"
SCHEME_NAME="chunks64_eps10_ntime128_neval60_ff32"
RUN_NAME="wave1_single_${SCHEME_NAME}"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "time_sampling=${TIME_SAMPLING}"
echo "causal_loss=${CAUSAL_LOSS}"
echo "causal_curriculum=${CAUSAL_CURRICULUM}"
echo "causal_n_chunks=${CAUSAL_N_CHUNKS}"
echo "causal_epsilon=${CAUSAL_EPSILON}"
echo "n_time=${N_TIME}"
echo "n_eval=${N_EVAL}"
echo "lr=${LR}"
echo "lr_scheduler=${LR_SCHEDULER}"
echo "fourier_num_features=32"

python -m scripts.train_pde_only_single_species \
  --input-dir validation/fixtures/pde_single_species \
  --n-steps 150000 \
  --n-time "${N_TIME}" \
  --n-eval "${N_EVAL}" \
  --lr "${LR}" \
  --residual-form log \
  --boundary-loss-form relative \
  --lambda-pde 1.0 \
  --lambda-ic 1.0 \
  --lambda-bc 1.0 \
  --lambda-timestep 0.0 \
  --collocation-strategy uniform \
  --time-sampling "${TIME_SAMPLING}" \
  --causal-loss "${CAUSAL_LOSS}" \
  --causal-n-chunks "${CAUSAL_N_CHUNKS}" \
  --causal-epsilon "${CAUSAL_EPSILON}" \
  --causal-curriculum "${CAUSAL_CURRICULUM}" \
  --causal-start-fraction 0.05 \
  --causal-ramp-steps 1500 \
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0" \
  --loss-weighting expert-grad-norm \
  --expert-weight-update-every 1000 \
  --expert-weight-alpha 0.7 \
  --expert-weight-batch fixed \
  --weight-min 1e-3 \
  --weight-max 1e3 \
  --model-arch fourier \
  --fourier-num-features 32 \
  --fourier-scale 1 \
  --fourier-include-raw-input \
  --fourier-seed 123 \
  --weight-factorization rwf \
  --rwf-mu 1.0 \
  --rwf-sigma 0.1 \
  --rwf-apply-to all \
  --rwf-base-init xavier_uniform \
  --hidden-width 384 \
  --hidden-layers 5 \
  --lr-scheduler "${LR_SCHEDULER}" \
  --lr-min "${LR_MIN}" \
  --seed 123 \
  --device cpu \
  --hpc

echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
