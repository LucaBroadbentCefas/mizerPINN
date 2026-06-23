#!/bin/bash
#SBATCH --job-name=pinn_wave1_causal_lr
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-27%5
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch


SCHEME_ID="${SLURM_ARRAY_TASK_ID}"

TIME_SAMPLING="stratified"
CAUSAL_LOSS="expert"
CAUSAL_CURRICULUM="linear"
CAUSAL_N_CHUNKS="32"
CAUSAL_EPSILON="1.0"
N_TIME="36"
N_EVAL="30"
LR="3e-4"
LR="3e-4"
LR_SCHEDULER="cosine"
LR_MIN="1e-6"
SCHEME_NAME="baseline"

case "${SCHEME_ID}" in
  0)
    SCHEME_NAME="baseline"
    ;;

  1)
    SCHEME_NAME="no_causal_control"
    TIME_SAMPLING="uniform"
    CAUSAL_LOSS="off"
    CAUSAL_CURRICULUM="off"
    ;;

  2)
    SCHEME_NAME="linear_curriculum_only"
    TIME_SAMPLING="uniform"
    CAUSAL_LOSS="off"
    CAUSAL_CURRICULUM="linear"
    ;;

  3)
    SCHEME_NAME="step_curriculum_only"
    TIME_SAMPLING="uniform"
    CAUSAL_LOSS="off"
    CAUSAL_CURRICULUM="step"
    ;;

  4)
    SCHEME_NAME="expert_causal_only"
    TIME_SAMPLING="stratified"
    CAUSAL_LOSS="expert"
    CAUSAL_CURRICULUM="off"
    ;;

  5)
    SCHEME_NAME="eps_0p01"
    CAUSAL_EPSILON="0.01"
    ;;

  6)
    SCHEME_NAME="eps_0p05"
    CAUSAL_EPSILON="0.05"
    ;;

  7)
    SCHEME_NAME="eps_0p1"
    CAUSAL_EPSILON="0.1"
    ;;

  8)
    SCHEME_NAME="eps_0p5"
    CAUSAL_EPSILON="0.5"
    ;;

  9)
    SCHEME_NAME="eps_2"
    CAUSAL_EPSILON="2.0"
    ;;

  10)
    SCHEME_NAME="eps_5"
    CAUSAL_EPSILON="5.0"
    ;;

  11)
    SCHEME_NAME="eps_10"
    CAUSAL_EPSILON="10.0"
    ;;

  12)
    SCHEME_NAME="chunks_4"
    CAUSAL_N_CHUNKS="4"
    ;;

  13)
    SCHEME_NAME="chunks_8"
    CAUSAL_N_CHUNKS="8"
    ;;

  14)
    SCHEME_NAME="chunks_16"
    CAUSAL_N_CHUNKS="16"
    ;;

  15)
    SCHEME_NAME="chunks_64"
    CAUSAL_N_CHUNKS="64"
    ;;

  16)
    SCHEME_NAME="chunks_128"
    CAUSAL_N_CHUNKS="128"
    ;;

  17)
    SCHEME_NAME="lr_3e-5"
    LR="3e-5"
    ;;

  18)
    SCHEME_NAME="lr_1e-4"
    LR="1e-4"
    ;;

  19)
    SCHEME_NAME="lr_1e-3"
    LR="1e-3"
    ;;

  20)
    SCHEME_NAME="lr_3e-3"
    LR="3e-3"
    ;;

  21)
    SCHEME_NAME="ntime_32_neval_30"
    N_TIME="32"
    N_EVAL="30"
    ;;

  22)
    SCHEME_NAME="ntime_96_neval_30"
    N_TIME="96"
    N_EVAL="30"
    ;;

  23)
    SCHEME_NAME="ntime_128_neval_30"
    N_TIME="128"
    N_EVAL="30"
    ;;

  24)
    SCHEME_NAME="ntime_36_neval_15"
    N_TIME="36"
    N_EVAL="15"
    ;;

  25)
    SCHEME_NAME="ntime_36_neval_60"
    N_TIME="36"
    N_EVAL="60"
    ;;

  26)
    SCHEME_NAME="ntime_96_neval_60"
    N_TIME="96"
    N_EVAL="60"
    ;;

  27)
    SCHEME_NAME="ntime_128_neval_60"
    N_TIME="128"
    N_EVAL="60"
    ;;

  *)
    echo "Unknown SCHEME_ID: ${SCHEME_ID}"
    exit 2
    ;;
esac

RUN_NAME="wave1_${SCHEME_ID}_${SCHEME_NAME}"

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

python -m scripts.train_pde_only_single_species \
  --input-dir validation/fixtures/pde_single_species \
  --n-steps 15000 \
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
  --fourier-num-features 16 \
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
