#!/bin/bash
#SBATCH --job-name=pinn_cv030_resume
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN
mkdir -p slurm_logs

module purge
module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"
conda activate mizer-torch

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Point this to the checkpoint from iteration 30,000.
MODEL_WEIGHTS="/gpfs/home/sfc26usu/mizerPINN/runs/pde_multispecies/20260730_163053_999056_job3770192_task3/model_step_30000.pt"

if [[ ! -f "${MODEL_WEIGHTS}" ]]; then
    echo "Checkpoint not found: ${MODEL_WEIGHTS}" >&2
    exit 1
fi

python -m scripts.train_pde_multispecies \
  --input-dir validation/fixtures/pde_multispecies \
  --species-mode all \
  --n-steps 35000 \
  --start-step 30000 \
  --load-weights "${MODEL_WEIGHTS}" \
  --load-optimizer-state \
  --n-time 128 \
  --n-eval 60 \
  --lr 1e-5 \
  --lr-scheduler none \
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
  --lambda-data 0.0 \
  --lambda-timestep 0.0 \
  --collocation-strategy uniform \
  --time-sampling stratified \
  --causal-loss expert \
  --causal-curriculum linear \
  --causal-start-fraction 0.05 \
  --causal-ramp-steps 20000 \
  --causal-step-fractions 0.05,0.10,0.20,0.40,0.70,1.0 \
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
  --diag-final-n-time 301 \
  --diag-final-n-eval 100 \
  --seed 123 \
  --device cpu \
  --print-every 500 \
  --checkpoint-every 5000 \
  --hpc
