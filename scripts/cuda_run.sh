#!/bin/bash
#SBATCH --job-name=cuda_pinn_first_run
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH  --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80


set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN

mkdir -p /gpfs/home/sfc26usu/mizerPINN/slurm_logs
module purge

module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"

conda activate mizer-torch

python - <<'PY'
import numpy as np
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

print("device:", torch.cuda.get_device_name(0))
PY


python -m scripts.train_pde_only_single_species \
  --input-dir validation/fixtures/pde_single_species \
  --n-steps 10000 \
  --n-time 36 \
  --n-eval 30 \
  --lr 3e-4 \
  --residual-form log \
  --boundary-loss-form relative \
  --lambda-pde 1.0 \
  --lambda-ic 1.0 \
  --lambda-bc 1.0 \
  --lambda-timestep 0.0 \
  --collocation-strategy uniform \
  --time-sampling stratified \
  --causal-loss expert \
  --causal-n-chunks 32 \
  --causal-epsilon 1.0 \
  --loss-weighting expert-grad-norm \
  --expert-weight-update-every 1000 \
  --expert-weight-alpha 0.7 \
  --expert-weight-batch fixed \
  --weight-min 1e-3 \
  --weight-max 1e3 \
  --model-arch fourier \
  --fourier-num-features 32 \
  --fourier-scale 1 \
  --fourier-seed 123 \
  --weight-factorization rwf \
  --rwf-mu 1.0 \
  --rwf-sigma 0.1 \
  --rwf-apply-to all \
  --rwf-base-init xavier_uniform \
  --hidden-width 384 \
  --hidden-layers 5 \
  --lr-scheduler cosine \
  --lr-min 1e-6 \
  --seed 123 \
  --device cuda \
  --hpc

echo "Finished: ${RUN_NAME}"
echo "End time: $(date)"
```
