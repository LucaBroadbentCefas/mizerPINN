#!/bin/bash
#SBATCH --job-name=pinn_arch_search
#SBATCH --array=1-12%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --mail-user=luke.broadbent@cefas.gov.uk
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80


set -euo pipefail

cd /gpfs/home/sfc26usu/mizerPINN



module purge

module add mamba/25.3.1-0

eval "$(conda shell.bash hook)"

conda activate mizer-torch

# Match PyTorch CPU threads to the CPU cores requested from SLURM.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Array task index.
ID="${SLURM_ARRAY_TASK_ID}"

WIDTH=(0 256 256 256 256 256 128 384 256 256 256 256 256)
LAYERS=(0 5 5 5 5 5 5 5 3 7 5 5 5)
FEATURES=(0 32 32 32 16 64 32 32 32 32 32 32 32)
SCALE=(0 1.0 0.5 2.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)
RWF=(none rwf rwf rwf rwf rwf rwf rwf rwf rwf none rwf rwf)
RWF_APPLY=(none all all all all all all all all all all hidden all)
RAW_INPUT=(none "" "" "" "" "" "" "" "" "" "" "" "--fourier-include-raw-input")


RUN_NAME="run_${ID}_w${WIDTH[$ID]}_l${LAYERS[$ID]}_f${FEATURES[$ID]}_s${SCALE[$ID]}_${RWF[$ID]}"

echo "Run name: ${RUN_NAME}"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "SLURM task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "Host: $(hostname)"

python -m scripts.train_pde_only_single_species \
  --input-dir validation/fixtures/pde_single_species \
  --n-steps 30000 \
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
  --fourier-num-features "${FEATURES[$ID]}" \
  --fourier-scale "${SCALE[$ID]}" \
  ${RAW_INPUT[$ID]} \
  --fourier-seed 123 \
  --weight-factorization "${RWF[$ID]}" \
  --rwf-mu 1.0 \
  --rwf-sigma 0.1 \
  --rwf-apply-to "${RWF_APPLY[$ID]}" \
  --rwf-base-init xavier_uniform \
  --hidden-width "${WIDTH[$ID]}" \
  --hidden-layers "${LAYERS[$ID]}" \
  --lr-scheduler cosine \
  --lr-min 1e-6 \
  --seed 123 \
  --device cpu \
  --hpc

echo "Finished: ${RUN_NAME}"
echo "End time: $(date)"
```
