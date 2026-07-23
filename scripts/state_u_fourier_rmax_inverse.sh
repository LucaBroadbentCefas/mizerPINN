#!/bin/bash
#SBATCH --job-name=pinn_u_rmax_inverse
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
RUN_NAME="u_fourier_s05_q3_rmax_inverse"
N_STEPS=30000; N_TIME=128; N_EVAL=60; LR=1e-3; RMAX_LR=1e-3; LR_MIN=1e-5; CAUSAL_RAMP_STEPS=20000
BASE_INPUT_DIR="validation/fixtures/pde_multispecies"
INPUT_DIR="validation/fixtures/pde_multispecies_tmax30_${SLURM_JOB_ID:-local}"
DATA_TIME_QUADRATURE_POINTS=3
LOAD_WEIGHTS="/gpfs/home/sfc26usu/mizerPINN/runs/pde_multispecies/REPLACE_WITH_REFERENCE_RUN/model_final.pt"
rm -rf "${INPUT_DIR}"; cp -a "${BASE_INPUT_DIR}" "${INPUT_DIR}"
printf '"value"\n30\n' > "${INPUT_DIR}/t_max.csv"
awk -F',' 'NR == 1 || ($5 + 0) <= 30' "${BASE_INPUT_DIR}/observations.csv" > "${INPUT_DIR}/observations.csv"
DATA_CSV="${INPUT_DIR}/observations.csv"
if [[ ! -f "${LOAD_WEIGHTS}" ]]; then echo "Missing reference checkpoint: ${LOAD_WEIGHTS}" >&2; exit 1; fi
echo "Starting ${RUN_NAME}"; echo "Start time: $(date)"; echo "input_dir=${INPUT_DIR}"; echo "load_weights=${LOAD_WEIGHTS}"
python -m scripts.train_pde_multispecies \
  --input-dir "${INPUT_DIR}" --species-mode all --n-steps "${N_STEPS}" --n-time "${N_TIME}" --n-eval "${N_EVAL}" --lr "${LR}" --load-weights "${LOAD_WEIGHTS}" \
  --state-parameterization log-u --state-scale-eps 1e-30 --residual-form scaled --pde-penalty squared --pde-pseudo-huber-delta 1.0 \
  --boundary-loss-form relative --bc-penalty squared --bc-pseudo-huber-delta 1.0 --lambda-pde 1.0 --lambda-ic 1.0 --lambda-bc 1.0 \
  --initial-w-pde 1.0 --initial-w-ic 1.0 --initial-w-bc 0.1 --data-csv "${DATA_CSV}" --lambda-data 1.0 --initial-w-data 1.0 \
  --data-default-cv 0.3 --data-loss-eps 1e-30 --data-time-quadrature-points "${DATA_TIME_QUADRATURE_POINTS}" --lambda-timestep 0.0 \
  --collocation-strategy uniform --time-sampling stratified --causal-loss expert --causal-curriculum linear --causal-start-fraction 0.05 --causal-ramp-steps "${CAUSAL_RAMP_STEPS}" \
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0" --causal-n-chunks 64 --causal-epsilon 1.0 --loss-weighting expert-grad-norm \
  --expert-weight-update-every 2000 --expert-weight-alpha 0.9 --expert-weight-batch fixed --weight-min 1e-3 --weight-max 1e2 --expert-weight-min 1e-3 --expert-weight-max 1e2 \
  --no-hard-set-first-weight-update --model-arch fourier --fourier-num-features 32 --fourier-scale 0.5 --fourier-include-raw-input --fourier-seed 123 \
  --weight-factorization rwf --rwf-mu 1.0 --rwf-sigma 0.1 --rwf-apply-to all --rwf-base-init xavier_uniform --hidden-width 384 --hidden-layers 5 \
  --lr-scheduler cosine --lr-min "${LR_MIN}" --estimate-rmax --rmax-lr "${RMAX_LR}" --rmax-log-lower 0 --rmax-log-upper 50 \
  --diag-final-n-time 301 --diag-final-n-eval 100 --seed 123 --device cpu --print-every 500 --checkpoint-every 5000
echo "Finished ${RUN_NAME}"; echo "End time: $(date)"
