#!/bin/bash
#SBATCH --job-name=pinn_rmax_recovery
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-2%3
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

RMAX_FACTORS=(
  "0.25"
  "1.0"
  "4.0"
)

RUN_NAMES=(
  "rmax_start_025"
  "rmax_start_true"
  "rmax_start_4"
)

RMAX_FACTOR="${RMAX_FACTORS[$RUN_ID]}"
RUN_NAME="${RUN_NAMES[$RUN_ID]}"

BASE_INPUT_DIR="validation/fixtures/pde_multispecies"
INPUT_DIR="validation/fixtures/rmax_recovery_${SLURM_ARRAY_JOB_ID:-local}_${RUN_ID}"

rm -rf "${INPUT_DIR}"
cp -a "${BASE_INPUT_DIR}" "${INPUT_DIR}"

# Use the same fitting period as the previous runs.
printf '"value"\n30\n' > "${INPUT_DIR}/t_max.csv"

awk -F',' \
  'NR == 1 || ($5 + 0) <= 30' \
  "${BASE_INPUT_DIR}/observations.csv" \
  > "${INPUT_DIR}/observations.csv"

# Keep the true values for checking recovery after fitting.
cp "${BASE_INPUT_DIR}/r_max.csv" "${INPUT_DIR}/r_max_true.csv"

# Change the initial r_max values used by the inverse fit.
python - "${INPUT_DIR}/r_max.csv" "${RMAX_FACTOR}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
factor = float(sys.argv[2])

with path.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.reader(file))

if len(rows) < 2:
    raise ValueError(f"No r_max values found in {path}")

header = rows[0]

if len(header) != 1:
    raise ValueError(
        f"Expected one column in {path}, found {len(header)} columns"
    )

updated = [header]

for row in rows[1:]:
    if not row:
        continue

    value = float(row[0]) * factor

    if value <= 0:
        raise ValueError("All initial r_max values must be positive.")

    updated.append([f"{value:.17g}"])

with path.open("w", newline="", encoding="utf-8") as file:
    csv.writer(file).writerows(updated)
PY

DATA_CSV="${INPUT_DIR}/observations.csv"

echo "Starting ${RUN_NAME}"
echo "Start time: $(date)"
echo "Initial r_max factor: ${RMAX_FACTOR}"
echo "Input directory: ${INPUT_DIR}"

python -m scripts.train_pde_multispecies \
  --input-dir "${INPUT_DIR}" \
  --species-mode all \
  --n-steps 30000 \
  --n-time 128 \
  --n-eval 60 \
  --lr 1e-3 \
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
  --data-csv "${DATA_CSV}" \
  --lambda-data 1.0 \
  --initial-w-data 1.0 \
  --data-default-cv 0.3 \
  --data-loss-eps 1e-30 \
  --data-time-quadrature-points 3 \
  --lambda-timestep 0.0 \
  --collocation-strategy uniform \
  --time-sampling stratified \
  --causal-loss expert \
  --causal-curriculum linear \
  --causal-start-fraction 0.05 \
  --causal-ramp-steps 20000 \
  --causal-step-fractions "0.05,0.10,0.20,0.40,0.70,1.0" \
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
  --lr-scheduler cosine \
  --lr-min 1e-5 \
  --estimate-rmax \
  --rmax-lr 1e-3 \
  --rmax-log-lower 0 \
  --rmax-log-upper 50 \
  --diag-final-n-time 301 \
  --diag-final-n-eval 100 \
  --seed 123 \
  --device cpu \
  --print-every 500 \
  --checkpoint-every 5000

echo "Finished ${RUN_NAME}"
echo "End time: $(date)"
