#!/usr/bin/env bash
#SBATCH --array=0-0
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
set -euo pipefail
SPECIES_INDEX="${SLURM_ARRAY_TASK_ID}"
python -m experiments.species_ensemble_pinn.train_species \
  --input-dir "${INPUT_DIR}" --known-state-csv "${KNOWN_STATE_CSV}" --experiment-name "${EXPERIMENT_NAME}" \
  --environment-state "${ENVIRONMENT_STATE:-dynamic-known}" --biology-label "${BIOLOGY_LABEL:-detailed}" \
  --known-state-interpolation "${KNOWN_STATE_INTERPOLATION:-linear}" --species-index "${SPECIES_INDEX}" \
  --state-parameterization log-u --residual-form scaled --state-scale-eps 1e-30 --hpc "$@"
