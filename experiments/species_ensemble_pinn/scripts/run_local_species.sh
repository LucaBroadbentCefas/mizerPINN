#!/usr/bin/env bash
set -euo pipefail
python -m experiments.species_ensemble_pinn.train_species --state-parameterization log-u --residual-form scaled --state-scale-eps 1e-30 "$@"
