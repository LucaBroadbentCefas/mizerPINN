#!/usr/bin/env bash
set -euo pipefail
python -m experiments.species_ensemble_pinn.assemble_species "$@"
