# Species ensemble PINN — Tranche 1

This experiment trains one independent scalar direct-`log_N` PINN per species while biological coefficients are computed from a known complete ecosystem state. Existing single-species and multispecies trainers are not imported or modified.

## Fixed method

- Input: `[x_scaled,t_scaled]`; output: `[P,1]` direct `log_N`.
- Fourier features: 16, scale 1, raw inputs included, seed 123.
- Five width-384 tanh hidden layers; RWF on all linear layers.
- PDE residual: `R_ref=(N/S_ref) R_log`, squared penalty only.
- IC: direct log-abundance MSE on active bins.
- BC: relative recruitment-flux residual.
- Dynamic-known or frozen-initial environmental state.
- Linear causal curriculum, exact stratified chunks, expert causal loss and expert gradient-norm weighting.

The residual reference scale is constructed from `n_init`, detached, and used only in PDE residual normalisation and diagnostics. It is not a model-state parameterisation and is never supplied to biology, IC, BC, or recruitment.

## Commands

```bash
python -m experiments.species_ensemble_pinn.train_species \
  --input-dir validation/fixtures/pde_multispecies \
  --known-state-csv /path/to/known_state.csv \
  --species-idx 0 --biology-label detailed
```

```bash
python -m experiments.species_ensemble_pinn.scripts.run_all_species \
  --input-dir validation/fixtures/pde_multispecies \
  --known-state-csv /path/to/known_state.csv \
  --biology-label detailed
```

Core and smoke validation require the parameter fixture and known-state CSV described in `fixtures/README.md`.

## Runtime-default assumption

The specification referenced an attached successful single-species command/SLURM script, but that artifact was not available. Defaults for step count, cosine horizon, checkpoints, and diagnostics are therefore explicit local assumptions, recorded in each `config.json`, rather than claimed copies of a successful run.
