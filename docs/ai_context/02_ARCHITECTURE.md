# Architecture

## Top-level project split

```text
PINNmizer/
  params.py
  io.py
  mizer_grid_ops.py
  biology/{kernels,encounter,growth,mortality,recruitment}.py
  pde_residual.py
  utils.py

validation/scripts/
  train_pde_only_single_species.py
  PINNmizer/diagnostics/{fixed_grid,metrics,outputs,plots,fields}.py
  pde_output_diagnostics.py
  other validation/export/debug scripts

docs/ai_context/
  project continuity documentation for ChatGPT and future contributors
```

## Architectural principle

The project has two biological-computation paths:

1. **Fixed-grid mizer/TMB-style path**
   - implemented mainly in `PINNmizer/mizer_grid_ops.py`;
   - uses FFT kernels and fixed grid arrays;
   - useful for validation and comparison against mizer/TMB conventions.

2. **Continuous/off-grid PDE path**
   - implemented mainly in `PINNmizer/biology/{kernels,encounter,growth,mortality,recruitment}.py` and `PINNmizer/pinn/* modules`;
   - evaluates biology at arbitrary `w_eval = exp(x_eval)` collocation points;
   - used by the PINN PDE residual and loss.

These paths should not be casually merged. Differences between them are scientifically and numerically meaningful.

## Module map

### `PINNmizer/params.py`

Purpose:

- Defines `MizerTorchParams`.
- Holds tensors for grids, FFT validation quantities, continuous biological parameters, interaction matrices, reproduction parameters, mortality parameters, and physical time bounds.
- Provides helpers for dtype/device consistency, species-vector extraction, scaling, and dimension counts.

Important functions/classes:

- `MizerTorchParams`
- `fish_start()`
- `_params_dtype_device()`
- `_to_param_tensor()`
- `_species_vector()`
- `_eval_weight_vector()`
- `_x_grid()`
- `_x_limits()`
- `_t_limits()`
- `scale_x()`
- `scale_t()`
- `_n_species()`
- `_n_w()`
- `_k_full()`
- `validate_params_shapes()`

Boundary:

- Should not implement biological operators.
- Should not run training.
- Should remain the central place for tensor shape and coordinate helper logic.

### `PINNmizer/io.py`

Purpose:

- Loads CSV exports from R/mizer/TMB into PyTorch tensors.
- Constructs `MizerTorchParams`.
- Returns `(params, n_init, n_pp)`.

Important functions:

- `load_mat()`
- `load_vec()`
- `load_complex_mat()`
- `maybe_vec()`
- `maybe_mat()`
- `load_mizer_inputs()`

Boundary:

- Should not perform model training.
- Should not transform biological equations.
- Should preserve dtype/device supplied by the caller.

### `PINNmizer/mizer_grid_ops.py`

Purpose:

- Implements fixed-grid, mizer/TMB-style operators.
- Contains FFT-based encounter and predation operations.
- Provides an AD-safe projection-style step.

Important functions:

- `fft_convolve_rows()`
- `compute_prey()`
- `get_encounter()`
- `feeding_level()`
- `e_repro_and_growth()`
- `e_repro()`
- `e_growth()`
- `compute_q_matrix()`
- `get_pred_rate()`
- `pred_mortality()`
- `resource_mortality()`
- `total_mortality()`
- `rdi()`
- `rdd()`
- `resource_semichemostat()`
- `mizer_operators()`
- `step()`

Boundary:

- This is the fixed-grid validation/reference path.
- Do not use it as the arbitrary off-grid PDE residual path unless explicitly testing that design.

### `PINNmizer/biology/{kernels,encounter,growth,mortality,recruitment}.py`

Purpose:

- Computes biological quantities at arbitrary continuous physical weights.
- Provides analytical/manual growth-side derivatives for PDE residual assembly.
- Computes direct continuous predation mortality and recruitment flux.

Important functions:

- `compute_encounter_direct_at_eval()`
- `evaluate_gamma_continuous()`
- `evaluate_intake_max_continuous()`
- `evaluate_metab_continuous()`
- `evaluate_psi_continuous()`
- `compute_phi_and_dphi_dw()`
- `evaluate_mu_b_continuous()`
- `compute_growth_direct_at_eval()`
- `compute_pred_mortality_direct_at_eval()`
- `compute_total_mortality_direct_at_eval()`
- `compute_pred_mortality_direct_at_eval_from_growth_grid()`
- `compute_total_mortality_direct_at_eval_from_growth_grid()`
- `compute_recruitment_direct_from_growth_grid()`

Boundary:

- Do not introduce NumPy.
- Preserve dtype/device.
- Do not detach inside computations used by the PDE loss unless deliberately producing diagnostics.
- Do not replace analytical/manual `dg_dw` with autograd without an explicit decision record.

### `PINNmizer/pinn/* modules`

Purpose:

- Samples collocation batches.
- Evaluates model outputs on collocation and fixed grids.
- Computes model derivatives with autograd.
- Builds a cached PDE state.
- Assembles PDE residuals, initial-condition loss, and recruitment-boundary loss.

Important functions:

- `sample_pde_batch()`
- `_make_model_inputs()`
- `evaluate_log_model_on_points()`
- `evaluate_log_model_with_derivatives_at_eval()`
- `compute_pde_state()`
- `compute_pde_residual_from_state()`
- `compute_pde_residual()`
- `compute_initial_condition_loss_from_state()`
- `compute_recruitment_boundary_loss_from_state()`
- `compute_pde_loss()`

Boundary:

- This module should not contain the full training loop.
- It can assemble loss components, but optimiser logic belongs elsewhere.
- It should remain the central source for residual shape and derivative-scaling conventions.

### `validation/scripts/train_pde_only_single_species.py`

Purpose:

- Current executable single-species training script.
- Defines the MLP.
- Runs sampling, loss computation, Wang-style weighting, optimisation, diagnostics, checkpoints, and final output exports.

Important functions/classes:

- `MLP`
- `make_run_dir()`
- `total_grad_norm_and_check()`
- `causal_time_fraction()`
- `causal_t_max_current()`
- `initialise_final_bias_from_ic()`
- `train_one_step()`
- `_flat_loss_grad()`
- `update_wang_gradient_weights_()`
- `save_checkpoint()`
- `save_final_predictions()`
- `save_final_residual_sample()`
- `parse_args()`
- `main()`

Boundary:

- This script is allowed to be operational and pragmatic.
- If loss weighting grows more complicated, it should be moved into a dedicated module.
- Do not hide major scientific assumptions in this script; document them in `docs/ai_context/`.

### `validation/scripts/PINNmizer/diagnostics/{fixed_grid,metrics,outputs,plots,fields}.py`

Purpose:

- Deterministic fixed-grid diagnostics.
- Component loss/gradient diagnostics.
- Diagnostic CSV and PNG exports.
- Final field exports/plots.

Important functions:

- `make_fixed_pde_batch()`
- `make_fixed_pde_batch_from_csv()`
- `compute_fixed_diagnostics()`
- `append_diagnostic_row()`
- `save_training_diagnostic_plots()`
- `save_latest_metrics_table()`
- `save_fixed_grid_fields_and_plots()`

Boundary:

- Diagnostics may use plotting, pandas, and NumPy.
- Differentiable training/PDE loss code should not inherit these dependencies unless necessary.

## Data flow

```text
R/mizer/TMB exports CSVs
        |
        v
PINNmizer.io.load_mizer_inputs()
        |
        v
MizerTorchParams + n_init + n_pp
        |
        +--> fixed-grid validation path: mizer_grid_ops.py
        |
        +--> continuous PDE path:
                pde_residual.sample_pde_batch()
                model([x_scaled, t_scaled]) -> log_N
                autograd derivatives wrt scaled coordinates
                convert derivatives to physical t and w
                continuous_biology computes g, dg_dw, mu, recruitment flux
                pde_residual assembles losses
                training script applies weighting and optimiser step
```

## Where new code should go

- New biological equation or derivative: `PINNmizer/biology/{kernels,encounter,growth,mortality,recruitment}.py`, plus ADR if it changes conventions.
- Fixed-grid mizer reference operation: `PINNmizer/mizer_grid_ops.py`.
- PDE residual assembly or loss component: `PINNmizer/pinn/* modules`.
- General training-loss weighting: a future `PINNmizer/loss_weighting.py` or current training script until split.
- Training script experiment only: `validation/scripts/train_pde_only_single_species.py`.
- Diagnostics/plots: `validation/scripts/PINNmizer/diagnostics/{fixed_grid,metrics,outputs,plots,fields}.py` or related diagnostic module.
- Cross-chat project state: `docs/ai_context/`.

## Current architectural debt

- Wang-style loss weighting currently lives inside the training script rather than a reusable module.
- The training script has become a combined experiment runner, diagnostics coordinator, and loss-weighting implementation.
- `pde_residual.py` contains both residual assembly and IC/BC loss assembly; acceptable for now, but could be split if more losses are added.
- Historical experiment results are not yet systematically encoded in `08_EXPERIMENT_LOG.jsonl`.
