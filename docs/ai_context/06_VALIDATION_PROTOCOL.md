# Validation Protocol

## Purpose

This file defines what counts as evidence that the PINNmizer PINN implementation is behaving correctly. Falling training loss alone is not sufficient evidence.

## Validation hierarchy

Use this hierarchy. Do not skip directly to training interpretation if lower-level checks are failing.

```text
1. Shape and dtype/device validation
2. Fixed-grid mizer/TMB operator validation
3. Continuous biological operator validation
4. Manual derivative validation
5. PDE residual validation on known/mizer-generated trajectories
5b. Fixed-grid timestep-consistency validation
5c. R3/Causal R3 paired-collocation validation, when implemented
6. Training-loss and optimisation diagnostics
7. Biological plausibility diagnostics on trained output
```

## Stage 1: shape and dtype/device validation

Goal: confirm all tensors have expected shapes and live on the expected dtype/device before interpreting numerical values.

Checks:

- `validate_params_shapes(params)` passes.
- `n_init` has shape `[n_species, n_w]` or is converted accordingly.
- `n_pp` has shape `[k_full]`.
- Cartesian model output has shape `[n_time * n_x, n_species]`.
- Cartesian reshaped model output has shape `[n_time, n_species, n_x]`.
- Cartesian PDE residual has shape `[n_time, n_species, n_eval]`.
- Paired R3 model input has shape `[n_pair, 2]`, with columns `[x_pair_scaled, t_pair_scaled]`.
- Paired R3 PDE residual has shape `[n_species, n_pair]`.
- Paired R3 grid state has shape `[n_pair, n_species, n_w]` where applicable.
- R3 score has shape `[n_pair]`.
- timestep tensors have expected shapes when timestep consistency is active:
  - `N0_pred`: `[n_pairs, n_species_or_1, n_w]`;
  - `N1_pred`: `[n_pairs, n_species_or_1, n_w]`;
  - `N1_step`: `[n_pairs, n_species_or_1, n_w]`.
- all differentiable path tensors are PyTorch tensors.
- no accidental NumPy arrays enter PDE loss calculation.
- dtype/device are consistent with `params.w`.

Failure interpretation:

- Shape failures should be fixed before any numerical debugging.
- Silent broadcasting is a major risk; prefer explicit asserts.
- A paired R3 shape failure is not a sampling issue only; it may indicate that Cartesian assumptions still exist in derivatives, PDE state, masking, or loss code.

## Stage 2: fixed-grid mizer/TMB operator validation

Goal: ensure the fixed-grid PyTorch path matches the exported mizer/TMB reference behaviour sufficiently well.

Relevant code:

- `PINNmizer/mizer_grid_ops.py`
- R export scripts that generate CSV reference inputs/outputs.

Compare:

- prey construction;
- encounter;
- feeding level;
- energy available for reproduction/growth;
- e_repro;
- e_growth;
- predation rate;
- predation mortality;
- resource mortality;
- total mortality;
- RDI/RDD;
- one-step projection if relevant.

Metrics:

```text
max_abs_error
mean_abs_error
relative_error_with_floor
range(reference - pytorch)
range((reference - pytorch) / pmax(abs(reference), floor))
```

Interpretation:

- Tiny absolute errors with large relative errors near zero are not necessarily important.
- Large errors in encounter, growth, or predation mortality must be resolved before training interpretation.
- Timestep consistency depends on `mizer_grid_ops.step(...)`; if `step(...)` is not validated, timestep-loss training is not interpretable.

## Stage 3: continuous biological operator validation

Goal: validate direct/off-grid biological outputs against fixed-grid reference outputs when evaluated at grid points.

Relevant code:

- `PINNmizer/biology/*`
- fixed-grid reference outputs from `mizer_grid_ops.py` or R/mizer exports.

Compare at `w_eval = params.w`:

- `gamma_eval` versus search/prefactor convention where applicable;
- `h_eval` versus `params.intake_max`;
- `metab_eval` versus `params.metab`;
- `psi_eval` versus `params.psi`;
- `encounter_eval` versus fixed-grid encounter;
- `feeding_eval` versus fixed-grid feeding;
- `e_growth_eval` versus fixed-grid growth;
- `mu_b_eval` versus `params.mu_b`;
- `pred_mort_eval` versus fixed-grid predation mortality;
- `mu_eval` versus fixed-grid total mortality.

Important caution:

The continuous direct path and fixed-grid FFT path are not guaranteed to match exactly because their numerical conventions can differ. Any accepted discrepancy should be documented.

## Stage 4: manual derivative validation

Goal: ensure manually implemented derivatives are correct enough for PDE residual assembly.

Validate with finite differences, not by asserting that training works.

Targets:

- `dgamma_dw`
- `dphi_dw`
- `dencounter_dw`
- `dh_dw`
- `dfeeding_dw`
- `dmetab_dw`
- `dpsi_dw`
- `derepog_dw`
- `dpos_erepog_dw` away from kink at zero
- `de_repro_dw`
- `dg_dw`

Finite-difference cautions:

- Avoid points exactly at piecewise boundaries.
- Avoid `erepog = 0` where the positive-part derivative is undefined.
- Avoid kernel activation/truncation boundaries.
- Use multiple step sizes; a single finite-difference epsilon can mislead.
- Compare absolute and relative errors.

Failure interpretation:

- A derivative mismatch in `dg_dw` invalidates PDE residual interpretation.
- A derivative mismatch near nondifferentiable boundaries may be acceptable if documented and isolated.

## Stage 5: PDE residual validation on known trajectories

Goal: test the PDE residual assembly independently of neural-network training.

The strongest current idea is:

1. Generate a mizer trajectory `N(w,t)` on the fixed log-weight grid.
2. Feed/interpolate this trajectory through the same residual assembly path as if it were model output.
3. Check whether the PDE residual is small relative to discretisation and interpolation error.

This is distinct from training. It tests the residual implementation.

Minimum outputs to inspect:

- `residual_log` RMS, mean absolute, p95, max;
- physical residual consistency check;
- `dlogN_dt` magnitude;
- `g * dlogN_dw` magnitude;
- `mu_eval` magnitude;
- `dg_dw` magnitude;
- sign and scale of `g_eval`;
- sign and scale of `mu_eval`;
- `N_eval` range.

Failure interpretation:

- Large residual on a known mizer trajectory can mean PDE mismatch, derivative scaling error, discretisation mismatch, biological operator mismatch, or time-stepping mismatch.
- Do not treat this as a neural-network problem until the residual implementation itself passes.
- R3 should not be used to work around a failed residual validation; it will simply concentrate points where the possibly-wrong residual is large.

## Stage 5b: fixed-grid timestep-consistency validation

Goal: check whether the fixed-grid one-step temporal loss is numerically meaningful before using it as a training term.

Relevant code:

- `PINNmizer/pinn/timestep_consistency.py`
- `PINNmizer/mizer_grid_ops.py`
- `PINNmizer/training/loop.py`

This validation stage is not the continuous PDE residual. It checks the optional fixed-grid temporal consistency loss:

```text
N0_pred = N_theta(w_grid, t0)
N1_pred = N_theta(w_grid, t0 + dt)
N1_step = step(n_pp, N0_pred, params, dt)
```

Checks:

- `compute_timestep_consistency_loss()` returns a scalar finite loss.
- `N0_pred`, `N1_pred`, and `N1_step` have expected shapes.
- physical/log/relative residual summaries are finite.
- backward pass gives nonzero gradients when `lambda_timestep > 0`.
- compare `detach_step_target=True` and `detach_step_target=False`.
- verify `dt` source: explicit `--timestep-dt` versus `params.dt`.
- verify no selected `t0` has `t0 + dt > t_max`.
- verify at least one valid timestep pair remains after filtering.
- compare `--timestep-loss-form physical`, `log`, and `relative` before deciding which form is usable for training.

Known source-code issue to watch, not patched here:

```python
valid = (t0 + dt_tensor) <= params.t_max
t0 = t0[valid]
```

If all sampled times are too close to `t_max`, `t0` may become empty before calling `compute_timestep_consistency_loss()`. That can lead to empty tensor reductions or NaN timestep loss. A smoke check must include a case where at least one valid pair remains, and a failure-path check should be added before relying on long training runs.

Acceptance criterion:

If timestep loss is active in training, the run must report and inspect `loss_timestep`, `w_timestep`, `weighted_loss_timestep`, `grad_timestep_mean_for_weighting`, `target_w_timestep`, and timestep residual summaries. A falling total loss is not interpretable unless these are inspected.

## Stage 5c: R3/Causal R3 paired-collocation validation

Goal: validate that the R3 implementation retains and resamples true paired collocation points and preserves the PDE residual graph.

Relevant intended code:

- `PINNmizer/pinn/r3.py`
- `PINNmizer/pinn/derivatives.py`
- `PINNmizer/pinn/pde_state.py`
- `PINNmizer/pinn/losses.py`
- `PINNmizer/training/loop.py`

Minimum checks:

- R3 population has shape `[n_pair, 2]` with physical columns `[x, t]`.
- `as_batch()` or equivalent returns `x_pair`, `t_pair`, `w_pair`, `x_pair_scaled`, and `t_pair_scaled` as `[n_pair]` vectors.
- paired derivative function returns `[n_species, n_pair]` tensors.
- paired PDE state returns `N_grid` as `[n_pair, n_species, n_w]`.
- paired residual and active mask both have shape `[n_species, n_pair]`.
- R3 score has shape `[n_pair]`.
- retain/resample mutates the persistent population across training steps.
- retained points are unchanged after an R3 update.
- released points are replaced inside the active physical domain.
- Causal R3 gate values are finite and shaped `[n_pair]`.
- Causal R3 gamma update uses detached PDE loss and does not add trainable model parameters.
- backward pass from paired PDE loss reaches model parameters.

Smoke commands after implementation:

```bash
python -m scripts.train_pde_only_single_species --n-steps 3 --collocation-strategy uniform --causal-curriculum off
python -m scripts.train_pde_only_single_species --n-steps 3 --collocation-strategy r3 --r3-population-size 20 --causal-curriculum off
python -m scripts.train_pde_only_single_species --n-steps 3 --collocation-strategy causal-r3 --r3-population-size 20 --causal-r3-weight-pde-loss --causal-curriculum off
```

Acceptance criterion:

R3/Causal R3 should not be treated as implemented correctly until finite loss, finite nonzero gradients, persistent population mutation, active-mask application, and minimal R3 diagnostics are confirmed.

## Stage 6: training diagnostics

Goal: determine whether optimisation is balancing PDE, IC, BC, timestep, and any R3/Causal R3 collocation effects, and whether the output is physically meaningful.

Current diagnostics exported by the training workflow include:

- training losses;
- PDE/IC/BC/timestep component losses;
- total gradient norm;
- sampled residual summaries;
- fixed-grid losses;
- fixed-grid residual summaries;
- component-wise gradient norms;
- Wang-weighting gradient statistics;
- timestep physical/log/relative residual summaries when active;
- PDE term RMS values;
- boundary flux mismatch metrics;
- predicted `N` range;
- fixed-grid field plots.

Planned R3 diagnostics include:

- `collocation_strategy`;
- `r3_population_size`;
- `r3_retained_fraction`;
- `r3_resampled`;
- `r3_score_mean`;
- `r3_score_max`;
- `causal_r3_gamma`;
- `causal_r3_gamma_update`;
- `causal_r3_gate_mean`.

Minimum comparison matrix:

| Run | Purpose |
|---|---|
| PDE only | Tests whether residual can be optimised without anchors. |
| PDE + IC | Tests temporal propagation from initial state. |
| PDE + IC + BC | Tests whether recruitment boundary stabilises or destabilises. |
| PDE + IC + timestep | Tests whether fixed-grid one-step consistency improves temporal propagation. |
| PDE + IC + BC + timestep | Tests whether all current constraints can be balanced together. |
| Uniform collocation | Baseline Cartesian sampling. |
| R3 collocation | Tests retain/resample focus on high residual paired points. |
| Causal R3 collocation | Tests causal residual focus without truncating the time domain. |
| Wang weights on | Tests adaptive balancing, including timestep if active. |
| Wang weights off | Tests baseline fixed weighting. |
| Causal curriculum on | Tests time-marching/curriculum benefit. |
| Causal curriculum off | Tests whether curriculum is masking failure. |

Timestep-active run checks:

- inspect `loss_timestep`;
- inspect `lambda_timestep`;
- inspect `w_timestep`;
- inspect `weighted_loss_timestep`;
- inspect `grad_timestep_mean_for_weighting`;
- inspect `target_w_timestep`;
- inspect physical/log/relative timestep residual summaries.

R3-active run checks:

- inspect R3 retained fraction and resampled count;
- inspect R3 score mean/max;
- confirm population size is stable;
- compare against uniform collocation using fixed-grid diagnostics, not only training loss;
- for Causal R3, inspect gamma and gate mean;
- do not stack old causal curriculum with Causal R3 unless deliberately testing that interaction.

## Stage 7: biological plausibility diagnostics

Goal: ensure the trained surface is not only numerically low-loss but biologically interpretable.

Inspect:

- `N(w,t)` heatmaps;
- `log_N(w,t)` heatmaps;
- time slices of abundance;
- growth `g(w,t)` range and sign;
- mortality `mu(w,t)` range;
- residual spatial/time localisation;
- boundary flux versus recruitment flux;
- timestep residual localisation if timestep loss is active;
- whether R3 concentrates residuals in biologically meaningful regions or artefactual regions;
- whether abundance collapses to near-zero;
- whether abundance explodes in small or large weights;
- whether dynamics are time-dependent enough to match expected behaviour.

## Acceptance criteria for a new implementation change

A change should not be treated as successful merely because it runs. Minimum acceptance should include:

1. No shape/dtype/device regression.
2. No NaN/Inf in core losses and diagnostics.
3. Expected gradient flow to model parameters.
4. Fixed-grid diagnostics saved successfully.
5. Relevant operator or derivative checks pass if the change touches biology.
6. Timestep smoke checks pass if the change touches `dt`, `step(...)`, timestep loss, or timestep training integration.
7. R3 paired-shape and retain/resample smoke checks pass if the change touches R3/Causal R3.
8. Training comparison against a baseline if the change touches optimisation or collocation strategy.
9. Documentation update in `01_CURRENT_STATE.md`, `02_ARCHITECTURE.md`, `03_FUNCTION_REGISTRY.yaml`, `04_DATA_AND_SHAPES.md`, `05_EQUATIONS.md`, and, if needed, an ADR.

## Current known weak points in validation system

The project needs an explicit, reusable script for PDE residual validation on mizer-generated trajectories. That should become a priority before relying heavily on training results.

The timestep-consistency path also needs explicit smoke coverage for `params.dt`, `--timestep-dt`, valid-pair filtering, residual-form scaling, and detach/no-detach gradient paths before interpreting timestep-active training runs.

R3/Causal R3 also needs explicit smoke coverage after implementation because it changes tensor geometry from Cartesian residuals to paired residuals.
