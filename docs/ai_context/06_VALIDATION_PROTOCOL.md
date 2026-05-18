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
6. Training-loss and optimisation diagnostics
7. Biological plausibility diagnostics on trained output
```

## Stage 1: shape and dtype/device validation

Goal: confirm all tensors have expected shapes and live on the expected dtype/device before interpreting numerical values.

Checks:

- `validate_params_shapes(params)` passes.
- `n_init` has shape `[n_species, n_w]` or is converted accordingly.
- `n_pp` has shape `[k_full]`.
- model output has shape `[n_time * n_x, n_species]`.
- reshaped model output has shape `[n_time, n_species, n_x]`.
- PDE residual has shape `[n_time, n_species, n_eval]`.
- all differentiable path tensors are PyTorch tensors.
- no accidental NumPy arrays enter PDE loss calculation.
- dtype/device are consistent with `params.w`.

Failure interpretation:

- Shape failures should be fixed before any numerical debugging.
- Silent broadcasting is a major risk; prefer explicit asserts.

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

## Stage 3: continuous biological operator validation

Goal: validate direct/off-grid biological outputs against fixed-grid reference outputs when evaluated at grid points.

Relevant code:

- `PINNmizer/continuous_biology.py`
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

## Stage 6: training diagnostics

Goal: determine whether optimisation is balancing PDE, IC, and BC constraints and whether the output is physically meaningful.

Current diagnostics already exported by the training workflow include:

- training losses;
- PDE/IC/BC component losses;
- total gradient norm;
- sampled residual summaries;
- fixed-grid losses;
- fixed-grid residual summaries;
- component-wise gradient norms;
- PDE term RMS values;
- boundary flux mismatch metrics;
- predicted `N` range;
- fixed-grid field plots.

Minimum comparison matrix:

| Run | Purpose |
|---|---|
| PDE only | Tests whether residual can be optimised without anchors. |
| PDE + IC | Tests temporal propagation from initial state. |
| PDE + IC + BC | Tests whether recruitment boundary stabilises or destabilises. |
| Wang weights on | Tests adaptive balancing. |
| Wang weights off | Tests baseline fixed weighting. |
| Causal curriculum on | Tests time-marching/curriculum benefit. |
| Causal curriculum off | Tests whether curriculum is masking failure. |

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
6. Training comparison against a baseline if the change touches optimisation.
7. Documentation update in `01_CURRENT_STATE.md` and, if needed, an ADR.

## Current known weak point in validation system

The project needs an explicit, reusable script for PDE residual validation on mizer-generated trajectories. That should become a priority before relying heavily on training results.
