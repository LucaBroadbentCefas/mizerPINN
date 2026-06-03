# Known Issues

## 1. Trivial, near-zero, or time-invariant solution risk

Status: active

### Description

The PINN can reduce parts of the objective while producing very small abundance values over much of the domain, or a nonzero but nearly time-invariant abundance surface. This is a known failure mode when constraints do not sufficiently identify the desired solution.

### Evidence so far

Earlier logs and discussion showed cases where predicted `N` approached extremely small values while PDE residual terms could remain partly satisfied. The current training script now includes adaptive loss weighting, causal curriculum, final-bias initialisation from IC, optional timestep consistency, R3/Causal R3, and diagnostics partly in response to this risk.

A Causal R3 run with population 300, update every 1, warmup 0, abs score, log residual, `lambda_pde = lambda_ic = lambda_bc = 1`, and `lambda_timestep = 0` still collapsed predicted `N` toward near-zero around approximately 300 iterations while scalar loss decreased.

A later all-feature Causal R3 run avoided the strict zero solution but produced nearly time-invariant outputs. Fixed-grid PDE diagnostics improved early and then worsened. Timestep loss was effectively inactive because lambda and Wang weighting made the objective contribution negligible. IC weight also became very small. BC diagnostics were not yet trustworthy because the BC formulation was still being revised.

### Interpretation

R3/Causal R3 is not, by itself, a fix for the trivial-solution problem. The failure is probably not only poor residual-point coverage. Strong anchoring, meaningful timestep consistency, valid recruitment-boundary handling, residual validation, and correct loss-scale diagnostics remain necessary before interpreting training success.

### Current mitigations

- Initial-condition loss.
- Recruitment-boundary loss implementation, inactive by default unless `lambda_bc` is overridden.
- Optional fixed-grid timestep-consistency loss.
- Wang-style gradient-statistic weighting.
- Causal time curriculum.
- Slab/time R3/Causal R3 collocation to focus on high-residual regions while reusing biological work across time slabs.
- Final-layer bias initialisation from IC.
- Fixed-grid diagnostics and output plots.

### Next checks

- Treat Causal R3 collapse and time-invariance runs as evidence that residual-focused collocation alone is insufficient.
- Validate the PDE residual on known/mizer-generated trajectories before interpreting training failures as optimisation-only.
- Compare uniform collocation against R3/Causal R3 using fixed-grid diagnostics, not only training batch loss.
- Check `N_eval_min`, `N_eval_max`, `log_N_eval_min`, temporal variance, and fixed-grid abundance heatmaps after every run.
- Inspect actual objective contributions using `objective_loss_*`, not only raw component losses.

## 2. Recruitment boundary loss remains scientifically unsettled

Status: active

### Description

The current code implements a recruitment-boundary loss under the continuous flux-boundary assumption:

```text
g(w_min,t) N(w_min,t) = R(t)
```

However, recent inspection showed that mizer-style exported values do not numerically support using the first-bin mizer `N` as if it satisfied this continuous boundary equation. In particular, `g(w_min)` can be extremely small while exported recruitment `R` and first-bin `N` are similar in magnitude, making `R/g` implausibly large as a direct density target.

### Current implementation

`compute_recruitment_boundary_loss_from_state()` extracts:

```text
N_left, log_N_left, g_left, recruitment_flux
```

at `egg_idx = params.w_min_idx - 1`, then excludes invalid samples:

```text
finite values
and g_left > bc_g_min
and recruitment_flux > 0
```

Loss forms currently mean:

```text
log:      log_N_left - log(R / g_left)
physical: N_left - R / g_left
relative: 1 - (N_left * g_left) / R
```

The relative form was changed on 2026-06-02 from a density-relative residual to a dimensionless flux-ratio residual. That is a scale improvement, not a proof that the boundary condition is biologically/numerically correct for the exported mizer data.

### Current risks

- `R/g` can become enormous when `g_left` is small but positive.
- Excluding `g_left <= bc_g_min` may remove most or all BC samples.
- The default `lambda_bc = 0.0`, so the BC implementation may be present but inactive.
- A low relative flux-ratio loss does not imply that mizer first-bin density has been reproduced.
- `gN/R` should not be treated as a mizer numerical validation metric unless the continuous-vs-discrete recruitment convention is explicitly reconciled.

### Required diagnostics

Check these before interpreting any BC-active run:

- `bc_valid_fraction`
- `bc_invalid_g_fraction`
- `bc_invalid_recruitment_fraction`
- `bc_nonfinite_fraction`
- `bc_target_N_min`, `bc_target_N_max`
- `bc_target_log_N_min`, `bc_target_log_N_max`
- `boundary_residual_abs_p95`, `boundary_residual_abs_max`
- `flux_left_min/max` and `recruitment_flux_min/max`

### Next decision needed

Decide whether recruitment should remain a strong pointwise boundary loss, be moved into a weak/integrated boundary formulation, be represented as a near-boundary source term, or be treated as diagnostic-only while other anchors are validated. Do not assume the current pointwise BC is final.

## 3. Timestep loss can dominate, vanish, or go invalid

Status: active

### Description

The timestep-consistency loss compares model-predicted `N(w,t+dt)` with the fixed-grid `step(...)` projection from `N(w,t)`. Depending on residual form, `dt`, abundance scale, lambda, and Wang weights, this term can dominate the total objective or become effectively irrelevant.

This loss is implemented in `PINNmizer/pinn/timestep_consistency.py`. It is an optional fixed-grid temporal regulariser, not part of the continuous/off-grid PDE residual.

Recent all-feature runs showed timestep loss can be effectively inactive even when configured if `lambda_timestep` is small and Wang weighting drives `w_timestep` down.

### Current risks

- physical-form residual can be very large on abundance scale;
- log-form residual depends on clamping;
- relative-form residual can explode where the step target is near zero;
- Wang weighting may amplify or suppress the term based on gradient statistics;
- selected `t0` values near `t_max_current` must satisfy `t0 + dt <= t_max_current`;
- `detach_step_target=False` changes the gradient path substantially;
- if `--timestep-dt` is omitted, behaviour depends on whether `params.dt` was loaded from `dt.csv`.

### Required checks

- inspect `loss_timestep`, `w_timestep`, `lambda_timestep`, and `objective_loss_timestep`;
- inspect `grad_timestep_mean_for_weighting` and `target_w_timestep`;
- check timestep residual physical/log/relative summaries;
- check whether valid timestep pairs remain after filtering;
- compare `--timestep-loss-form physical`, `log`, and `relative`;
- compare `--detach-step-target` against `--no-detach-step-target`;
- verify explicit `--timestep-dt` versus fixture `params.dt`;
- run the timestep smoke check before interpreting timestep-active training.

## 4. PDE residual on mizer trajectory not yet formalised

Status: active

### Description

The project needs a formal script that takes mizer-generated `N(w,t)` trajectories and evaluates the PDE residual without training a neural network.

### Why it matters

If the residual is large on a known mizer trajectory, training failures may come from the residual implementation, derivative scaling, or biological operator mismatch rather than optimisation.

### Required output

- residual summaries over time and size;
- decomposition into `dlogN_dt`, `g*dlogN_dw`, `mu`, and `dg_dw`;
- biological operator comparisons at the same states;
- finite-difference sensitivity to time/weight resolution.

## 5. Continuous kernel convention may not fully match mizer

Status: open

### Description

Earlier discussion included kernel truncation/support conventions. The current source implementation in `compute_phi_and_dphi_dw()` uses:

```text
active = ppmr > 1
```

and does not encode an upper truncation such as a beta/sigma cutoff.

### Risk

The direct continuous encounter and predation mortality path may differ from the mizer reference path if mizer uses additional kernel support logic.

### Next checks

- Compare direct continuous encounter at `w_eval = params.w` against fixed-grid/mizer outputs.
- Compare predation mortality at `w_eval = params.w`.
- Document accepted discrepancy or add an ADR if kernel support changes.

## 6. Wang-style weighting diagnostics need objective-scale interpretation

Status: active design risk

### Description

The weighting code lives in `PINNmizer/training/weighting.py` and can update non-PDE loss weights using PDE gradients as anchor. Current training constructs lambda-scaled losses for the adaptive weighting calculation.

For each non-PDE loss component included in the weighting dictionary, including IC, BC, and timestep, the target weight is effectively based on:

```text
target_weight = max_abs_grad(lambda_pde * loss_pde_anchor) / mean_abs_grad(lambda_component * loss_component)
```

The target is clipped to `[weight_min, weight_max]`, then either hard-set or exponentially smoothed.

### Diagnostics distinction

Use:

```text
objective_loss_component = lambda_component * w_component * loss_component
```

to interpret actual optimizer contribution.

Use:

```text
wang_scaled_loss_component = w_component * loss_component
```

only to inspect adaptive-weight scaling before lambda multiplication.

The older label `weighted_loss_*` is kept as a backward-compatible alias for objective contribution.

### Risk

A diagnostic-only or experimental scalar loss should not be added to the weighting dictionary unless it is intended to participate in optimisation and Wang weighting.

If Causal R3 weights the PDE loss, current training uses `loss_pde_ungated` as the Wang anchor if available, while the optimised PDE term may be gated. Interpret `loss_pde_for_weighting`, `loss_pde_ungated`, `loss_pde_gated`, and `pde_gate_mean` together.

## 7. Training script is single-species only

Status: intentional current limitation

### Description

The lower-level code is mostly species-general, but the current training script raises an error unless `n_species == 1`.

### Risk

Do not assume multi-species training works simply because tensor shapes allow species dimensions.

### Next checks before multi-species extension

- Validate tensor shapes for multi-species outputs.
- Revisit loss aggregation across species.
- Revisit boundary condition per species.
- Revisit timestep-consistency species slicing and aggregation.
- Revisit R3 score aggregation across species.
- Revisit diagnostics and plotting.
- Test gradient weighting across species and loss components.

## 8. R3/Causal R3 can focus training on artefacts if the residual is wrong

Status: active feature risk

### Description

R3 retains high-residual collocation regions and resamples low-residual regions. This can improve coverage of difficult regions, but it also means the method will concentrate training effort wherever the current residual is large.

Current implementation is slab/time R3:

```text
t_points: [K]
x_points: [K, M]
residual: [K, n_species, M]
```

At each R3/Causal R3 step, time slabs are resampled under `t_max_current`; R3 retain/resample applies only to `x_points`.

### Risk

If the residual implementation is wrong, the biological operator is mismatched, or active-size masking is incomplete, R3 will concentrate points on artefactual errors rather than fixing the underlying model. R3 is therefore not a replacement for residual validation.

Do not keep recommending fixes for the old flat-paired performance bottleneck unless source inspection shows the slab/time path has regressed or is not being used.

The old advice to forbid `R3/Causal R3 + causal_curriculum` is obsolete for current source: the ordinary causal curriculum now bounds R3 time resampling through `t_max_current`. Causal time truncation and Causal R3 gating still need separate interpretation.

### Required checks

- Validate PDE residual on known/mizer-generated trajectories before treating R3 results as meaningful.
- Compare R3 against uniform collocation using fixed-grid diagnostics, not only training batch loss.
- Inspect where retained `x_points` concentrate if results look pathological.
- Check `r3_n_time`, `r3_n_eval_per_time`, `r3_population_size`, and `r3_biology_time_loops` to understand actual work per step.
- Keep active `w_max` masking in both sampling and loss/scoring.
