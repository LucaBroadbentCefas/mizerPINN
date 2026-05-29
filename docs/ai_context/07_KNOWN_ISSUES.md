# Known Issues

## 1. Trivial or near-zero solution risk

Status: active

### Description

The PINN can reduce parts of the objective while producing very small abundance values over much of the domain. This is a known failure mode in PDE-constrained neural networks when constraints do not sufficiently identify the desired solution.

### Evidence so far

Earlier logs and discussion showed cases where predicted `N` approached extremely small values while PDE residual terms could remain partly satisfied. The current training script now includes adaptive loss weighting, causal curriculum, final-bias initialisation from IC, optional timestep consistency, and diagnostics partly in response to this risk.

### Plausible causes

- PDE residual alone does not identify a unique biologically correct solution.
- Initial, boundary, and timestep constraints may be too weak, badly scaled, inactive, or underweighted.
- Log-form losses and clamping can hide physically bad flux behaviour.
- Temporal propagation may be weak over long time horizons.
- The network may satisfy local derivative constraints without learning the intended trajectory.
- Biological operator or derivative mismatch may create a residual that is easier to satisfy than the true mizer dynamics.
- Collocation sampling may underrepresent difficult residual regions.

### Current mitigations

- Initial-condition loss.
- Recruitment-boundary loss implementation.
- Optional fixed-grid timestep-consistency loss.
- Wang-style gradient-statistic weighting.
- Causal time curriculum.
- Planned R3/Causal R3 collocation to focus on high-residual paired points.
- Final-layer bias initialisation from IC.
- Fixed-grid diagnostics and output plots.

### Next checks

- Compare PDE+IC with and without causal curriculum.
- Compare Wang weights on/off.
- Compare timestep loss inactive versus active.
- Compare uniform collocation against R3/Causal R3 after R3 is implemented.
- Run explicit PDE residual validation on mizer-generated trajectories.
- Run timestep smoke checks before interpreting timestep-active training.
- Check `N_eval_min`, `N_eval_max`, `log_N_eval_min`, and fixed-grid abundance heatmaps after every run.

## 2. Boundary loss can be numerically misleading

Status: active

### Description

The recruitment-boundary loss has log, physical, and relative forms. The log form requires clamping. If `flux_left` is clamped heavily, the scalar loss can misrepresent actual boundary behaviour.

### Current diagnostics

The current code exports:

- `bc_eps`
- `flux_left_min`
- `flux_left_max`
- `recruitment_flux_min`
- `recruitment_flux_max`
- `frac_flux_left_clamped`
- `frac_recruitment_flux_clamped`
- `boundary_residual_abs_p95`
- `boundary_residual_abs_max`
- flux mismatch metrics in fixed diagnostics.

### Next checks

- Do not interpret BC loss without checking clamp fractions.
- Compare log, physical, and relative boundary loss forms.
- Confirm whether default `lambda_bc = 0.0` is intentional for current experiments.

## 3. Timestep loss can dominate or go invalid

Status: active

### Description

The timestep-consistency loss compares model-predicted `N(w,t+dt)` with the fixed-grid `step(...)` projection from `N(w,t)`. Depending on residual form, `dt`, abundance scale, and Wang weights, this term can dominate the total objective or produce unstable gradients.

This loss is implemented in `PINNmizer/pinn/timestep_consistency.py`. It is an optional fixed-grid temporal regulariser, not part of the continuous/off-grid PDE residual.

### Current risks

- physical-form residual can be very large on abundance scale;
- log-form residual depends on clamping;
- relative-form residual can explode where the step target is near zero;
- Wang weighting may amplify or suppress the term based on gradient statistics;
- selected `t0` values near `t_max` are filtered, and all pairs may be removed;
- `detach_step_target=False` changes the gradient path substantially;
- if `--timestep-dt` is omitted, behaviour depends on whether `params.dt` was loaded from `dt.csv`.

### Required checks

- inspect `loss_timestep`, `w_timestep`, `weighted_loss_timestep`;
- inspect `grad_timestep_mean_for_weighting` and `target_w_timestep`;
- check timestep residual physical/log/relative summaries;
- check whether valid timestep pairs remain after filtering;
- compare `--timestep-loss-form physical`, `log`, and `relative`;
- compare `--detach-step-target` against `--no-detach-step-target`;
- verify explicit `--timestep-dt` versus fixture `params.dt`;
- run the timestep smoke check before interpreting training.

### Source-code issue recorded, not fixed here

Current `PINNmizer/training/loop.py` filters sampled timestep pairs with logic equivalent to:

```python
valid = (t0 + dt_tensor) <= params.t_max
t0 = t0[valid]
```

If all sampled times are too close to `t_max`, `t0` may become empty before calling `compute_timestep_consistency_loss()`. That could produce empty tensor reductions or NaN timestep loss. This should be patched separately if timestep-active training is used seriously.

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

## 6. Wang-style weighting now supports arbitrary non-PDE losses

Status: active design risk, no longer module-location debt

### Description

The weighting code lives in `PINNmizer/training/weighting.py` and can update any non-PDE loss key included in the raw loss dictionary. PDE remains the anchor. For each non-PDE loss component included in `raw_losses`, including IC, BC, and timestep, the target weight is:

```text
target_weight = max_abs_grad(loss_pde) / mean_abs_grad(loss_component)
```

The target is clipped to `[weight_min, weight_max]`, then either hard-set or exponentially smoothed.

### Risk

This generality is useful for timestep loss, but future losses must be added deliberately because every included loss affects adaptive weights. A diagnostic-only or experimental scalar loss should not be added to `raw_losses` unless it is intended to participate in optimisation and Wang weighting.

If Causal R3 weights the PDE loss, Wang weighting should see the actual PDE objective being optimised; otherwise adaptive weighting diagnostics become hard to interpret.

### Required checks

- Confirm which loss keys are in `raw_losses` before interpreting weights.
- Inspect `grad_*_mean_for_weighting` and `target_w_*` for every active component.
- Do not describe Wang weighting as hard-coded only for PDE/IC/BC.

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

Status: planned feature risk

### Description

R3 retains high-residual paired collocation points and resamples low-residual points. This can improve coverage of difficult regions, but it also means the method will concentrate training effort wherever the current residual is large.

### Risk

If the residual implementation is wrong, the biological operator is mismatched, or active-size masking is incomplete, R3 will concentrate points on artefactual errors rather than fixing the underlying model. R3 is therefore not a replacement for residual validation.

### Required checks

- Validate PDE residual on known/mizer-generated trajectories before treating R3 results as meaningful.
- Compare R3 against uniform collocation using fixed-grid diagnostics, not only training batch loss.
- Inspect where retained points concentrate if results look pathological.
- Keep active `w_max` masking in both sampling and loss/scoring.

## 9. R3 is paired geometry, not retained marginal x/t vectors

Status: planned feature risk

### Description

Daw-style R3 retains/replaces collocation points as pairs `(x_j, t_j)`. Retaining separate `x_eval` and `t_eval` vectors would create new Cartesian combinations that were never scored.

### Risk

A superficially simple implementation that only changes `sample_pde_batch()` to retain `x_eval` and `t_eval` separately is not equivalent to R3.

### Required checks

- R3 population should be stored as `[n_pair, 2]` with physical columns `[x, t]`.
- Paired residual tensors should be `[n_species, n_pair]`.
- Paired derivative/state/loss paths should be used for R3.
- Retained points should remain unchanged across an update.
- Released points should be replaced as full `(x,t)` pairs.

## 10. Causal R3 and old causal curriculum should not be silently stacked

Status: planned feature risk

### Description

The existing causal curriculum truncates the sampled time horizon. Causal R3 uses a smooth gate over paired points. These are not the same mechanism.

### Risk

Using both at once makes it unclear whether improvements or failures come from time-domain truncation, causal gate weighting, R3 resampling, or their interaction.

### Required checks

- Default R3/Causal R3 runs should use `--causal-curriculum off`.
- If stacking is later allowed, document it explicitly and run ablations.
- Add a CLI guard unless a future ADR deliberately permits stacking.

## 11. Exact paired R3 may be much slower than Cartesian sampling

Status: planned feature risk

### Description

The current Cartesian path evaluates nonlocal biology once per sampled time and multiple weights per time. Exact paired R3 may require biology at many unique paired times.

### Risk

A large `--r3-population-size` may make training much slower than expected.

### Required checks

- Start R3 with small smoke runs.
- Compare runtime per step against uniform collocation.
- Do not call a cheaper Cartesian approximation Daw-style R3. Name it explicitly if added later.

## 12. Historical experiment logging is incomplete

Status: active

### Description

The repository contains run outputs and configs, but the new `08_EXPERIMENT_LOG.jsonl` is not yet a complete historical record.

### Risk

Future chats may infer too much from sparse logs unless new experiments are appended systematically.

### Required habit

After each important run, append one JSONL row with:

- date;
- commit;
- branch;
- run directory;
- command/config summary;
- key metrics;
- result interpretation;
- next action.

## 13. Matplotlib/diagnostic dependencies may fail in some environments

Status: lower priority

### Description

Past discussion included a missing `matplotlib` error during post-training diagnostics. Current diagnostic modules import `matplotlib.pyplot`.

### Risk

A training run may finish but fail during plotting/output diagnostics if the environment lacks plotting dependencies.

### Options

1. Install plotting dependencies in the `mizer-torch` environment.
2. Guard plotting imports and allow non-plot training completion.
3. Separate training from post-training diagnostics.

### Current recommendation

If this failure recurs, use option 2 or 3 rather than letting plotting failure invalidate completed training.

## 14. Root-level py_* validation-output folders are legacy artifacts

Status: resolved as context/documentation issue; watch for regression

### Description

Generated validation outputs have been moved out of repository root and into:

```text
validation/outputs/
```

Root-level generated folders such as:

```text
py_growth_derivative*
py_known*
py_mizer*
py_pred*
```

should be treated as legacy generated artifacts and should not be expected at repository root.

### Risk

Future sessions may misinterpret missing root-level `py_*` folders as a broken validation setup. That assumption is wrong under the current layout.

### Required habit

- Put generated validation outputs under `validation/outputs/`.
- Do not reintroduce root-level generated validation-output folders.
- Do not diagnose missing root-level `py_*` folders as a source-code failure.
