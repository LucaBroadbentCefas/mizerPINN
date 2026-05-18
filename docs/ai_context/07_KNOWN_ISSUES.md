# Known Issues

## 1. Trivial or near-zero solution risk

Status: active

### Description

The PINN can reduce parts of the objective while producing very small abundance values over much of the domain. This is a known failure mode in PDE-constrained neural networks when constraints do not sufficiently identify the desired solution.

### Evidence so far

Earlier logs and discussion showed cases where predicted `N` approached extremely small values while PDE residual terms could remain partly satisfied. The current training script now includes adaptive loss weighting, causal curriculum, final-bias initialisation from IC, and diagnostics partly in response to this risk.

### Plausible causes

- PDE residual alone does not identify a unique biologically correct solution.
- Initial and boundary conditions may be too weak, badly scaled, or underweighted.
- Log-form losses and clamping can hide physically bad flux behaviour.
- Temporal propagation may be weak over long time horizons.
- The network may satisfy local derivative constraints without learning the intended trajectory.
- Biological operator or derivative mismatch may create a residual that is easier to satisfy than the true mizer dynamics.

### Current mitigations

- Initial-condition loss.
- Recruitment-boundary loss implementation.
- Wang-style gradient-statistic weighting.
- Causal time curriculum.
- Final-layer bias initialisation from IC.
- Fixed-grid diagnostics and output plots.

### Next checks

- Compare PDE+IC with and without causal curriculum.
- Compare Wang weights on/off.
- Run explicit PDE residual validation on mizer-generated trajectories.
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

## 3. PDE residual on mizer trajectory not yet formalised

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

## 4. Continuous kernel convention may not fully match mizer

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

## 5. Wang-style weighting is embedded in training script

Status: technical debt

### Description

`update_wang_gradient_weights_()` currently lives in `validation_steps/train_pde_only_single_species.py`.

### Risk

As experiments grow, loss weighting may become harder to test and reuse.

### Options

1. Leave it in the script while experiments are changing quickly.
2. Move it into `PINNmizer/loss_weighting.py` once stable.
3. Move all training utilities into a training package if multiple scripts start sharing them.

### Current recommendation

Do not refactor immediately unless the next task modifies weighting substantially. Refactor after the weighting design is stable.

## 6. Training script is single-species only

Status: intentional current limitation

### Description

The lower-level code is mostly species-general, but the current training script raises an error unless `n_species == 1`.

### Risk

Do not assume multi-species training works simply because tensor shapes allow species dimensions.

### Next checks before multi-species extension

- Validate tensor shapes for multi-species outputs.
- Revisit loss aggregation across species.
- Revisit boundary condition per species.
- Revisit diagnostics and plotting.
- Test gradient weighting across species and loss components.

## 7. Historical experiment logging is incomplete

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

## 8. Matplotlib/diagnostic dependencies may fail in some environments

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
