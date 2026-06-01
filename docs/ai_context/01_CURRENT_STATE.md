# Current State

## Last updated

2026-05-29

## Repository

- Canonical GitHub repository: `LucaBroadbentCefas/mizerPINN`
- Default branch inspected: `main`
- Previous repository `LucaBroadbentCefas/PINNs` is deprecated and expected to be deleted. Do not use it for new project summaries, source inspection, or future context updates.

## Active project phase

The project is a package-style PyTorch PINN implementation for a mizer-style marine size-spectrum PDE. Reusable code belongs under `PINNmizer/`, thin executable wrappers under `scripts/`, validation fixtures and checks under `validation/`, generated validation outputs under `validation/outputs/`, generated training outputs under `runs/`, and continuity documentation under `docs/ai_context/`.

The active scientific problem remains whether the PINN is learning a physically meaningful size-time abundance surface rather than exploiting weak constraints, loss imbalance, timestep inconsistency, poor collocation coverage, or near-zero/collapsed solutions.

## Current source-derived state

### Implemented in the current repository

- `MizerTorchParams` stores fixed-grid, FFT, continuous biological, interaction, reproduction, mortality, time-domain, optional timestep, and active-size parameters.
- `load_mizer_inputs()` reads CSV exports into `MizerTorchParams`, `n_init`, and `n_pp`.
- `load_mizer_inputs()` optionally loads `dt.csv` into `params.dt` when the fixture/export supplies it.
- Fixed-grid mizer-like operators exist in `PINNmizer/mizer_grid_ops.py` for validation/reference and one-step projection.
- Continuous/off-grid biological functions are split across `PINNmizer/biology/`.
- PINN sampling, model evaluation, autograd derivatives, PDE-state construction, residual assembly, and losses are split across `PINNmizer/pinn/`.
- Optional fixed-grid timestep-consistency loss is implemented in `PINNmizer/pinn/timestep_consistency.py`.
- The timestep loss compares model-predicted `N(w_grid, t + dt)` against the fixed-grid mizer-style `step(...)` projection from model-predicted `N(w_grid, t)`.
- The timestep loss supports physical, log, and relative residual forms.
- The timestep loss is opt-in through `--lambda-timestep`; default `lambda_timestep = 0.0`.
- The timestep loss is separate from the continuous/off-grid PDE residual path. It intentionally uses the fixed mizer grid and the fixed-grid `step(...)` operator.
- The model output is treated as `log_N`.
- Autograd is used for neural-network derivatives.
- Manual/analytical derivatives are used for biological growth-side `dg_dw`.
- PDE residuals are returned in log, physical, and physical-check forms.
- Initial-condition and recruitment-boundary losses are implemented through cached PDE state.
- Active-size masking exists through `active_grid_mask(params)` and `active_eval_mask(w_eval, params)`; these prevent inactive weights above species `w_max` from contributing to relevant losses.
- Single-species training is package-backed under `PINNmizer/training/`, with a thin script wrapper under `scripts/`.
- Training includes Wang-style gradient-statistic adaptive weighting for arbitrary non-PDE loss keys present in `raw_losses`, including IC, BC, and timestep.
- Training records timestep diagnostics and includes timestep loss in Wang-style adaptive weighting when active.
- Training includes the original causal time curriculum over the sampled PDE time horizon.
- Final-layer bias can be initialised from the mean initial log abundance.
- Diagnostics live in package modules under `PINNmizer/diagnostics/`.
- Training and fixed-grid diagnostic outputs are saved into timestamped run directories under `runs/`.
- Generated validation outputs belong under `validation/outputs/`.
- Root-level generated folders such as `py_growth_derivative*`, `py_known*`, `py_mizer*`, and `py_pred*` are legacy artifacts and should not be expected or reintroduced at repository root.

### Accepted design / implementation target: R3 and Causal R3 collocation

R3/Causal R3 is now a recorded implementation target for the PINN collocation workflow. Future assistants must still inspect source files before assuming it has been merged.

The intended R3 addition is not just a new sampler. It changes the residual tensor geometry from Cartesian batches to paired collocation batches.

Current Cartesian path:

```text
t_eval: [n_time]
x_eval: [n_eval]
w_eval: [n_eval]
residual: [n_time, n_species, n_eval]
```

R3 paired path target:

```text
R3Population.points: [n_pair, 2] with columns [x, t]
x_pair: [n_pair]
t_pair: [n_pair]
w_pair: [n_pair]
residual: [n_species, n_pair]
```

The intended R3 source layout is:

```text
PINNmizer/pinn/r3.py
PINNmizer/pinn/derivatives.py          add evaluate_log_model_with_derivatives_at_pairs
PINNmizer/pinn/pde_state.py            add compute_pde_state_paired
PINNmizer/pinn/losses.py               add compute_pde_loss_paired
PINNmizer/training/loop.py             add collocation_strategy branch
PINNmizer/training/train_pde_only_single_species.py add CLI args and state setup
```

The original uniform Cartesian path must remain available and unchanged as the default validation/training baseline.

## Current training entry point

```bash
python scripts/train_pde_only_single_species.py
```

The script delegates to:

```text
PINNmizer.training.train_pde_only_single_species.main
```

The default input directory is:

```text
validation/fixtures/pde_single_species
```

The default run directory pattern is:

```text
runs/pde_only_single_species/YYYYMMDD_HHMMSS
```

## Current important defaults in the training script

These are current source defaults unless later changed:

- `n_steps = 2000`
- `n_time = 10`
- `n_eval = 30`
- `lr = 1e-3`
- `hidden_width = 64`
- `hidden_layers = 3`
- `residual_form = log`
- `boundary_loss_form = log`
- `lambda_pde = 1.0`
- `lambda_ic = 1.0`
- `lambda_bc = 0.0`
- `lambda_timestep = 0.0`
- `initial_w_pde = 1.0`
- `initial_w_ic = 1.0`
- `initial_w_bc = 1e-3`
- `initial_w_timestep = 1.0`
- `timestep_loss_form = physical`
- `detach_step_target = True`
- `timestep_dt = None`
- `timestep_n_pairs = 1`
- `weighting = wang_gradient_statistics` unless `--disable-wang-weights` is used
- `causal_curriculum = linear`
- `causal_start_fraction = 0.05`
- `causal_ramp_steps = 1500`
- `init_final_bias_from_ic = True`

Important implications:

- Boundary-loss machinery is implemented, but the default command has `lambda_bc = 0.0` unless overridden.
- Timestep-consistency loss is implemented, but inactive by default because `lambda_timestep = 0.0`.
- When `--timestep-dt` is omitted, timestep consistency should use `params.dt` loaded from `dt.csv` when present.
- `params.dt` should represent the mizer timestep used by the fixture/export; `--timestep-dt` is an explicit experimental override.
- The original causal curriculum truncates the sampled time horizon. It should not be silently stacked with Causal R3 unless a later explicit design decision allows it.

## Planned R3/Causal R3 CLI additions

When R3 is implemented, the intended CLI additions are:

```text
--collocation-strategy {uniform,r3,causal-r3}
--r3-population-size INT
--r3-update-every INT
--r3-warmup-steps INT
--r3-score-form {abs,squared}
--r3-seed INT
--causal-r3-alpha FLOAT
--causal-r3-gamma-init FLOAT
--causal-r3-gamma-max FLOAT
--causal-r3-weight-pde-loss
--no-causal-r3-score
```

Recommended guard:

```text
if collocation_strategy != "uniform" and causal_curriculum != "off":
    raise ValueError
```

Reason: the old causal curriculum restricts the sampled time interval; Causal R3 applies a smooth gate over paired points. Stacking both changes the meaning of both methods.

## Current timestep diagnostics in training rows

When timestep loss is active or configured, training rows may include:

- `loss_timestep`
- `lambda_timestep`
- `w_timestep`
- `weighted_loss_timestep`
- `grad_timestep_mean_for_weighting`
- `target_w_timestep`
- `timestep_physical_abs_mean`
- `timestep_physical_abs_max`
- `timestep_log_abs_mean`
- `timestep_log_abs_max`
- `timestep_relative_abs_mean`
- `timestep_relative_abs_max`

## Planned R3 diagnostics in training rows

When R3/Causal R3 is implemented, keep initial diagnostics minimal:

- `collocation_strategy`
- `r3_population_size`
- `r3_retained_fraction`
- `r3_resampled`
- `r3_score_mean`
- `r3_score_max`
- `causal_r3_gamma`
- `causal_r3_gamma_update`
- `causal_r3_gate_mean`

Do not save large retained-point snapshots unless debugging requires it. If snapshots are saved, place them under the run directory, not repository root.

## Recent repository transition and cleanup

- `LucaBroadbentCefas/mizerPINN` is now the canonical repository.
- `LucaBroadbentCefas/PINNs` is deprecated and expected to be deleted.
- Do not inspect `PINNs` when asked about the active project unless explicitly asked for historical comparison.
- Code was moved toward a clearer package structure: reusable package code under `PINNmizer/`, scripts under `scripts/`, fixtures/checks/comparisons under `validation/`, diagnostics under `PINNmizer/diagnostics/`, and generated training outputs under `runs/`.
- Generated validation outputs were moved out of root-level `py_*` folders and into `validation/outputs/`.
- Missing root-level `py_*` validation-output folders are not a problem.
- A stale post-run diagnostics import was fixed by moving diagnostics into package modules rather than importing them from `validation/scripts/`.

## Known live concerns

### 1. Near-zero or collapsed solutions

Previous training runs and discussion indicated that the PINN can push `N` toward near-zero values while reducing parts of the objective. This remains a central modelling/training risk.

Potential contributors:

- loss-term imbalance;
- weak temporal propagation;
- timestep-consistency loss inactive or badly scaled;
- PDE residual admitting trivial or near-trivial solutions without sufficient anchoring;
- boundary/IC scaling problems;
- log-space clamping choices;
- mismatch between continuous PDE residual and discretised mizer trajectory generation;
- insufficient data/trajectory anchoring away from IC/BC;
- collocation points under-sampling difficult regions.

### 2. Boundary loss sensitivity

The recruitment-boundary loss uses clamping floors for log-form residuals and exports diagnostics such as clamped fractions, flux minima, and boundary residual summaries. These diagnostics should be checked before trusting BC loss values.

### 3. Timestep loss scale and flat-gradient risk

The timestep term can dominate or destabilise training depending on residual form, `dt`, abundance scale, and Wang weighting. Future debugging must inspect `loss_timestep`, `w_timestep`, `weighted_loss_timestep`, `grad_timestep_mean_for_weighting`, and whether valid timestep pairs remain after filtering.

### 4. R3/Causal R3 cost and interpretation risk

Exact paired R3 is more expensive than Cartesian sampling because nonlocal biology may be evaluated at up to `n_pair` unique times. A Cartesian approximation should not be called Daw-style R3. If a cheaper approximation is added, name it explicitly, for example `cartesian-r3-approx`.

R3 will concentrate points where the current residual is high. If the residual implementation is wrong, R3 will concentrate training on artefacts. Therefore, R3 is not a substitute for PDE residual validation.

### 5. Validation against mizer trajectories still matters

The strongest diagnostic is not just whether training loss falls. A key validation route is to use mizer-generated `N(w,t)` as a surrogate model output and check whether the assembled PDE residual is small within expected discretisation/interpolation error.

### 6. Documentation and run logs remain incomplete

`08_EXPERIMENT_LOG.jsonl` is intentionally sparse and should be appended after future runs.

### 7. Recent structure refactor increases import-regression risk

Because code was moved between package, script, diagnostics, and validation areas, future edits should check imports against the current repository before proposing patches.

## Current next sensible tasks

1. Confirm the current training defaults are intentional, especially `lambda_bc = 0.0` and `lambda_timestep = 0.0` while both loss mechanisms exist.
2. Run controlled comparisons:
   - PDE + IC only;
   - PDE + IC + BC;
   - PDE + IC + timestep;
   - PDE + IC + BC + timestep;
   - Wang weights on/off;
   - causal curriculum on/off.
3. Implement R3/Causal R3 as a small source change, preserving the original uniform Cartesian path.
4. Add paired derivative/state/loss functions only where needed; do not rewrite the whole PDE stack.
5. Keep R3/Causal R3 initially minimal: one sampler/state module, paired derivatives, paired PDE state, paired loss, and a training-loop branch.
6. Run timestep smoke checks before interpreting timestep-active training.
7. Add explicit validation that evaluates the PDE residual on mizer-generated trajectories without training.
8. Keep package imports clean:
   - `PINNmizer/*` should not import from `scripts/*`;
   - `PINNmizer/*` should not import from `validation/*`;
   - diagnostics used by training should live under `PINNmizer/diagnostics/`.
9. Keep context files updated at the end of each major debugging or implementation session.

## Do not revisit without new evidence

- Do not replace the whole biological operator path to solve a training issue.
- Do not infer that falling PDE loss means correct dynamics.
- Do not assume the fixed-grid FFT path and continuous off-grid path are interchangeable without validation.
- Do not treat timestep consistency as part of the continuous PDE residual.
- Do not treat R3 as merely a different loader; it requires paired tensor geometry.
- Do not silently stack Causal R3 with the old causal curriculum.
- Do not expect root-level `py_*` validation-output folders to exist.
- Do not treat ChatGPT memory or chat history as canonical project state.
- Do not use `LucaBroadbentCefas/PINNs` as the active repository.
