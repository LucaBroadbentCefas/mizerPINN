# Current State

## Last updated

2026-05-19

## Repository

- Canonical GitHub repository: `LucaBroadbentCefas/mizerPINN`
- Default branch inspected: `main`
- Main head used for this context layer: `79c027aa3c12cc6675a7ae023394b186159c4b84`
- Previous repository `LucaBroadbentCefas/PINNs` is deprecated and expected to be deleted. Do not use it for new project summaries, source inspection, or future context updates.

## Active project phase

The project is no longer just a standalone PDE-residual implementation. The current repository is now organised as a package-style project with reusable code under `PINNmizer/`, thin executable entry points under `scripts/`, validation assets under `validation/`, generated outputs under `runs/`, and project/context documentation under `docs/`.

The active scientific problem remains whether the PINN is learning a physically meaningful size-time abundance surface rather than exploiting weak constraints, loss imbalance, or near-zero/collapsed solutions.

## Current source-derived state

### Implemented

- `MizerTorchParams` stores fixed-grid, FFT, continuous biological, interaction, reproduction, mortality, and time-domain parameters.
- `load_mizer_inputs()` reads CSV exports into `MizerTorchParams`, `n_init`, and `n_pp`.
- Fixed-grid mizer-like operators exist for validation and projection in `PINNmizer/mizer_grid_ops.py`.
- Continuous/off-grid biological functions are split across `PINNmizer/biology/`.
- PINN sampling, model evaluation, autograd derivatives, PDE-state construction, residual assembly, and losses are split across `PINNmizer/pinn/`.
- The model output is treated as `log_N`.
- Autograd is used for network derivatives.
- Manual/analytical derivatives are used for the biological growth-side `dg_dw` path.
- PDE residuals are returned in both log and physical/check forms.
- Initial-condition and recruitment-boundary losses are implemented through cached PDE state.
- Single-species training is now package-backed under `PINNmizer/training/`, with a thin script wrapper under `scripts/`.
- Training includes Wang-style gradient-statistic adaptive weighting for PDE/IC/BC terms.
- Training includes causal time curriculum over the sampled PDE time horizon.
- Final-layer bias can be initialised from the mean initial log abundance.
- Diagnostics have been moved into package modules under `PINNmizer/diagnostics/`.
- Training and fixed-grid diagnostic outputs are saved into timestamped run directories under `runs/`.

### Current training entry point

```bash
python scripts/train_pde_only_single_species.py
```

The script is intentionally thin and delegates to:

```text
PINNmizer.training.train_pde_only_single_species.main
```

The default input directory is now:

```text
validation/fixtures/pde_single_species
```

The default run directory pattern is:

```text
runs/pde_only_single_species/YYYYMMDD_HHMMSS
```

## Current important defaults in the training script

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
- `initial_w_pde = 1.0`
- `initial_w_ic = 1.0`
- `initial_w_bc = 1e-3`
- `weighting = wang_gradient_statistics` unless `--disable-wang-weights` is used
- `causal_curriculum = linear`
- `causal_start_fraction = 0.05`
- `causal_ramp_steps = 1500`
- `init_final_bias_from_ic = True`

Important implication: although boundary-loss machinery is implemented, the default command currently has `lambda_bc = 0.0` unless overridden.

## Recent repository transition and cleanup

- `LucaBroadbentCefas/mizerPINN` is now the canonical repository.
- `LucaBroadbentCefas/PINNs` is deprecated and expected to be deleted.
- Do not inspect `PINNs` when asked about the active project unless explicitly asked for historical comparison.
- Recent `mizerPINN` work moved code toward a clearer package structure: reusable package code under `PINNmizer/`, scripts under `scripts/`, fixtures/checks/comparisons under `validation/`, diagnostics under `PINNmizer/diagnostics/`, and generated outputs under `runs/`.
- A stale post-run diagnostics import was fixed by moving diagnostics into package modules rather than importing them from `validation/scripts/`.

## Known live concerns

### 1. Near-zero or collapsed solutions

Previous training runs and discussion indicated that the PINN can push `N` toward near-zero values while reducing parts of the objective. This remains a central modelling/training risk.

Potential contributors:

- loss-term imbalance;
- weak temporal propagation;
- PDE residual admitting trivial or near-trivial solutions without sufficient anchoring;
- boundary/IC scaling problems;
- log-space clamping choices;
- mismatch between continuous PDE residual and discretised mizer trajectory generation;
- insufficient data/trajectory anchoring away from IC/BC.

### 2. Boundary loss sensitivity

The recruitment-boundary loss uses clamping floors for log-form residuals and exports diagnostics such as clamped fractions, flux minima, and boundary residual summaries. These diagnostics should be checked before trusting BC loss values.

### 3. Validation against mizer trajectories still matters

The strongest diagnostic is not just whether training loss falls. A key validation route is to use mizer-generated `N(w,t)` as a surrogate model output and check whether the assembled PDE residual is small within expected discretisation/interpolation error.

### 4. Documentation and run logs remain incomplete

This context layer now records the repository transition and package structure, but it does not yet encode every historical experiment. `08_EXPERIMENT_LOG.jsonl` is intentionally sparse and should be appended after future runs.

### 5. Recent structure refactor increases import-regression risk

Because code was moved between package, script, diagnostics, and validation areas, future edits should check imports against the current repository before proposing patches.

## Current next sensible tasks

1. Confirm the current training defaults are intentional, especially `lambda_bc = 0.0` while BC machinery exists.
2. Run a controlled comparison:
   - PDE + IC only;
   - PDE + IC + BC;
   - Wang weights on/off;
   - causal curriculum on/off.
3. Add an explicit validation script that evaluates the PDE residual on mizer-generated trajectories without training.
4. Keep package imports clean:
   - `PINNmizer/*` should not import from `scripts/*`;
   - `PINNmizer/*` should not import from `validation/*`;
   - diagnostics used by training should live under `PINNmizer/diagnostics/`.
5. Keep this context layer updated at the end of each major debugging or implementation session.

## Do not revisit without new evidence

- Do not replace the whole biological operator path to solve a training issue.
- Do not infer that falling PDE loss means correct dynamics.
- Do not assume the fixed-grid FFT path and continuous off-grid path are interchangeable without validation.
- Do not treat ChatGPT memory or chat history as canonical project state.
- Do not use `LucaBroadbentCefas/PINNs` as the active repository.
