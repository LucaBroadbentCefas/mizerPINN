# Current State

## Last updated

2026-05-18

## Repository

- GitHub repository: `LucaBroadbentCefas/PINNs`
- Default branch inspected: `main`
- Main head used for this context layer: `cf437f8da7ba73678971b69f56d746eb93ee0703`
- Documentation branch: `docs/add-ai-context`

## Active project phase

The project is no longer just a standalone PDE-residual implementation. The current repository includes a single-species training workflow using PDE, initial-condition, and recruitment-boundary losses, with adaptive loss weighting and diagnostic exports.

The main active problem remains whether the PINN is learning a physically meaningful size-time abundance surface rather than exploiting weak constraints, loss imbalance, or near-zero/collapsed solutions.

## Current source-derived state

### Implemented

- `MizerTorchParams` stores fixed-grid, FFT, continuous biological, interaction, reproduction, mortality, and time-domain parameters.
- `load_mizer_inputs()` reads CSV exports into `MizerTorchParams`, `n_init`, and `n_pp`.
- Fixed-grid mizer-like operators exist for validation and projection.
- Continuous/direct biological functions exist for off-grid PDE residual evaluation.
- The model output is treated as `log_N`.
- Autograd is used for network derivatives.
- Manual/analytical derivatives are used for the biological growth-side `dg_dw` path.
- PDE residuals are returned in both log and physical/check forms.
- Initial-condition and recruitment-boundary losses are implemented through cached PDE state.
- Single-species training includes Wang-style gradient-statistic adaptive weighting for PDE/IC/BC terms.
- Single-species training includes causal time curriculum over the sampled PDE time horizon.
- Final-layer bias can be initialised from the mean initial log abundance.
- Training and fixed-grid diagnostic outputs are saved into timestamped run directories.

### Current training entry point

```bash
python -m validation_steps.train_pde_only_single_species
```

The default input directory in the script is:

```text
py_inputs_ns_first_species
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
- `weighting = wang_gradient_statistics`
- `causal_curriculum = linear`
- `causal_start_fraction = 0.05`
- `causal_ramp_steps = 1500`
- `init_final_bias_from_ic = True`

Important implication: although boundary-loss machinery is implemented, the default command currently has `lambda_bc = 0.0` unless overridden.

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

### 4. Documentation and run logs are incomplete

This context layer starts the documentation system, but it does not yet encode every historical experiment. `08_EXPERIMENT_LOG.jsonl` is intentionally sparse and should be appended after future runs.

## Current next sensible tasks

1. Confirm the current training defaults are intentional, especially `lambda_bc = 0.0` while BC machinery exists.
2. Run a controlled comparison:
   - PDE + IC only;
   - PDE + IC + BC;
   - Wang weights on/off;
   - causal curriculum on/off.
3. Add an explicit validation script that evaluates the PDE residual on mizer-generated trajectories without training.
4. Split Wang-style loss weighting out of `train_pde_only_single_species.py` into a dedicated module if it grows further.
5. Keep this context layer updated at the end of each major debugging or implementation session.

## Do not revisit without new evidence

- Do not replace the whole biological operator path to solve a training issue.
- Do not infer that falling PDE loss means correct dynamics.
- Do not assume the fixed-grid FFT path and continuous off-grid path are interchangeable without validation.
- Do not treat ChatGPT memory or chat history as canonical project state.
