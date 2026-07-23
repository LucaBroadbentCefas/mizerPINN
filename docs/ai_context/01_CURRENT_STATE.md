# Current State

## Last updated

2026-06-03

## Repository

- Canonical GitHub repository: `LucaBroadbentCefas/mizerPINN`
- Default branch inspected: `main`
- Previous repository `LucaBroadbentCefas/PINNs` is deprecated and expected to be deleted. Do not use it for new project summaries, source inspection, or future context updates.

## Active project phase

The project is a package-style PyTorch PINN implementation for a mizer-style marine size-spectrum PDE. Reusable code belongs under `PINNmizer/`, thin executable wrappers under `scripts/`, validation fixtures and checks under `validation/`, generated validation outputs under `validation/outputs/`, generated training outputs under `runs/`, and continuity documentation under `docs/ai_context/`.

The active scientific problem remains whether the PINN is learning a physically meaningful size-time abundance surface rather than exploiting weak constraints, loss imbalance, timestep inconsistency, poor collocation coverage, invalid recruitment-boundary scaling, or near-zero/collapsed solutions.

## Current source-derived state

### Implemented in the current repository

- `MizerTorchParams` stores fixed-grid, FFT, continuous biological, interaction, reproduction, mortality, time-domain, optional timestep, and active-size parameters.
- `load_mizer_inputs()` reads CSV exports into `MizerTorchParams`, `n_init`, and `n_pp` and optionally loads `dt.csv` into `params.dt`.
- Fixed-grid mizer-like operators live in `PINNmizer/mizer_grid_ops.py` for validation/reference and one-step projection.
- Continuous/off-grid biological functions are split across `PINNmizer/biology/`.
- PINN sampling, model evaluation, autograd derivatives, PDE-state construction, residual assembly, and losses are split across `PINNmizer/pinn/`.
- Optional fixed-grid timestep-consistency loss is implemented in `PINNmizer/pinn/timestep_consistency.py`.
- The timestep loss compares model-predicted `N(w_grid, t + dt)` against the fixed-grid mizer-style `step(...)` projection from model-predicted `N(w_grid, t)`.
- The timestep loss supports physical, log, and relative residual forms and is opt-in through `--lambda-timestep`; default `lambda_timestep = 0.0`.
- The model output is treated as `log_N`.
- Autograd is used for neural-network derivatives; manual/analytical derivatives are used for biological growth-side `dg_dw`.
- PDE residuals are returned in log, physical, and physical-check forms.
- Initial-condition and recruitment-boundary losses are implemented through cached PDE state.
- Active-size masking exists through `active_grid_mask(params)` and `active_eval_mask(w_eval, params)`; these prevent inactive weights above species `w_max` from contributing to relevant losses and R3 scores.
- Single-species training is package-backed under `PINNmizer/training/`, with a thin script wrapper under `scripts/`.
- Training includes Wang-style gradient-statistic adaptive weighting for non-PDE loss keys present in the weighting dictionary, including IC, BC, and timestep.
- Training records objective-scale diagnostics: `objective_loss_*` includes lambda and Wang weights; `wang_scaled_loss_*` excludes lambda and records only adaptive-weight scaling.
- Training includes the original causal time curriculum over the sampled PDE time horizon.
- Final-layer bias can be initialised from the mean initial log abundance.
- Diagnostics live in package modules under `PINNmizer/diagnostics/`.
- Training and fixed-grid diagnostic outputs are saved into timestamped run directories under `runs/`.
- Generated validation outputs belong under `validation/outputs/`.
- Root-level generated folders such as `py_growth_derivative*`, `py_known*`, `py_mizer*`, and `py_pred*` are legacy artifacts and should not be expected or reintroduced at repository root.

## R3 and Causal R3 collocation status

R3/Causal R3 is implemented experimentally and currently uses a slab/time population, not the older flat-paired point design.

Current implemented files include:

```text
PINNmizer/pinn/r3.py
PINNmizer/pinn/derivatives.py          paired/slab derivative support
PINNmizer/pinn/pde_state.py            paired/slab-aware PDE-state support
PINNmizer/pinn/losses.py               paired/slab-aware PDE loss support
PINNmizer/training/loop.py             collocation_strategy branch
PINNmizer/training/train_pde_only_single_species.py CLI args and state setup
```

Current slab/time R3 structure:

```text
R3Population.t_points: [K]
R3Population.x_points: [K, M]
R3Population.population_size = K * M
batch["t_slab"]: [K]
batch["x_slab"]: [K, M]
batch["w_slab"]: [K, M]
residual: [K, n_species, M]
```

The implemented rule is:

- `K = args.n_time`.
- If `--r3-population-size` is omitted, `n_pair = args.n_time * args.n_eval` and therefore `M = args.n_eval`.
- If `--r3-population-size` is supplied, `M = ceil(r3_population_size / args.n_time)` and the effective population is `K * M`, which may exceed the requested total.
- At every training step in R3 or Causal R3 mode, `r3_population.resample_time_points_(..., t_max_current=current_t_max)` is called before `as_batch()`.
- R3 retain/resample acts only on `x_points`; time slabs are resampled each step.
- `t_max_current` comes from the ordinary causal time curriculum, so R3 can currently be combined with `--causal-curriculum linear` or `step` as a bounded-time sampler.

Important correction to older context: do **not** keep recommending a hard guard that forbids `collocation_strategy != uniform` with `causal_curriculum != off`. That was an earlier design concern. The current source deliberately uses causal curriculum to bound R3 time resampling. Causal R3's smooth gate remains a separate mechanism and should still be interpreted separately from time-domain truncation.

The original uniform Cartesian path remains the default validation/training baseline and must remain available.

Current uniform Cartesian path:

```text
t_eval: [n_time]
x_eval: [n_eval]
w_eval: [n_eval]
residual: [n_time, n_species, n_eval]
```

Historical flat-paired R3 target, retained only as context:

```text
x_pair: [n_pair]
t_pair: [n_pair]
w_pair: [n_pair]
residual: [n_species, n_pair]
```

Do not advise reverting to flat-paired R3 for performance. The previous flat-paired bottleneck was addressed by the slab/time implementation, which reuses biological work by time slab.

## Recruitment boundary condition state

The code currently implements a recruitment-boundary loss, but it is still an active scientific risk rather than a validated final formulation.

Current implementation in `compute_recruitment_boundary_loss_from_state()`:

```text
egg_idx = params.w_min_idx - 1
N_left = N_grid[:, species, egg_idx]
g_left = growth_grid["e_growth_eval"][:, species, egg_idx]
R = recruitment["rdd_flux"][:, species]
valid = finite(log_N_left, N_left, g_left, R) and g_left > bc_g_min and R > 0
```

Invalid samples are excluded, not clamped. If no valid samples exist, the BC contribution is a graph-connected zero.

Loss forms currently mean:

```text
log:      log_N_left - log(R / g_left)
physical: N_left - R / g_left
relative: 1 - (N_left * g_left) / R
```

The relative form was changed on 2026-06-02 to use a dimensionless flux ratio. This avoids the old density-relative expression that could still produce misleading scale behaviour when `R/g` is huge.

Do not infer that mizer first-bin values validate `g(w_min)N(w_min)=R`. Recent investigation found mizer-like exported values with `g(w_min)` extremely small while `R` and first-bin `N` are similar in magnitude, so `R/g` can be numerically absurd. Treat the current BC loss as an experimental implementation under the continuous flux-boundary assumption, not as a validated mizer-matching boundary condition. Default `lambda_bc` remains `0.0` unless explicitly overridden.

Useful BC diagnostics now include:

```text
bc_g_min
bc_valid_count
bc_total_count
bc_valid_fraction
bc_invalid_fraction
bc_invalid_g_fraction
bc_invalid_recruitment_fraction
bc_nonfinite_fraction
bc_target_log_N_min / max
bc_target_N_min / max
boundary_residual_abs_p95 / max
flux_left_min / max
recruitment_flux_min / max
```

Backward-compatible names `frac_g_left_clamped`, `frac_recruitment_flux_clamped`, and `frac_flux_left_clamped` may still appear, but they should be interpreted as validity/clamp-risk diagnostics, not evidence that the new BC target clamps invalid samples into the loss.

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
- `bc_g_min = 1e-12`

Important implications:

- Boundary-loss machinery is implemented, but the default command has `lambda_bc = 0.0` unless overridden.
- Timestep-consistency loss is implemented, but inactive by default because `lambda_timestep = 0.0`.
- When `--timestep-dt` is omitted, timestep consistency should use `params.dt` loaded from `dt.csv` when present.
- `params.dt` should represent the mizer timestep used by the fixture/export; `--timestep-dt` is an explicit experimental override.
- R3/Causal R3 are inactive by default because `collocation_strategy = uniform`.

## R3/Causal R3 CLI additions

Current R3/Causal R3 CLI family:

```text
--collocation-strategy {uniform,r3,causal-r3}
--r3-population-size INT or omitted
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

## Current timestep diagnostics in training rows

When timestep loss is active or configured, training rows may include:

- `loss_timestep`
- `lambda_timestep`
- `w_timestep`
- `objective_loss_timestep`
- `wang_scaled_loss_timestep`
- `weighted_loss_timestep` as a backward-compatible alias for objective contribution
- `grad_timestep_mean_for_weighting`
- `target_w_timestep`
- `timestep_physical_abs_mean`
- `timestep_physical_abs_max`
- `timestep_log_abs_mean`
- `timestep_log_abs_max`
- `timestep_relative_abs_mean`
- `timestep_relative_abs_max`

## Current R3 diagnostics in training rows

When R3/Causal R3 is active, training rows may include:

- `collocation_strategy`
- `r3_population_size`
- `r3_n_time`
- `r3_n_eval_per_time`
- `r3_biology_time_loops`
- `r3_retained_fraction`
- `r3_resampled`
- `r3_score_mean`
- `r3_score_max`
- `causal_r3_gamma`
- `causal_r3_gamma_update`
- `causal_r3_gate_mean`
- `loss_pde_ungated`
- `loss_pde_gated`
- `pde_gate_mean`
- `pde_gate_min`
- `pde_gate_max`

Do not save large retained-point snapshots unless debugging requires it. If snapshots are saved, place them under the run directory, not repository root.

## Optional per-species inverse `r_max` mode

The multispecies training entry point can optionally estimate one `r_max` per species with `--estimate-rmax`. The inverse parameter is represented as a raw trainable logit whose sigmoid maps `log(r_max)` into configured hard bounds (defaults `[0, 50]`), then exponentiates back to physical `r_max`.

In this mode, data and initial-condition losses constrain the PINN abundance surface but do not directly differentiate with respect to `r_max`. The direct `r_max` gradient is intentionally restricted to the recruitment boundary condition: `R_DI` and boundary growth `g_left` are detached, while `R_DD = R_DI / (1 + R_DI / r_max)` is recomputed from detached `R_DI` and live `r_max`. Full target-side gradients remain disabled.

The reference inverse run keeps `--lambda-timestep 0.0`; timestep consistency is not used to estimate `r_max`.
