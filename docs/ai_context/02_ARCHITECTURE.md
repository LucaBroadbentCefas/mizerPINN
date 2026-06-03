# Architecture

## Purpose of this document

This document records the intended structure of `mizerPINN` so that future edits keep scientific code, training workflows, diagnostics, validation scripts, generated outputs, and project context separated.

The main rule is simple: reusable package code belongs under `PINNmizer/`; thin executable entry points belong under `scripts/`; validation fixtures and one-off comparison scripts belong under `validation/`; generated validation outputs belong under `validation/outputs/`; generated training outputs belong under `runs/`; project context and design notes belong under `docs/ai_context/`.

## Top-level project split

```text
PINNmizer/
  biology/
    kernels.py
    encounter.py
    growth.py
    mortality.py
    recruitment.py
  pinn/
    sampling.py
    r3.py
    model_eval.py
    derivatives.py
    pde_state.py
    residual.py
    losses.py
    timestep_consistency.py
  training/
    train_pde_only_single_species.py
    loop.py
    weighting.py
    config.py
    outputs.py
    checkpointing.py
  diagnostics/
    fixed_grid.py
    metrics.py
    outputs.py
    plots.py
    fields.py
  params.py
  io.py
  mizer_grid_ops.py
  utils.py

scripts/
  train_pde_only_single_species.py

validation/
  fixtures/
  outputs/
  scripts/
    checks/
    comparisons/
    export/
    legacy/

runs/
  generated training and diagnostic outputs only

docs/
  ai_context/
  design notes, architecture notes, prompts, and experiment records
```

## Architectural principle

The project has three deliberately separate computation paths:

1. fixed-grid mizer/TMB-style reference operators;
2. continuous/off-grid PINN PDE residual operators;
3. optional fixed-grid timestep-consistency regularisation.

R3/Causal R3 is a collocation strategy layered over the PINN residual path. It should not change biological equations or fixed-grid reference operators.

## 1. Fixed-grid mizer/TMB-style path

Implemented mainly in:

```text
PINNmizer/mizer_grid_ops.py
```

Purpose:

- reproduce fixed-grid mizer/TMB-style operators;
- use fixed mizer grids and FFT-based kernel operations;
- support validation, comparison, and reference behaviour;
- provide timestep/reference operators where needed.

Boundary:

- This path is not the arbitrary off-grid PINN residual path.
- Keep FFT/reference logic here unless a function is genuinely shared by both paths.
- If a function is shared by both fixed-grid and continuous paths, consider moving it to a neutral module rather than making continuous biology depend on `mizer_grid_ops.py`.

## 2. Continuous/off-grid Cartesian PINN PDE path

Implemented mainly in:

```text
PINNmizer/biology/*
PINNmizer/pinn/sampling.py
PINNmizer/pinn/model_eval.py
PINNmizer/pinn/derivatives.py
PINNmizer/pinn/pde_state.py
PINNmizer/pinn/residual.py
PINNmizer/pinn/losses.py
```

Purpose:

- evaluate biological quantities at arbitrary physical weights `w_eval = exp(x_eval)`;
- compute model derivatives using PyTorch autograd;
- assemble the PDE residual and PDE/IC/BC loss terms;
- preserve gradient flow from the loss back to model parameters;
- keep the current uniform Cartesian collocation path as the default baseline.

Boundary:

- Do not introduce NumPy into differentiable PDE-loss calculations.
- Do not detach tensors inside differentiable loss paths unless deliberately producing diagnostics.
- Do not replace analytical/manual `dg_dw` with autograd without an explicit design decision.
- Diagnostics may detach tensors and use pandas, NumPy, or matplotlib after differentiable computation is complete.

## 3. Fixed-grid timestep-consistency loss

Implemented in:

```text
PINNmizer/pinn/timestep_consistency.py
```

Purpose:

- compare `N_theta(w_grid, t + dt)` with `mizer_grid_ops.step(n_pp, N_theta(w_grid, t), params, dt)`;
- provide optional temporal regularisation and diagnostics on the fixed mizer grid;
- reuse the fixed-grid `step(...)` operator rather than approximating this with the continuous PDE residual.

Boundary:

- The timestep-consistency loss intentionally uses the fixed-grid `mizer_grid_ops.step(...)` operator.
- This is an exception to the usual separation between the continuous/off-grid PDE residual path and fixed-grid validation path.
- Treat it as a separate temporal regularisation/validation loss, not as part of the continuous PDE residual.
- Do not document or import it as `PINNmizer/timestep_consistency.py`; the actual implementation is `PINNmizer/pinn/timestep_consistency.py`.

## 4. Slab/time R3 and Causal R3 collocation path

Implemented across:

```text
PINNmizer/pinn/r3.py
PINNmizer/pinn/derivatives.py
PINNmizer/pinn/pde_state.py
PINNmizer/pinn/losses.py
PINNmizer/training/loop.py
PINNmizer/training/train_pde_only_single_species.py
```

Current source uses slab/time R3, not the earlier flat-paired design.

Core geometry:

```text
R3Population.t_points: [K]
R3Population.x_points: [K, M]
batch["t_slab"]: [K]
batch["x_slab"]: [K, M]
batch["w_slab"]: [K, M]
residual: [K, n_species, M]
```

Purpose:

- maintain persistent log-weight points within time slabs;
- score cells by PDE residual magnitude;
- retain high-residual `x_points` and resample low-residual `x_points`;
- resample time slabs every step under the current causal curriculum horizon `t_max_current`;
- optionally apply a smooth Causal R3 gate to scores and/or PDE loss;
- expose R3 as a selectable collocation strategy while preserving uniform Cartesian collocation.

Boundary:

- R3 is not a biological operator.
- R3 should not alter recruitment, growth, mortality, or derivative definitions.
- Current R3 time slabs are resampled every step; only `x_points` persist under retain/resample.
- The old flat-paired performance bottleneck should not be treated as current unless source inspection shows regression.
- The old guard forbidding `R3/Causal R3 + causal_curriculum` is obsolete. Current source uses the ordinary causal curriculum to bound R3 time resampling through `t_max_current`. Causal time truncation and Causal R3 gating still have different meanings.

## Package module map

### `PINNmizer/params.py`

Purpose:

- define `MizerTorchParams`;
- hold grids, FFT/reference tensors, continuous biological parameters, interaction matrices, reproduction parameters, mortality parameters, physical time bounds, active-size vectors, and optional fixture timestep `dt`;
- provide coordinate-scaling, active-mask, dtype/device, and dimension helpers.

Important responsibilities:

- `MizerTorchParams`
- optional `dt: Optional[torch.Tensor] = None`
- `fish_start()`
- `_params_dtype_device()`
- `_to_param_tensor()`
- `_species_vector()`
- `_eval_weight_vector()`
- `_x_grid()` / `_x_limits()`
- `_t_limits()`
- `scale_x()` / `scale_t()`
- `_n_species()` / `_n_w()` / `_k_full()`
- `active_grid_mask()` / `active_eval_mask()`
- `validate_params_shapes()`

Boundary:

- Should not implement biological operators.
- Should not run training.
- Should not contain experiment-specific logic.

### `PINNmizer/io.py`

Purpose:

- load CSV exports from R/mizer/TMB into PyTorch tensors;
- construct `MizerTorchParams`;
- optionally load `dt.csv` into `params.dt`;
- return `(params, n_init, n_pp)`.

Boundary:

- Preserve caller-supplied dtype and device.
- Do not perform biological transformations beyond loading and basic construction.
- Do not run training or diagnostics.

### `PINNmizer/mizer_grid_ops.py`

Purpose:

- implement fixed-grid mizer/TMB-style reference operators;
- provide FFT-based encounter and predation operations;
- provide AD-safe fixed-grid projection/timestep utilities.

Boundary:

- This is the fixed-grid validation/reference path.
- Do not use this as the arbitrary off-grid PDE residual path.
- It is deliberately reused by `PINNmizer/pinn/timestep_consistency.py` for optional fixed-grid temporal consistency.
- Do not place continuous collocation-point biology here.

### `PINNmizer/biology/*`

Purpose:

- compute continuous/off-grid biological quantities at arbitrary physical weights;
- provide analytical/manual derivative chains for growth-side terms;
- compute direct continuous encounter, growth, mortality, and recruitment flux quantities.

Current modules:

- `kernels.py`: predation kernel and kernel derivative terms.
- `encounter.py`: continuous encounter/search terms.
- `growth.py`: intake, metabolism, reproduction allocation, growth, and `dg_dw`.
- `mortality.py`: continuous background and predation mortality.
- `recruitment.py`: recruitment flux from grid-based reproduction output.

Boundary:

- Keep these functions torch-based and differentiable where they feed the PDE loss.
- Keep diagnostic-only detaches out of these modules.
- If a function is only for fixed-grid reference comparison, it belongs in `mizer_grid_ops.py` or diagnostics, not here.

### `PINNmizer/pinn/*`

Purpose:

- sample collocation points;
- evaluate the neural network and autograd derivatives;
- assemble cached PDE state;
- compute PDE residuals and loss terms;
- manage R3/Causal R3 collocation state.

Boundary:

- Preserve model input order `[x_scaled, t_scaled]`.
- Preserve model output as `log_N` unless an explicit decision record changes it.
- Preserve active-size masking in loss and R3 scoring.
- Keep differentiable operations torch-native.

### `PINNmizer/training/*`

Purpose:

- command-line training workflow;
- per-step optimisation;
- adaptive weighting;
- run outputs and checkpoints.

Boundary:

- Do not put biological equations here.
- Do not silently change scientific loss definitions when adding CLI options.
- Training script is currently single-species only.

### `PINNmizer/diagnostics/*`

Purpose:

- deterministic fixed-grid diagnostics;
- output tables and plots;
- final field exports.

Boundary:

- Diagnostics may detach tensors and use pandas/matplotlib.
- Diagnostics should not define the training objective unless explicitly reused by a loss module.
