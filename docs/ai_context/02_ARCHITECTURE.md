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
    r3.py                         # planned/minimal R3 and Causal R3 state/sampling module
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

`PINNmizer/pinn/r3.py` is listed as the intended location for the R3/Causal R3 addition. Future assistants must inspect the source before assuming it already exists in the checked-out repository.

## Architectural principle

The project has two biological-computation paths, one deliberate fixed-grid temporal consistency loss, and a planned paired-collocation path for R3. These are related, but they should not be casually merged.

### 1. Fixed-grid mizer/TMB-style path

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

### 2. Continuous/off-grid Cartesian PINN PDE path

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

### 3. Fixed-grid timestep-consistency loss

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

### 4. Planned paired R3 / Causal R3 collocation path

Intended implementation locations:

```text
PINNmizer/pinn/r3.py
PINNmizer/pinn/derivatives.py
PINNmizer/pinn/pde_state.py
PINNmizer/pinn/losses.py
PINNmizer/training/loop.py
PINNmizer/training/train_pde_only_single_species.py
```

Purpose:

- maintain a fixed-size population of paired collocation points `(x_j, t_j)`;
- score points by PDE residual magnitude;
- retain high-residual points and resample low-residual points;
- optionally apply a smooth causal gate to R3 scores and/or PDE loss;
- expose R3 as a selectable training collocation strategy while preserving the original uniform Cartesian strategy.

Boundary:

- R3 is not merely a different loader for `sample_pde_batch()`.
- Exact R3 needs paired tensor geometry and therefore paired derivative, PDE-state, and loss functions.
- The original Cartesian residual shape is `[n_time, n_species, n_eval]`.
- The paired R3 residual shape is `[n_species, n_pair]`.
- A Cartesian approximation to R3 must not be called Daw-style R3; if added, name it explicitly such as `cartesian-r3-approx`.
- Causal R3 should not be silently stacked with the existing causal time curriculum, because the old curriculum truncates the sampled time horizon while Causal R3 gates paired points smoothly.

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
- `recruitment.py`: recruitment boundary flux from grid-based reproduction output.

Boundary:

- Keep these functions torch-based and differentiable where they feed the PDE loss.
- Keep diagnostic-only detaches out of these modules.
- If a function is only for fixed-grid reference comparison, it belongs in `mizer_grid_ops.py` or diagnostics, not here.

### `PINNmizer/pinn/*`

Purpose:

- sample PDE collocation batches;
- maintain optional R3 paired collocation populations once implemented;
- evaluate the neural network on scaled coordinates `[x_scaled, t_scaled]`;
- compute model derivatives with autograd;
- build cached PDE state;
- assemble PDE residuals and loss components;
- provide the optional fixed-grid timestep-consistency loss.

Current and intended modules:

- `sampling.py`: current random Cartesian PDE collocation batches and scaled/physical coordinate vectors.
- `r3.py`: intended minimal R3/Causal R3 population, scoring, retain/resample, and causal gate utilities.
- `model_eval.py`: model input construction and Cartesian grid/eval-point model calls.
- `derivatives.py`: autograd derivatives of model outputs with respect to scaled coordinates, then conversion to physical derivatives. Add paired derivative evaluation here for R3.
- `pde_state.py`: cached state containing model outputs, growth, mortality, recruitment, and optional IC outputs. Add paired PDE state here for R3.
- `residual.py`: PDE residual assembly shared by Cartesian and paired state where possible.
- `losses.py`: PDE loss, initial-condition loss, recruitment-boundary loss, and paired PDE loss for R3 once implemented.
- `timestep_consistency.py`: fixed-grid temporal consistency loss comparing `N_theta(w,t+dt)` with `step(N_theta(w,t), dt)` on the mizer weight grid.

Boundary:

- `PINNmizer/pinn/*` should not contain the full optimiser/training loop.
- It may assemble loss terms, but optimiser steps and experiment orchestration belong under `PINNmizer/training/`.
- This package area should remain the source of truth for residual shape conventions and derivative-scaling conventions.
- `timestep_consistency.py` is not part of the continuous/off-grid PDE residual path, even though it lives under `PINNmizer/pinn/` because it is a neural-network loss term.
- `r3.py` should manage collocation population/state and causal gating; it should not compute biology or perform optimiser steps.

### `PINNmizer/training/*`

Purpose:

- provide reusable training workflow components;
- keep command-line scripts thin;
- coordinate sampling, R3 population state, loss computation, adaptive weighting, optimisation, diagnostics, checkpoints, and output saving.

Current modules:

- `train_pde_only_single_species.py`: package-level single-species training workflow and CLI argument parsing.
- `loop.py`: per-step training logic, including optional timestep-consistency loss integration and, once implemented, a `collocation_strategy` branch for `uniform`, `r3`, and `causal-r3`.
- `weighting.py`: Wang-style gradient-statistics weighting for arbitrary non-PDE scalar losses included in `raw_losses`.
- `config.py`: causal curriculum and configuration helpers.
- `outputs.py`: final training-output exports.
- `checkpointing.py`: checkpoint saving.

Boundary:

- Training code can be pragmatic, but scientific assumptions should be documented.
- Keep reusable training pieces in package modules, not in top-level scripts.
- Keep exploratory or obsolete training scripts under `validation/scripts/legacy/` rather than mixing them into the package.

### `PINNmizer/diagnostics/*`

Purpose:

- compute deterministic diagnostic grids;
- compute fixed-grid diagnostics;
- summarise loss/gradient/residual quantities;
- save diagnostic tables and plots;
- save final field/surface outputs.

Current modules:

- `fixed_grid.py`: deterministic diagnostic batches and fixed-grid diagnostic summaries.
- `metrics.py`: scalar metric helpers and diagnostic gradient summaries.
- `outputs.py`: diagnostic table/file outputs.
- `plots.py`: diagnostic plotting utilities.
- `fields.py`: final fixed-grid fields and plots.

Boundary:

- Package training code should import diagnostics from `PINNmizer.diagnostics.*`.
- Package code should not import diagnostics from `validation/scripts/*`.
- Diagnostics may use pandas, NumPy, matplotlib, and detached tensors.
- Diagnostics should not be required for core differentiable PDE loss assembly unless deliberately designed that way.

## Scripts, validation, docs, and runs

### `scripts/`

Purpose:

- thin executable wrappers only.

Example:

```text
scripts/train_pde_only_single_species.py
```

should mostly do:

```python
from PINNmizer.training.train_pde_only_single_species import main

if __name__ == "__main__":
    main()
```

Boundary:

- Do not put core biological equations here.
- Do not put training-loop implementation here.
- Do not put reusable diagnostics here.

### `validation/`

Purpose:

- fixtures;
- R/mizer export checks;
- fixed-grid comparisons;
- smoke checks;
- generated validation outputs;
- legacy scripts;
- one-off debugging scripts.

Generated validation-output layout:

- Generated validation outputs belong under `validation/outputs/`.
- Root-level `py_*` validation outputs are legacy artifacts and should not be reintroduced.
- Missing root-level folders such as `py_growth_derivative*`, `py_known*`, `py_mizer*`, and `py_pred*` should not be treated as a repository problem.

Boundary:

- `validation/` may import from `PINNmizer/`.
- `PINNmizer/` should not import from `validation/`.
- Code needed by the package should be promoted into `PINNmizer/`, not imported from validation.

### `runs/`

Purpose:

- generated training and training-diagnostic outputs only.

Boundary:

- Do not import from `runs/`.
- Do not require files in `runs/` for package import.
- Keep run artifacts out of source-code APIs.

### `docs/ai_context/`

Purpose:

- project continuity documentation for future AI-assisted work and contributors;
- architecture notes;
- function registries;
- experiment logs;
- prompts and design records.

Boundary:

- Keep this documentation consistent with current code paths.
- Stale architecture notes cause bad future edits.

## Data flow

```text
R/mizer/TMB CSV exports
        |
        v
PINNmizer.io.load_mizer_inputs()
        |
        v
MizerTorchParams + n_init + n_pp + optional params.dt
        |
        +--> fixed-grid reference/validation path
        |       PINNmizer.mizer_grid_ops
        |
        +--> fixed-grid timestep-consistency loss
        |       PINNmizer.pinn.timestep_consistency
        |       N_theta(w_grid, t + dt) versus step(n_pp, N_theta(w_grid, t), params, dt)
        |
        +--> continuous/off-grid Cartesian PINN PDE path
        |       PINNmizer.pinn.sampling.sample_pde_batch()
        |       PINNmizer.pinn.model_eval / derivatives
        |       model([x_scaled, t_scaled]) -> log_N
        |       autograd derivatives wrt scaled coordinates
        |       convert derivatives to physical t and w
        |       PINNmizer.biology computes g, dg_dw, mu, recruitment flux
        |       PINNmizer.pinn.residual assembles PDE residuals
        |       PINNmizer.pinn.losses assembles PDE/IC/BC losses
        |       PINNmizer.training applies weighting and optimiser steps
        |       PINNmizer.diagnostics writes optional diagnostics
        |
        +--> planned paired R3/Causal R3 PINN PDE path
                PINNmizer.pinn.r3 maintains paired population [(x_j,t_j)]
                paired derivatives/state/loss use [n_species, n_pair] residual tensors
                same biology/residual equations, different collocation geometry
                PINNmizer.training.loop branches by collocation_strategy
```

## Where new code should go

- New continuous biological equation or analytical derivative: `PINNmizer/biology/*`.
- Fixed-grid mizer/TMB reference operation: `PINNmizer/mizer_grid_ops.py`.
- PDE residual assembly, derivative-scaling convention, or loss component: `PINNmizer/pinn/*`.
- R3/Causal R3 collocation population and scoring: `PINNmizer/pinn/r3.py`.
- Paired derivative/state/loss support for R3: `PINNmizer/pinn/derivatives.py`, `PINNmizer/pinn/pde_state.py`, and `PINNmizer/pinn/losses.py`.
- Fixed-grid timestep-consistency changes: `PINNmizer/pinn/timestep_consistency.py`, with the fixed-grid operator itself remaining in `PINNmizer/mizer_grid_ops.py`.
- Training-loop logic, adaptive weighting, checkpointing, or CLI-backed training workflow: `PINNmizer/training/*`.
- Thin executable entry point: `scripts/*`.
- Diagnostic table/plot/surface output used by package training workflows: `PINNmizer/diagnostics/*`.
- One-off validation, fixture export, comparison, or legacy debugging script: `validation/scripts/*`.
- Generated validation outputs: `validation/outputs/*`.
- Test/smoke check: `tests/` if a test suite exists, otherwise `validation/scripts/checks/` until tests are formalised.
- Design decision or cross-chat context: `docs/ai_context/`.

## Import rules

- `scripts/*` may import from `PINNmizer/*`.
- `validation/*` may import from `PINNmizer/*`.
- `PINNmizer/*` should not import from `scripts/*`.
- `PINNmizer/*` should not import from `validation/*`.
- `PINNmizer/biology/*` should not depend on plotting, pandas, or run-output code.
- `PINNmizer/pinn/*` should not depend on training-loop orchestration.
- Diagnostics should be optional around training completion: a failed optional plot should not make a successfully completed training run look like a failed optimisation run unless diagnostics were explicitly required.

## Current architectural debt and code-practice priorities

These are known improvement targets, not blockers for current experiments.

1. **State dictionaries are too loose.**
   - `compute_pde_state()` and downstream code pass large nested dictionaries.
   - Prefer dataclasses or `TypedDict` for PDE state, growth outputs, mortality outputs, recruitment outputs, and future paired R3 state.

2. **Loss outputs and diagnostics are mixed.**
   - Core loss functions return many diagnostic fields.
   - Prefer separating minimal loss outputs from expanded diagnostic snapshots.

3. **`losses.py` has too many responsibilities.**
   - IC loss, boundary loss, composite loss, and helper functions currently share one module.
   - Adding paired R3 loss is acceptable as a small extension, but split if further losses accumulate.

4. **Shape checks rely heavily on `assert`.**
   - Prefer explicit shape-check helpers with informative `ValueError` messages for user-facing paths.

5. **Some shared biology/reference utilities need clearer ownership.**
   - If continuous biology and fixed-grid operators both need the same prey-construction logic, consider moving that logic to a neutral shared module.

6. **Fixture IO needs a clearer contract.**
   - `load_mizer_inputs()` expects a large set of CSV files.
   - Add a fixture manifest and clearer missing-file messages when the export format stabilises.

7. **Architecture docs must stay current.**
   - When moving files between `validation/`, `scripts/`, `PINNmizer/`, and generated-output locations, update this document and the function registry.

8. **Timestep-consistency boundary must remain explicit.**
   - The timestep loss deliberately crosses from a neural-network loss into the fixed-grid `step(...)` operator.
   - Future edits should preserve the distinction between this fixed-grid temporal regulariser and the continuous/off-grid PDE residual.

9. **R3/Causal R3 must remain explicit about paired geometry.**
   - R3 is not equivalent to retaining separate `x_eval` and `t_eval` vectors.
   - Paired collocation should keep `(x_j,t_j)` together through derivative evaluation, PDE state, residual scoring, and retain/resample updates.

## Recent issue to avoid repeating

A previous successful training run was followed by a crash from a stale post-run diagnostics import:

```text
ModuleNotFoundError: No module named 'validation.scripts.pde_output_diagnostics'
```

The lesson is structural: package training code should not import post-run diagnostics from `validation/scripts/*`. If a diagnostic is part of the package training workflow, place it under `PINNmizer/diagnostics/` and import it from there. If a diagnostic is optional, guard only the optional diagnostic import/call and report a clear warning without hiding unrelated diagnostic errors.

A separate documentation issue is resolved: root-level `py_*` validation-output folders are legacy generated artifacts. Future sessions should expect generated validation outputs under `validation/outputs/` instead.
