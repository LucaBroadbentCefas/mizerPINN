# PINNmizer Project Brief

## Purpose

This repository develops a PyTorch PINN workflow for a mizer-style size-spectrum model. The immediate aim is to make the PDE residual, biological operators, training diagnostics, and experimental decisions explicit enough that a new ChatGPT session or a future contributor can recover the current project state quickly.

This document is a high-level entry point. More detailed conventions are in the other files in `docs/ai_context/`.

## Core modelling target

For species `i`, the continuous size-spectrum PDE currently used by the PINN is:

```text
dN_i/dt + g_i(w,t) dN_i/dw + [mu_i(w,t) + dg_i/dw] N_i(w,t) = 0
```

The implemented log-form residual is:

```text
dlogN_i/dt + g_i(w,t) dlogN_i/dw + mu_i(w,t) + dg_i/dw = 0
```

The physical residual is also computed as a check:

```text
dN_i/dt + g_i(w,t) dN_i/dw + [mu_i(w,t) + dg_i/dw] N_i(w,t)
```

## Current implementation status on `main`

Current source inspection of `main` shows:

- `PINNmizer/params.py` defines `MizerTorchParams`, grid helpers, scaling helpers, and shape validation.
- `PINNmizer/io.py` loads exported mizer/TMB CSV inputs into tensors.
- `PINNmizer/mizer_grid_ops.py` contains fixed-grid FFT-style mizer operators used as validation/reference machinery.
- `PINNmizer/continuous_biology.py` contains the continuous/off-grid biological path used by the PDE residual.
- `PINNmizer/pde_residual.py` samples PDE batches, evaluates the model and derivatives, assembles PDE/IC/recruitment-boundary losses, and returns diagnostics.
- `validation_steps/train_pde_only_single_species.py` contains the current single-species training script with Wang-style gradient-statistic loss weighting, causal time curriculum, diagnostics, checkpointing, and final-output saving.
- `validation_steps/pinn_diagnostics.py` contains deterministic fixed-grid diagnostics, gradient diagnostics, and plotting/export utilities.

## Neural-network convention

The active training script defines an MLP with:

- input dimension 2;
- input columns `[x_scaled, t_scaled]`;
- one `nn.Tanh()` activation after each hidden `nn.Linear` layer;
- output dimension equal to the number of species;
- model output interpreted as `log_N`.

## Coordinate convention

```text
x = log(w)
w = exp(x)
x_scaled = (x - x_min) / (x_max - x_min)
t_scaled = (t - t_min) / (t_max - t_min)
```

Autograd gives derivatives with respect to scaled coordinates. The PDE code converts these to physical derivatives before assembling the residual.

## Differentiation convention

- `dlogN_dt`, `dlogN_dw`, `dN_dt`, and `dN_dw` come from PyTorch autograd through the neural network.
- `g_eval` and `dg_dw` come from the continuous biological path.
- The current biological derivative path for `dg_dw` is analytical/manual rather than obtained by differentiating `g` through autograd.
- The differentiable loss path should remain PyTorch-only.

## Biological-operator convention

The PDE path currently uses continuous/direct functions for:

- encounter;
- search/predation prefactor `gamma_i(w)`;
- intake maximum `h_i(w)`;
- metabolism;
- reproduction allocation `psi_i(w)`;
- growth `g_i(w,t)`;
- direct predation mortality;
- background mortality;
- recruitment flux used in the boundary loss.

The fixed-grid FFT-style mizer operators remain important as validation/reference machinery, not as the preferred off-grid PDE residual path.

## Current training convention

The current single-species training script supports:

- PDE loss;
- initial-condition loss;
- recruitment-boundary loss;
- log or physical PDE residual form;
- log, physical, or relative boundary-loss form;
- Wang-style gradient-statistic adaptive weighting;
- optional disabling of Wang weights;
- causal time curriculum over the sampled PDE time horizon;
- final-layer bias initialisation from the initial condition;
- fixed-grid diagnostics and output surface diagnostics.

## Non-negotiable project rules

- Do not infer file structure from memory when repository files are available.
- Do not rewrite broad source files when a small patch is enough.
- Do not change biological equations while implementing unrelated training or documentation changes.
- Do not introduce NumPy into the differentiable PDE loss path.
- Do not detach tensors in the PDE loss path unless the reason is explicit and documented.
- Preserve dtype/device consistency.
- Preserve tensor shape contracts documented in `04_DATA_AND_SHAPES.md`.
- Treat this documentation as a working context layer, not as a replacement for Git history or tests.

## Scope boundary

This AI context layer is intended to help with project continuity across chats. It is not a formal scientific manuscript, not a substitute for unit tests, and not a guarantee that the current numerical implementation is biologically or mathematically correct.
