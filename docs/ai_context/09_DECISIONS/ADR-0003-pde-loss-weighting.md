# ADR-0003: Use Wang-style gradient-statistic weighting for composite PINN losses

## Status

Accepted as current implementation for the single-species training script

## Date

2026-05-18

## Context

The training objective can include PDE residual loss, initial-condition loss, and recruitment-boundary loss. These terms have different scales and can produce very different gradient magnitudes. Equal scalar weights are not meaningful if one component dominates the optimisation signal.

Earlier discussion identified solution collapse and loss imbalance as central risks. The current training script includes Wang-style gradient-statistic weighting as a response.

## Decision

Use PDE loss as the anchor and adapt IC/BC scalar weights from gradient statistics.

Current implementation:

```text
w_pde = 1

target_w_component = max_abs_grad(loss_pde) / mean_abs_grad(loss_component)
```

where component is `ic` or `bc`.

The target is clipped to `[weight_min, weight_max]`. Weight updates can be hard-set on the first update and smoothed afterwards:

```text
w_component = (1 - alpha) * w_component + alpha * target_w_component
```


### Weight batch source for R3 training

Wang adaptive weights can be computed from either the current training batch
(`--wang-weight-batch training`) or a fixed calibration batch
(`--wang-weight-batch fixed`). The default is now the fixed calibration batch.

Reason: R3 deliberately changes the sampled PDE objective during training, so
using R3 collocation points for gradient-statistic weight estimation can make
scalar weights chase sampling noise rather than global component imbalance. R3
remains the training collocation strategy; the fixed batch only affects scalar
loss-weight updates.

Validation should compare `--wang-weight-batch fixed`,
`--wang-weight-batch training`, and `--disable-wang-weights`.

## Current source location

`PINNmizer/training/train_pde_only_single_species.py`

Key functions:

- `_flat_loss_grad()`
- `update_wang_gradient_weights_()`
- `train_one_step()`

## Important current default

The machinery supports PDE, IC, and BC weighting, but the default command-line settings currently include:

```text
lambda_pde = 1.0
lambda_ic = 1.0
lambda_bc = 0.0
```

Therefore, BC machinery is implemented but off by default unless `--lambda-bc` is set.

## Alternatives considered

### Option 1: Fixed equal weights

Pros:

- Simple.
- Easy to reason about mechanically.

Cons:

- Raw loss values and gradient magnitudes are not comparable.
- Can let one constraint dominate.
- Can encourage trivial or collapsed solutions.

### Option 2: Manual scalar weights

Pros:

- Simple to test.
- Useful for controlled ablations.

Cons:

- Brittle across runs.
- Requires repeated tuning.
- Does not directly track optimisation signal.

### Option 3: Loss-value normalisation

Pros:

- Easy to implement.
- Can stop raw-value domination.

Cons:

- Loss magnitude is not the same as gradient influence.
- Can still produce poor optimisation balance.

### Option 4: Wang-style gradient-statistic weighting

Pros:

- Balances based on gradients, closer to optimiser behaviour.
- Relatively compact implementation.
- Directly targets component training signal imbalance.

Cons:

- Requires extra gradient computations.
- Adds hyperparameters: update frequency, warmup, alpha, min/max clipping.
- Can become unstable if one component has tiny or pathological gradients.

### Option 5: NTK-based weighting or GradNorm-style alternatives

Pros:

- Potentially more principled in some settings.
- Useful if Wang-style weighting fails.

Cons:

- More complex.
- Heavier to compute.
- Not the first debugging step.

## Consequences

- Diagnostics must record weights, weighted losses, raw losses, and gradient statistics.
- Wang weights should be compared against a fixed-weight baseline.
- Adaptive weights can mask deeper modelling problems; they do not prove the PDE residual is correct.
- If weighting code grows further, move it to a dedicated module.

## Validation requirement

Any claim that weighting improved training should compare at least:

- Wang weights on/off;
- causal curriculum on/off;
- PDE+IC versus PDE+IC+BC;
- fixed-grid residual diagnostics;
- abundance range and surface plots.
