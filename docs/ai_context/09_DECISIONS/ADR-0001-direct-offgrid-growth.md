# ADR-0001: Use direct off-grid continuous growth for PDE residual

## Status

Accepted as current implementation

## Date

2026-05-18

## Context

The mizer-style growth term is nonlocal. It depends on the community spectrum over prey weights at the same time. The PINN PDE residual is evaluated at arbitrary continuous collocation weights `w_eval = exp(x_eval)`, not only at the fixed mizer fish grid.

A fixed-grid FFT operator is available and useful for matching mizer/TMB outputs, but direct use of fixed-grid interpolated biological quantities inside the off-grid PDE residual can obscure which errors come from biology, interpolation, derivatives, or the neural network.

## Decision

For the PDE residual path, compute growth-side quantities directly at continuous `w_eval` using `PINNmizer/continuous_biology.py`.

The fixed-grid mizer/TMB-style path in `PINNmizer/mizer_grid_ops.py` remains available as a validation/reference path.

## Current implementation

The continuous path computes, at arbitrary `w_eval`:

- `gamma_i(w)`;
- encounter;
- intake maximum;
- feeding level;
- metabolism;
- reproduction allocation;
- available energy for reproduction/growth;
- reproduction energy;
- growth energy;
- `dg_dw`;
- background mortality;
- direct predation mortality;
- recruitment flux from grid growth when needed for boundary loss.

## Alternatives considered

### Option 1: Interpolate all biological quantities from fixed-grid mizer arrays

Pros:

- Simpler to implement.
- Closer to fixed-grid mizer outputs by construction.

Cons:

- Does not cleanly define off-grid biology.
- Makes `dg_dw` depend on interpolation choices.
- Can hide inconsistencies between fixed-grid and continuous equations.

### Option 2: Use fixed-grid FFT operators only

Pros:

- Closer to mizer/TMB implementation.
- Easier to compare against exported reference outputs.

Cons:

- PDE collocation points are continuous/off-grid.
- Requires interpolation or restricting residuals to the fixed grid.
- Less natural for PINN collocation training.

### Option 3: Direct continuous/off-grid biological operators

Pros:

- Gives explicit biology at each residual point.
- Separates PDE-path equations from validation/reference operators.
- Enables manual derivatives for the residual.

Cons:

- More implementation burden.
- Must be validated carefully against fixed-grid/reference outputs.
- Direct quadrature can differ numerically from FFT conventions.

## Consequences

- Validation must compare direct continuous outputs against fixed-grid/mizer outputs at grid points.
- Differences between the two paths should be documented rather than silently ignored.
- Any change to kernel support, integration weights, encounter convention, or mortality convention should be recorded in a new ADR or an update to this one.

## Current unresolved caveat

The current continuous kernel implementation uses an active mask `ppmr > 1`. Earlier discussion included possible upper-support/truncation conventions. This remains a validation issue and should not be treated as settled without comparison against mizer outputs.
