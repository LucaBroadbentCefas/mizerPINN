# ADR-0002: Compute `dg_dw` manually in the biological path

## Status

Accepted as current implementation

## Date

2026-05-18

## Context

The PDE residual requires `dg_i/dw`. The growth term `g_i(w,t)` is not a simple pointwise neural-network output. It is a biological quantity built from encounter, feeding, metabolism, reproduction allocation, and the positive part of available energy.

Using autograd through all biological operations is possible in principle, but the project has deliberately separated:

- neural-network derivatives, which come from autograd;
- biological growth derivatives, which are computed analytically/manually.

## Decision

Use manual/analytical derivatives for the biological growth-side derivative `dg_dw` in the PDE residual path.

Autograd remains responsible for:

- `dlogN_dt`;
- `dlogN_dw`;
- `dN_dt`;
- `dN_dw`.

Manual biological derivatives remain responsible for:

- `dgamma_dw`;
- `dphi_dw`;
- `dencounter_dw`;
- `dh_dw`;
- `dfeeding_dw`;
- `dmetab_dw`;
- `dpsi_dw`;
- `derepog_dw`;
- `dpos_erepog_dw`;
- `de_repro_dw`;
- `dg_dw`.

## Reason

This gives explicit control over the PDE term and makes the biological derivative chain inspectable. That is preferable while debugging equation conventions and validating against mizer.

## Current derivative chain

```text
E = encounter
h = intake_max
f = E / (E + h + eps)

erepog = alpha * (1 - f) * E - metab
pos_erepog = max(erepog, 0)
e_repro = pos_erepog * psi
g = e_growth = pos_erepog - e_repro
```

Derivative chain:

```text
df/dw = [dE/dw * (h + eps) - E * dh/dw] / (E + h + eps)^2

derepog/dw = alpha * ((1 - f) * dE/dw - E * df/dw) - dmetab/dw

dpos_erepog/dw = derepog/dw if erepog > 0 else 0

de_repro/dw = dpos_erepog/dw * psi + pos_erepog * dpsi/dw

dg/dw = dpos_erepog/dw - de_repro/dw
```

Equivalent final expression:

```text
dg/dw = (1 - psi) * dpos_erepog/dw - pos_erepog * dpsi/dw
```

## Alternatives considered

### Option 1: Autograd through all biological calculations

Pros:

- Less manual derivative code.
- Reduces algebraic derivative mistakes if implementation is smooth.

Cons:

- Harder to inspect and compare derivative components.
- Piecewise functions and masks still need explicit choices.
- Can hide whether the derivative matches the intended biological equation.

### Option 2: Finite-difference `dg_dw`

Pros:

- Simple for checking.
- Useful as validation.

Cons:

- Too slow/noisy for training.
- Sensitive to finite-difference step size.
- Poor near discontinuities or kinks.

### Option 3: Manual derivative chain

Pros:

- Transparent.
- Matches mathematical derivation directly.
- Easier to validate component by component.

Cons:

- More code.
- More risk of algebraic mistakes.
- Requires finite-difference validation.

## Consequences

- Manual derivative tests are mandatory for changes touching growth biology.
- Piecewise boundaries must be treated carefully.
- Any change to `psi`, kernel support, positive-part logic, or feeding formula requires rechecking `dg_dw`.

## Validation requirement

Before treating a training result as evidence, validate `dg_dw` against finite differences away from nondifferentiable points.
