# ADR-0004: Power-law reference scale for log-U state

- Status: Proposed / experimental
- Date: 2026-09-21
- PR: #50

## Context

The working PINN uses

[
N_i(t,w) = S_i(w) U_i(t,w)
]

with the network predicting `log U`. The legacy implementation sets
`S_i(w) = N_i(0,w)`. This makes `U_i(0,w)=1` everywhere and was important
for optimisation, but it also embeds the complete known initial size-spectrum
shape in the state parameterisation. In no-PDE/data-only ablations this can
make unobserved sizes look artificially well reconstructed.

Direct `log N` with the log residual was used extensively before log-U and
did not provide the same fitting performance. The aim is therefore to retain
log-U preconditioning while replacing the detailed IC-shaped reference.

## Proposed formulation

Use a generic power-law reference

[
S_i(w) = C_i left(rac{w}{w_{ref}}ight)^{-p}.
]

Default experimental values:

- `p = 2.05`
- `w_ref = 1 g`

The exponent is a generic Sheldon/community size-spectrum reference used by
mizer. It is not claimed to be the exact spectrum slope of each species;
species slopes depend on growth and mortality and curve around maturation.

Only the scalar amplitude `C_i` is obtained from the initial condition:

[
log C_i =
operatorname{mean}_{win active}
left[log N_i(0,w) + plog(w/w_{ref})ight].
]

Thus the initial condition contributes one scalar abundance scale per species,
not its local size-dependent shape. This choice centres the active-grid
`log U_i(0,w)` at zero.

The existing scaled residual remains

[
R_{scaled} = rac{R_{physical}}{S_i(w)}
           = U_i R_{log N}.
]

All biological operators continue to receive reconstructed physical
`N = S U`.

## Compatibility

Legacy `S=N(0,w)` remains the default CLI behaviour so historical
`run_command.txt` files retain their meaning. The experimental formulation is
selected explicitly with:

```
--state-parameterization log-u \
--state-scale-source power-law \
--state-scale-power 2.05 \
--state-scale-reference-weight 1.0
```

The final-suite shell script exposes the same selection through environment
variables while defaulting to the legacy source.

Checkpoint loading rejects incompatible log-U state-scale metadata.

## Validation required before acceptance

1. Unit tests for power-law slope and analytic derivative
   `d log S / dw = -p/w`.
2. Confirm physical reconstruction `N=S U` and mizer operator baselines remain
   unchanged for a supplied physical state.
3. Run matched legacy-vs-power-law log-U fits on at least the representative
   single-species and multispecies cases.
4. Compare initial and trained distributions of `U` and `R_scaled`; the
   power-law reference may leave materially larger pointwise `U` variation
   than the legacy IC-shaped scale.
5. Re-run the no-PDE control to determine whether the previous near-perfect
   unobserved-state reconstruction disappears.

## Decision

Do not replace the legacy log-U formulation until the matched validation above
shows that the generic reference retains the optimisation benefit.
