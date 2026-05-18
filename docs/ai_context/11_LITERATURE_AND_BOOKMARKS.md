# Literature and Bookmarks

## Purpose

This file records research directions and literature categories relevant to the PINNmizer project. It is not yet a complete bibliography.

Use this as a parking lot for ideas, not as proof that an approach is correct.

## PINN loss balancing

### Wang-style gradient-statistic weighting

Why relevant:

- The current training script uses this family of idea.
- It targets imbalance between PDE, initial-condition, and boundary-condition training signals.
- It is directly connected to observed loss imbalance and near-zero/collapsed solutions.

Project use:

- Current implementation anchors PDE weight at 1 and adapts IC/BC weights using gradient statistics.
- Must be compared against fixed-weight baselines.

Open checks:

- Does it improve fixed-grid residual diagnostics, or only sampled training loss?
- Does it reduce trivial solution risk?
- Does it destabilise when component gradients are very small?

### GradNorm-style multitask weighting

Why relevant:

- PINN composite losses are effectively multitask objectives.
- GradNorm-like methods may provide an alternative when Wang-style weighting fails.

Project use:

- Candidate alternative, not current implementation.

### NTK-based PINN weighting

Why relevant:

- More theoretically motivated loss weighting approaches exist for PINNs.
- Could be useful if gradient-statistic weighting is insufficient.

Project use:

- Later-stage candidate because implementation and computation are heavier.

## Temporal and causal PINNs

### Causal training / time-marching curricula

Why relevant:

- The project has shown concern that the model may not propagate time dependence correctly.
- The current training script includes a causal time curriculum over `t_max_current`.

Project use:

- Current implementation supports `off`, `linear`, and `step` modes.
- Needs ablation against full-domain sampling.

Open checks:

- Does the model learn early times first and then extend correctly?
- Does curriculum merely hide late-time residual failures?
- Does it interact badly with adaptive loss weighting?

## PINN trivial-solution and collapse problems

Why relevant:

- PDE residuals can be satisfied by undesirable or weakly constrained solutions.
- Your project has an explicit risk of near-zero abundance collapse.

Project use:

- Use IC/BC losses, trajectory anchors, causal curriculum, and diagnostics.
- Do not rely on PDE residual alone.

Potential avenues:

- Add supervised anchor points from mizer trajectories.
- Use sequential/curricular time-domain training.
- Non-dimensionalise or rescale residual terms.
- Add conservation or integral constraints if biologically justified.
- Use adaptive sampling near high-residual regions.

## Size-spectrum and mizer references

Why relevant:

- The PINN is based on mizer-style size-spectrum equations.
- Biological equations and conventions should be grounded in mizer/TMB/marine ecosystem modelling literature.

Project use:

- Keep mizer-specific conventions separate from generic PINN practice.
- Record any deviations from mizer equations in ADRs.

Needed additions:

- mizer core paper/package references.
- mizer size-spectrum equation references.
- TMB/mizerTMB implementation references if using as source of truth.

## Numerical validation references

Why relevant:

- The project mixes continuous PDE residuals, fixed-grid mizer operators, FFT convolution, direct quadrature, manual derivatives, and neural-network derivatives.

Project use:

- Finite-difference derivative checks.
- Grid-convergence checks.
- Direct-vs-FFT biological operator comparisons.
- PDE residual checks on known trajectories.

Needed additions:

- References on method-of-lines or finite-volume validation for size-spectrum PDEs.
- References on residual validation for PINNs on known numerical solutions.

## Bookmarks to maintain

Add links or citations under these headings as the project develops:

```text
PINN gradient pathologies / loss balancing
PINN causal training / temporal curriculum
PINN trivial-solution failure modes
Adaptive collocation sampling
Size-spectrum PDE references
mizer implementation references
Manual derivative / finite-difference validation notes
```

## Rule for future citation additions

When adding a paper or source, include:

```text
- Full citation or stable link
- One-line why it matters
- Which project decision or issue it informs
- Whether it is background, method, or direct implementation support
```
