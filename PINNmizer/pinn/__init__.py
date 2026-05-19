"""
PINN-side components for the mizer-style size-spectrum model.

This package is for neural-network evaluation, coordinate handling, autograd
quantities, PDE state construction, residual assembly, and PINN losses. It should
not contain mizer biological operators or runnable experiment scripts.

The split is intentionally thematic:
- `models` defines neural-network architectures.
- `sampling` creates PDE collocation batches.
- `model_eval` evaluates the network on scaled coordinates.
- `pde_state` builds the cached state used by residuals and losses.
- `residual` assembles the PDE residual from an already-built state.
- `losses` assembles PDE, initial-condition, and boundary-condition losses.
"""
