"""
PDE state construction for the mizer-style PINN.

A PDE state is the cached bridge between the neural network and the biological
operators. It contains:

- off-grid model values and autograd derivatives used in the residual;
- fixed-grid model values used by nonlocal biological operators;
- growth, mortality, and recruitment terms computed from the predicted spectrum;
- optional initial-condition values at t_min.

Shape convention
----------------
Most state tensors use time-major order:

    [n_time, n_species, n_eval] for off-grid residual points
    [n_time, n_species, n_w]    for fixed mizer-grid values

This module re-exports the current implementation from `PINNmizer.pde_residual`
so this structural PR does not alter biological or numerical behaviour.
"""

from PINNmizer.pde_residual import compute_pde_state

__all__ = ["compute_pde_state"]
