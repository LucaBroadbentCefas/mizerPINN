"""
Loss assembly for the mizer-style PINN.

This module is the thematic home for losses built from PDE state objects:

- PDE residual loss;
- initial-condition loss at t_min;
- recruitment boundary-condition loss;
- combined loss assembly.

The biological equation, neural-network evaluation, and batch sampling are kept
outside this module. This separation makes it clearer whether a future change is
mathematical, biological, numerical, or training-related.
"""

from PINNmizer.pde_residual import (
    compute_initial_condition_loss_from_state,
    compute_recruitment_boundary_loss_from_state,
    compute_pde_loss,
)

__all__ = [
    "compute_initial_condition_loss_from_state",
    "compute_recruitment_boundary_loss_from_state",
    "compute_pde_loss",
]
