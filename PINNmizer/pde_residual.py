"""Compatibility re-exports for PINN PDE modules.

Real implementations live in PINNmizer.pinn.*.
"""

from PINNmizer.pinn.derivatives import evaluate_log_model_with_derivatives_at_eval
from PINNmizer.pinn.losses import (
    compute_initial_condition_loss_from_state,
    compute_pde_loss,
    compute_recruitment_boundary_loss_from_state,
)
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.pde_state import compute_pde_state
from PINNmizer.pinn.residual import compute_pde_residual, compute_pde_residual_from_state
from PINNmizer.pinn.sampling import sample_pde_batch

__all__ = [
    "sample_pde_batch",
    "evaluate_log_model_on_points",
    "evaluate_log_model_with_derivatives_at_eval",
    "compute_pde_residual",
    "compute_pde_residual_from_state",
    "compute_initial_condition_loss_from_state",
    "compute_recruitment_boundary_loss_from_state",
    "compute_pde_loss",
    "compute_pde_state",
]
