"""
PDE collocation sampling for the PINN.

This module is the thematic home for functions that choose where the PDE
residual is evaluated. The current implementation is re-exported from the legacy
`PINNmizer.pde_residual` module so this refactor does not change numerical
behaviour.

Important coordinate convention
-------------------------------
The residual is evaluated at off-grid physical weights `w_eval = exp(x_eval)`.
The neural network receives `x_eval_scaled`, not `w_eval` directly. Biological
operators receive physical weights.
"""

from PINNmizer.pde_residual import sample_pde_batch

__all__ = ["sample_pde_batch"]
