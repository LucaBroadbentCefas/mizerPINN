"""
PDE residual assembly for the mizer-style size-spectrum equation.

For species i, the physical residual is

    dN_i/dt + g_i dN_i/dw + (mu_i + dg_i/dw) N_i = 0

The log-state form used for numerical scaling is

    dlogN_i/dt + g_i dlogN_i/dw + mu_i + dg_i/dw = 0

This module should only assemble residuals from an already-built PDE state. It
should not sample collocation points, evaluate the neural network, or compute the
biology terms. The current implementation is re-exported from the legacy module
for compatibility.
"""

from PINNmizer.pde_residual import (
    compute_pde_residual,
    compute_pde_residual_from_state,
)

__all__ = ["compute_pde_residual", "compute_pde_residual_from_state"]
