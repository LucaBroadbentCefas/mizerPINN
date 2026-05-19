"""
Autograd derivative utilities for the PINN residual.

The mizer PDE is written in physical coordinates, but PyTorch autograd first
sees derivatives with respect to the scaled neural-network inputs. The relevant
chain-rule conversions are:

    d/dt = (1 / (t_max - t_min)) d/dt_scaled
    d/dx = (1 / (x_max - x_min)) d/dx_scaled
    d/dw = (1 / w) d/dx

Only neural-network derivatives should live here. Biological derivatives such as
`dg_dw` are hand-coded in the biology modules and should not be obtained from
autograd.
"""

from PINNmizer.pde_residual import evaluate_log_model_with_derivatives_at_eval

__all__ = ["evaluate_log_model_with_derivatives_at_eval"]
