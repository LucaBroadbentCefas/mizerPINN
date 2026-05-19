"""
Predation-kernel functions for the continuous mizer operators.

The log-normal kernel describes feeding preference as a function of predator and
prey mass. The current implementation is re-exported from the legacy continuous
biology module to avoid changing numerical results in this structural PR.

Expected kernel tensors have shape:

    phi:          [n_species, n_pred, n_prey]
    dphi_dw_pred: [n_species, n_pred, n_prey]

The derivative is with respect to predator physical mass `w_pred`, not scaled
PINN input coordinates.
"""

from PINNmizer.continuous_biology import compute_phi_and_dphi_dw

__all__ = ["compute_phi_and_dphi_dw"]
