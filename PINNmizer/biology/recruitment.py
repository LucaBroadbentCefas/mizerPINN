"""
Recruitment-flux functions for the continuous mizer operator.

Recruitment here is treated as a boundary flux used by the PINN boundary loss,
not as a finite-difference grid-cell update. The function operates on the
current predicted fish spectrum and cached growth quantities.

The implementation is re-exported from the legacy continuous biology module to
avoid changing the numerical path in this structural PR.
"""

from PINNmizer.continuous_biology import compute_recruitment_direct_from_growth_grid

__all__ = ["compute_recruitment_direct_from_growth_grid"]
