"""
Growth-side biological functions for the continuous mizer operator.

This module groups the pieces that turn encounter into growth:

- maximum intake h(w);
- metabolism;
- reproduction allocation psi(w);
- feeding level;
- available energy;
- growth energy and dg/dw.

The current public functions are re-exported from the legacy implementation.
The important ownership rule is that `dg_dw` is a biological/manual derivative,
whereas dN/dt and dN/dw are PINN/autograd derivatives.
"""

from PINNmizer.continuous_biology import (
    evaluate_intake_max_continuous,
    evaluate_metab_continuous,
    evaluate_psi_continuous,
    compute_growth_direct_at_eval,
)

__all__ = [
    "evaluate_intake_max_continuous",
    "evaluate_metab_continuous",
    "evaluate_psi_continuous",
    "compute_growth_direct_at_eval",
]
