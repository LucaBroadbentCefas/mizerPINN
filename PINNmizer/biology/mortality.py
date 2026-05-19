"""
Mortality functions for continuous/off-grid mizer biology.

This module groups background and predation mortality terms used by the PDE
residual. Mortality is evaluated at physical weights and returns tensors shaped
[n_species, n_eval].

The public functions are re-exported from the legacy implementation so this PR
only changes structure and documentation, not the mortality calculation.
"""

from PINNmizer.continuous_biology import (
    evaluate_mu_b_continuous,
    compute_pred_mortality_direct_at_eval,
    compute_total_mortality_direct_at_eval,
    compute_pred_mortality_direct_at_eval_from_growth_grid,
    compute_total_mortality_direct_at_eval_from_growth_grid,
)

__all__ = [
    "evaluate_mu_b_continuous",
    "compute_pred_mortality_direct_at_eval",
    "compute_total_mortality_direct_at_eval",
    "compute_pred_mortality_direct_at_eval_from_growth_grid",
    "compute_total_mortality_direct_at_eval_from_growth_grid",
]
