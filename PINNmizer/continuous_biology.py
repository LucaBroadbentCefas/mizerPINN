"""Compatibility re-exports for continuous biology operators.

Real implementations live in PINNmizer.biology.*.
"""

from PINNmizer.biology.encounter import compute_encounter_direct_at_eval, evaluate_gamma_continuous
from PINNmizer.biology.growth import (
    compute_growth_direct_at_eval,
    evaluate_intake_max_continuous,
    evaluate_metab_continuous,
    evaluate_psi_continuous,
)
from PINNmizer.biology.kernels import compute_phi_and_dphi_dw
from PINNmizer.biology.mortality import (
    compute_pred_mortality_direct_at_eval,
    compute_pred_mortality_direct_at_eval_from_growth_grid,
    compute_total_mortality_direct_at_eval,
    compute_total_mortality_direct_at_eval_from_growth_grid,
    evaluate_mu_b_continuous,
)
from PINNmizer.biology.recruitment import compute_recruitment_direct_from_growth_grid

__all__ = [
    "compute_phi_and_dphi_dw",
    "compute_encounter_direct_at_eval",
    "evaluate_gamma_continuous",
    "evaluate_intake_max_continuous",
    "evaluate_metab_continuous",
    "evaluate_psi_continuous",
    "compute_growth_direct_at_eval",
    "evaluate_mu_b_continuous",
    "compute_pred_mortality_direct_at_eval",
    "compute_total_mortality_direct_at_eval",
    "compute_pred_mortality_direct_at_eval_from_growth_grid",
    "compute_total_mortality_direct_at_eval_from_growth_grid",
    "compute_recruitment_direct_from_growth_grid",
]
