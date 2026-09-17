from __future__ import annotations

import torch


def compute_residual_from_fields(state: dict[str, torch.Tensor], biology: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute log, physical, and U-scaled residuals for N=S U."""
    g = biology["g"]; mu = biology["mu_total"]; dg_dw = biology["dg_dw"]
    residual_log = state["dlogU_dt"] + g * (state["dlogU_dw"] + state["dlogS_dw"]) + mu + dg_dw
    residual_physical = state["N_eval"] * residual_log
    residual_physical_check = state["dN_dt"] + g * state["dN_dw"] + (mu + dg_dw) * state["N_eval"]
    residual_scaled = state["U_eval"] * residual_log
    residual_scaled_check = state["dU_dt"] + g * state["dU_dw"] + (g * state["dlogS_dw"] + mu + dg_dw) * state["U_eval"]
    return {"residual_log": residual_log, "residual_physical": residual_physical, "residual_physical_check": residual_physical_check, "residual_scaled": residual_scaled, "residual_scaled_check": residual_scaled_check}
