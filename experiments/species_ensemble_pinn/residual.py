from __future__ import annotations

import torch

from .residual_scale import interpolate_log_residual_scale


def compute_pde_residual_from_state(state: dict[str, object], params, *, species_idx: int) -> dict[str, torch.Tensor]:
    derivatives = state["eval_derivs"]
    growth = state["growth_eval"]
    mortality = state["mortality"]
    log_n, n = derivatives["log_N_eval"], derivatives["N_eval"]
    dlogn_dt, dlogn_dw = derivatives["dlogN_dt"], derivatives["dlogN_dw"]
    dn_dt, dn_dw = derivatives["dN_dt"], derivatives["dN_dw"]
    g, dg_dw = growth["e_growth_eval"], growth["dg_dw"]
    mu = mortality["mu_eval"]
    residual_log = dlogn_dt + g * dlogn_dw + mu + dg_dw
    residual_physical = n * residual_log
    residual_physical_check = dn_dt + g * dn_dw + (mu + dg_dw) * n
    log_scale_all, scale_all = interpolate_log_residual_scale(params, state["batch"]["w_eval"])
    log_scale = log_scale_all[species_idx:species_idx + 1]
    scale = scale_all[species_idx:species_idx + 1]
    log_scale = log_scale[None, :, :].expand(log_n.shape[0], -1, -1).detach()
    scale = scale[None, :, :].expand(log_n.shape[0], -1, -1).detach()
    n_over_scale = torch.exp(log_n - log_scale)
    residual_reference_scaled = n_over_scale * residual_log
    return {
        "residual_log": residual_log,
        "residual_physical": residual_physical,
        "residual_physical_check": residual_physical_check,
        "residual_reference_scaled": residual_reference_scaled,
        "log_reference_scale_eval": log_scale,
        "reference_scale_eval": scale,
        "N_over_reference_scale": n_over_scale,
        "log_N_eval": log_n,
        "N_eval": n,
        "dlogN_dt": dlogn_dt,
        "dlogN_dw": dlogn_dw,
        "dN_dt": dn_dt,
        "dN_dw": dn_dw,
        "g_eval": g,
        "dg_dw": dg_dw,
        "mu_eval": mu,
        "mu_b_eval": mortality["mu_b_eval"],
        "pred_mort_eval": mortality["pred_mort_eval"],
        "f_mort_eval": mortality["f_mort_eval"],
    }
