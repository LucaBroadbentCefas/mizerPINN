from __future__ import annotations

"""PDE residual assembly.

dN_i/dt + g_i dN_i/dw + (mu_i + dg_i/dw) N_i = 0
and log-form dlogN_i/dt + g_i dlogN_i/dw + mu_i + dg_i/dw = 0.
"""

import torch

from PINNmizer.params import MizerTorchParams
from PINNmizer.pinn.pde_state import compute_pde_state
from PINNmizer.pinn.residual_scale import interpolate_log_residual_scale


def compute_pde_residual(model, batch: dict[str, torch.Tensor], params: MizerTorchParams, n_pp: torch.Tensor) -> dict[str, torch.Tensor]:
    state = compute_pde_state(model=model, batch=batch, params=params, n_pp=n_pp, include_ic=False)
    return compute_pde_residual_from_state(state, params)


def _reference_scale_for_state(state: dict[str, object], params: MizerTorchParams, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = state.get("batch", {})
    if "w_eval" in batch:
        log_s, s = interpolate_log_residual_scale(params, batch["w_eval"])
        log_s = log_s.unsqueeze(0)
        s = s.unsqueeze(0)
    elif "w_pair" in batch:
        log_s, s = interpolate_log_residual_scale(params, batch["w_pair"])
    elif "w_slab" in batch:
        log_s, s = interpolate_log_residual_scale(params, batch["w_slab"])
        log_s = log_s.permute(1, 0, 2).contiguous()
        s = s.permute(1, 0, 2).contiguous()
    else:
        raise ValueError("Cannot determine residual reference scale: batch lacks w_eval, w_pair, or w_slab.")
    log_s = log_s.to(dtype=target.dtype, device=target.device).detach()
    s = s.to(dtype=target.dtype, device=target.device).detach()
    if log_s.shape != target.shape:
        try:
            log_s = log_s.expand_as(target)
            s = s.expand_as(target)
        except RuntimeError as exc:
            raise ValueError(f"Reference scale shape {tuple(log_s.shape)} is not compatible with target shape {tuple(target.shape)}.") from exc
    return log_s.detach(), s.detach()


def compute_pde_residual_from_state(state: dict[str, object], params: MizerTorchParams) -> dict[str, torch.Tensor]:
    eval_derivs = state["eval_derivs"]
    growth = state["growth_eval"]
    mortality = state["mortality"]
    log_N_eval, N_eval = eval_derivs["log_N_eval"], eval_derivs["N_eval"]
    dlogN_dt, dlogN_dw = eval_derivs["dlogN_dt"], eval_derivs["dlogN_dw"]
    dN_dt, dN_dw = eval_derivs["dN_dt"], eval_derivs["dN_dw"]
    g_eval, dg_dw, mu_eval = growth["e_growth_eval"], growth["dg_dw"], mortality["mu_eval"]
    residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
    residual = N_eval * residual_log
    S_eval = eval_derivs.get("S_eval", torch.ones_like(N_eval))
    residual_scaled = residual / S_eval
    residual_physical_check = dN_dt + g_eval * dN_dw + (mu_eval + dg_dw) * N_eval
    log_reference_scale_eval, reference_scale_eval = _reference_scale_for_state(state, params, log_N_eval)
    N_over_reference_scale = torch.exp(log_N_eval - log_reference_scale_eval)
    residual_reference_scaled = N_over_reference_scale * residual_log
    out = {"residual": residual, "residual_log": residual_log, "residual_physical_check": residual_physical_check, "residual_scaled": residual_scaled, "residual_reference_scaled": residual_reference_scaled, "reference_scale_eval": reference_scale_eval, "log_reference_scale_eval": log_reference_scale_eval, "N_over_reference_scale": N_over_reference_scale, "log_N_eval": log_N_eval, "log_N_grid": state["log_N_grid"], "N_eval": N_eval, "N_grid": state["N_grid"], "dlogN_dt": dlogN_dt, "dlogN_dw": dlogN_dw, "dN_dt": dN_dt, "dN_dw": dN_dw, "g_eval": g_eval, "dg_dw": dg_dw, "mu_eval": mu_eval}
    out.update({f"growth_eval_{k}": v for k, v in growth.items()})
    out.update({f"growth_grid_{k}": v for k, v in state["growth_grid"].items()})
    out.update({f"mortality_{k}": v for k, v in mortality.items()})
    out.update({f"recruitment_{k}": v for k, v in state["recruitment"].items()})
    return out
