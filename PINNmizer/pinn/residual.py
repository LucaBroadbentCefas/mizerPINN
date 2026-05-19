from __future__ import annotations

"""PDE residual assembly.

dN_i/dt + g_i dN_i/dw + (mu_i + dg_i/dw) N_i = 0
and log-form dlogN_i/dt + g_i dlogN_i/dw + mu_i + dg_i/dw = 0.
"""

import torch

from PINNmizer.params import MizerTorchParams
from PINNmizer.pinn.pde_state import compute_pde_state


def compute_pde_residual(model, batch: dict[str, torch.Tensor], params: MizerTorchParams, n_pp: torch.Tensor) -> dict[str, torch.Tensor]:
    state = compute_pde_state(model=model, batch=batch, params=params, n_pp=n_pp, include_ic=False)
    return compute_pde_residual_from_state(state)


def compute_pde_residual_from_state(state: dict[str, object]) -> dict[str, torch.Tensor]:
    eval_derivs = state["eval_derivs"]
    growth = state["growth_eval"]
    mortality = state["mortality"]
    log_N_eval, N_eval = eval_derivs["log_N_eval"], eval_derivs["N_eval"]
    dlogN_dt, dlogN_dw = eval_derivs["dlogN_dt"], eval_derivs["dlogN_dw"]
    dN_dt, dN_dw = eval_derivs["dN_dt"], eval_derivs["dN_dw"]
    g_eval, dg_dw, mu_eval = growth["e_growth_eval"], growth["dg_dw"], mortality["mu_eval"]
    residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
    residual = N_eval * residual_log
    residual_physical_check = dN_dt + g_eval * dN_dw + (mu_eval + dg_dw) * N_eval
    out = {"residual": residual, "residual_log": residual_log, "residual_physical_check": residual_physical_check, "log_N_eval": log_N_eval, "log_N_grid": state["log_N_grid"], "N_eval": N_eval, "N_grid": state["N_grid"], "dlogN_dt": dlogN_dt, "dlogN_dw": dlogN_dw, "dN_dt": dN_dt, "dN_dw": dN_dw, "g_eval": g_eval, "dg_dw": dg_dw, "mu_eval": mu_eval}
    out.update({f"growth_eval_{k}": v for k, v in growth.items()})
    out.update({f"growth_grid_{k}": v for k, v in state["growth_grid"].items()})
    out.update({f"mortality_{k}": v for k, v in mortality.items()})
    out.update({f"recruitment_{k}": v for k, v in state["recruitment"].items()})
    return out
