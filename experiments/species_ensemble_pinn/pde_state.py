from __future__ import annotations

import torch

from PINNmizer.params import _params_dtype_device, _t_limits, scale_t

from .biology_adapter import compute_environment_biology
from .derivatives import evaluate_log_model_with_derivatives_at_eval
from .model_eval import evaluate_log_model_on_points
from .recruitment import compute_hybrid_recruitment


def compute_pde_state(model, batch: dict[str, torch.Tensor], params, n_init: torch.Tensor,
                      n_pp: torch.Tensor, known_state, *, species_idx: int,
                      include_ic: bool = True) -> dict[str, object]:
    dtype, device = _params_dtype_device(params)
    t_eval = batch["t_eval"].to(dtype=dtype, device=device)
    eval_derivs = evaluate_log_model_with_derivatives_at_eval(
        model, batch["x_eval_scaled"], batch["t_scaled"], batch["w_eval"], params
    )
    t_grid_scaled = batch["t_scaled"]
    if include_ic:
        t_min, _ = _t_limits(params)
        t0 = torch.as_tensor(t_min, dtype=dtype, device=device).reshape(1)
        t_grid_scaled = torch.cat([t_grid_scaled, scale_t(t0, params)])
    grid_all = evaluate_log_model_on_points(model, batch["x_grid_scaled"], t_grid_scaled, params)
    n_time = t_eval.numel()
    log_n_grid, n_grid = grid_all["log_N"][:n_time], grid_all["N"][:n_time]
    if include_ic:
        log_n_ic, n_ic = grid_all["log_N"][n_time:], grid_all["N"][n_time:]
    else:
        log_n_ic = n_ic = None
    environment = known_state.at(t_eval)
    biology = compute_environment_biology(
        N_environment_grid=environment, n_pp=n_pp, w_eval=batch["w_eval"],
        t_eval=t_eval, params=params, species_idx=species_idx,
    )
    recruitment = compute_hybrid_recruitment(
        N_target_grid=n_grid,
        growth_grid_environment=biology["growth_grid"],
        params=params,
        species_idx=species_idx,
    )
    return {
        "batch": batch,
        "eval_derivs": eval_derivs,
        "log_N_grid": log_n_grid,
        "N_grid": n_grid,
        "log_N_ic": log_n_ic,
        "N_ic": n_ic,
        "N_environment_grid": environment,
        "growth_eval": biology["growth_eval"],
        "growth_grid": biology["growth_grid"],
        "mortality": biology["mortality"],
        "recruitment": recruitment,
        "n_init_target": n_init[species_idx:species_idx + 1],
    }
