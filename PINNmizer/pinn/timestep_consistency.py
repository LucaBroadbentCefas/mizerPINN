from __future__ import annotations

import torch

from PINNmizer.mizer_grid_ops import step
from PINNmizer.params import MizerTorchParams, _params_dtype_device, scale_t, scale_x
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points


def compute_timestep_consistency_loss(
    model,
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    t0: torch.Tensor,
    dt: torch.Tensor | float | None = None,
    *,
    loss_form: str = "physical",
    detach_step_target: bool = True,
    eps: float = 1e-30,
    relative_eps: float = 1e-12,
    species_idx: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str | bool]]:
    dtype, device = _params_dtype_device(params)

    t0 = torch.as_tensor(t0, dtype=dtype, device=device).reshape(-1)
    if dt is None:
        if not hasattr(params, "dt"):
            raise ValueError("dt is None and params has no dt attribute.")
        dt = getattr(params, "dt")
    dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)

    t1 = t0 + dt_tensor
    t_max = torch.as_tensor(params.t_max, dtype=dtype, device=device)
    if not torch.all(t1 <= t_max):
        raise ValueError("All t0 + dt must be <= params.t_max.")

    x_grid = torch.log(params.w).to(dtype=dtype, device=device)
    x_grid_scaled = scale_x(x_grid, params)

    pred0 = evaluate_log_model_on_points(model, x_grid_scaled, scale_t(t0, params), params)
    pred1 = evaluate_log_model_on_points(model, x_grid_scaled, scale_t(t1, params), params)

    N0_pred = pred0["N"]
    N1_pred = pred1["N"]

    if species_idx is not None:
        N0_pred = N0_pred[:, species_idx:species_idx + 1, :]
        N1_pred = N1_pred[:, species_idx:species_idx + 1, :]

    n_pairs = t0.numel()
    stepped = []
    n_pp_step_list = []
    ops_growth, ops_mort, ops_rdd = [], [], []

    n_pp_in = n_pp.to(dtype=dtype, device=device)

    for i in range(n_pairs):
        n0_i = N0_pred[i]
        if detach_step_target:
            n0_i = n0_i.detach()
        n_pp_new_i, n1_step_i, ops_i = step(
            n_pp=n_pp_in,
            n=n0_i,
            params=params,
            dt=dt_tensor,
        )
        stepped.append(n1_step_i)
        n_pp_step_list.append(n_pp_new_i)
        if "e_growth" in ops_i:
            ops_growth.append(ops_i["e_growth"])
        if "mort" in ops_i:
            ops_mort.append(ops_i["mort"])
        if "rdd" in ops_i:
            ops_rdd.append(ops_i["rdd"])

    N1_step = torch.stack(stepped, dim=0)
    n_pp_step = torch.stack(n_pp_step_list, dim=0)

    residual_physical = N1_pred - N1_step
    residual_log = torch.log(torch.clamp(N1_pred, min=eps)) - torch.log(torch.clamp(N1_step, min=eps))
    residual_relative = residual_physical / torch.clamp(torch.abs(N1_step), min=relative_eps)

    if loss_form == "physical":
        loss_timestep = torch.mean(residual_physical ** 2)
    elif loss_form == "log":
        loss_timestep = torch.mean(residual_log ** 2)
    elif loss_form == "relative":
        loss_timestep = torch.mean(residual_relative ** 2)
    else:
        raise ValueError("loss_form must be one of {'physical','log','relative'}")

    diagnostics: dict[str, torch.Tensor | str | bool] = {
        "loss_timestep": loss_timestep,
        "residual_timestep_physical": residual_physical,
        "residual_timestep_log": residual_log,
        "residual_timestep_relative": residual_relative,
        "N0_pred": N0_pred,
        "N1_pred": N1_pred,
        "N1_step": N1_step,
        "n_pp_step": n_pp_step,
        "t0": t0,
        "t1": t1,
        "dt": dt_tensor,
        "loss_form": loss_form,
        "detach_step_target": detach_step_target,
        "physical_abs_mean": torch.mean(torch.abs(residual_physical)),
        "physical_abs_max": torch.max(torch.abs(residual_physical)),
        "log_abs_mean": torch.mean(torch.abs(residual_log)),
        "log_abs_max": torch.max(torch.abs(residual_log)),
        "relative_abs_mean": torch.mean(torch.abs(residual_relative)),
        "relative_abs_max": torch.max(torch.abs(residual_relative)),
        "has_nan_loss": torch.isnan(loss_timestep),
        "has_inf_loss": torch.isinf(loss_timestep),
        "has_nan_residual": torch.isnan(residual_physical).any(),
        "has_inf_residual": torch.isinf(residual_physical).any(),
    }

    if ops_growth:
        diagnostics["ops_e_growth"] = torch.stack(ops_growth, dim=0)
    if ops_mort:
        diagnostics["ops_mort"] = torch.stack(ops_mort, dim=0)
    if ops_rdd:
        diagnostics["ops_rdd"] = torch.stack(ops_rdd, dim=0)

    return loss_timestep, diagnostics
