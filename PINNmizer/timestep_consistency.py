from __future__ import annotations

import torch

from PINNmizer.mizer_grid_ops import step
from PINNmizer.params import MizerTorchParams, _params_dtype_device, scale_t, scale_x
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points


def _as_1d_time(x: torch.Tensor | float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim != 1:
        raise ValueError(f"Expected scalar or 1D time tensor, got shape {tuple(t.shape)}")
    return t


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

    t0_vec = _as_1d_time(t0, dtype=dtype, device=device)
    dt_raw = getattr(params, "dt", None) if dt is None else dt
    if dt_raw is None:
        raise ValueError("dt is required: pass dt explicitly because params.dt is unavailable.")
    dt_vec = _as_1d_time(dt_raw, dtype=dtype, device=device)
    if dt_vec.numel() == 1 and t0_vec.numel() > 1:
        dt_vec = dt_vec.expand_as(t0_vec)
    if dt_vec.shape != t0_vec.shape:
        raise ValueError(f"dt shape {tuple(dt_vec.shape)} must match t0 shape {tuple(t0_vec.shape)} or be scalar.")

    t1_vec = t0_vec + dt_vec
    t_max = torch.as_tensor(params.t_max, dtype=dtype, device=device)
    if torch.any(t1_vec > t_max):
        raise ValueError("All t0 + dt must satisfy t0 + dt <= t_max.")

    x_grid = torch.log(params.w.to(dtype=dtype, device=device))
    x_scaled = scale_x(x_grid, params)

    t_scaled_0 = scale_t(t0_vec, params)
    t_scaled_1 = scale_t(t1_vec, params)
    out0 = evaluate_log_model_on_points(model=model, x_scaled=x_scaled, t_scaled=t_scaled_0, params=params)
    out1 = evaluate_log_model_on_points(model=model, x_scaled=x_scaled, t_scaled=t_scaled_1, params=params)

    N0_pred_all = out0["N"]
    N1_pred_all = out1["N"]

    if species_idx is not None:
        N0_pred_all = N0_pred_all[:, species_idx : species_idx + 1, :]
        N1_pred_all = N1_pred_all[:, species_idx : species_idx + 1, :]

    stepped = []
    npp_steps = []
    ops_e_growth = []
    ops_mort = []
    ops_rdd = []
    for i in range(t0_vec.numel()):
        n_for_step = N0_pred_all[i]
        if detach_step_target:
            n_for_step = n_for_step.detach()
        n_pp_new, n_new, ops0 = step(n_pp=n_pp, n=n_for_step, params=params, dt=dt_vec[i])
        stepped.append(n_new)
        npp_steps.append(n_pp_new)
        ops_e_growth.append(ops0["e_growth"])
        ops_mort.append(ops0["mort"])
        ops_rdd.append(ops0["rdd"])

    N1_step_all = torch.stack(stepped, dim=0)
    n_pp_step = torch.stack(npp_steps, dim=0)

    residual_physical = N1_pred_all - N1_step_all
    log_N1_pred = torch.log(torch.clamp(N1_pred_all, min=eps))
    log_N1_step = torch.log(torch.clamp(N1_step_all, min=eps))
    residual_log = log_N1_pred - log_N1_step
    residual_relative = residual_physical / torch.clamp(torch.abs(N1_step_all), min=relative_eps)

    if loss_form == "physical":
        loss_timestep = torch.mean(residual_physical ** 2)
    elif loss_form == "log":
        loss_timestep = torch.mean(residual_log ** 2)
    elif loss_form == "relative":
        loss_timestep = torch.mean(residual_relative ** 2)
    else:
        raise ValueError("loss_form must be one of: physical, log, relative")

    out_diag: dict[str, torch.Tensor | str | bool] = {
        "loss_timestep": loss_timestep,
        "residual_timestep_physical": residual_physical,
        "residual_timestep_log": residual_log,
        "residual_timestep_relative": residual_relative,
        "N0_pred": N0_pred_all,
        "N1_pred": N1_pred_all,
        "N1_step": N1_step_all,
        "n_pp_step": n_pp_step,
        "t0": t0_vec,
        "t1": t1_vec,
        "dt": dt_vec,
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
        "has_nan_physical": torch.isnan(residual_physical).any(),
        "has_inf_physical": torch.isinf(residual_physical).any(),
        "ops_e_growth": torch.stack(ops_e_growth, dim=0),
        "ops_mort": torch.stack(ops_mort, dim=0),
        "ops_rdd": torch.stack(ops_rdd, dim=0),
    }

    return loss_timestep, out_diag
