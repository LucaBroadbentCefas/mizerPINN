from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _eval_weight_vector, _params_dtype_device


def _n_gears(params: MizerTorchParams) -> int:
    if params.catchability is not None:
        return int(params.catchability.shape[0])
    if params.selectivity is not None:
        return int(params.selectivity.shape[0])
    if params.initial_effort is not None and params.initial_effort.ndim > 0:
        return int(params.initial_effort.numel())
    if params.fishing_effort is not None:
        return int(params.fishing_effort.shape[-1]) if params.fishing_effort.ndim > 1 else 1
    return 1


def _as_effort_by_gear(effort: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    effort = effort.to(dtype=dtype, device=device)
    n_gear = _n_gears(params)
    if effort.ndim == 0:
        return effort.expand(n_gear)
    if effort.ndim == 1:
        if effort.numel() == 1 and n_gear != 1:
            return effort.reshape(()).expand(n_gear)
        return effort
    return effort


def evaluate_effort_at_time(t_eval, params: MizerTorchParams) -> torch.Tensor:
    """Evaluate fishing effort in physical time.

    Returns [gear] for scalar/constant effort and [n_time, gear] for vector times.
    """
    dtype, device = _params_dtype_device(params)

    if params.fishing_effort is None or params.fishing_effort_time is None:
        if params.initial_effort is None:
            return torch.zeros(_n_gears(params), dtype=dtype, device=device)
        return _as_effort_by_gear(params.initial_effort, params)

    effort = params.fishing_effort.to(dtype=dtype, device=device)
    effort_time = params.fishing_effort_time.to(dtype=dtype, device=device).reshape(-1)
    if effort.ndim == 1:
        effort = effort[:, None]
    if effort.shape[0] != effort_time.numel():
        raise ValueError("params.fishing_effort first dimension must match params.fishing_effort_time")

    if t_eval is None:
        if params.initial_effort is not None:
            return _as_effort_by_gear(params.initial_effort, params)
        return effort[0]

    t = torch.as_tensor(t_eval, dtype=dtype, device=device)
    scalar = t.ndim == 0
    t_flat = t.reshape(-1)
    idx_hi = torch.searchsorted(effort_time, t_flat, right=False).clamp(1, effort_time.numel() - 1)
    idx_lo = idx_hi - 1
    t0 = effort_time[idx_lo]
    t1 = effort_time[idx_hi]
    frac = ((t_flat - t0) / (t1 - t0).clamp_min(torch.finfo(dtype).tiny)).clamp(0.0, 1.0)
    out = effort[idx_lo] + frac[:, None] * (effort[idx_hi] - effort[idx_lo])
    return out[0] if scalar else out.reshape(t.shape + (effort.shape[1],))


def compute_fishing_mortality_grid(params: MizerTorchParams, effort=None) -> torch.Tensor:
    """Compute mizer fishing mortality on the fixed species/weight grid."""
    dtype, device = _params_dtype_device(params)
    if params.catchability is None or params.selectivity is None:
        if params.f_mort is None:
            return torch.zeros_like(params.mu_b)
        return params.f_mort.to(dtype=dtype, device=device)

    effort_t = evaluate_effort_at_time(None, params) if effort is None else _as_effort_by_gear(torch.as_tensor(effort), params)
    catchability = params.catchability.to(dtype=dtype, device=device)
    selectivity = params.selectivity.to(dtype=dtype, device=device)
    if effort_t.ndim != 1:
        raise ValueError("compute_fishing_mortality_grid expects effort shaped [gear]")
    return (effort_t[:, None, None] * catchability[:, :, None] * selectivity).sum(dim=0)


def _interp_rows_log_weight(values: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    x_grid = torch.log(params.w.to(dtype=w_eval.dtype, device=w_eval.device))
    x = torch.log(w_eval)
    idx_hi = torch.searchsorted(x_grid, x, right=False).clamp(1, x_grid.numel() - 1)
    idx_lo = idx_hi - 1
    x0 = x_grid[idx_lo]
    x1 = x_grid[idx_hi]
    frac = ((x - x0) / (x1 - x0).clamp_min(torch.finfo(w_eval.dtype).tiny)).clamp(0.0, 1.0)
    return values[..., idx_lo] + frac * (values[..., idx_hi] - values[..., idx_lo])


def evaluate_fishing_mortality_direct(w_eval: torch.Tensor, params: MizerTorchParams, t_eval=None) -> torch.Tensor:
    """Evaluate fishing mortality at off-grid weights without interpolating F_grid."""
    dtype, device = _params_dtype_device(params)
    w_eval = _eval_weight_vector(w_eval, params)

    # Placeholder for future analytical selectivity exports: callers may attach a callable.
    selectivity_fn = getattr(params, "selectivity_fn", None)
    if selectivity_fn is not None and params.catchability is not None:
        effort = evaluate_effort_at_time(t_eval, params)
        catchability = params.catchability.to(dtype=dtype, device=device)
        selectivity_eval = selectivity_fn(w_eval, params).to(dtype=dtype, device=device)
        return (effort[:, None, None] * catchability[:, :, None] * selectivity_eval).sum(dim=0)

    if params.catchability is not None and params.selectivity is not None:
        effort = evaluate_effort_at_time(t_eval, params)
        catchability = params.catchability.to(dtype=dtype, device=device)
        selectivity_eval = _interp_rows_log_weight(params.selectivity.to(dtype=dtype, device=device), w_eval, params)
        if effort.ndim != 1:
            raise ValueError("time-varying fishing mortality at direct eval currently expects scalar t_eval")
        return (effort[:, None, None] * catchability[:, :, None] * selectivity_eval).sum(dim=0)

    if params.f_mort is not None:
        return _interp_rows_log_weight(params.f_mort.to(dtype=dtype, device=device), w_eval, params)
    return torch.zeros((params.interaction.shape[0], w_eval.numel()), dtype=dtype, device=device)
