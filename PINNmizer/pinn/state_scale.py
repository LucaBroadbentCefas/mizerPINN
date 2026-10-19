from __future__ import annotations

import math

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _n_w, _params_dtype_device, active_grid_mask

DEFAULT_STATE_SCALE_EPS = 1e-30
DEFAULT_STATE_SCALE_POWER = 2.05
DEFAULT_STATE_SCALE_REFERENCE_WEIGHT = 1.0

STATE_SCALE_SOURCE = "initial_condition"
STATE_SCALE_SOURCE_INITIAL_CONDITION = "initial_condition"
STATE_SCALE_SOURCE_POWER_LAW = "power_law"
STATE_SCALE_INTERPOLATION = "linear_log_weight"


def state_parameterization(params: MizerTorchParams) -> str:
    value = getattr(params, "state_parameterization", "log-n")
    if value not in {"log-n", "log-u"}:
        raise ValueError(f"state_parameterization must be 'log-n' or 'log-u', got {value!r}.")
    return value


def state_scale_eps(params: MizerTorchParams) -> float:
    return float(getattr(params, "state_scale_eps", DEFAULT_STATE_SCALE_EPS))


def _coerce_initial_state(
    params: MizerTorchParams,
    n_init: torch.Tensor,
) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    n_species, n_w = _n_species(params), _n_w(params)
    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    if n_init.ndim == 1:
        if n_species != 1:
            raise ValueError("1D n_init is only valid for one species.")
        n_init = n_init.reshape(1, n_w)
    if n_init.shape != (n_species, n_w):
        raise ValueError(f"n_init has shape {tuple(n_init.shape)}, expected {(n_species, n_w)}.")
    return n_init


def set_state_scale_from_initial_condition(
    params: MizerTorchParams,
    n_init: torch.Tensor,
    *,
    eps: float | None = None,
) -> None:
    n_init = _coerce_initial_state(params, n_init)
    eps_v = state_scale_eps(params) if eps is None else float(eps)
    mask = active_grid_mask(params).to(device=n_init.device)
    log_s = torch.log(torch.clamp(n_init, min=eps_v))
    log_s = torch.where(mask, log_s, torch.zeros_like(log_s))

    params.state_scale_log = log_s.detach().clone()
    params.state_scale_eps = eps_v
    params.state_scale_source = STATE_SCALE_SOURCE_INITIAL_CONDITION
    params.state_scale_interpolation = STATE_SCALE_INTERPOLATION
    params.state_scale_power = None
    params.state_scale_reference_weight = None
    params.state_scale_amplitude_source = "full_initial_condition"


def set_state_scale_from_power_law(
    params: MizerTorchParams,
    n_init: torch.Tensor,
    *,
    exponent: float = DEFAULT_STATE_SCALE_POWER,
    reference_weight: float = DEFAULT_STATE_SCALE_REFERENCE_WEIGHT,
    eps: float | None = None,
) -> None:
    """Set S_i(w) = C_i (w / w_ref)^(-exponent).

    The slope is fixed a priori and therefore contains no local information from
    the initial spectrum. Only one scalar amplitude C_i per species is estimated
    from the initial condition. C_i is the least-squares intercept in log space,
    chosen so that mean_active(log(N_i(0,w) / S_i(w))) == 0.
    """
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("state-scale power-law exponent must be finite and strictly positive.")
    if not math.isfinite(reference_weight) or reference_weight <= 0.0:
        raise ValueError("state-scale reference weight must be finite and strictly positive.")

    n_init = _coerce_initial_state(params, n_init)
    eps_v = state_scale_eps(params) if eps is None else float(eps)
    mask = active_grid_mask(params).to(device=n_init.device)

    log_n = torch.log(torch.clamp(n_init, min=eps_v))
    w = params.w.to(dtype=log_n.dtype, device=log_n.device)
    log_w_ratio = torch.log(w / reference_weight)

    mask_f = mask.to(dtype=log_n.dtype)
    denom = mask_f.sum(dim=1)
    if not bool((denom > 0).all().detach().cpu()):
        raise ValueError("Cannot build power-law state scale: a species has no active weight bins.")

    log_c = (
        ((log_n + exponent * log_w_ratio[None, :]) * mask_f).sum(dim=1)
        / denom
    )
    log_s = log_c[:, None] - exponent * log_w_ratio[None, :]

    params.state_scale_log = log_s.detach().clone()
    params.state_scale_eps = eps_v
    params.state_scale_source = STATE_SCALE_SOURCE_POWER_LAW
    params.state_scale_interpolation = STATE_SCALE_INTERPOLATION
    params.state_scale_power = float(exponent)
    params.state_scale_reference_weight = float(reference_weight)
    params.state_scale_amplitude_source = "initial_condition_scalar_log_fit"


def set_state_scale(
    params: MizerTorchParams,
    n_init: torch.Tensor,
    *,
    source: str = STATE_SCALE_SOURCE_INITIAL_CONDITION,
    eps: float | None = None,
    power: float = DEFAULT_STATE_SCALE_POWER,
    reference_weight: float = DEFAULT_STATE_SCALE_REFERENCE_WEIGHT,
) -> None:
    source_key = source.replace("-", "_")
    if source_key == STATE_SCALE_SOURCE_INITIAL_CONDITION:
        set_state_scale_from_initial_condition(params, n_init, eps=eps)
        return
    if source_key == STATE_SCALE_SOURCE_POWER_LAW:
        set_state_scale_from_power_law(
            params,
            n_init,
            exponent=power,
            reference_weight=reference_weight,
            eps=eps,
        )
        return
    raise ValueError(
        "state scale source must be 'initial-condition'/'initial_condition' "
        "or 'power-law'/'power_law'."
    )


def grid_state_scale(params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    log_s = getattr(params, "state_scale_log", None)
    if log_s is None:
        log_s = torch.zeros((_n_species(params), _n_w(params)), dtype=dtype, device=device)
    log_s = log_s.to(dtype=dtype, device=device)
    return log_s, torch.exp(log_s)


def interpolate_log_state_scale(params: MizerTorchParams, w_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log_S, S and d log_S / dw with shapes [n_species, *w_eval.shape]."""
    dtype, device = _params_dtype_device(params)
    w_eval = w_eval.to(dtype=dtype, device=device)
    orig_shape = w_eval.shape
    w_flat = w_eval.reshape(-1)
    x_grid = torch.log(params.w.to(dtype=dtype, device=device))
    x = torch.log(w_flat)
    log_s_grid, _ = grid_state_scale(params)
    idx = torch.searchsorted(x_grid, x).clamp(1, x_grid.numel() - 1)
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    y0 = log_s_grid[:, idx - 1]
    y1 = log_s_grid[:, idx]
    dx = x1 - x0
    delta_log_s = y1 - y0

    frac = (x - x0) / dx
    log_s = y0 + delta_log_s * frac[None, :]

    slope = delta_log_s / dx[None, :]
    dlogS_dw = slope / w_flat[None, :]
    log_s = log_s.reshape((_n_species(params),) + orig_shape)
    dlogS_dw = dlogS_dw.reshape((_n_species(params),) + orig_shape)
    return log_s, torch.exp(log_s), dlogS_dw


def reconstruct_from_model_output(
    raw: torch.Tensor,
    params: MizerTorchParams,
    *,
    w: torch.Tensor | None = None,
    grid: bool = False,
) -> dict[str, torch.Tensor]:
    if state_parameterization(params) == "log-n":
        return {"log_N": raw, "N": torch.exp(raw)}

    log_U = raw
    U = torch.exp(log_U)
    if grid:
        log_s, s = grid_state_scale(params)
        while log_s.ndim < log_U.ndim:
            log_s = log_s.unsqueeze(0)
            s = s.unsqueeze(0)
    else:
        if w is None:
            raise ValueError("w is required for off-grid log-u reconstruction.")
        log_s, s, dlogS_dw = interpolate_log_state_scale(params, w)
        if log_U.ndim == 3 and log_s.ndim == 2:
            log_s = log_s.unsqueeze(0)
            s = s.unsqueeze(0)
            dlogS_dw = dlogS_dw.unsqueeze(0)
        elif log_U.ndim == 3 and log_s.ndim == 3:
            log_s = log_s.permute(1, 0, 2).contiguous()
            s = s.permute(1, 0, 2).contiguous()
            dlogS_dw = dlogS_dw.permute(1, 0, 2).contiguous()

    out = {
        "log_U": log_U,
        "U": U,
        "log_S": log_s.expand_as(log_U),
        "S": s.expand_as(U),
    }
    out["log_N"] = out["log_U"] + out["log_S"]
    out["N"] = out["S"] * out["U"]
    if not grid and "dlogS_dw" in locals():
        out["dlogS_dw"] = dlogS_dw.expand_as(log_U)
    return out
