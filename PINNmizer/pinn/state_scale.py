from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _n_w, _params_dtype_device, active_grid_mask

DEFAULT_STATE_SCALE_EPS = 1e-30
STATE_SCALE_SOURCE = "initial_condition"
STATE_SCALE_INTERPOLATION = "linear_log_weight"


def state_parameterization(params: MizerTorchParams) -> str:
    value = getattr(params, "state_parameterization", "log-n")
    if value not in {"log-n", "log-u"}:
        raise ValueError(f"state_parameterization must be 'log-n' or 'log-u', got {value!r}.")
    return value


def state_scale_eps(params: MizerTorchParams) -> float:
    return float(getattr(params, "state_scale_eps", DEFAULT_STATE_SCALE_EPS))


def set_state_scale_from_initial_condition(params: MizerTorchParams, n_init: torch.Tensor, *, eps: float | None = None) -> None:
    dtype, device = _params_dtype_device(params)
    n_species, n_w = _n_species(params), _n_w(params)
    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    if n_init.ndim == 1:
        if n_species != 1:
            raise ValueError("1D n_init is only valid for one species.")
        n_init = n_init.reshape(1, n_w)
    if n_init.shape != (n_species, n_w):
        raise ValueError(f"n_init has shape {tuple(n_init.shape)}, expected {(n_species, n_w)}.")
    eps_v = state_scale_eps(params) if eps is None else float(eps)
    mask = active_grid_mask(params).to(device=device)
    log_s = torch.log(torch.clamp(n_init, min=eps_v))
    log_s = torch.where(mask, log_s, torch.zeros_like(log_s))
    params.state_scale_log = log_s.detach().clone()
    params.state_scale_eps = eps_v
    params.state_scale_source = STATE_SCALE_SOURCE
    params.state_scale_interpolation = STATE_SCALE_INTERPOLATION


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
    x0 = x_grid[idx - 1]; x1 = x_grid[idx]
    y0 = log_s_grid[:, idx - 1]; y1 = log_s_grid[:, idx]
    slope = (y1 - y0) / (x1 - x0)[None, :]
    frac = (x - x0) / (x1 - x0)
    log_s = y0 + slope * frac[None, :]
    dlogS_dw = slope / w_flat[None, :]
    log_s = log_s.reshape((_n_species(params),) + orig_shape)
    dlogS_dw = dlogS_dw.reshape((_n_species(params),) + orig_shape)
    return log_s, torch.exp(log_s), dlogS_dw


def reconstruct_from_model_output(raw: torch.Tensor, params: MizerTorchParams, *, w: torch.Tensor | None = None, grid: bool = False) -> dict[str, torch.Tensor]:
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
            log_s = log_s.unsqueeze(0); s = s.unsqueeze(0); dlogS_dw = dlogS_dw.unsqueeze(0)
        elif log_U.ndim == 3 and log_s.ndim == 3:  # slab [species,K,M] -> [K,species,M]
            log_s = log_s.permute(1,0,2).contiguous(); s = s.permute(1,0,2).contiguous(); dlogS_dw = dlogS_dw.permute(1,0,2).contiguous()
    out = {"log_U": log_U, "U": U, "log_S": log_s.expand_as(log_U), "S": s.expand_as(U)}
    out["log_N"] = out["log_U"] + out["log_S"]
    out["N"] = out["S"] * out["U"]
    if not grid and "dlogS_dw" in locals():
        out["dlogS_dw"] = dlogS_dw.expand_as(log_U)
    return out
