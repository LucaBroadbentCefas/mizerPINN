from __future__ import annotations

import torch
from PINNmizer.params import MizerTorchParams, _n_species, _n_w, _params_dtype_device, active_grid_mask

DEFAULT_STATE_SCALE_EPS = 1e-30
STATE_SCALE_SOURCE = "initial_condition"
STATE_SCALE_INTERPOLATION = "linear_log_weight"
STATE_SCALE_EXTRAPOLATION = "constant_nearest_active"


def set_state_scale_from_initial_condition(params: MizerTorchParams, n_init: torch.Tensor, *, eps: float = DEFAULT_STATE_SCALE_EPS) -> None:
    """Store fixed log(S) from n_init, continuing nearest active values into inactive tails."""
    dtype, device = _params_dtype_device(params)
    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    if n_init.ndim == 1:
        n_init = n_init.reshape(1, -1)
    expected = (_n_species(params), _n_w(params))
    if tuple(n_init.shape) != expected:
        raise ValueError(f"n_init has shape {tuple(n_init.shape)}, expected {expected}.")
    if eps <= 0:
        raise ValueError("state-scale eps must be positive.")
    active = active_grid_mask(params).to(device=device)
    scale = torch.empty_like(n_init)
    for i in range(expected[0]):
        idx = torch.nonzero(active[i], as_tuple=False).flatten()
        if idx.numel() == 0:
            raise ValueError(f"Species {i} has no active grid bins.")
        vals = torch.clamp(n_init[i], min=float(eps))
        scale[i] = vals
        first, last = int(idx[0]), int(idx[-1])
        scale[i, :first] = vals[first]
        scale[i, last + 1 :] = vals[last]
    log_s = torch.log(scale).detach().clone()
    if not torch.isfinite(log_s).all() or not torch.all(torch.exp(log_s) > 0):
        raise ValueError("state scale must be finite and strictly positive.")
    params.state_parameterization = "log-u"
    params.state_scale_log = log_s
    params.state_scale_eps = float(eps)
    params.state_scale_source = STATE_SCALE_SOURCE
    params.state_scale_interpolation = STATE_SCALE_INTERPOLATION
    params.state_scale_extrapolation = STATE_SCALE_EXTRAPOLATION
    params.state_scale_is_trainable = False


def grid_state_scale(params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    log_s = getattr(params, "state_scale_log", None)
    if log_s is None:
        raise ValueError("params.state_scale_log is missing; call set_state_scale_from_initial_condition first.")
    log_s = log_s.detach().clone().to(dtype=dtype, device=device)
    if tuple(log_s.shape) != (_n_species(params), _n_w(params)):
        raise ValueError("state_scale_log has incompatible shape.")
    return log_s, torch.exp(log_s)


def interpolate_log_state_scale(params: MizerTorchParams, w_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate detached log_S(log w) and analytical dlogS/dw; returns [S,*w_eval.shape]."""
    dtype, device = _params_dtype_device(params)
    w_eval = torch.as_tensor(w_eval, dtype=dtype, device=device)
    if not torch.all(w_eval > 0):
        raise ValueError("All evaluation weights must be positive.")
    shape = tuple(w_eval.shape)
    w_flat = w_eval.reshape(-1)
    x_grid = torch.log(params.w.to(dtype=dtype, device=device))
    x = torch.log(w_flat).clamp(min=x_grid[0], max=x_grid[-1])
    log_s_grid, _ = grid_state_scale(params)
    idx = torch.searchsorted(x_grid, x).clamp(1, x_grid.numel() - 1)
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    y0, y1 = log_s_grid[:, idx - 1], log_s_grid[:, idx]
    slope_x = (y1 - y0) / (x1 - x0)[None, :]
    frac = (x - x0) / (x1 - x0)
    log_s = (y0 + slope_x * frac[None, :]).detach()
    dlogS_dw = (slope_x / w_flat[None, :]).detach()
    log_s = log_s.reshape((_n_species(params),) + shape)
    dlogS_dw = dlogS_dw.reshape((_n_species(params),) + shape)
    return log_s, torch.exp(log_s), dlogS_dw


def reconstruct_scalar_state(log_U: torch.Tensor, params: MizerTorchParams, *, species_idx: int, w: torch.Tensor | None = None, grid: bool = False) -> dict[str, torch.Tensor]:
    """Reconstruct log_U, U, log_S, S, log_N and N for one scalar species model."""
    U = torch.exp(log_U)
    if grid:
        log_s_all, s_all = grid_state_scale(params)
        log_s = log_s_all[species_idx].reshape((1, 1, -1))
        s = s_all[species_idx].reshape((1, 1, -1))
        dlogS_dw = None
    else:
        if w is None:
            raise ValueError("w is required for off-grid reconstruction.")
        log_s_all, s_all, d_all = interpolate_log_state_scale(params, w)
        log_s, s, dlogS_dw = log_s_all[species_idx], s_all[species_idx], d_all[species_idx]
        while log_s.ndim < log_U.ndim:
            log_s = log_s.unsqueeze(0); s = s.unsqueeze(0); dlogS_dw = dlogS_dw.unsqueeze(0)
    log_s = log_s.expand_as(log_U).detach(); s = s.expand_as(U).detach()
    out = {"log_U": log_U, "U": U, "log_S": log_s, "S": s, "log_N": log_s + log_U, "N": s * U}
    if dlogS_dw is not None:
        out["dlogS_dw"] = dlogS_dw.expand_as(log_U).detach()
    return out
