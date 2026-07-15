from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _n_w, _params_dtype_device, active_grid_mask

RESIDUAL_SCALE_SOURCE = "initial_condition"
RESIDUAL_SCALE_INTERPOLATION = "linear_log_weight"
RESIDUAL_SCALE_EXTRAPOLATION = "constant_last_active"
DEFAULT_RESIDUAL_SCALE_FLOOR_FRACTION = 1e-12


def set_residual_scale_from_initial_condition(
    params: MizerTorchParams,
    n_init: torch.Tensor,
    *,
    floor_fraction: float | None = None,
) -> None:
    dtype, device = _params_dtype_device(params)
    n_species, n_w = _n_species(params), _n_w(params)
    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    if n_init.ndim == 1:
        if n_species != 1:
            raise ValueError("1D n_init is only valid for one species.")
        n_init = n_init.reshape(1, n_w)
    if n_init.shape != (n_species, n_w):
        raise ValueError(f"n_init has shape {tuple(n_init.shape)}, expected {(n_species, n_w)}.")

    eta = float(getattr(params, "residual_scale_floor_fraction", DEFAULT_RESIDUAL_SCALE_FLOOR_FRACTION) if floor_fraction is None else floor_fraction)
    if eta <= 0.0:
        raise ValueError("residual_scale_floor_fraction must be strictly positive.")

    mask = active_grid_mask(params).to(device=device)
    s_ref = torch.empty_like(n_init)
    for i in range(n_species):
        active_idx = torch.nonzero(mask[i], as_tuple=False).flatten()
        if active_idx.numel() == 0:
            raise ValueError(f"Species {i} has no active weight-grid entries.")
        last = int(active_idx[-1].item())
        active_values = n_init[i, active_idx]
        active_max = active_values.max()
        floor = eta * active_max
        active_scale = torch.maximum(active_values, floor.expand_as(active_values))
        s_ref[i, active_idx] = active_scale
        if last + 1 < n_w:
            s_ref[i, last + 1 :] = active_scale[-1]
        if active_idx[0] > 0:
            s_ref[i, : int(active_idx[0].item())] = active_scale[0]

    params.residual_scale_log = torch.log(s_ref).detach().clone()
    params.residual_scale_floor_fraction = eta
    params.residual_scale_source = RESIDUAL_SCALE_SOURCE
    params.residual_scale_interpolation = RESIDUAL_SCALE_INTERPOLATION
    params.residual_scale_extrapolation = RESIDUAL_SCALE_EXTRAPOLATION


def grid_residual_scale(params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    log_s = getattr(params, "residual_scale_log", None)
    if log_s is None:
        log_s = torch.zeros((_n_species(params), _n_w(params)), dtype=dtype, device=device)
    log_s = log_s.to(dtype=dtype, device=device).detach()
    expected = (_n_species(params), _n_w(params))
    if log_s.shape != expected:
        raise ValueError(f"residual_scale_log has shape {tuple(log_s.shape)}, expected {expected}.")
    return log_s, torch.exp(log_s).detach()


def interpolate_log_residual_scale(params: MizerTorchParams, w_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached log_S_reference and S_reference with shape [n_species, *w_eval.shape]."""
    dtype, device = _params_dtype_device(params)
    w_eval = w_eval.to(dtype=dtype, device=device)
    orig_shape = w_eval.shape
    w_flat = w_eval.reshape(-1)
    x_grid = torch.log(params.w.to(dtype=dtype, device=device))
    x = torch.log(w_flat)
    log_s_grid, _ = grid_residual_scale(params)
    idx = torch.searchsorted(x_grid, x).clamp(1, x_grid.numel() - 1)
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    y0 = log_s_grid[:, idx - 1]
    y1 = log_s_grid[:, idx]
    frac = (x - x0) / (x1 - x0)
    log_s = y0 + (y1 - y0) * frac[None, :]
    log_s = log_s.reshape((_n_species(params),) + orig_shape).detach()
    return log_s, torch.exp(log_s).detach()
