from __future__ import annotations

import torch

from PINNmizer.params import _n_species, _n_w, _params_dtype_device, active_grid_mask

RESIDUAL_SCALE_SOURCE = "initial_condition"
RESIDUAL_SCALE_INTERPOLATION = "linear_log_weight"
RESIDUAL_SCALE_EXTRAPOLATION = "constant_last_active"
DEFAULT_RESIDUAL_SCALE_FLOOR_FRACTION = 1e-12


def set_residual_scale_from_initial_condition(params, n_init: torch.Tensor, *,
                                              floor_fraction: float | None = None) -> None:
    dtype, device = _params_dtype_device(params)
    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    expected = (_n_species(params), _n_w(params))
    if n_init.shape != expected:
        raise ValueError(f"n_init has shape {tuple(n_init.shape)}, expected {expected}.")
    eta = DEFAULT_RESIDUAL_SCALE_FLOOR_FRACTION if floor_fraction is None else float(floor_fraction)
    if eta <= 0:
        raise ValueError("Residual-scale floor fraction must be strictly positive.")
    if not torch.isfinite(n_init).all() or bool((n_init < 0).any().detach().cpu()):
        raise ValueError("n_init must be finite and non-negative.")
    mask = active_grid_mask(params).to(device=device)
    scale = torch.empty_like(n_init)
    for species_idx in range(expected[0]):
        active = torch.nonzero(mask[species_idx], as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError(f"Species {species_idx} has no active grid bins.")
        values = n_init[species_idx, active]
        maximum = values.max()
        floor = eta * maximum
        if not bool((maximum > 0).detach().cpu()):
            raise ValueError(f"Species {species_idx} has non-positive active initial maximum.")
        active_scale = torch.maximum(values, floor.expand_as(values))
        first, last = int(active[0]), int(active[-1])
        scale[species_idx, active] = active_scale
        scale[species_idx, :first] = active_scale[0]
        scale[species_idx, last + 1:] = active_scale[-1]
    if not torch.isfinite(scale).all() or not bool((scale > 0).all().detach().cpu()):
        raise ValueError("Constructed residual reference scale is not finite and positive.")
    params.residual_scale_log = torch.log(scale).detach().clone()
    params.residual_scale_floor_fraction = eta
    params.residual_scale_source = RESIDUAL_SCALE_SOURCE
    params.residual_scale_interpolation = RESIDUAL_SCALE_INTERPOLATION
    params.residual_scale_extrapolation = RESIDUAL_SCALE_EXTRAPOLATION
    params.residual_scale_is_differentiated = False


def grid_residual_scale(params) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    log_scale = getattr(params, "residual_scale_log", None)
    if log_scale is None:
        raise ValueError("Residual reference scale has not been initialised.")
    log_scale = log_scale.to(dtype=dtype, device=device).detach()
    expected = (_n_species(params), _n_w(params))
    if log_scale.shape != expected:
        raise ValueError(f"residual_scale_log has shape {tuple(log_scale.shape)}, expected {expected}.")
    scale = torch.exp(log_scale).detach()
    if not torch.isfinite(log_scale).all() or not bool((scale > 0).all().detach().cpu()):
        raise ValueError("Stored residual reference scale is invalid.")
    return log_scale, scale


def interpolate_log_residual_scale(params, w_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    w_eval = torch.as_tensor(w_eval, dtype=dtype, device=device)
    original_shape = w_eval.shape
    flat = w_eval.reshape(-1)
    if not torch.isfinite(flat).all() or not bool((flat > 0).all().detach().cpu()):
        raise ValueError("Evaluation weights must be finite and positive.")
    w_grid = params.w.to(dtype=dtype, device=device)
    if bool(((flat < w_grid[0]) | (flat > w_grid[-1])).any().detach().cpu()):
        raise ValueError("Residual-scale evaluation weights must remain inside params.w domain.")
    x_grid, x = torch.log(w_grid), torch.log(flat)
    log_grid, _ = grid_residual_scale(params)
    idx = torch.searchsorted(x_grid, x).clamp(1, x_grid.numel() - 1)
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    y0, y1 = log_grid[:, idx - 1], log_grid[:, idx]
    fraction = (x - x0) / (x1 - x0)
    log_scale = (y0 + (y1 - y0) * fraction[None, :]).reshape((_n_species(params),) + original_shape).detach()
    return log_scale, torch.exp(log_scale).detach()
