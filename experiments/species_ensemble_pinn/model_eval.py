from __future__ import annotations

import torch
from PINNmizer.params import MizerTorchParams, _params_dtype_device, scale_x, scale_t
from .state_scale import reconstruct_scalar_state


def make_model_inputs(x_scaled: torch.Tensor, t_scaled: torch.Tensor) -> torch.Tensor:
    xx = x_scaled[None, :].expand(t_scaled.numel(), x_scaled.numel())
    tt = t_scaled[:, None].expand(t_scaled.numel(), x_scaled.numel())
    return torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)


def evaluate_scalar_model_on_points(model, x_scaled: torch.Tensor, t_scaled: torch.Tensor, params: MizerTorchParams, *, species_idx: int, w: torch.Tensor | None = None, grid: bool = True) -> dict[str, torch.Tensor]:
    """Evaluate scalar log_U model and reconstruct U, S and physical N as [T,1,M]."""
    dtype, device = _params_dtype_device(params)
    x_scaled = x_scaled.to(dtype=dtype, device=device); t_scaled = t_scaled.to(dtype=dtype, device=device)
    raw = model(make_model_inputs(x_scaled, t_scaled))
    if tuple(raw.shape) != (t_scaled.numel() * x_scaled.numel(), 1):
        raise ValueError(f"Scalar species model must return log_U [P,1], got {tuple(raw.shape)}.")
    log_U = raw.reshape(t_scaled.numel(), x_scaled.numel(), 1).permute(0, 2, 1).contiguous()
    if w is None:
        w = params.w if grid else torch.exp(x_scaled)
    return reconstruct_scalar_state(log_U, params, species_idx=species_idx, w=w, grid=grid)


def evaluate_scalar_model_physical(model, t: torch.Tensor, w: torch.Tensor, params: MizerTorchParams, *, species_idx: int) -> dict[str, torch.Tensor]:
    return evaluate_scalar_model_on_points(model, scale_x(torch.log(w), params), scale_t(t, params), params, species_idx=species_idx, w=w, grid=torch.equal(w, params.w))
