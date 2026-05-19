from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _params_dtype_device


def _check_batch_vector(x: torch.Tensor, name: str) -> None:
    assert x.ndim == 1, f"{name} must be 1D, got shape {tuple(x.shape)}"


def _make_model_inputs(x_scaled: torch.Tensor, t_scaled: torch.Tensor) -> torch.Tensor:
    """Build time-major model inputs [n_time*n_x,2] with columns [x_scaled,t_scaled]."""
    _check_batch_vector(x_scaled, "x_scaled")
    _check_batch_vector(t_scaled, "t_scaled")
    n_time = t_scaled.numel()
    n_x = x_scaled.numel()
    xx = x_scaled[None, :].expand(n_time, n_x)
    tt = t_scaled[:, None].expand(n_time, n_x)
    return torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)


def evaluate_log_model_on_points(model, x_scaled: torch.Tensor, t_scaled: torch.Tensor, params: MizerTorchParams) -> dict[str, torch.Tensor]:
    """Evaluate model(log_N) on scaled points; returns log_N and N as [n_time,n_species,n_x]."""
    dtype, device = _params_dtype_device(params)
    x_scaled = x_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    inputs = _make_model_inputs(x_scaled, t_scaled)
    log_N_flat = model(inputs)
    n_time = t_scaled.numel()
    n_x = x_scaled.numel()
    n_species = _n_species(params)
    assert log_N_flat.shape == (n_time * n_x, n_species)
    log_N = log_N_flat.reshape(n_time, n_x, n_species).permute(0, 2, 1).contiguous()
    N = torch.exp(log_N)
    return {"log_N": log_N, "N": N}
