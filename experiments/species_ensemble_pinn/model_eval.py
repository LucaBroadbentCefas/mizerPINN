from __future__ import annotations

import torch

from PINNmizer.params import _params_dtype_device


def make_model_inputs(x_scaled: torch.Tensor, t_scaled: torch.Tensor) -> torch.Tensor:
    if x_scaled.ndim != 1 or t_scaled.ndim != 1:
        raise ValueError("x_scaled and t_scaled must both be one-dimensional.")
    n_time, n_x = t_scaled.numel(), x_scaled.numel()
    xx = x_scaled[None, :].expand(n_time, n_x)
    tt = t_scaled[:, None].expand(n_time, n_x)
    return torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)


def evaluate_log_model_on_points(model, x_scaled: torch.Tensor, t_scaled: torch.Tensor, params) -> dict[str, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    x_scaled = x_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    raw = model(make_model_inputs(x_scaled, t_scaled))
    expected = (t_scaled.numel() * x_scaled.numel(), 1)
    if raw.shape != expected:
        raise ValueError(f"Scalar model returned {tuple(raw.shape)}, expected {expected}.")
    log_n = raw.reshape(t_scaled.numel(), x_scaled.numel(), 1).permute(0, 2, 1).contiguous()
    return {"log_N": log_n, "N": torch.exp(log_n)}
