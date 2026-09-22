from __future__ import annotations

import torch

from PINNmizer.params import _params_dtype_device, _t_limits, _x_limits

from .model_eval import make_model_inputs


def evaluate_log_model_with_derivatives_at_eval(model, x_eval_scaled: torch.Tensor,
                                                  t_scaled: torch.Tensor,
                                                  w_eval: torch.Tensor,
                                                  params) -> dict[str, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    x_eval_scaled = x_eval_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    w_eval = w_eval.to(dtype=dtype, device=device)
    if w_eval.ndim != 1 or w_eval.numel() != x_eval_scaled.numel():
        raise ValueError("w_eval and x_eval_scaled must be matching vectors.")
    inputs = make_model_inputs(x_eval_scaled, t_scaled).detach().requires_grad_(True)
    raw_flat = model(inputs)
    expected = (t_scaled.numel() * w_eval.numel(), 1)
    if raw_flat.shape != expected:
        raise ValueError(f"Scalar model returned {tuple(raw_flat.shape)}, expected {expected}.")
    grad = torch.autograd.grad(raw_flat.sum(), inputs, create_graph=True, retain_graph=True)[0]
    log_n = raw_flat.reshape(t_scaled.numel(), w_eval.numel(), 1).permute(0, 2, 1).contiguous()
    d_dx_scaled = grad[:, 0].reshape(t_scaled.numel(), 1, w_eval.numel())
    d_dt_scaled = grad[:, 1].reshape(t_scaled.numel(), 1, w_eval.numel())
    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)
    dlogn_dt = d_dt_scaled / (t_max - t_min)
    dlogn_dw = (d_dx_scaled / (x_max - x_min)) / w_eval[None, None, :]
    n = torch.exp(log_n)
    return {
        "log_N_eval": log_n,
        "N_eval": n,
        "dlogN_dt": dlogn_dt,
        "dlogN_dw": dlogn_dw,
        "dN_dt": n * dlogn_dt,
        "dN_dw": n * dlogn_dw,
    }
