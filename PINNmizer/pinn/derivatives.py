from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _params_dtype_device, _t_limits, _x_limits
from PINNmizer.pinn.model_eval import _make_model_inputs


def evaluate_log_model_with_derivatives_at_eval(model, x_eval_scaled: torch.Tensor, t_scaled: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams) -> dict[str, torch.Tensor]:
    """Autograd derivatives of NN output; converts scaled derivatives to physical coords.

    Returns tensors shaped [n_time, n_species, n_eval].
    """
    dtype, device = _params_dtype_device(params)
    x_eval_scaled = x_eval_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    w_eval = w_eval.to(dtype=dtype, device=device)

    inputs = _make_model_inputs(x_eval_scaled, t_scaled)
    inputs.requires_grad_(True)
    log_N_flat = model(inputs)

    n_time = t_scaled.numel()
    n_eval = x_eval_scaled.numel()
    n_species = _n_species(params)

    dlogN_dx_scaled_rows, dlogN_dt_scaled_rows = [], []
    for i in range(n_species):
        grad_i = torch.autograd.grad(log_N_flat[:, i].sum(), inputs, inputs, create_graph=True, retain_graph=True, allow_unused=False)[0]
        dlogN_dx_scaled_rows.append(grad_i[:, 0])
        dlogN_dt_scaled_rows.append(grad_i[:, 1])

    log_N = log_N_flat.reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()
    N = torch.exp(log_N)

    dlogN_dx_scaled = torch.stack(dlogN_dx_scaled_rows, dim=1).reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()
    dlogN_dt_scaled = torch.stack(dlogN_dt_scaled_rows, dim=1).reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)
    dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
    dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
    dlogN_dw = dlogN_dx / w_eval[None, None, :]
    dN_dt = N * dlogN_dt
    dN_dw = N * dlogN_dw
    return {"log_N_eval": log_N, "N_eval": N, "dlogN_dt": dlogN_dt, "dlogN_dw": dlogN_dw, "dN_dt": dN_dt, "dN_dw": dN_dw}
