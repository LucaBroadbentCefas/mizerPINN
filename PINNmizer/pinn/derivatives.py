from __future__ import annotations

import torch

from PINNmizer.params import (
    MizerTorchParams,
    _n_species,
    _params_dtype_device,
    _t_limits,
    _x_limits,
)
from PINNmizer.pinn.model_eval import _make_model_inputs


def evaluate_log_model_with_derivatives_at_eval(
    model,
    x_eval_scaled: torch.Tensor,
    t_scaled: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """
    Evaluate log_N and its neural-network derivatives at off-grid PDE points.

    Autograd differentiates with respect to the scaled network inputs
    [x_scaled, t_scaled]. The derivatives are then converted to physical
    coordinates before residual assembly:

        d/dt = (1 / (t_max - t_min)) d/dt_scaled
        d/dx = (1 / (x_max - x_min)) d/dx_scaled
        d/dw = (1 / w) d/dx

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

    assert log_N_flat.shape == (n_time * n_eval, n_species), (
        f"model output must be {(n_time * n_eval, n_species)}, "
        f"got {tuple(log_N_flat.shape)}"
    )

    dlogN_dx_scaled_rows = []
    dlogN_dt_scaled_rows = []

    for i in range(n_species):
        grad_i = torch.autograd.grad(
            log_N_flat[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        dlogN_dx_scaled_rows.append(grad_i[:, 0])
        dlogN_dt_scaled_rows.append(grad_i[:, 1])

    log_N = (
        log_N_flat
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )
    N = torch.exp(log_N)

    dlogN_dx_scaled = (
        torch.stack(dlogN_dx_scaled_rows, dim=1)
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )
    dlogN_dt_scaled = (
        torch.stack(dlogN_dt_scaled_rows, dim=1)
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)

    dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
    dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
    dlogN_dw = dlogN_dx / w_eval[None, None, :]

    dN_dt = N * dlogN_dt
    dN_dw = N * dlogN_dw

    expected = (n_time, n_species, n_eval)
    for name, value in {
        "log_N": log_N,
        "N": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,
        "dN_dt": dN_dt,
        "dN_dw": dN_dw,
    }.items():
        assert value.shape == expected, f"{name}: expected {expected}, got {value.shape}"

    return {
        "log_N_eval": log_N,
        "N_eval": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,
        "dN_dt": dN_dt,
        "dN_dw": dN_dw,
    }
