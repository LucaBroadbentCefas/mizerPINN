from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _params_dtype_device, _t_limits, _x_grid, scale_t, scale_x


def sample_pde_batch(
    params: MizerTorchParams,
    n_time: int,
    n_eval: int,
    *,
    t_max_current=None,
) -> dict[str, torch.Tensor]:
    """Sample PDE collocation points in time and log-weight.

    Returns off-grid residual points and fixed grid vectors in both physical and
    scaled coordinates.
    """
    dtype, device = _params_dtype_device(params)
    x_grid = _x_grid(params)
    x_min, x_max = x_grid[0], x_grid[-1]
    t_min, t_max = _t_limits(params)

    t_min_t = torch.as_tensor(t_min, dtype=dtype, device=device)
    t_max_t = torch.as_tensor(t_max, dtype=dtype, device=device)
    if t_max_current is None:
        t_upper = t_max_t
    else:
        t_upper = torch.as_tensor(t_max_current, dtype=dtype, device=device)
        t_upper = torch.maximum(t_upper, t_min_t)
        t_upper = torch.minimum(t_upper, t_max_t)

    if not bool((t_upper > t_min_t).detach().cpu()):
        raise ValueError(f"t_max_current must be greater than t_min. Got t_min={float(t_min_t.detach().cpu())}, t_max_current={float(t_upper.detach().cpu())}.")

    t_eval = t_min_t + (t_upper - t_min_t) * torch.rand(n_time, dtype=dtype, device=device)
    x_eval = x_min + (x_max - x_min) * torch.rand(n_eval, dtype=dtype, device=device)
    w_eval = torch.exp(x_eval)
    return {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": w_eval,
        "x_grid": x_grid,
        "x_grid_scaled": scale_x(x_grid, params),
        "w_grid": params.w,
    }
