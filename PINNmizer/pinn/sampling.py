from __future__ import annotations

import math

import torch

from PINNmizer.params import MizerTorchParams, _params_dtype_device, _t_limits, _x_grid, scale_t, scale_x


def make_fixed_pde_batch(
    params: MizerTorchParams,
    n_time: int,
    n_eval: int,
    *,
    t_max_current=None,
    use_mizer_x_grid: bool = False,
) -> dict[str, torch.Tensor]:
    """Build deterministic Cartesian PDE collocation points for training."""
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

    t_eval = torch.linspace(t_min_t, t_upper, n_time, dtype=dtype, device=device)
    if use_mizer_x_grid:
        x_eval = x_grid
    else:
        x_eval = torch.linspace(x_min, x_max, n_eval, dtype=dtype, device=device)
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


def sample_pde_batch(
    params: MizerTorchParams,
    n_time: int,
    n_eval: int,
    *,
    t_max_current=None,
    time_sampling: str = "uniform",
    causal_n_chunks: int = 32,
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

    if time_sampling == "uniform":
        t_eval = t_min_t + (t_upper - t_min_t) * torch.rand(n_time, dtype=dtype, device=device)
        t_chunk_idx = None
    elif time_sampling == "stratified":
        if causal_n_chunks <= 0:
            raise ValueError(f"causal_n_chunks must be positive, got {causal_n_chunks}.")
        samples_per_chunk = max(1, math.ceil(n_time / causal_n_chunks))
        chunk_edges = torch.linspace(
            t_min_t,
            t_upper,
            causal_n_chunks + 1,
            dtype=dtype,
            device=device,
        )
        chunks = []
        idx = []
        for i in range(causal_n_chunks):
            lo = chunk_edges[i]
            hi = chunk_edges[i + 1]
            chunks.append(lo + (hi - lo) * torch.rand(samples_per_chunk, dtype=dtype, device=device))
            idx.append(torch.full((samples_per_chunk,), i, dtype=torch.long, device=device))
        t_eval = torch.cat(chunks, dim=0)
        t_chunk_idx = torch.cat(idx, dim=0)
    else:
        raise ValueError("time_sampling must be 'uniform' or 'stratified'.")

    x_eval = x_min + (x_max - x_min) * torch.rand(n_eval, dtype=dtype, device=device)
    w_eval = torch.exp(x_eval)
    batch = {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": w_eval,
        "x_grid": x_grid,
        "x_grid_scaled": scale_x(x_grid, params),
        "w_grid": params.w,
    }
    if t_chunk_idx is not None:
        batch["t_chunk_idx"] = t_chunk_idx
        batch["effective_n_time"] = torch.as_tensor(t_eval.numel(), device=device)
    return batch
