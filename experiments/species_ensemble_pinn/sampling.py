from __future__ import annotations

import torch

from PINNmizer.params import _params_dtype_device, _t_limits, _x_grid, scale_t, scale_x


def _rand(shape, *, dtype, device, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.rand(shape, dtype=dtype, device=device)
    sample = torch.rand(shape, dtype=dtype, device="cpu", generator=generator)
    return sample.to(device=device)


def _domain(params, t_max_current=None):
    dtype, device = _params_dtype_device(params)
    x_grid = _x_grid(params)
    t_min, t_max = _t_limits(params)
    upper = t_max if t_max_current is None else torch.as_tensor(t_max_current, dtype=dtype, device=device)
    upper = torch.minimum(torch.maximum(upper, t_min), t_max)
    if not bool((upper > t_min).detach().cpu()):
        raise ValueError("t_max_current must exceed t_min.")
    return dtype, device, x_grid, t_min, upper


def sample_pde_batch(params, n_time: int, n_eval: int, *, t_max_current=None,
                     causal_n_chunks: int = 32,
                     generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    if n_time % causal_n_chunks != 0:
        raise ValueError("n_time must be exactly divisible by causal_n_chunks.")
    dtype, device, x_grid, t_min, t_upper = _domain(params, t_max_current)
    per_chunk = n_time // causal_n_chunks
    edges = torch.linspace(t_min, t_upper, causal_n_chunks + 1, dtype=dtype, device=device)
    times, indices = [], []
    for chunk in range(causal_n_chunks):
        u = _rand((per_chunk,), dtype=dtype, device=device, generator=generator)
        times.append(edges[chunk] + (edges[chunk + 1] - edges[chunk]) * u)
        indices.append(torch.full((per_chunk,), chunk, dtype=torch.long, device=device))
    t_eval = torch.cat(times)
    x_eval = x_grid[0] + (x_grid[-1] - x_grid[0]) * _rand(
        (n_eval,), dtype=dtype, device=device, generator=generator
    )
    return {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "t_chunk_idx": torch.cat(indices),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": torch.exp(x_eval),
        "x_grid": x_grid,
        "x_grid_scaled": scale_x(x_grid, params),
        "w_grid": params.w,
    }


def make_fixed_pde_batch(params, n_time: int, n_eval: int, *, t_max_current=None,
                         causal_n_chunks: int = 32,
                         generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Deterministic relative to the supplied dedicated generator."""
    return sample_pde_batch(
        params, n_time, n_eval, t_max_current=t_max_current,
        causal_n_chunks=causal_n_chunks, generator=generator,
    )
