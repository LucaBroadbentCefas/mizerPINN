from __future__ import annotations

import torch

from PINNmizer.biology.growth import compute_growth_direct_at_eval
from PINNmizer.biology.mortality import compute_total_mortality_direct_at_eval_from_growth_grid
from PINNmizer.biology.recruitment import compute_recruitment_direct_from_growth_grid
from PINNmizer.params import (
    MizerTorchParams,
    _params_dtype_device,
    _t_limits,
    scale_t,
    active_grid_mask,
)
from PINNmizer.pinn.derivatives import (
    evaluate_log_model_with_derivatives_at_eval,
    evaluate_log_model_with_derivatives_at_pairs,
    evaluate_log_model_with_derivatives_at_slabs,
)
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points


def _stack_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = dicts[0].keys()
    return {key: torch.stack([d[key] for d in dicts], dim=0) for key in keys}

def _cat_eval_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = dicts[0].keys()
    return {key: torch.cat([d[key] for d in dicts], dim=1) for key in keys}


def _physical_time_from_batch(batch: dict[str, torch.Tensor], physical_key: str, scaled_key: str, idx: int, params: MizerTorchParams) -> torch.Tensor:
    if physical_key in batch:
        return batch[physical_key][idx]
    t_min, t_max = _t_limits(params)
    t_min = torch.as_tensor(t_min, dtype=batch[scaled_key].dtype, device=batch[scaled_key].device)
    t_max = torch.as_tensor(t_max, dtype=batch[scaled_key].dtype, device=batch[scaled_key].device)
    return t_min + batch[scaled_key][idx] * (t_max - t_min)

def compute_pde_state(model, batch: dict[str, torch.Tensor], params: MizerTorchParams, n_pp: torch.Tensor, *, include_ic: bool = False) -> dict[str, object]:
    """Build cached PDE state: off-grid NN derivs, fixed-grid NN values, growth, mortality, recruitment, and optional IC."""
    dtype, device = _params_dtype_device(params)
    w_eval = batch["w_eval"]
    x_eval_scaled = batch["x_eval_scaled"]
    t_scaled = batch["t_scaled"]
    x_grid_scaled = batch["x_grid_scaled"]
    n_time = t_scaled.numel()

    eval_derivs = evaluate_log_model_with_derivatives_at_eval(model=model, x_eval_scaled=x_eval_scaled, t_scaled=t_scaled, w_eval=w_eval, params=params)

    if include_ic:
        t_min, _ = _t_limits(params)
        t0 = t_min.reshape(1).to(dtype=dtype, device=device) if torch.is_tensor(t_min) else torch.tensor([t_min], dtype=dtype, device=device)
        t0_scaled = scale_t(t0, params)
        t_grid_scaled = torch.cat([t_scaled, t0_scaled], dim=0)
    else:
        t_grid_scaled = t_scaled

    grid_eval_all = evaluate_log_model_on_points(model=model, x_scaled=x_grid_scaled, t_scaled=t_grid_scaled, params=params)
    if include_ic:
        log_N_grid, N_grid = grid_eval_all["log_N"][:n_time], grid_eval_all["N"][:n_time]
        log_N_ic, N_ic = grid_eval_all["log_N"][n_time:], grid_eval_all["N"][n_time:]
    else:
        log_N_grid, N_grid = grid_eval_all["log_N"], grid_eval_all["N"]
        log_N_ic, N_ic = None, None

    growth_eval_by_time, growth_grid_by_time, mortality_by_time, recruitment_by_time = [], [], [], []
    for tt in range(n_time):
        N_t = N_grid[tt]
        active_mask = active_grid_mask(params).to(dtype=N_t.dtype, device=N_t.device)
        N_t_bio = N_t * active_mask
        growth_eval_t = compute_growth_direct_at_eval(n_pp=n_pp, n_grid=N_t_bio, w_eval=w_eval, params=params)
        growth_grid_t = compute_growth_direct_at_eval(n_pp=n_pp, n_grid=N_t_bio, w_eval=params.w, params=params)
        mortality_t = compute_total_mortality_direct_at_eval_from_growth_grid(N_pred_grid=N_t_bio, w_eval=w_eval, params=params, growth_grid=growth_grid_t, t_eval=_physical_time_from_batch(batch, "t_eval", "t_scaled", tt, params))
        recruitment_t = compute_recruitment_direct_from_growth_grid(N_grid=N_t_bio, params=params, growth_grid=growth_grid_t)
        growth_eval_by_time.append(growth_eval_t)
        growth_grid_by_time.append(growth_grid_t)
        mortality_by_time.append(mortality_t)
        recruitment_by_time.append(recruitment_t)

    return {
        "batch": batch,
        "eval_derivs": eval_derivs,
        "log_N_grid": log_N_grid,
        "N_grid": N_grid,
        "log_N_ic": log_N_ic,
        "N_ic": N_ic,
        **({k + "_grid": v[:n_time] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}} if "n_time" in locals() else {}),
        **({k + "_ic": v[n_time:] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}} if include_ic and "n_time" in locals() else {}),
        "growth_eval": _stack_dicts(growth_eval_by_time),
        "growth_grid": _stack_dicts(growth_grid_by_time),
        "mortality": _stack_dicts(mortality_by_time),
        "recruitment": _stack_dicts(recruitment_by_time),
    }

def compute_pde_state_paired(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    *,
    include_ic: bool = False,
) -> dict[str, object]:
    """PDE state for paired R3 points. Residual terms are [n_species, n_pair]."""
    dtype, device = _params_dtype_device(params)

    w_pair = batch["w_pair"].to(dtype=dtype, device=device)
    x_pair_scaled = batch["x_pair_scaled"].to(dtype=dtype, device=device)
    t_pair_scaled = batch["t_pair_scaled"].to(dtype=dtype, device=device)
    x_grid_scaled = batch["x_grid_scaled"].to(dtype=dtype, device=device)

    n_pair = w_pair.numel()

    eval_derivs = evaluate_log_model_with_derivatives_at_pairs(
        model=model,
        x_scaled_pair=x_pair_scaled,
        t_scaled_pair=t_pair_scaled,
        w_pair=w_pair,
        params=params,
    )

    if include_ic:
        t_min, _ = _t_limits(params)
        t0 = t_min.reshape(1).to(dtype=dtype, device=device)
        t_grid_scaled = torch.cat([t_pair_scaled, scale_t(t0, params)], dim=0)
    else:
        t_grid_scaled = t_pair_scaled

    grid_eval_all = evaluate_log_model_on_points(
        model=model,
        x_scaled=x_grid_scaled,
        t_scaled=t_grid_scaled,
        params=params,
    )

    if include_ic:
        log_N_grid = grid_eval_all["log_N"][:n_pair]
        N_grid = grid_eval_all["N"][:n_pair]
        log_N_ic = grid_eval_all["log_N"][n_pair:]
        N_ic = grid_eval_all["N"][n_pair:]
    else:
        log_N_grid = grid_eval_all["log_N"]
        N_grid = grid_eval_all["N"]
        log_N_ic = None
        N_ic = None

    active_mask = active_grid_mask(params).to(dtype=dtype, device=device)

    growth_eval = []
    growth_grid = []
    mortality = []
    recruitment = []

    for j in range(n_pair):
        N_t = N_grid[j] * active_mask
        w_j = w_pair[j : j + 1]

        growth_eval_j = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=w_j,
            params=params,
        )

        growth_grid_j = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=params.w,
            params=params,
        )

        mortality_j = compute_total_mortality_direct_at_eval_from_growth_grid(
            N_pred_grid=N_t,
            w_eval=w_j,
            params=params,
            growth_grid=growth_grid_j,
            t_eval=_physical_time_from_batch(batch, "t_pair", "t_pair_scaled", j, params),
        )

        recruitment_j = compute_recruitment_direct_from_growth_grid(
            N_grid=N_t,
            params=params,
            growth_grid=growth_grid_j,
        )

        growth_eval.append(growth_eval_j)
        growth_grid.append(growth_grid_j)
        mortality.append(mortality_j)
        recruitment.append(recruitment_j)

    return {
        "batch": batch,
        "eval_derivs": eval_derivs,
        "log_N_grid": log_N_grid,
        "N_grid": N_grid,
        "log_N_ic": log_N_ic,
        "N_ic": N_ic,
        **({k + "_grid": v[:n_pair] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}}),
        **({k + "_ic": v[n_pair:] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}} if include_ic else {}),
        "growth_eval": _cat_eval_dicts(growth_eval),
        "growth_grid": _stack_dicts(growth_grid),
        "mortality": _cat_eval_dicts(mortality),
        "recruitment": _stack_dicts(recruitment),
    }

def compute_pde_state_r3_slabbed(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    *,
    include_ic: bool = False,
) -> dict[str, object]:
    """PDE state for slabbed R3 points.

    Residual terms are [K, n_species, M], where:

        K = number of fixed time slabs
        M = number of R3 x-points per time slab

    The biological operators run once per time slab, not once per residual point.
    """
    dtype, device = _params_dtype_device(params)

    w_slab = batch["w_slab"].to(dtype=dtype, device=device)
    x_slab_scaled = batch["x_slab_scaled"].to(dtype=dtype, device=device)
    t_slab_scaled = batch["t_slab_scaled"].to(dtype=dtype, device=device)
    x_grid_scaled = batch["x_grid_scaled"].to(dtype=dtype, device=device)

    if w_slab.ndim != 2:
        raise ValueError(f"w_slab must be [K, M], got {tuple(w_slab.shape)}.")
    if x_slab_scaled.shape != w_slab.shape:
        raise ValueError(
            f"x_slab_scaled shape {tuple(x_slab_scaled.shape)} must match "
            f"w_slab shape {tuple(w_slab.shape)}."
        )
    if t_slab_scaled.ndim != 1:
        raise ValueError(f"t_slab_scaled must be [K], got {tuple(t_slab_scaled.shape)}.")
    if t_slab_scaled.numel() != w_slab.shape[0]:
        raise ValueError(
            f"t_slab_scaled length {t_slab_scaled.numel()} does not match "
            f"w_slab K={w_slab.shape[0]}."
        )

    n_time = t_slab_scaled.numel()

    eval_derivs = evaluate_log_model_with_derivatives_at_slabs(
        model=model,
        x_slab_scaled=x_slab_scaled,
        t_slab_scaled=t_slab_scaled,
        w_slab=w_slab,
        params=params,
    )

    if include_ic:
        t_min, _ = _t_limits(params)
        t0 = t_min.reshape(1).to(dtype=dtype, device=device)
        t_grid_scaled = torch.cat([t_slab_scaled, scale_t(t0, params)], dim=0)
    else:
        t_grid_scaled = t_slab_scaled

    grid_eval_all = evaluate_log_model_on_points(
        model=model,
        x_scaled=x_grid_scaled,
        t_scaled=t_grid_scaled,
        params=params,
    )

    if include_ic:
        log_N_grid = grid_eval_all["log_N"][:n_time]
        N_grid = grid_eval_all["N"][:n_time]
        log_N_ic = grid_eval_all["log_N"][n_time:]
        N_ic = grid_eval_all["N"][n_time:]
    else:
        log_N_grid = grid_eval_all["log_N"]
        N_grid = grid_eval_all["N"]
        log_N_ic = None
        N_ic = None

    active_mask = active_grid_mask(params).to(dtype=dtype, device=device)

    growth_eval_by_time = []
    growth_grid_by_time = []
    mortality_by_time = []
    recruitment_by_time = []

    for k in range(n_time):
        N_t = N_grid[k] * active_mask
        w_eval_k = w_slab[k]

        growth_eval_k = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=w_eval_k,
            params=params,
        )

        growth_grid_k = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=params.w,
            params=params,
        )

        mortality_k = compute_total_mortality_direct_at_eval_from_growth_grid(
            N_pred_grid=N_t,
            w_eval=w_eval_k,
            params=params,
            growth_grid=growth_grid_k,
            t_eval=_physical_time_from_batch(batch, "t_slab", "t_slab_scaled", k, params),
        )

        recruitment_k = compute_recruitment_direct_from_growth_grid(
            N_grid=N_t,
            params=params,
            growth_grid=growth_grid_k,
        )

        growth_eval_by_time.append(growth_eval_k)
        growth_grid_by_time.append(growth_grid_k)
        mortality_by_time.append(mortality_k)
        recruitment_by_time.append(recruitment_k)

    return {
        "batch": batch,
        "eval_derivs": eval_derivs,
        "log_N_grid": log_N_grid,
        "N_grid": N_grid,
        "log_N_ic": log_N_ic,
        "N_ic": N_ic,
        **({k + "_grid": v[:n_time] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}}),
        **({k + "_ic": v[n_time:] for k, v in grid_eval_all.items() if k in {"log_U", "U", "log_S", "S"}} if include_ic else {}),
        "growth_eval": _stack_dicts(growth_eval_by_time),
        "growth_grid": _stack_dicts(growth_grid_by_time),
        "mortality": _stack_dicts(mortality_by_time),
        "recruitment": _stack_dicts(recruitment_by_time),
    }
