from __future__ import annotations

import torch

from PINNmizer.biology.growth import compute_growth_direct_at_eval
from PINNmizer.biology.mortality import compute_total_mortality_direct_at_eval_from_growth_grid
from PINNmizer.params import active_grid_mask


def _stack(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def compute_environment_biology(*, N_environment_grid: torch.Tensor, n_pp: torch.Tensor,
                                w_eval: torch.Tensor, t_eval: torch.Tensor,
                                params, species_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    if N_environment_grid.ndim != 3:
        raise ValueError("N_environment_grid must have shape [T,S,W].")
    n_time, n_species, n_w = N_environment_grid.shape
    if n_species != params.interaction.shape[0] or n_w != params.w.numel():
        raise ValueError("Environmental state does not match the complete ecosystem dimensions.")
    if t_eval.shape != (n_time,):
        raise ValueError("t_eval must contain one physical time per environmental state.")
    if not 0 <= species_idx < n_species:
        raise IndexError("species_idx is outside the ecosystem species range.")
    active = active_grid_mask(params).to(dtype=N_environment_grid.dtype,
                                         device=N_environment_grid.device)
    growth_eval_all, growth_grid_all, mortality_all = [], [], []
    for tt in range(n_time):
        environment = N_environment_grid[tt] * active
        growth_eval = compute_growth_direct_at_eval(
            n_pp=n_pp, n_grid=environment, w_eval=w_eval, params=params
        )
        growth_grid = compute_growth_direct_at_eval(
            n_pp=n_pp, n_grid=environment, w_eval=params.w, params=params
        )
        mortality = compute_total_mortality_direct_at_eval_from_growth_grid(
            N_pred_grid=environment,
            w_eval=w_eval,
            params=params,
            growth_grid=growth_grid,
            t_eval=t_eval[tt],
        )
        sl = slice(species_idx, species_idx + 1)
        growth_eval_all.append({key: value[sl] for key, value in growth_eval.items()})
        growth_grid_all.append({key: value[sl] for key, value in growth_grid.items()})
        mortality_all.append({key: value[sl] for key, value in mortality.items()})
    return {
        "growth_eval": _stack(growth_eval_all),
        "growth_grid": _stack(growth_grid_all),
        "mortality": _stack(mortality_all),
    }
