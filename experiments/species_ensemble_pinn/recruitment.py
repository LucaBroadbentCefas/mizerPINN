from __future__ import annotations

import torch


def compute_hybrid_recruitment(*, N_target_grid: torch.Tensor,
                               growth_grid_environment: dict[str, torch.Tensor],
                               params, species_idx: int) -> dict[str, torch.Tensor]:
    if N_target_grid.ndim != 3 or N_target_grid.shape[1] != 1:
        raise ValueError("N_target_grid must have shape [T,1,W].")
    e_repro = growth_grid_environment["e_repro_eval"]
    if e_repro.shape != N_target_grid.shape:
        raise ValueError("Environmental e_repro and predicted target abundance shapes differ.")
    integrand = e_repro * N_target_grid
    integral = torch.trapz(integrand, x=params.w, dim=2)
    egg_idx = int(params.w_min_idx[species_idx].item()) - 1
    if not 0 <= egg_idx < params.w.numel():
        raise ValueError("Target egg index is outside params.w.")
    egg_w = params.w[egg_idx]
    erepro = params.erepro[species_idx]
    r_max = params.r_max[species_idx]
    rdi = 0.5 * erepro * integral / egg_w
    rdd = rdi / (1.0 + rdi / r_max)
    return {
        "e_repro_environment": e_repro,
        "repro_integrand": integrand,
        "repro_integral": integral,
        "rdi_flux": rdi,
        "rdd_flux": rdd,
    }
