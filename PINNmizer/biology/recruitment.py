from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _params_dtype_device


def compute_recruitment_direct_from_growth_grid(N_grid: torch.Tensor, params: MizerTorchParams, growth_grid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute recruitment boundary flux from growth-grid reproduction output.

    Recruitment here is a boundary flux, not a finite-difference state update.
    """
    dtype, device = _params_dtype_device(params)
    N_grid = N_grid.to(dtype=dtype, device=device)
    e_repro_grid = growth_grid["e_repro_eval"]
    assert e_repro_grid.shape == N_grid.shape
    repro_integrand = e_repro_grid * N_grid
    repro_integral = (repro_integrand * params.dw[None, :]).sum(dim=1)
    egg_idx = params.w_min_idx.to(torch.long) - 1
    egg_w = params.w[egg_idx]
    rdi_flux = 0.5 * repro_integral * params.erepro / egg_w
    rdd_flux = rdi_flux / (1.0 + rdi_flux / params.r_max)
    return {"e_repro_grid": e_repro_grid, "repro_integrand": repro_integrand, "repro_integral": repro_integral, "rdi_flux": rdi_flux, "rdd_flux": rdd_flux}
