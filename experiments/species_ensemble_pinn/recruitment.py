from __future__ import annotations

import torch


def hybrid_recruitment(params, *, species_idx: int, e_repro_environment: torch.Tensor, N_pred_grid: torch.Tensor, active_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Hybrid recruitment integral uses environmental e_repro and predicted physical N=S*U."""
    n_phys = N_pred_grid.squeeze(1) if N_pred_grid.ndim == 3 else N_pred_grid
    integrand = e_repro_environment * n_phys
    integrand = torch.where(active_mask.reshape(1, -1), integrand, torch.zeros_like(integrand))
    integral = torch.trapz(integrand, x=params.w, dim=-1)
    rdi_flux = params.erepro[species_idx] * integral
    rdd_flux = rdi_flux / (1.0 + rdi_flux / torch.clamp(params.r_max[species_idx], min=torch.finfo(rdi_flux.dtype).tiny))
    return {"e_repro_environment": e_repro_environment, "repro_integrand": integrand, "repro_integral": integral, "rdi_flux": rdi_flux, "rdd_flux": rdd_flux}
