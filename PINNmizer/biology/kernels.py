from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _params_dtype_device, _species_vector


def compute_phi_and_dphi_dw(
    w_pred_eval: torch.Tensor,
    w_prey_grid_or_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-normal predation kernel.

    Shapes:
      phi: [species, n_pred, n_prey]
      dphi_dw_pred: [species, n_pred, n_prey]
    """
    dtype, device = _params_dtype_device(params)
    w_pred_eval = w_pred_eval.to(dtype=dtype, device=device)
    w_prey_grid_or_eval = w_prey_grid_or_eval.to(dtype=dtype, device=device)
    assert w_pred_eval.ndim == 1
    assert w_prey_grid_or_eval.ndim == 1
    assert torch.all(w_pred_eval > 0)
    assert torch.all(w_prey_grid_or_eval > 0)

    beta = _species_vector(params, "beta")[:, None, None]
    sigma = _species_vector(params, "sigma")[:, None, None]
    w_pred = w_pred_eval[None, :, None]
    w_prey = w_prey_grid_or_eval[None, None, :]
    ppmr = w_pred / w_prey
    log_term = torch.log(ppmr) - torch.log(beta)
    phi_raw = torch.exp(-(log_term ** 2) / (2.0 * sigma ** 2))
    active = ppmr > 1.0
    phi = torch.where(active, phi_raw, torch.zeros_like(phi_raw))
    dphi_dw_pred = -phi * log_term / (sigma ** 2 * w_pred)
    dphi_dw_pred = torch.where(active, dphi_dw_pred, torch.zeros_like(dphi_dw_pred))
    return phi, dphi_dw_pred
