from __future__ import annotations

import torch

from PINNmizer.biology.kernels import compute_phi_and_dphi_dw
from PINNmizer.mizer_grid_ops import compute_prey
from PINNmizer.params import MizerTorchParams, _eval_weight_vector, _params_dtype_device, _species_vector


def evaluate_gamma_continuous(w_eval: torch.Tensor, params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    """gamma_i(w)=gamma_i*w^q_i, returning [species,n_eval] for value and derivative."""
    w_eval = _eval_weight_vector(w_eval, params)
    gamma = _species_vector(params, "gamma")[:, None]
    q = _species_vector(params, "q")[:, None]
    w = w_eval[None, :]
    gamma_eval = gamma * w.pow(q)
    dgamma_dw = gamma * q * w.pow(q - 1.0)
    return gamma_eval, dgamma_dw


def compute_encounter_direct_at_eval(n_pp: torch.Tensor, n_grid: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct encounter at physical eval weights.

    Returns:
      encounter_eval: [species, n_eval]
      dencounter_dw:  [species, n_eval]
    """
    dtype, device = _params_dtype_device(params)
    n_pp = n_pp.to(dtype=dtype, device=device)
    n_grid = n_grid.to(dtype=dtype, device=device)
    w_eval = _eval_weight_vector(w_eval, params)
    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    k_full = params.w_full.numel()
    assert n_pp.shape == (k_full,)
    assert n_grid.shape == (n_species, n_w)

    prey_full = compute_prey(n_pp, n_grid, params)
    phi, dphi_dw = compute_phi_and_dphi_dw(w_pred_eval=w_eval, w_prey_grid_or_eval=params.w_full, params=params)
    conv = (prey_full[:, None, :] * phi).sum(dim=-1)
    dconv_dw = (prey_full[:, None, :] * dphi_dw).sum(dim=-1)
    gamma_eval, dgamma_dw = evaluate_gamma_continuous(w_eval, params)
    encounter_eval = gamma_eval * conv
    dencounter_dw = dgamma_dw * conv + gamma_eval * dconv_dw
    return encounter_eval, dencounter_dw
