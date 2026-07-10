from __future__ import annotations

import torch

from PINNmizer.biology.growth import compute_growth_direct_at_eval
from PINNmizer.biology.kernels import compute_phi_and_dphi_dw
from PINNmizer.biology.fishing import evaluate_fishing_mortality_direct
from PINNmizer.params import MizerTorchParams, _eval_weight_vector, _params_dtype_device, _species_vector


def evaluate_mu_b_continuous(w_eval: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    """Background mortality at physical weights, output [species, n_eval]."""
    w_eval = _eval_weight_vector(w_eval, params)
    if not params.mu_b_allometric:
        z0 = _species_vector(params, "z0")[:, None]
        return z0.expand(-1, w_eval.numel())
    z0_pre = _species_vector(params, "z0_pre")[:, None]
    w_inf = _species_vector(params, "w_inf")[:, None]
    n_exp = _species_vector(params, "n_exp")[:, None]
    w = w_eval[None, :]
    z0_i = z0_pre * w_inf.pow(1.0 - n_exp)
    return z0_i * w.pow(n_exp - 1.0)


def compute_pred_mortality_direct_at_eval(n_pp: torch.Tensor, N_pred_grid: torch.Tensor, w_prey_eval: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    n_pp = n_pp.to(dtype=dtype, device=device)
    N_pred_grid = N_pred_grid.to(dtype=dtype, device=device)
    w_prey_eval = _eval_weight_vector(w_prey_eval, params)
    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    n_eval = w_prey_eval.numel()
    assert N_pred_grid.shape == (n_species, n_w)
    growth_grid = compute_growth_direct_at_eval(n_pp=n_pp, n_grid=N_pred_grid, w_eval=params.w, params=params)
    feeding_grid = growth_grid["feeding_eval"]
    gamma_grid = growth_grid["gamma_eval"]
    q_grid = (1.0 - feeding_grid) * gamma_grid * N_pred_grid
    phi, _ = compute_phi_and_dphi_dw(w_pred_eval=params.w, w_prey_grid_or_eval=w_prey_eval, params=params)
    pred_rate = (phi * q_grid[:, :, None] * params.dw[None, :, None]).sum(dim=1)
    assert pred_rate.shape == (n_species, n_eval)
    pred_mort_eval = params.interaction.T @ pred_rate
    assert pred_mort_eval.shape == (n_species, n_eval)
    return pred_mort_eval


def compute_total_mortality_direct_at_eval(n_pp: torch.Tensor, N_pred_grid: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams, t_eval=None) -> dict[str, torch.Tensor]:
    mu_b_eval = evaluate_mu_b_continuous(w_eval, params)
    pred_mort_eval = compute_pred_mortality_direct_at_eval(n_pp=n_pp, N_pred_grid=N_pred_grid, w_prey_eval=w_eval, params=params)
    f_mort_eval = evaluate_fishing_mortality_direct(w_eval, params, t_eval=t_eval)
    mu_eval = mu_b_eval + pred_mort_eval + f_mort_eval
    return {"mu_b_eval": mu_b_eval, "pred_mort_eval": pred_mort_eval, "f_mort_eval": f_mort_eval, "mu_eval": mu_eval}


def compute_pred_mortality_direct_at_eval_from_growth_grid(N_pred_grid: torch.Tensor, w_prey_eval: torch.Tensor, params: MizerTorchParams, growth_grid: dict[str, torch.Tensor]) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    N_pred_grid = N_pred_grid.to(dtype=dtype, device=device)
    w_prey_eval = _eval_weight_vector(w_prey_eval, params)
    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    n_eval = w_prey_eval.numel()
    assert N_pred_grid.shape == (n_species, n_w)
    feeding_grid = growth_grid["feeding_eval"]
    gamma_grid = growth_grid["gamma_eval"]
    q_grid = (1.0 - feeding_grid) * gamma_grid * N_pred_grid
    phi, _ = compute_phi_and_dphi_dw(w_pred_eval=params.w, w_prey_grid_or_eval=w_prey_eval, params=params)
    pred_rate = (phi * q_grid[:, :, None] * params.dw[None, :, None]).sum(dim=1)
    assert pred_rate.shape == (n_species, n_eval)
    pred_mort_eval = params.interaction.T @ pred_rate
    assert pred_mort_eval.shape == (n_species, n_eval)
    return pred_mort_eval


def compute_total_mortality_direct_at_eval_from_growth_grid(N_pred_grid: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams, growth_grid: dict[str, torch.Tensor], t_eval=None) -> dict[str, torch.Tensor]:
    """Total mortality at physical weights, output terms shaped [species, n_eval]."""
    mu_b_eval = evaluate_mu_b_continuous(w_eval, params)
    pred_mort_eval = compute_pred_mortality_direct_at_eval_from_growth_grid(N_pred_grid=N_pred_grid, w_prey_eval=w_eval, params=params, growth_grid=growth_grid)
    f_mort_eval = evaluate_fishing_mortality_direct(w_eval, params, t_eval=t_eval)
    mu_eval = mu_b_eval + pred_mort_eval + f_mort_eval
    return {"mu_b_eval": mu_b_eval, "pred_mort_eval": pred_mort_eval, "f_mort_eval": f_mort_eval, "mu_eval": mu_eval}
