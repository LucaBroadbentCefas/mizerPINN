import torch

from .params import (
    MizerTorchParams,
    _params_dtype_device,
    _species_vector,
    _eval_weight_vector,
)

from .mizer_grid_ops import compute_prey
from .utils import pos

def compute_encounter_direct_at_eval(
    n_pp: torch.Tensor,
    n_grid: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Direct encounter at arbitrary continuous predator weights.

    E_i(w,t) = gamma_i(w) * sum_p prey_full[i,p,t] * phi_i(w, w_full[p])

    returns:
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

    prey_full = compute_prey(n_pp, n_grid, params)  # [species, k_full]

    phi, dphi_dw = compute_phi_and_dphi_dw(
        w_pred_eval=w_eval,
        w_prey_grid_or_eval=params.w_full,
        params=params,
    )

    # prey_full[:, None, :] -> [species, 1, k_full]
    # phi                  -> [species, n_eval, k_full]
    conv = (prey_full[:, None, :] * phi).sum(dim=-1)
    dconv_dw = (prey_full[:, None, :] * dphi_dw).sum(dim=-1)

    gamma_eval, dgamma_dw = evaluate_gamma_continuous(w_eval, params)

    encounter_eval = gamma_eval * conv
    dencounter_dw = dgamma_dw * conv + gamma_eval * dconv_dw

    return encounter_eval, dencounter_dw

def evaluate_gamma_continuous(
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    gamma_i(w) = gamma_i * w^q_i

    returns:
        gamma_eval: [species, n_eval]
        dgamma_dw:  [species, n_eval]
    """
    w_eval = _eval_weight_vector(w_eval, params)

    gamma = _species_vector(params, "gamma")[:, None]
    q = _species_vector(params, "q")[:, None]

    w = w_eval[None, :]

    gamma_eval = gamma * w.pow(q)
    dgamma_dw = gamma * q * w.pow(q - 1.0)

    return gamma_eval, dgamma_dw

def evaluate_intake_max_continuous(
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    h_i(w) = h_i * w^n_i

    returns:
        h_eval: [species, n_eval]
        dh_dw:  [species, n_eval]
    """
    w_eval = _eval_weight_vector(w_eval, params)

    h = _species_vector(params, "h")[:, None]
    n_exp = _species_vector(params, "n_exp")[:, None]

    w = w_eval[None, :]

    h_eval = h * w.pow(n_exp)
    dh_dw = h * n_exp * w.pow(n_exp - 1.0)

    return h_eval, dh_dw

def evaluate_metab_continuous(
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    metab_i(w) = ks_i * w^p_i + k_i * w

    returns:
        metab_eval: [species, n_eval]
        dmetab_dw:  [species, n_eval]
    """
    w_eval = _eval_weight_vector(w_eval, params)

    ks = _species_vector(params, "ks")[:, None]
    p_exp = _species_vector(params, "p_exp")[:, None]
    k_metab = _species_vector(params, "k_metab")[:, None]

    w = w_eval[None, :]

    metab_eval = ks * w.pow(p_exp) + k_metab * w
    dmetab_dw = ks * p_exp * w.pow(p_exp - 1.0) + k_metab

    return metab_eval, dmetab_dw

def evaluate_psi_continuous(
    w_eval: torch.Tensor,
    params: MizerTorchParams,
    maturity_floor: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    psi_i(w) = maturity_i(w) * repro_prop_i(w)

    Piecewise:
        maturity < 1e-8 -> psi = 0, dpsi/dw = 0
        w >= w_repro_max -> psi = 1, dpsi/dw = 0

    returns:
        psi_eval: [species, n_eval]
        dpsi_dw:  [species, n_eval]
    """
    w_eval = _eval_weight_vector(w_eval, params)

    w_mat = _species_vector(params, "w_mat")[:, None]
    U = _species_vector(params, "U")[:, None]
    w_repro_max = _species_vector(params, "w_repro_max")[:, None]
    m_exp = _species_vector(params, "m_exp")[:, None]
    n_exp = _species_vector(params, "n_exp")[:, None]

    w = w_eval[None, :]

    A = (w / w_mat).pow(-U)
    maturity_raw = 1.0 / (1.0 + A)

    dmaturity_dw_raw = U * maturity_raw * (1.0 - maturity_raw) / w

    exponent = m_exp - n_exp

    repro_prop = (w / w_repro_max).pow(exponent)
    drepro_prop_dw = exponent * repro_prop / w

    psi_raw = maturity_raw * repro_prop
    dpsi_dw_raw = dmaturity_dw_raw * repro_prop + maturity_raw * drepro_prop_dw

    immature_mask = maturity_raw < maturity_floor
    above_repro_mask = w >= w_repro_max

    psi_eval = torch.where(
        immature_mask,
        torch.zeros_like(psi_raw),
        psi_raw,
    )
    dpsi_dw = torch.where(
        immature_mask,
        torch.zeros_like(dpsi_dw_raw),
        dpsi_dw_raw,
    )

    psi_eval = torch.where(
        above_repro_mask,
        torch.ones_like(psi_eval),
        psi_eval,
    )
    dpsi_dw = torch.where(
        above_repro_mask,
        torch.zeros_like(dpsi_dw),
        dpsi_dw,
    )

    return psi_eval, dpsi_dw

def compute_phi_and_dphi_dw(
    w_pred_eval: torch.Tensor,
    w_prey_grid_or_eval: torch.Tensor,
    params: MizerTorchParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log-normal predation kernel.

    returns:
        phi:          [species, n_pred, n_prey]
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

    w_pred = w_pred_eval[None, :, None]          # [1, n_pred, 1]
    w_prey = w_prey_grid_or_eval[None, None, :]  # [1, 1, n_prey]

    ppmr = w_pred / w_prey
    log_term = torch.log(ppmr) - torch.log(beta)

    phi_raw = torch.exp(-(log_term ** 2) / (2.0 * sigma ** 2))

    active = ppmr > 1.0

    phi = torch.where(active, phi_raw, torch.zeros_like(phi_raw))

    dphi_dw_pred = -phi * log_term / (sigma ** 2 * w_pred)
    dphi_dw_pred = torch.where(active, dphi_dw_pred, torch.zeros_like(dphi_dw_pred))

    return phi, dphi_dw_pred

def evaluate_mu_b_continuous(
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    Background mortality.

    If params.mu_b_allometric is False:

        mu_b_i(w) = z0_i

    If params.mu_b_allometric is True:

        z0_i = z0_pre_i * w_inf_i^(1 - n_i)
        mu_b_i(w) = z0_i * w^(n_i - 1)

    returns:
        mu_b_eval: [species, n_eval]
    """
    w_eval = _eval_weight_vector(w_eval, params)

    if not params.mu_b_allometric:
        z0 = _species_vector(params, "z0")[:, None]
        return z0.expand(-1, w_eval.numel())

    z0_pre = _species_vector(params, "z0_pre")[:, None]
    w_inf = _species_vector(params, "w_inf")[:, None]
    n_exp = _species_vector(params, "n_exp")[:, None]

    w = w_eval[None, :]

    z0_i = z0_pre * w_inf.pow(1.0 - n_exp)
    mu_b_eval = z0_i * w.pow(n_exp - 1.0)

    return mu_b_eval


def compute_growth_direct_at_eval(
    n_pp: torch.Tensor,
    n_grid: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
    eps: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Continuous analytical growth-side path.

    No interpolation.
    No autograd for biological derivatives.
    """
    dtype, device = _params_dtype_device(params)

    w_eval = _eval_weight_vector(w_eval, params)
    eps_t = torch.as_tensor(eps, dtype=dtype, device=device)

    gamma_eval, dgamma_dw = evaluate_gamma_continuous(w_eval, params)

    encounter_eval, dencounter_dw = compute_encounter_direct_at_eval(
        n_pp=n_pp,
        n_grid=n_grid,
        w_eval=w_eval,
        params=params,
    )

    h_eval, dh_dw = evaluate_intake_max_continuous(w_eval, params)
    metab_eval, dmetab_dw = evaluate_metab_continuous(w_eval, params)
    psi_eval, dpsi_dw = evaluate_psi_continuous(w_eval, params)

    denom = encounter_eval + h_eval + eps_t

    feeding_eval = encounter_eval / denom

    # For f = E / (E + h + eps):
    # df/dw = [dE * (h + eps) - E * dh] / (E + h + eps)^2
    dfeeding_dw = (
        dencounter_dw * (h_eval + eps_t)
        - encounter_eval * dh_dw
    ) / (denom ** 2)

    alpha = params.alpha.to(dtype=dtype, device=device)[:, None]

    erepog_eval = alpha * (1.0 - feeding_eval) * encounter_eval - metab_eval

    derepog_dw = (
        alpha * ((1.0 - feeding_eval) * dencounter_dw - encounter_eval * dfeeding_dw)
        - dmetab_dw
    )

    pos_erepog = pos(erepog_eval)

    dpos_erepog_dw = torch.where(
        erepog_eval > 0.0,
        derepog_dw,
        torch.zeros_like(derepog_dw),
    )

    e_repro_eval = pos_erepog * psi_eval

    de_repro_dw = dpos_erepog_dw * psi_eval + pos_erepog * dpsi_dw

    e_growth_eval = pos_erepog - e_repro_eval

    dg_dw = dpos_erepog_dw - de_repro_dw

    out = {
        "gamma_eval": gamma_eval,
        "dgamma_dw": dgamma_dw,
        "encounter_eval": encounter_eval,
        "dencounter_dw": dencounter_dw,
        "h_eval": h_eval,
        "dh_dw": dh_dw,
        "feeding_eval": feeding_eval,
        "dfeeding_dw": dfeeding_dw,
        "metab_eval": metab_eval,
        "dmetab_dw": dmetab_dw,
        "erepog_eval": erepog_eval,
        "derepog_dw": derepog_dw,
        "pos_erepog": pos_erepog,
        "dpos_erepog_dw": dpos_erepog_dw,
        "psi_eval": psi_eval,
        "dpsi_dw": dpsi_dw,
        "e_repro_eval": e_repro_eval,
        "de_repro_dw": de_repro_dw,
        "e_growth_eval": e_growth_eval,
        "dg_dw": dg_dw,
    }

    expected_shape = encounter_eval.shape
    for key, value in out.items():
        assert value.shape == expected_shape, f"{key}: expected {expected_shape}, got {value.shape}"

    return out

def compute_pred_mortality_direct_at_eval(
    n_pp: torch.Tensor,
    N_pred_grid: torch.Tensor,
    w_prey_eval: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    Continuous direct predation mortality.

    Uses params.w as the predator integration grid.

    pred_rate_j(w_prey)
        = integral phi_j(w_pred, w_prey)
          * (1 - f_j(w_pred))
          * gamma_j(w_pred)
          * N_j(w_pred, t)
          dw_pred

    Discretised using params.dw.

    returns:
        pred_mort_eval: [species, n_eval]
    """
    dtype, device = _params_dtype_device(params)

    n_pp = n_pp.to(dtype=dtype, device=device)
    N_pred_grid = N_pred_grid.to(dtype=dtype, device=device)
    w_prey_eval = _eval_weight_vector(w_prey_eval, params)

    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    n_eval = w_prey_eval.numel()

    assert N_pred_grid.shape == (n_species, n_w)

    growth_grid = compute_growth_direct_at_eval(
        n_pp=n_pp,
        n_grid=N_pred_grid,
        w_eval=params.w,
        params=params,
    )

    feeding_grid = growth_grid["feeding_eval"]      # [species, n_w]
    gamma_grid = growth_grid["gamma_eval"]          # [species, n_w]

    q_grid = (1.0 - feeding_grid) * gamma_grid * N_pred_grid

    phi, _ = compute_phi_and_dphi_dw(
        w_pred_eval=params.w,
        w_prey_grid_or_eval=w_prey_eval,
        params=params,
    )

    # phi                  -> [predator_species, n_w, n_eval]
    # q_grid[:, :, None]   -> [predator_species, n_w, 1]
    # params.dw[None,:,None] -> [1, n_w, 1]
    pred_rate = (
        phi
        * q_grid[:, :, None]
        * params.dw[None, :, None]
    ).sum(dim=1)

    assert pred_rate.shape == (n_species, n_eval)

    pred_mort_eval = params.interaction.T @ pred_rate

    assert pred_mort_eval.shape == (n_species, n_eval)

    return pred_mort_eval

def compute_total_mortality_direct_at_eval(
    n_pp: torch.Tensor,
    N_pred_grid: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """
    Total continuous mortality for PDE residual:

        mu = mu_b + pred_mort

    Fishing mortality is zero here.
    """
    mu_b_eval = evaluate_mu_b_continuous(w_eval, params)

    pred_mort_eval = compute_pred_mortality_direct_at_eval(
        n_pp=n_pp,
        N_pred_grid=N_pred_grid,
        w_prey_eval=w_eval,
        params=params,
    )

    mu_eval = mu_b_eval + pred_mort_eval

    return {
        "mu_b_eval": mu_b_eval,
        "pred_mort_eval": pred_mort_eval,
        "mu_eval": mu_eval,
    }


def compute_pred_mortality_direct_at_eval_from_growth_grid(
    N_pred_grid: torch.Tensor,
    w_prey_eval: torch.Tensor,
    params: MizerTorchParams,
    growth_grid: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Direct predation mortality using already-computed growth_grid.

    Avoids recomputing compute_growth_direct_at_eval(..., w_eval=params.w).
    """
    dtype, device = _params_dtype_device(params)

    N_pred_grid = N_pred_grid.to(dtype=dtype, device=device)
    w_prey_eval = _eval_weight_vector(w_prey_eval, params)

    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    n_eval = w_prey_eval.numel()

    assert N_pred_grid.shape == (n_species, n_w)

    feeding_grid = growth_grid["feeding_eval"]  # [species, n_w]
    gamma_grid = growth_grid["gamma_eval"]      # [species, n_w]

    q_grid = (1.0 - feeding_grid) * gamma_grid * N_pred_grid

    phi, _ = compute_phi_and_dphi_dw(
        w_pred_eval=params.w,
        w_prey_grid_or_eval=w_prey_eval,
        params=params,
    )

    pred_rate = (
        phi
        * q_grid[:, :, None]
        * params.dw[None, :, None]
    ).sum(dim=1)

    assert pred_rate.shape == (n_species, n_eval)

    pred_mort_eval = params.interaction.T @ pred_rate

    assert pred_mort_eval.shape == (n_species, n_eval)

    return pred_mort_eval
  
def compute_total_mortality_direct_at_eval_from_growth_grid(
    N_pred_grid: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
    growth_grid: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Total continuous mortality using cached grid growth.

        mu = mu_b + pred_mort
    """
    mu_b_eval = evaluate_mu_b_continuous(w_eval, params)

    pred_mort_eval = compute_pred_mortality_direct_at_eval_from_growth_grid(
        N_pred_grid=N_pred_grid,
        w_prey_eval=w_eval,
        params=params,
        growth_grid=growth_grid,
    )

    mu_eval = mu_b_eval + pred_mort_eval

    return {
        "mu_b_eval": mu_b_eval,
        "pred_mort_eval": pred_mort_eval,
        "mu_eval": mu_eval,
    }
    
    
def compute_recruitment_direct_from_growth_grid(
    N_grid: torch.Tensor,
    params: MizerTorchParams,
    growth_grid: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Continuous/direct recruitment flux from the current predicted spectrum.

    Uses cached continuous growth output:

        e_repro(w, t)

    Returns recruitment fluxes, not finite-difference grid-cell updates.
    """
    dtype, device = _params_dtype_device(params)

    N_grid = N_grid.to(dtype=dtype, device=device)

    e_repro_grid = growth_grid["e_repro_eval"]  # [species, n_w]

    assert e_repro_grid.shape == N_grid.shape

    repro_integrand = e_repro_grid * N_grid

    repro_integral = torch.trapz(
        repro_integrand,
        x=params.w,
        dim=1,
    )

    egg_idx = params.w_min_idx.to(torch.long) - 1
    egg_w = params.w[egg_idx]

    rdi_flux = 0.5 * repro_integral * params.erepro / egg_w
    rdd_flux = rdi_flux / (1.0 + rdi_flux / params.r_max)

    return {
        "e_repro_grid": e_repro_grid,
        "repro_integrand": repro_integrand,
        "repro_integral": repro_integral,
        "rdi_flux": rdi_flux,
        "rdd_flux": rdd_flux,
    }
