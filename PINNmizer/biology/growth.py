from __future__ import annotations

import torch

from PINNmizer.biology.encounter import compute_encounter_direct_at_eval, evaluate_gamma_continuous
from PINNmizer.params import MizerTorchParams, _eval_weight_vector, _params_dtype_device, _species_vector
from PINNmizer.utils import pos


def evaluate_intake_max_continuous(w_eval: torch.Tensor, params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    w_eval = _eval_weight_vector(w_eval, params)
    h = _species_vector(params, "h")[:, None]
    n_exp = _species_vector(params, "n_exp")[:, None]
    w = w_eval[None, :]
    h_eval = h * w.pow(n_exp)
    dh_dw = h * n_exp * w.pow(n_exp - 1.0)
    return h_eval, dh_dw


def evaluate_metab_continuous(w_eval: torch.Tensor, params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    w_eval = _eval_weight_vector(w_eval, params)
    ks = _species_vector(params, "ks")[:, None]
    p_exp = _species_vector(params, "p_exp")[:, None]
    k_metab = _species_vector(params, "k_metab")[:, None]
    w = w_eval[None, :]
    metab_eval = ks * w.pow(p_exp) + k_metab * w
    dmetab_dw = ks * p_exp * w.pow(p_exp - 1.0) + k_metab
    return metab_eval, dmetab_dw


def evaluate_psi_continuous(w_eval: torch.Tensor, params: MizerTorchParams, maturity_floor: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
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
    psi_eval = torch.where(immature_mask, torch.zeros_like(psi_raw), psi_raw)
    dpsi_dw = torch.where(immature_mask, torch.zeros_like(dpsi_dw_raw), dpsi_dw_raw)
    psi_eval = torch.where(above_repro_mask, torch.ones_like(psi_eval), psi_eval)
    dpsi_dw = torch.where(above_repro_mask, torch.zeros_like(dpsi_dw), dpsi_dw)
    return psi_eval, dpsi_dw


def compute_growth_direct_at_eval(n_pp: torch.Tensor, n_grid: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams, eps: float = 0.0) -> dict[str, torch.Tensor]:
    """Continuous analytical growth path with manual biology derivatives.

    Derivative chain:
      f = E / (E + h)
      erepog = alpha * (1 - f) * E - metab
      pos_erepog = max(erepog, 0)
      e_repro = pos_erepog * psi
      g = pos_erepog - e_repro
      dg_dw = dpos_erepog_dw - de_repro_dw

    Note: non-differentiable kink at erepog = 0 due to positive-part operator.
    """
    dtype, device = _params_dtype_device(params)
    w_eval = _eval_weight_vector(w_eval, params)
    eps_t = torch.as_tensor(eps, dtype=dtype, device=device)
    gamma_eval, dgamma_dw = evaluate_gamma_continuous(w_eval, params)
    encounter_eval, dencounter_dw = compute_encounter_direct_at_eval(n_pp=n_pp, n_grid=n_grid, w_eval=w_eval, params=params)
    h_eval, dh_dw = evaluate_intake_max_continuous(w_eval, params)
    metab_eval, dmetab_dw = evaluate_metab_continuous(w_eval, params)
    psi_eval, dpsi_dw = evaluate_psi_continuous(w_eval, params)
    denom = encounter_eval + h_eval + eps_t
    feeding_eval = encounter_eval / denom
    dfeeding_dw = (dencounter_dw * (h_eval + eps_t) - encounter_eval * dh_dw) / (denom ** 2)
    alpha = params.alpha.to(dtype=dtype, device=device)[:, None]
    erepog_eval = alpha * (1.0 - feeding_eval) * encounter_eval - metab_eval
    derepog_dw = alpha * ((1.0 - feeding_eval) * dencounter_dw - encounter_eval * dfeeding_dw) - dmetab_dw
    pos_erepog = pos(erepog_eval)
    dpos_erepog_dw = torch.where(erepog_eval > 0.0, derepog_dw, torch.zeros_like(derepog_dw))
    e_repro_eval = pos_erepog * psi_eval
    de_repro_dw = dpos_erepog_dw * psi_eval + pos_erepog * dpsi_dw
    e_growth_eval = pos_erepog - e_repro_eval
    dg_dw = dpos_erepog_dw - de_repro_dw
    out = {"gamma_eval": gamma_eval, "dgamma_dw": dgamma_dw, "encounter_eval": encounter_eval, "dencounter_dw": dencounter_dw, "h_eval": h_eval, "dh_dw": dh_dw, "feeding_eval": feeding_eval, "dfeeding_dw": dfeeding_dw, "metab_eval": metab_eval, "dmetab_dw": dmetab_dw, "erepog_eval": erepog_eval, "derepog_dw": derepog_dw, "pos_erepog": pos_erepog, "dpos_erepog_dw": dpos_erepog_dw, "psi_eval": psi_eval, "dpsi_dw": dpsi_dw, "e_repro_eval": e_repro_eval, "de_repro_dw": de_repro_dw, "e_growth_eval": e_growth_eval, "dg_dw": dg_dw}
    expected_shape = encounter_eval.shape
    for key, value in out.items():
        assert value.shape == expected_shape, f"{key}: expected {expected_shape}, got {value.shape}"
    return out
