from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.params import MizerTorchParams
from PINNmizer.io_observations import load_observation_csv
from PINNmizer.pinn.observation_operators import biomass_prediction, catch_prediction, observation_time_grid
from PINNmizer.pinn.data_losses import lognormal_nll
from PINNmizer.biology.fishing import evaluate_fishing_mortality_direct


def make_params(time_varying: bool = True):
    dtype = torch.float64
    w = torch.tensor([1.0, 2.0, 4.0], dtype=dtype)
    nsp = 1
    return MizerTorchParams(
        w_full=w, w=w, dw_full=torch.ones_like(w), dw=torch.tensor([0.5, 1.0, 2.0], dtype=dtype), w_min_idx=torch.tensor([1]),
        ft_pred_kernel_e=torch.zeros((1, 3), dtype=torch.complex128), ft_pred_kernel_p=torch.zeros((1, 3), dtype=torch.complex128), ft_mask=torch.ones((1, 3), dtype=dtype),
        search_vol=torch.ones((1, 3), dtype=dtype), intake_max=torch.ones((1, 3), dtype=dtype), alpha=torch.ones(1, dtype=dtype), metab=torch.zeros((1, 3), dtype=dtype), psi=torch.zeros((1, 3), dtype=dtype), mu_b=torch.zeros((1, 3), dtype=dtype),
        interaction_resource=torch.ones(1, dtype=dtype), interaction=torch.ones((1, 1), dtype=dtype), erepro=torch.ones(1, dtype=dtype), r_max=torch.ones(1, dtype=dtype), rr_pp=torch.ones(3, dtype=dtype), cc_pp=torch.ones(3, dtype=dtype),
        catchability=torch.tensor([[2.0]], dtype=dtype), selectivity=torch.tensor([[[1.0, 0.5, 0.25]]], dtype=dtype), initial_effort=torch.tensor([1.0], dtype=dtype),
        fishing_effort_time=torch.tensor([0.0, 1.0], dtype=dtype) if time_varying else None,
        fishing_effort=torch.tensor([[1.0], [3.0]], dtype=dtype) if time_varying else None,
        f_mort=None, gamma=torch.ones(1, dtype=dtype), q=torch.ones(1, dtype=dtype), h=torch.ones(1, dtype=dtype), n_exp=torch.ones(1, dtype=dtype), ks=torch.ones(1, dtype=dtype), p_exp=torch.ones(1, dtype=dtype), k_metab=torch.ones(1, dtype=dtype), beta=torch.ones(1, dtype=dtype), sigma=torch.ones(1, dtype=dtype), w_max=torch.tensor([4.0], dtype=dtype), w_mat=torch.ones(1, dtype=dtype), U=torch.ones(1, dtype=dtype), w_repro_max=torch.ones(1, dtype=dtype), m_exp=torch.ones(1, dtype=dtype), z0=torch.zeros(1, dtype=dtype), z0_pre=torch.zeros(1, dtype=dtype), w_inf=torch.ones(1, dtype=dtype), t_min=0.0, t_max=1.0,
    )


def main():
    params = make_params()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "obs.csv"
        pd.DataFrame([
            {"obs_type": "biomass", "species_idx": 0, "t_start": 0.0, "value": 10.0},
            {"obs_type": "survey_biomass", "species_idx": 0, "t_start": 0.0, "w_min": 1.0, "w_max": 2.0, "value": 5.0, "q": 0.5},
            {"obs_type": "catch_total", "species_idx": 0, "t_start": 0.0, "value": 3.0, "cv": 0.2},
            {"obs_type": "catch_gear", "species_idx": 0, "gear_idx": 0, "t_start": 0.0, "value": 3.0, "sd_log": 0.1},
        ]).to_csv(path, index=False)
        obs = load_observation_csv(path, params, default_cv=0.3)
        assert obs["value"].dtype == params.w.dtype and obs["value"].device == params.w.device
        assert torch.all(obs["t_end"] == obs["t_start"])
        for bad_rows in [[{"obs_type": "gamma", "species_idx": 0, "t_start": 0, "value": 1}], [{"obs_type": "catch_gear", "species_idx": 0, "t_start": 0, "value": 1}]]:
            bad = Path(td) / "bad.csv"; pd.DataFrame(bad_rows).to_csv(bad, index=False)
            try:
                load_observation_csv(bad, params)
                raise AssertionError("invalid CSV was accepted")
            except ValueError:
                pass

    N_grid = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]], dtype=torch.float64)
    t_grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    b = biomass_prediction(N_grid, t_grid, params, torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([4.0]))
    manual_b = (N_grid[0, 0] * params.w * params.dw).sum()
    assert torch.allclose(b, manual_b.reshape(1))

    y = catch_prediction(N_grid, t_grid, params, torch.tensor([0]), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([4.0]), gear_specific=True)
    F = torch.tensor([3.0 * 2.0 * 1.0, 3.0 * 2.0 * 0.5, 3.0 * 2.0 * 0.25], dtype=torch.float64)
    manual_y = (F * N_grid[1, 0] * params.w * params.dw).sum()
    assert torch.allclose(y, manual_y.reshape(1))

    annual = catch_prediction(N_grid, t_grid, params, torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([4.0]), gear_specific=True)
    F0 = torch.tensor([1.0 * 2.0 * 1.0, 1.0 * 2.0 * 0.5, 1.0 * 2.0 * 0.25], dtype=torch.float64)
    rate0 = (F0 * N_grid[0, 0] * params.w * params.dw).sum()
    rate1 = (F * N_grid[1, 0] * params.w * params.dw).sum()
    assert torch.allclose(annual, (0.5 * (rate0 + rate1)).reshape(1))

    N_with_midpoint = torch.stack([N_grid[0], torch.full_like(N_grid[0], 1e6), N_grid[1]])
    annual_with_midpoint = catch_prediction(N_with_midpoint, torch.tensor([0.0, 0.5, 1.0]), params, torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([4.0]), gear_specific=True)
    assert torch.allclose(annual_with_midpoint, annual)

    q = 10
    interval_batch = {
        "t_start": torch.tensor([0.0], dtype=torch.float64),
        "t_end": torch.tensor([1.0], dtype=torch.float64),
    }
    t_quadrature = observation_time_grid(interval_batch, data_time_quadrature_points=q)
    assert t_quadrature.numel() == q + 1
    assert torch.allclose(t_quadrature, torch.linspace(0.0, 1.0, q + 1, dtype=torch.float64))

    N_quadrature = torch.stack([N_grid[0] + t * (N_grid[1] - N_grid[0]) for t in t_quadrature])
    annual_10 = catch_prediction(
        N_quadrature,
        t_quadrature,
        params,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([4.0]),
        gear_specific=True,
        data_time_quadrature_points=q,
    )
    rates_10 = []
    selectivity = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    for k in range(q):
        effort_k = 1.0 + 2.0 * t_quadrature[k]
        F_k = effort_k * 2.0 * selectivity
        rates_10.append((F_k * N_quadrature[k, 0] * params.w * params.dw).sum())
    assert torch.allclose(annual_10, torch.stack(rates_10).mean().reshape(1))

    equal = lognormal_nll(torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([0.5]))
    assert torch.allclose(equal["loss_data"], torch.log(torch.tensor(0.5)))
    twice = lognormal_nll(torch.tensor([4.0]), torch.tensor([2.0]), torch.tensor([1.0]))
    assert torch.allclose(torch.abs(twice["log_residual"]), torch.log(torch.tensor([2.0])))

    F0_direct = evaluate_fishing_mortality_direct(params.w, params, t_eval=torch.tensor(0.0))
    F1_direct = evaluate_fishing_mortality_direct(params.w, params, t_eval=torch.tensor(1.0))
    assert not torch.allclose(F0_direct, F1_direct)
    fixed = make_params(time_varying=False)
    assert torch.allclose(evaluate_fishing_mortality_direct(fixed.w, fixed, t_eval=torch.tensor(0.0)), evaluate_fishing_mortality_direct(fixed.w, fixed, t_eval=torch.tensor(1.0)))
    print("data likelihood smoke checks passed")


if __name__ == "__main__":
    main()
