#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected the target block exactly once, found {count}. "
            "The repository may have changed; inspect the file manually."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    print(f"Updated {path}")


root = Path.cwd()
if not (root / "PINNmizer").is_dir():
    raise RuntimeError("Run this script from the mizerPINN repository root.")

observation_path = root / "PINNmizer/pinn/observation_operators.py"

replace_once(
    observation_path,
    '''def _time_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    mask = (t_grid >= torch.minimum(t0, t1)) & (t_grid <= torch.maximum(t0, t1))
    if bool(mask.any().detach().cpu()):
        return torch.nonzero(mask, as_tuple=False).reshape(-1)
    mid = 0.5 * (t0 + t1)
    return torch.argmin(torch.abs(t_grid - mid)).reshape(1)


def _endpoint_indices''',
    '''def _time_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    mask = (t_grid >= torch.minimum(t0, t1)) & (t_grid <= torch.maximum(t0, t1))
    if bool(mask.any().detach().cpu()):
        return torch.nonzero(mask, as_tuple=False).reshape(-1)
    mid = 0.5 * (t0 + t1)
    return torch.argmin(torch.abs(t_grid - mid)).reshape(1)


def observation_time_grid(
    observation_batch: dict[str, object],
    data_time_quadrature_points: int = 1,
) -> torch.Tensor:
    """Build observation evaluation times, including left-closed interval points."""
    if data_time_quadrature_points < 1:
        raise ValueError("data_time_quadrature_points must be at least 1.")

    t_start = observation_batch["t_start"]
    t_end = observation_batch["t_end"]
    obs_times = [t_start, t_end]

    if data_time_quadrature_points > 1:
        fractions = torch.arange(
            data_time_quadrature_points,
            dtype=t_start.dtype,
            device=t_start.device,
        ) / float(data_time_quadrature_points)
        interval_times = (
            t_start[:, None]
            + (t_end - t_start)[:, None] * fractions[None, :]
        )
        obs_times.append(interval_times.reshape(-1))

    return torch.unique(torch.cat(obs_times)).sort().values


def _endpoint_indices''',
)

replace_once(
    observation_path,
    '''def _endpoint_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """Return the grid indices for an instantaneous time or interval endpoints."""
    i0 = torch.argmin(torch.abs(t_grid - t0)).reshape(1)
    i1 = torch.argmin(torch.abs(t_grid - t1)).reshape(1)
    if bool((i0 == i1).all().detach().cpu()):
        return i0
    return torch.cat([i0, i1])


def biomass_prediction''',
    '''def _endpoint_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """Return the grid indices for an instantaneous time or interval endpoints."""
    i0 = torch.argmin(torch.abs(t_grid - t0)).reshape(1)
    i1 = torch.argmin(torch.abs(t_grid - t1)).reshape(1)
    if bool((i0 == i1).all().detach().cpu()):
        return i0
    return torch.cat([i0, i1])


def _left_closed_interval_indices(
    t_grid: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
) -> torch.Tensor:
    """Return evaluation times in [min(t0, t1), max(t0, t1))."""
    lower = torch.minimum(t0, t1)
    upper = torch.maximum(t0, t1)
    mask = (t_grid >= lower) & (t_grid < upper)
    if bool(mask.any().detach().cpu()):
        return torch.nonzero(mask, as_tuple=False).reshape(-1)
    return torch.argmin(torch.abs(t_grid - lower)).reshape(1)


def biomass_prediction''',
)

replace_once(
    observation_path,
    '''def catch_prediction(N_grid: torch.Tensor, t_grid: torch.Tensor, params: MizerTorchParams, species_idx: torch.Tensor, gear_idx: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, w_min: torch.Tensor, w_max: torch.Tensor, *, gear_specific: bool) -> torch.Tensor:
    """Predict instantaneous or annual catch/yield observations.

    Instantaneous observations return Y(t). Interval observations use the mean
    of the catch rates at the interval endpoints multiplied by interval length.
    For the intended annual data this is the annual mean-rate approximation
    C_y = 0.5 * [Y(y) + Y(y + 1)] * 1 year.
    Output shape [n_obs].
    """
    w = params.w.to(dtype=N_grid.dtype, device=N_grid.device)
    dw = params.dw.to(dtype=N_grid.dtype, device=N_grid.device)
    out = []
    for j in range(species_idx.numel()):
        sp = int(species_idx[j].detach().cpu())
        gear = int(gear_idx[j].detach().cpu()) if gear_specific else None
        tidx = _endpoint_indices(t_grid, t_start[j], t_end[j])
        wmask = (w >= w_min[j]) & (w <= w_max[j])
        rates = []
        for ti in tidx:
            F = _gear_fishing(params, gear, sp, t_grid[ti]).to(dtype=N_grid.dtype, device=N_grid.device)
            rates.append((F[wmask] * N_grid[ti, sp, wmask] * w[wmask] * dw[wmask]).sum())
        mean_rate = torch.stack(rates).mean()
        duration = torch.abs(t_end[j] - t_start[j])
        out.append(torch.where(duration > 0, mean_rate * duration, mean_rate))
    return torch.stack(out)


def predict_observations(state: dict[str, torch.Tensor], observation_batch: dict[str, object], params: MizerTorchParams) -> torch.Tensor:''',
    '''def catch_prediction(
    N_grid: torch.Tensor,
    t_grid: torch.Tensor,
    params: MizerTorchParams,
    species_idx: torch.Tensor,
    gear_idx: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    w_min: torch.Tensor,
    w_max: torch.Tensor,
    *,
    gear_specific: bool,
    data_time_quadrature_points: int = 1,
) -> torch.Tensor:
    """Predict instantaneous or annual catch/yield observations.

    With q > 1, interval observations use q equally spaced left-closed times:

        t_k = t_start + k * (t_end - t_start) / q,  k = 0, ..., q - 1
        C = mean_k Y(t_k) * abs(t_end - t_start)

    For q=10 and a one-year interval, this matches mizer output at
    y, y+0.1, ..., y+0.9, with the y+1 endpoint excluded.
    """
    if data_time_quadrature_points < 1:
        raise ValueError("data_time_quadrature_points must be at least 1.")

    w = params.w.to(dtype=N_grid.dtype, device=N_grid.device)
    dw = params.dw.to(dtype=N_grid.dtype, device=N_grid.device)
    out = []
    for j in range(species_idx.numel()):
        sp = int(species_idx[j].detach().cpu())
        gear = int(gear_idx[j].detach().cpu()) if gear_specific else None
        duration = torch.abs(t_end[j] - t_start[j])
        if data_time_quadrature_points > 1 and bool((duration > 0).detach().cpu()):
            tidx = _left_closed_interval_indices(t_grid, t_start[j], t_end[j])
        else:
            tidx = _endpoint_indices(t_grid, t_start[j], t_end[j])
        wmask = (w >= w_min[j]) & (w <= w_max[j])
        rates = []
        for ti in tidx:
            F = _gear_fishing(params, gear, sp, t_grid[ti]).to(dtype=N_grid.dtype, device=N_grid.device)
            rates.append((F[wmask] * N_grid[ti, sp, wmask] * w[wmask] * dw[wmask]).sum())
        mean_rate = torch.stack(rates).mean()
        out.append(torch.where(duration > 0, mean_rate * duration, mean_rate))
    return torch.stack(out)


def predict_observations(
    state: dict[str, torch.Tensor],
    observation_batch: dict[str, object],
    params: MizerTorchParams,
    *,
    data_time_quadrature_points: int = 1,
) -> torch.Tensor:''',
)

replace_once(
    observation_path,
    'p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=False)',
    'p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=False, data_time_quadrature_points=data_time_quadrature_points)',
)
replace_once(
    observation_path,
    'p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=True)',
    'p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=True, data_time_quadrature_points=data_time_quadrature_points)',
)

loop_path = root / "PINNmizer/training/loop_multispecies.py"
replace_once(loop_path, "from PINNmizer.pinn.observation_operators import predict_observations", "from PINNmizer.pinn.observation_operators import observation_time_grid, predict_observations")
replace_once(
    loop_path,
    '''            obs_times = torch.cat([observation_batch["t_start"], observation_batch["t_end"]])

            if data_time_quadrature_points > 1:
                qs = []
                for a, b in zip(observation_batch["t_start"], observation_batch["t_end"]):
                    qs.append(torch.linspace(a, b, data_time_quadrature_points, dtype=obs_times.dtype, device=obs_times.device))
                obs_times = torch.cat([obs_times, torch.cat(qs)])

            t_grid = torch.unique(obs_times).sort().values''',
    '''            t_grid = observation_time_grid(
                observation_batch,
                data_time_quadrature_points=data_time_quadrature_points,
            )''',
)
replace_once(
    loop_path,
    '            pred = predict_observations({"N_grid": grid_eval["N"], "t_grid": t_grid}, observation_batch, params)',
    '''            pred = predict_observations(
                {"N_grid": grid_eval["N"], "t_grid": t_grid},
                observation_batch,
                params,
                data_time_quadrature_points=data_time_quadrature_points,
            )''',
)

train_path = root / "PINNmizer/training/train_pde_multispecies.py"
replace_once(train_path, "from PINNmizer.pinn.observation_operators import predict_observations", "from PINNmizer.pinn.observation_operators import observation_time_grid, predict_observations")
replace_once(
    train_path,
    '''    obs_times = torch.cat([observation_batch["t_start"], observation_batch["t_end"]])
    if data_time_quadrature_points > 1:
        obs_times = torch.cat([obs_times, torch.cat([torch.linspace(a, b, data_time_quadrature_points, dtype=obs_times.dtype, device=obs_times.device) for a, b in zip(observation_batch["t_start"], observation_batch["t_end"])])])
    t_grid = torch.unique(obs_times).sort().values''',
    '''    t_grid = observation_time_grid(
        observation_batch,
        data_time_quadrature_points=data_time_quadrature_points,
    )''',
)
replace_once(
    train_path,
    '        pred = predict_observations({"N_grid": grid_eval["N"], "t_grid": t_grid}, observation_batch, params)',
    '''        pred = predict_observations(
            {"N_grid": grid_eval["N"], "t_grid": t_grid},
            observation_batch,
            params,
            data_time_quadrature_points=data_time_quadrature_points,
        )''',
)

verify_path = root / "validation/scripts/checks/verify_log_u_grid_offgrid_v2.py"
replace_once(verify_path, "from PINNmizer.pinn.observation_operators import predict_observations", "from PINNmizer.pinn.observation_operators import observation_time_grid, predict_observations")
replace_once(
    verify_path,
    '''    obs_times = torch.cat(
        [observation_batch["t_start"], observation_batch["t_end"]]
    )
    if quadrature_points > 1:
        quadrature = [
            torch.linspace(
                a,
                b,
                quadrature_points,
                dtype=obs_times.dtype,
                device=obs_times.device,
            )
            for a, b in zip(
                observation_batch["t_start"],
                observation_batch["t_end"],
            )
        ]
        obs_times = torch.cat([obs_times, torch.cat(quadrature)])

    t_grid = torch.unique(obs_times).sort().values''',
    '''    t_grid = observation_time_grid(
        observation_batch,
        data_time_quadrature_points=quadrature_points,
    )''',
)
replace_once(
    verify_path,
    '''        prediction = predict_observations(
            {"N_grid": grid_eval["N"], "t_grid": t_grid},
            observation_batch,
            params,
        )''',
    '''        prediction = predict_observations(
            {"N_grid": grid_eval["N"], "t_grid": t_grid},
            observation_batch,
            params,
            data_time_quadrature_points=quadrature_points,
        )''',
)

smoke_path = root / "validation/scripts/checks/validate_data_likelihood_smoke.py"
replace_once(smoke_path, "from PINNmizer.pinn.observation_operators import biomass_prediction, catch_prediction", "from PINNmizer.pinn.observation_operators import biomass_prediction, catch_prediction, observation_time_grid")
replace_once(
    smoke_path,
    '''    N_with_midpoint = torch.stack([N_grid[0], torch.full_like(N_grid[0], 1e6), N_grid[1]])
    annual_with_midpoint = catch_prediction(N_with_midpoint, torch.tensor([0.0, 0.5, 1.0]), params, torch.tensor([0]), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([4.0]), gear_specific=True)
    assert torch.allclose(annual_with_midpoint, annual)

    equal = lognormal_nll''',
    '''    N_with_midpoint = torch.stack([N_grid[0], torch.full_like(N_grid[0], 1e6), N_grid[1]])
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

    equal = lognormal_nll''',
)

print("\nRun:")
print("python validation/scripts/checks/validate_data_likelihood_smoke.py")
