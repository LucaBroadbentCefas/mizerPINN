from __future__ import annotations

import torch

from PINNmizer.biology.fishing import evaluate_effort_at_time
from PINNmizer.params import MizerTorchParams, _params_dtype_device


def _time_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
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


def _endpoint_indices(t_grid: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
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


def biomass_prediction(N_grid: torch.Tensor, t_grid: torch.Tensor, params: MizerTorchParams, species_idx: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, w_min: torch.Tensor, w_max: torch.Tensor, *, abundance: bool = False) -> torch.Tensor:
    """Predict biomass/abundance observations.

    N_grid [n_time, n_species, n_w], t_grid [n_time], output [n_obs].
    Biomass integrates sum N_i(w,t) * w * dw; abundance integrates sum N_i(w,t) * dw.
    species_idx < 0 requests an all-species/grouped sum.
    """
    w = params.w.to(dtype=N_grid.dtype, device=N_grid.device)
    dw = params.dw.to(dtype=N_grid.dtype, device=N_grid.device)
    out = []
    for j in range(species_idx.numel()):
        tidx = _time_indices(t_grid, t_start[j], t_end[j])
        wmask = (w >= w_min[j]) & (w <= w_max[j])
        N = N_grid[tidx]
        if int(species_idx[j].detach().cpu()) >= 0:
            N = N[:, int(species_idx[j].detach().cpu()):int(species_idx[j].detach().cpu()) + 1, :]
        factor = dw if abundance else w * dw
        vals = (N[..., wmask] * factor[wmask]).sum(dim=(-1, -2))
        out.append(vals.mean())
    return torch.stack(out)


def _gear_fishing(params: MizerTorchParams, gear_idx: int | None, species_idx: int, t: torch.Tensor) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    if params.catchability is None or params.selectivity is None:
        raise ValueError("catch observations require gear-level catchability and selectivity inputs.")
    effort = evaluate_effort_at_time(t, params).to(dtype=dtype, device=device)
    catchability = params.catchability.to(dtype=dtype, device=device)
    selectivity = params.selectivity.to(dtype=dtype, device=device)
    if gear_idx is None:
        return (effort[:, None] * catchability[:, species_idx:species_idx + 1] * selectivity[:, species_idx, :]).sum(dim=0)
    if gear_idx < 0 or gear_idx >= catchability.shape[0]:
        raise ValueError(f"gear_idx {gear_idx} is out of range for {catchability.shape[0]} gears.")
    return effort[gear_idx] * catchability[gear_idx, species_idx] * selectivity[gear_idx, species_idx, :]


def catch_prediction(
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
) -> torch.Tensor:
    """Dispatch deterministic observation operators; returns prediction [n_obs]."""
    N_grid = state["N_grid"]
    t_grid = state["t_grid"]
    obs_types = observation_batch["obs_type"]
    pred = []
    for j, typ in enumerate(obs_types):
        sl = slice(j, j + 1)
        if typ in {"biomass", "survey_biomass"}:
            p = biomass_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], abundance=False)
            if typ == "survey_biomass": p = p * observation_batch["q"][j]
        elif typ == "survey_abundance":
            p = biomass_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], abundance=True) * observation_batch["q"][j]
        elif typ == "catch_total":
            p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=False, data_time_quadrature_points=data_time_quadrature_points)
        elif typ == "catch_gear":
            p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=True, data_time_quadrature_points=data_time_quadrature_points)
        else:
            raise ValueError(f"Unsupported obs_type: {typ}")
        pred.append(p.reshape(()))
    return torch.stack(pred)
