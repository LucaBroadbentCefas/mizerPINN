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


def catch_prediction(N_grid: torch.Tensor, t_grid: torch.Tensor, params: MizerTorchParams, species_idx: torch.Tensor, gear_idx: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, w_min: torch.Tensor, w_max: torch.Tensor, *, gear_specific: bool) -> torch.Tensor:
    """Predict catch/yield for total or gear-specific observations.

    Instantaneous observations return Y(t). Intervals with t_start != t_end return a
    simple trapezoid/mean-rate quadrature over available t_grid points times interval length.
    Output shape [n_obs].
    """
    w = params.w.to(dtype=N_grid.dtype, device=N_grid.device)
    dw = params.dw.to(dtype=N_grid.dtype, device=N_grid.device)
    out = []
    for j in range(species_idx.numel()):
        sp = int(species_idx[j].detach().cpu())
        gear = int(gear_idx[j].detach().cpu()) if gear_specific else None
        tidx = _time_indices(t_grid, t_start[j], t_end[j])
        wmask = (w >= w_min[j]) & (w <= w_max[j])
        rates = []
        for ti in tidx:
            F = _gear_fishing(params, gear, sp, t_grid[ti]).to(dtype=N_grid.dtype, device=N_grid.device)
            rates.append((F[wmask] * N_grid[ti, sp, wmask] * w[wmask] * dw[wmask]).sum())
        rate = torch.stack(rates).mean()
        duration = torch.abs(t_end[j] - t_start[j])
        out.append(torch.where(duration > 0, rate * duration, rate))
    return torch.stack(out)


def predict_observations(state: dict[str, torch.Tensor], observation_batch: dict[str, object], params: MizerTorchParams) -> torch.Tensor:
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
            p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=False)
        elif typ == "catch_gear":
            p = catch_prediction(N_grid, t_grid, params, observation_batch["species_idx"][sl], observation_batch["gear_idx"][sl], observation_batch["t_start"][sl], observation_batch["t_end"][sl], observation_batch["w_min"][sl], observation_batch["w_max"][sl], gear_specific=True)
        else:
            raise ValueError(f"Unsupported obs_type: {typ}")
        pred.append(p.reshape(()))
    return torch.stack(pred)
