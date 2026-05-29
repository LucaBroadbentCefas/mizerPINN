from __future__ import annotations

from dataclasses import dataclass

import torch

from PINNmizer.params import (
    MizerTorchParams,
    _params_dtype_device,
    _t_limits,
    _x_grid,
    scale_x,
    scale_t,
    active_eval_mask,
)


@dataclass
class R3Population:
    points: torch.Tensor  # [n_pair, 2], physical columns [x, t]
    generator: torch.Generator | None = None

    def as_batch(self, *, params: MizerTorchParams) -> dict[str, torch.Tensor]:
        dtype, device = _params_dtype_device(params)

        points = self.points.to(dtype=dtype, device=device)
        x_pair = points[:, 0]
        t_pair = points[:, 1]
        w_pair = torch.exp(x_pair)

        x_grid = _x_grid(params).to(dtype=dtype, device=device)

        return {
            "x_pair": x_pair,
            "t_pair": t_pair,
            "w_pair": w_pair,
            "x_pair_scaled": scale_x(x_pair, params),
            "t_pair_scaled": scale_t(t_pair, params),
            "x_grid": x_grid,
            "x_grid_scaled": scale_x(x_grid, params),
            "w_grid": params.w.to(dtype=dtype, device=device),
        }


@dataclass
class CausalR3:
    gamma: float = -0.5
    gamma_max: float = 1.5
    alpha: float = 5.0
    lr: float = 1e-3
    tolerance: float = 20.0
    update_clip: float = 0.1

    def gate(self, t: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
        dtype, device = _params_dtype_device(params)
        t = t.to(dtype=dtype, device=device)

        t_min, t_max = _t_limits(params)
        t_min = t_min.to(dtype=dtype, device=device)
        t_max = t_max.to(dtype=dtype, device=device)

        t_scaled = (t - t_min) / (t_max - t_min)
        return (1.0 - torch.tanh(self.alpha * (t_scaled - self.gamma))) / 2.0

    def update_(self, loss_pde: torch.Tensor) -> dict[str, float]:
        loss = loss_pde.detach()

        raw = torch.exp(-self.tolerance * loss)
        clipped = torch.minimum(
            raw,
            torch.as_tensor(self.update_clip, dtype=loss.dtype, device=loss.device),
        )

        update = self.lr * clipped
        gamma_before = self.gamma
        self.gamma = min(self.gamma + float(update.detach().cpu()), self.gamma_max)

        return {
            "causal_r3_gamma": float(self.gamma),
            "causal_r3_gamma_update": float(self.gamma - gamma_before),
        }


def _r3_domain(params: MizerTorchParams, species_idx: int = 0):
    dtype, device = _params_dtype_device(params)

    x_grid = _x_grid(params).to(dtype=dtype, device=device)
    x_min = x_grid[0]
    x_max = x_grid[-1]

    if getattr(params, "w_max", None) is not None:
        w_max = torch.as_tensor(params.w_max, dtype=dtype, device=device)
        if w_max.ndim == 0:
            w_max_species = w_max
        else:
            w_max_species = w_max[species_idx]
        x_max = torch.minimum(x_max, torch.log(w_max_species))

    t_min, t_max = _t_limits(params)
    t_min = t_min.to(dtype=dtype, device=device)
    t_max = t_max.to(dtype=dtype, device=device)

    return x_min, x_max, t_min, t_max


def make_r3_population(
    *,
    params: MizerTorchParams,
    n_pair: int,
    species_idx: int = 0,
    seed: int | None = None,
) -> R3Population:
    dtype, device = _params_dtype_device(params)
    x_min, x_max, t_min, t_max = _r3_domain(params, species_idx=species_idx)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    x = x_min + (x_max - x_min) * torch.rand(
        n_pair,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    t = t_min + (t_max - t_min) * torch.rand(
        n_pair,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    return R3Population(points=torch.stack([x, t], dim=1), generator=generator)


def _r3_score(
    *,
    residual: torch.Tensor,
    w_pair: torch.Tensor,
    params: MizerTorchParams,
    score_form: str,
) -> torch.Tensor:
    if score_form == "abs":
        raw = residual.abs()
    elif score_form == "squared":
        raw = residual.square()
    else:
        raise ValueError("score_form must be 'abs' or 'squared'.")

    mask = active_eval_mask(w_pair, params).to(dtype=raw.dtype, device=raw.device)
    denom = mask.sum(dim=0).clamp_min(1.0)

    return (raw * mask).sum(dim=0) / denom


def update_r3_population_(
    *,
    population: R3Population,
    residual: torch.Tensor,  # [n_species, n_pair]
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    score_form: str = "abs",
    causal: CausalR3 | None = None,
    causal_score: bool = True,
) -> dict[str, float]:
    scores = _r3_score(
        residual=residual.detach(),
        w_pair=batch["w_pair"].detach(),
        params=params,
        score_form=score_form,
    )

    if causal is not None and causal_score:
        scores = scores * causal.gate(batch["t_pair"].detach(), params)

    threshold = scores.mean()
    retain = scores > threshold
    release = ~retain

    n_release = int(release.sum().detach().cpu())
    n_total = int(scores.numel())

    if n_release > 0:
        dtype = population.points.dtype
        device = population.points.device
        x_min, x_max, t_min, t_max = _r3_domain(params, species_idx=0)

        x_new = x_min + (x_max - x_min) * torch.rand(
            n_release,
            dtype=dtype,
            device=device,
            generator=population.generator,
        )
        t_new = t_min + (t_max - t_min) * torch.rand(
            n_release,
            dtype=dtype,
            device=device,
            generator=population.generator,
        )

        population.points[release] = torch.stack([x_new, t_new], dim=1)

    return {
        "r3_retained_fraction": float(retain.to(torch.float64).mean().detach().cpu()),
        "r3_score_mean": float(scores.mean().detach().cpu()),
        "r3_score_max": float(scores.max().detach().cpu()),
        "r3_resampled": float(n_release),
        "r3_population_size": float(n_total),
    }
