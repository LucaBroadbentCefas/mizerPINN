from __future__ import annotations

from dataclasses import dataclass
import math
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
    t_points: torch.Tensor          # [K], physical time slabs
    x_points: torch.Tensor          # [K, M], physical log-weight points per slab
    generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        if self.t_points.ndim != 1:
            raise ValueError(f"t_points must be [K], got {tuple(self.t_points.shape)}.")
        if self.x_points.ndim != 2:
            raise ValueError(f"x_points must be [K, M], got {tuple(self.x_points.shape)}.")
        if self.x_points.shape[0] != self.t_points.numel():
            raise ValueError(
                "x_points.shape[0] must equal t_points.numel(). "
                f"Got x_points={tuple(self.x_points.shape)}, "
                f"t_points={tuple(self.t_points.shape)}."
            )

    @property
    def n_time(self) -> int:
        return int(self.t_points.numel())

    @property
    def n_eval_per_time(self) -> int:
        return int(self.x_points.shape[1])

    @property
    def population_size(self) -> int:
        return int(self.x_points.numel())

    def as_batch(self, *, params: MizerTorchParams) -> dict[str, torch.Tensor]:
        dtype, device = _params_dtype_device(params)

        t_slab = self.t_points.to(dtype=dtype, device=device)
        x_slab = self.x_points.to(dtype=dtype, device=device)
        w_slab = torch.exp(x_slab)

        x_grid = _x_grid(params).to(dtype=dtype, device=device)

        return {
            "t_slab": t_slab,
            "t_slab_scaled": scale_t(t_slab, params),
            "x_slab": x_slab,
            "x_slab_scaled": scale_x(x_slab, params),
            "w_slab": w_slab,
            "x_grid": x_grid,
            "x_grid_scaled": scale_x(x_grid, params),
            "w_grid": params.w.to(dtype=dtype, device=device),
        }
        
    def resample_time_points_(
        self,
        *,
        params: MizerTorchParams,
        t_max_current: float | torch.Tensor | None = None,
    ) -> None:
        self.t_points = _sample_stratified_time_points(
            params=params,
            n_time=self.n_time,
            generator=self.generator,
            t_max_current=t_max_current,
        )        

def _sample_stratified_time_points(
    *,
    params: MizerTorchParams,
    n_time: int,
    generator: torch.Generator | None = None,
    t_max_current: float | torch.Tensor | None = None,
) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)

    t_min, t_max = _t_limits(params)
    t_min = torch.as_tensor(t_min, dtype=dtype, device=device)
    t_max = torch.as_tensor(t_max, dtype=dtype, device=device)

    if t_max_current is None:
        t_upper = t_max
    else:
        t_upper = torch.as_tensor(t_max_current, dtype=dtype, device=device)
        t_upper = torch.maximum(t_upper, t_min)
        t_upper = torch.minimum(t_upper, t_max)

    if not bool((t_upper > t_min).detach().cpu()):
        raise ValueError(
            "R3 time resampling requires t_max_current > t_min. "
            f"Got t_min={float(t_min.detach().cpu())}, "
            f"t_upper={float(t_upper.detach().cpu())}."
        )

    t_unit = (
        torch.arange(n_time, dtype=dtype, device=device)
        + torch.rand(n_time, dtype=dtype, device=device, generator=generator)
    ) / n_time

    return t_min + (t_upper - t_min) * t_unit

def make_r3_population(
    *,
    params: MizerTorchParams,
    n_pair: int,
    n_time: int,
    species_idx: int | None = 0,
    seed: int | None = None,
) -> R3Population:
    dtype, device = _params_dtype_device(params)

    if n_pair <= 0:
        raise ValueError(f"n_pair must be positive, got {n_pair}.")
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}.")

    n_eval_per_time = int(math.ceil(n_pair / n_time))
    x_min, x_max, t_min, t_max = _r3_domain(params, species_idx=species_idx)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Stratified time slabs: fixed after initialisation.
    # This gives better time-domain coverage than fully independent random times,
    # while still sampling uniformly over the physical time interval.
    t_points = _sample_stratified_time_points(
        params=params,
        n_time=n_time,
        generator=generator,
        t_max_current=None,
    )

    x_points = x_min + (x_max - x_min) * torch.rand(
        n_time,
        n_eval_per_time,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    return R3Population(
        t_points=t_points,
        x_points=x_points,
        generator=generator,
    )

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


def _r3_domain(params: MizerTorchParams, species_idx: int | None = 0):
    dtype, device = _params_dtype_device(params)

    x_grid = _x_grid(params).to(dtype=dtype, device=device)
    x_min = x_grid[0]
    x_max = x_grid[-1]

    if getattr(params, "w_max", None) is not None:
        w_max = torch.as_tensor(params.w_max, dtype=dtype, device=device)
        if w_max.ndim == 0:
            w_max_species = w_max
        elif species_idx is None:
            w_max_species = torch.max(w_max)
        else:
            w_max_species = w_max[species_idx]
        x_max = torch.minimum(x_max, torch.log(w_max_species))

    t_min, t_max = _t_limits(params)
    t_min = t_min.to(dtype=dtype, device=device)
    t_max = t_max.to(dtype=dtype, device=device)

    return x_min, x_max, t_min, t_max

def _active_eval_mask_slab(w_slab: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    if w_slab.ndim != 2:
        raise ValueError(f"w_slab must be [K, M], got {tuple(w_slab.shape)}.")

    k, m = w_slab.shape
    flat_mask = active_eval_mask(w_slab.reshape(-1), params)
    n_species = flat_mask.shape[0]

    return flat_mask.reshape(n_species, k, m).permute(1, 0, 2).contiguous()

def _r3_score(
    *,
    residual: torch.Tensor,
    w_slab: torch.Tensor,
    params: MizerTorchParams,
    score_form: str,
) -> torch.Tensor:
    if residual.ndim != 3:
        raise ValueError(
            "Slabbed R3 residual must be [K, n_species, M], "
            f"got {tuple(residual.shape)}."
        )

    if score_form == "abs":
        raw = residual.abs()
    elif score_form == "squared":
        raw = residual.square()
    else:
        raise ValueError("score_form must be 'abs' or 'squared'.")

    mask = _active_eval_mask_slab(w_slab, params).to(
        dtype=raw.dtype,
        device=raw.device,
    )

    if mask.shape != raw.shape:
        raise ValueError(
            f"R3 score mask shape {tuple(mask.shape)} does not match "
            f"residual shape {tuple(raw.shape)}."
        )

    denom = mask.sum(dim=1).clamp_min(1.0)

    return (raw * mask).sum(dim=1) / denom

def update_r3_population_(
    *,
    population: R3Population,
    residual: torch.Tensor,  # [K, n_species, M]
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    score_form: str = "abs",
    causal: CausalR3 | None = None,
    causal_score: bool = True,
    species_idx: int | None = 0,
) -> dict[str, float]:
    scores = _r3_score(
        residual=residual.detach(),
        w_slab=batch["w_slab"].detach(),
        params=params,
        score_form=score_form,
    )

    if causal is not None and causal_score:
        gate = causal.gate(batch["t_slab"].detach(), params).to(
            dtype=scores.dtype,
            device=scores.device,
        )
        scores = scores * gate[:, None]

    threshold = scores.mean()
    retain = scores > threshold
    release = ~retain

    n_release = int(release.sum().detach().cpu())
    n_total = int(scores.numel())

    if n_release > 0:
        dtype = population.x_points.dtype
        device = population.x_points.device

        x_min, x_max, _, _ = _r3_domain(params, species_idx=species_idx)
        x_min = x_min.to(dtype=dtype, device=device)
        x_max = x_max.to(dtype=dtype, device=device)

        x_new = x_min + (x_max - x_min) * torch.rand(
            n_release,
            dtype=dtype,
            device=device,
            generator=population.generator,
        )

        # Retain/resample only x positions inside fixed time slabs.
        population.x_points[release] = x_new

    return {
        "r3_retained_fraction": float(retain.to(torch.float64).mean().detach().cpu()),
        "r3_score_mean": float(scores.mean().detach().cpu()),
        "r3_score_max": float(scores.max().detach().cpu()),
        "r3_resampled": float(n_release),
        "r3_population_size": float(n_total),
        "r3_n_time": float(population.n_time),
        "r3_n_eval_per_time": float(population.n_eval_per_time),
        "r3_biology_time_loops": float(population.n_time),
    }
