from __future__ import annotations

import torch
import torch.nn as nn


class BoundedLogRMax(nn.Module):
    """Per-species inverse r_max parameter with bounded log transform."""

    def __init__(self, initial_r_max: torch.Tensor, *, lower: float = 0.0, upper: float = 50.0, eps: float = 1e-12):
        super().__init__()
        if not lower < upper:
            raise ValueError(f"r_max log bounds require lower < upper, got {lower} >= {upper}.")
        if initial_r_max.ndim != 1:
            raise ValueError(f"initial_r_max must have shape [n_species], got {tuple(initial_r_max.shape)}.")
        if not torch.isfinite(initial_r_max).all() or not (initial_r_max > 0).all():
            raise ValueError("initial_r_max must be finite and strictly positive.")
        initial_log = torch.log(initial_r_max)
        lower_t = torch.as_tensor(lower, dtype=initial_log.dtype, device=initial_log.device)
        upper_t = torch.as_tensor(upper, dtype=initial_log.dtype, device=initial_log.device)
        tol = 10 * torch.finfo(initial_log.dtype).eps * max(1.0, abs(float(upper - lower)))
        if bool(((initial_log < lower_t - tol) | (initial_log > upper_t + tol)).any().detach().cpu()):
            lo = float(initial_log.min().detach().cpu()); hi = float(initial_log.max().detach().cpu())
            raise ValueError(f"initial log(r_max) values [{lo}, {hi}] are outside requested bounds [{lower}, {upper}].")
        p = (initial_log - lower_t) / (upper_t - lower_t)
        p = torch.clamp(p, eps, 1.0 - eps)
        raw = torch.logit(p)
        self.raw_logit = nn.Parameter(raw.clone())
        self.lower = float(lower)
        self.upper = float(upper)
        self.eps = float(eps)
        self.register_buffer("initial_r_max", initial_r_max.detach().clone())
        self.register_buffer("initial_log_r_max", initial_log.detach().clone())

    def current_log_r_max(self) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.raw_logit)

    def current_r_max(self) -> torch.Tensor:
        return torch.exp(self.current_log_r_max())

    @property
    def ratio_to_initial(self) -> torch.Tensor:
        return self.current_r_max() / self.initial_r_max

    def config(self) -> dict:
        return {"type": self.__class__.__name__, "lower": self.lower, "upper": self.upper, "eps": self.eps}
