from __future__ import annotations

import math
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


class BoundedDataCV(nn.Module):
    """Bounded global or per-species observation CV in log-CV space."""

    def __init__(self, initial_cv: torch.Tensor, *, lower: float = 0.02, upper: float = 1.5, scope: str = "species", eps: float = 1e-12):
        super().__init__()
        if scope not in {"species", "global"}:
            raise ValueError("data CV scope must be 'species' or 'global'.")
        if not (0.0 < lower < upper):
            raise ValueError(f"data CV bounds require 0 < lower < upper, got {lower}, {upper}.")
        if initial_cv.ndim != 1 or initial_cv.numel() < 1:
            raise ValueError(f"initial_cv must be a non-empty vector, got {tuple(initial_cv.shape)}.")
        if not torch.isfinite(initial_cv).all() or bool(((initial_cv < lower) | (initial_cv > upper)).any()):
            raise ValueError(f"initial CV must be finite and within [{lower}, {upper}].")
        lo = torch.as_tensor(lower, dtype=initial_cv.dtype, device=initial_cv.device).log()
        hi = torch.as_tensor(upper, dtype=initial_cv.dtype, device=initial_cv.device).log()
        initial_log = initial_cv.log()
        transform_eps = max(float(eps), float(torch.finfo(initial_cv.dtype).eps))
        p = ((initial_log - lo) / (hi - lo)).clamp(transform_eps, 1.0 - transform_eps)
        self.raw_parameter = nn.Parameter(torch.logit(p))
        self.lower, self.upper, self.scope, self.eps = float(lower), float(upper), scope, float(eps)
        self.register_buffer("initial_cv", initial_cv.detach().clone())
        self.register_buffer("initial_log_cv", initial_log.detach().clone())

    def current_log_cv(self) -> torch.Tensor:
        lo = math.log(self.lower)
        return lo + (math.log(self.upper) - lo) * torch.sigmoid(self.raw_parameter)

    def current_cv(self) -> torch.Tensor:
        return self.current_log_cv().exp()

    def current_sd_log(self) -> torch.Tensor:
        cv = self.current_cv()
        return torch.sqrt(torch.log1p(cv.square()))

    def config(self) -> dict:
        return {"type": self.__class__.__name__, "scope": self.scope, "lower": self.lower, "upper": self.upper, "eps": self.eps}
