from __future__ import annotations

import math

import torch


def _to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def parse_fraction_schedule(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Causal fraction schedule is empty.")
    if any(v <= 0.0 or v > 1.0 for v in values):
        raise ValueError("All causal fraction schedule values must be in (0, 1]. " f"Got {values}.")
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def causal_time_fraction(*, step: int, mode: str, start_fraction: float, ramp_steps: int, step_fractions: str) -> float:
    if mode == "off":
        return 1.0
    if not (0.0 < start_fraction <= 1.0):
        raise ValueError(f"start_fraction must be in (0, 1], got {start_fraction}.")
    if ramp_steps <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (step - 1) / ramp_steps))
    if mode == "linear":
        return start_fraction + progress * (1.0 - start_fraction)
    if mode == "step":
        levels = parse_fraction_schedule(step_fractions)
        idx = min(len(levels) - 1, int(math.floor(progress * len(levels))))
        return levels[idx]
    raise ValueError("mode must be 'off', 'linear', or 'step'.")


def causal_t_max_current(*, params, step: int, mode: str, start_fraction: float, ramp_steps: int, step_fractions: str) -> tuple[float, float]:
    t_min = _to_float(params.t_min)
    t_max = _to_float(params.t_max)
    frac = causal_time_fraction(
        step=step,
        mode=mode,
        start_fraction=start_fraction,
        ramp_steps=ramp_steps,
        step_fractions=step_fractions,
    )
    t_current = t_min + frac * (t_max - t_min)
    return frac, t_current
