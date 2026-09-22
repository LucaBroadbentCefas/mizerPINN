from __future__ import annotations


def parse_fraction_schedule(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 or value > 1 for value in values):
        raise ValueError("Causal fractions must be in (0,1].")
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def causal_time_fraction(step: int, start_fraction: float, ramp_steps: int) -> float:
    if not 0 < start_fraction <= 1:
        raise ValueError("start_fraction must be in (0,1].")
    if ramp_steps <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (step - 1) / ramp_steps))
    return start_fraction + progress * (1.0 - start_fraction)


def causal_t_max_current(params, step: int, start_fraction: float, ramp_steps: int) -> tuple[float, float]:
    fraction = causal_time_fraction(step, start_fraction, ramp_steps)
    t_min, t_max = float(params.t_min), float(params.t_max)
    return fraction, t_min + fraction * (t_max - t_min)
