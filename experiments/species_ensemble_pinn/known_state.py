from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from PINNmizer.params import _n_species, _n_w, _params_dtype_device, _t_limits

REQUIRED_COLUMNS = {"time", "species_idx", "species", "weight", "N"}


def _names_by_index(frame: pd.DataFrame, n_species: int) -> list[str]:
    mapping = frame[["species_idx", "species"]].drop_duplicates()
    counts = mapping.groupby("species_idx")["species"].nunique()
    if len(counts) != n_species or (counts != 1).any():
        raise ValueError("Each zero-based species_idx must map to exactly one species name.")
    mapping = mapping.sort_values("species_idx")
    expected = list(range(n_species))
    if mapping["species_idx"].tolist() != expected:
        raise ValueError(f"species_idx must span {expected[0]} to {expected[-1]} without gaps.")
    return mapping["species"].astype(str).tolist()


class KnownStateProvider:
    def __init__(self, csv_path: str | Path, params, n_init: torch.Tensor, *,
                 mode: str, interpolation: str = "linear", log_floor: float = 1e-30,
                 consistency_atol: float = 1e-10, consistency_rtol: float = 1e-8,
                 allow_initial_mismatch: bool = False) -> None:
        if mode not in {"dynamic-known", "frozen-initial"}:
            raise ValueError("mode must be dynamic-known or frozen-initial.")
        if interpolation not in {"linear", "log-linear"}:
            raise ValueError("interpolation must be linear or log-linear.")
        if log_floor <= 0:
            raise ValueError("log_floor must be strictly positive.")
        self.path = Path(csv_path).expanduser().resolve()
        frame = pd.read_csv(self.path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Known-state CSV is missing columns: {sorted(missing)}")
        frame = frame[list(REQUIRED_COLUMNS)].copy()
        for column in ["time", "species_idx", "weight", "N"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame.duplicated(["time", "species_idx", "weight"]).any():
            raise ValueError("Known-state CSV has duplicate (time, species_idx, weight) rows.")
        n_species, n_w = _n_species(params), _n_w(params)
        names = _names_by_index(frame, n_species)
        current_names = getattr(params, "species", None)
        if current_names is not None and list(current_names) != names:
            raise ValueError("Known-state species names do not match params.species.")
        params.species = names
        times = sorted(frame["time"].unique().tolist())
        dtype, device = _params_dtype_device(params)
        if not times or not torch.isfinite(torch.tensor(times, dtype=dtype)).all():
            raise ValueError("Known-state times must be finite.")
        expected_weights = params.w.detach().cpu()
        states = []
        for time in times:
            block = frame.loc[frame["time"] == time].sort_values(["species_idx", "weight"])
            if len(block) != n_species * n_w:
                raise ValueError(f"Time {time} does not contain a complete [S,W] state.")
            state_rows = []
            for idx in range(n_species):
                species_block = block.loc[block["species_idx"] == idx].sort_values("weight")
                weights = torch.tensor(species_block["weight"].to_numpy(), dtype=dtype)
                if weights.shape != expected_weights.shape or not torch.allclose(weights, expected_weights, rtol=1e-10, atol=1e-12):
                    raise ValueError("Known-state weights do not match params.w; regrid during fixture preparation.")
                values = torch.tensor(species_block["N"].to_numpy(), dtype=dtype)
                state_rows.append(values)
            states.append(torch.stack(state_rows))
        known = torch.stack(states).to(device=device)
        if not torch.isfinite(known).all() or bool((known < 0).any().detach().cpu()):
            raise ValueError("Known abundance must be finite and non-negative.")
        self.known_times = torch.tensor(times, dtype=dtype, device=device)
        self.known_N = known
        t_min, t_max = _t_limits(params)
        if self.known_times[0] > t_min or self.known_times[-1] < t_max:
            raise ValueError("Known-state time coverage must include params.t_min and params.t_max.")
        n_init_t = torch.as_tensor(n_init, dtype=dtype, device=device)
        diff = torch.abs(known[0] - n_init_t)
        relative = diff / torch.clamp(torch.abs(n_init_t), min=torch.finfo(dtype).tiny)
        self.initial_max_abs_difference = float(diff.max().detach().cpu())
        self.initial_max_relative_difference = float(relative.max().detach().cpu())
        consistent = torch.allclose(known[0], n_init_t, atol=consistency_atol, rtol=consistency_rtol)
        if not consistent and not allow_initial_mismatch:
            raise ValueError(
                "Known initial state differs from n_init: "
                f"max_abs={self.initial_max_abs_difference:.6e}, "
                f"max_rel={self.initial_max_relative_difference:.6e}."
            )
        self.mode = mode
        self.interpolation = interpolation
        self.log_floor = float(log_floor)

    def at(self, t_eval: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t_eval, dtype=self.known_times.dtype, device=self.known_times.device)
        if t.ndim != 1:
            raise ValueError("t_eval must be one-dimensional.")
        if bool(((t < self.known_times[0]) | (t > self.known_times[-1])).any().detach().cpu()):
            raise ValueError("Known-state interpolation does not extrapolate in time.")
        if self.mode == "frozen-initial":
            return self.known_N[0][None, :, :].expand(t.numel(), -1, -1).clone()
        upper = torch.searchsorted(self.known_times, t, right=False).clamp(0, self.known_times.numel() - 1)
        lower = torch.clamp(upper - 1, min=0)
        exact = self.known_times[upper] == t
        lower = torch.where(exact, upper, lower)
        t0, t1 = self.known_times[lower], self.known_times[upper]
        denom = torch.where(t1 > t0, t1 - t0, torch.ones_like(t1))
        alpha = torch.where(t1 > t0, (t - t0) / denom, torch.zeros_like(t))[:, None, None]
        n0, n1 = self.known_N[lower], self.known_N[upper]
        if self.interpolation == "linear":
            out = (1 - alpha) * n0 + alpha * n1
        else:
            floor = torch.as_tensor(self.log_floor, dtype=n0.dtype, device=n0.device)
            out = torch.exp((1 - alpha) * torch.log(n0 + floor) + alpha * torch.log(n1 + floor))
            out = torch.where(exact[:, None, None], n1, out)
        return out
