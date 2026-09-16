"""Canonical truth normalisation and deterministic state-grid alignment."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["time", "species_idx", "species", "w", "x", "N", "log_N", "log10_N"]


def find_truth_source(project_root: Path) -> Path | None:
    """Find an explicit full truth export, never an observation or PINN output."""
    candidates = (
        "final_runs/truth_state.csv", "final_runs/mizer_truth_state.csv",
        "validation/fixtures/pde_multispecies/truth_state.csv",
    )
    return next((project_root / name for name in candidates if (project_root / name).is_file()), None)


def normalise_state(df: pd.DataFrame, source: str = "state") -> pd.DataFrame:
    aliases = {"t": "time", "t_eval": "time", "sp": "species", "weight": "w", "w_eval": "w", "x_eval": "x", "n": "N", "logN": "log_N", "log10N": "log10_N"}
    out = df.rename(columns={column: aliases.get(column, column) for column in df.columns}).copy()
    if "time" not in out or not ({"w", "x"} & set(out)) or not ({"N", "log_N", "log10_N"} & set(out)):
        raise ValueError(f"{source} needs time, w or x, and N/log_N/log10_N")
    if "species_idx" not in out:
        if "species" in out:
            # Final-suite single-species folders use names such as ``sp_7``.
            # Preserve that biological index rather than silently renumbering
            # species by their order of appearance in a CSV.  Non-numeric
            # names still receive a deterministic appearance-order index.
            labels = out["species"].astype(str)
            parsed = labels.str.extract(r"(?:^|_)sp(?:ecies)?_?(\d+)$", flags=re.IGNORECASE)[0]
            if parsed.notna().all():
                out["species_idx"] = pd.to_numeric(parsed)
            else:
                names = list(pd.unique(labels))
                out["species_idx"] = labels.map({name: i for i, name in enumerate(names)})
        else:
            out["species_idx"] = 0
    if "species" not in out:
        out["species"] = out["species_idx"].map(lambda value: f"species_{int(value)}")
    for column in ("time", "species_idx", "w", "x", "N", "log_N", "log10_N"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "w" not in out:
        out["w"] = np.exp(out["x"])
    if "x" not in out:
        out["x"] = np.log(out["w"].where(out["w"] > 0))
    if "N" not in out:
        out["N"] = np.exp(out["log_N"]) if "log_N" in out else np.power(10.0, out["log10_N"])
    # Zero bins are masks in these fixtures, not observations on a log scale.
    out = out[np.isfinite(out["time"]) & np.isfinite(out["x"]) & np.isfinite(out["N"]) & (out["N"] > 0)].copy()
    out["log_N"] = np.log(out["N"])
    out["log10_N"] = np.log10(out["N"])
    return out[STANDARD_COLUMNS].sort_values(["species_idx", "time", "x"]).reset_index(drop=True)


def load_truth_state(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("The configurable truth source currently supports long-form CSV only")
    return normalise_state(pd.read_csv(path), str(path))


def align_states(prediction: pd.DataFrame, truth: pd.DataFrame, *, w_max: Mapping[int, float] | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Place truth on prediction cells using linear interpolation in time then x.

    Exact time/x values are naturally preserved by ``numpy.interp``. Values
    outside truth support are NaN (never extrapolated). Both states are first
    restricted to positive abundance and species-specific active weights.
    """
    pred, true = normalise_state(prediction, "prediction"), normalise_state(truth, "truth")
    w_max = dict(w_max or {})
    if w_max:
        pred = pred[pred.apply(lambda row: row.w <= w_max.get(int(row.species_idx), np.inf), axis=1)]
        true = true[true.apply(lambda row: row.w <= w_max.get(int(row.species_idx), np.inf), axis=1)]
    rows: list[dict[str, object]] = []
    exact_time, interpolated_time = 0, 0
    for species_idx, targets in pred.groupby("species_idx"):
        source = true[true.species_idx == species_idx]
        if source.empty:
            continue
        source_times = np.sort(source.time.unique())
        # First interpolate each truth time profile in log-weight to target x.
        for target in targets.itertuples(index=False):
            values = []
            for time in source_times:
                profile = source[np.isclose(source.time, time)].groupby("x", as_index=False).log10_N.mean().sort_values("x")
                xs, ys = profile.x.to_numpy(), profile.log10_N.to_numpy()
                values.append(np.interp(target.x, xs, ys, left=np.nan, right=np.nan) if len(xs) else np.nan)
            values = np.asarray(values, dtype=float)
            valid = np.isfinite(values)
            if not valid.any() or target.time < source_times[valid].min() or target.time > source_times[valid].max():
                truth_value = np.nan
            else:
                truth_value = float(np.interp(target.time, source_times[valid], values[valid]))
                if np.any(np.isclose(source_times[valid], target.time)):
                    exact_time += 1
                else:
                    interpolated_time += 1
            if np.isfinite(truth_value):
                rows.append({"time": target.time, "species_idx": int(species_idx), "species": target.species, "w": target.w, "x": target.x, "pred_log10_N": target.log10_N, "true_log10_N": truth_value, "error_log10_N": target.log10_N - truth_value})
    aligned = pd.DataFrame(rows)
    metadata = {"time_method": "exact where available, otherwise linear interpolation", "weight_method": "exact where available, otherwise linear interpolation in x=log(w)", "extrapolation": "none", "target_grid": "prediction", "valid_cells": len(aligned), "exact_time_cells": exact_time, "interpolated_time_cells": interpolated_time, "active_weight_limits": w_max}
    return aligned, metadata
