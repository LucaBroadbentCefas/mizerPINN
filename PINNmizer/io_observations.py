from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from PINNmizer.params import MizerTorchParams, _params_dtype_device

SUPPORTED_OBS_TYPES = {"biomass", "survey_biomass", "survey_abundance", "catch_total", "catch_gear"}
REQUIRED_COLUMNS = {"obs_type", "species_idx", "t_start", "value"}


def load_observation_csv(path: str | Path, params: MizerTorchParams, *, default_cv: float | None = 0.3) -> dict[str, object]:
    """Load long-format observation CSV as tensors matching params dtype/device.

    Returned tensors are length [n_obs]. Times and weights are physical units.
    """
    dtype, device = _params_dtype_device(params)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Observation CSV missing required columns: {sorted(missing)}")
    if "include" in df.columns:
        include = df["include"].fillna(True).astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])
        df = df[include].copy()
    else:
        df = df.copy()
    if df.empty:
        raise ValueError("Observation CSV contains no included observations.")

    for col, default in [
        ("dataset", ""), ("gear_idx", float("nan")), ("t_end", None),
        ("w_min", float(params.w.min().detach().cpu())), ("w_max", float(params.w.max().detach().cpu())),
        ("cv", float("nan")), ("sd_log", float("nan")), ("unit", ""), ("q", 1.0),
    ]:
        if col not in df.columns:
            df[col] = df["t_start"] if col == "t_end" and default is None else default
    df["t_end"] = df["t_end"].fillna(df["t_start"])
    df["q"] = df["q"].fillna(1.0)

    bad_types = sorted(set(df["obs_type"].astype(str)) - SUPPORTED_OBS_TYPES)
    if bad_types:
        raise ValueError(f"Unsupported obs_type values: {bad_types}; supported={sorted(SUPPORTED_OBS_TYPES)}")

    n_species = int(params.interaction.shape[0])
    species = df["species_idx"].astype(int)
    if ((species < 0) | (species >= n_species)).any():
        raise ValueError(f"species_idx must be in [0, {n_species - 1}].")
    catch_gear = df["obs_type"].astype(str).eq("catch_gear")
    if catch_gear.any() and df.loc[catch_gear, "gear_idx"].isna().any():
        raise ValueError("gear_idx is required for obs_type='catch_gear'.")

    value = pd.to_numeric(df["value"], errors="coerce")
    if (~(value > 0) | ~value.apply(lambda x: pd.notna(x))).any():
        raise ValueError("Observation value must be finite and positive.")

    cv = pd.to_numeric(df["cv"], errors="coerce")
    sd_log = pd.to_numeric(df["sd_log"], errors="coerce")
    missing_sd = sd_log.isna()
    if missing_sd.any():
        if default_cv is None and cv[missing_sd].isna().any():
            raise ValueError("Each observation needs sd_log, cv, or --data-default-cv.")
        cv_fill = cv.copy().fillna(float(default_cv))
        sd_log[missing_sd] = (1.0 + cv_fill[missing_sd] ** 2).apply(lambda x: __import__('math').sqrt(__import__('math').log(x)))
    if (~(sd_log > 0) | sd_log.isna()).any():
        raise ValueError("sd_log must be finite and positive after cv/default conversion.")

    def ten(col, kind=float):
        vals = pd.to_numeric(df[col], errors="coerce")
        return torch.as_tensor(vals.to_numpy(), dtype=(torch.long if kind is int else dtype), device=device)

    gear = pd.to_numeric(df["gear_idx"], errors="coerce").fillna(-1).astype(int)
    return {
        "obs_type": df["obs_type"].astype(str).tolist(),
        "dataset": df["dataset"].fillna("").astype(str).tolist(),
        "unit": df["unit"].fillna("").astype(str).tolist(),
        "species_idx": ten("species_idx", int),
        "gear_idx": torch.as_tensor(gear.to_numpy(), dtype=torch.long, device=device),
        "t_start": ten("t_start"), "t_end": ten("t_end"),
        "w_min": ten("w_min"), "w_max": ten("w_max"),
        "value": ten("value"), "cv": torch.as_tensor(cv.fillna(float("nan")).to_numpy(), dtype=dtype, device=device),
        "sd_log": torch.as_tensor(sd_log.to_numpy(), dtype=dtype, device=device),
        "q": ten("q"),
    }
