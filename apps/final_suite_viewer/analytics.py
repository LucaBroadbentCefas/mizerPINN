"""Pure dataframe calculations shared by State, PDE, Data, and Training pages."""
from __future__ import annotations

from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import fold_error, state_rmse


def mask_domain(data: pd.DataFrame, time_range=None, weight_range=None) -> pd.DataFrame:
    out = data.copy()
    if time_range is not None:
        out = out[out.time.between(*time_range)]
    if weight_range is not None:
        out = out[out.w.between(*weight_range)]
    return out


def state_error_table(aligned: pd.DataFrame) -> pd.DataFrame:
    out = aligned.copy()
    out["error_log10_N"] = out.pred_log10_N - out.true_log10_N
    return out


def error_by_time(aligned: pd.DataFrame) -> pd.DataFrame:
    return aligned.groupby("time", as_index=False).error_log10_N.agg(lambda x: state_rmse(x)).rename(columns={"error_log10_N": "RMSE_log10N"})


def error_by_weight(aligned: pd.DataFrame) -> pd.DataFrame:
    return aligned.groupby("w", as_index=False).error_log10_N.agg(lambda x: state_rmse(x)).rename(columns={"error_log10_N": "RMSE_log10N"})


def error_by_species(aligned: pd.DataFrame) -> pd.DataFrame:
    keys = ["species_idx"] + (["species"] if "species" in aligned else [])
    return aligned.groupby(keys, as_index=False).error_log10_N.agg(lambda x: state_rmse(x)).rename(columns={"error_log10_N": "RMSE_log10N"})


def fold_by_species(aligned: pd.DataFrame) -> pd.DataFrame:
    keys = ["species_idx"] + (["species"] if "species" in aligned else [])
    return aligned.groupby(keys, as_index=False).error_log10_N.agg(lambda x: fold_error(x)).rename(columns={"error_log10_N": "fold_error"})


def common_comparison_domain(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["species_idx", "time", "w"]
    common = first[keys].merge(second[keys], on=keys).drop_duplicates()
    return first.merge(common, on=keys), second.merge(common, on=keys)


def residual_summary(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float); x = x[np.isfinite(x)]; a = np.abs(x)
    if not x.size:
        return {name: np.nan for name in ("rms", "mean_abs", "median_abs", "p90_abs", "p95_abs", "p99_abs", "max_abs")}
    return {"rms": float(np.sqrt(np.mean(x*x))), "mean_abs": float(a.mean()), "median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)), "p95_abs": float(np.percentile(a, 95)), "p99_abs": float(np.percentile(a, 99)), "max_abs": float(a.max())}


def residual_aggregate(data: pd.DataFrame, by: str) -> pd.DataFrame:
    def one(group):
        a = np.abs(pd.to_numeric(group.residual_log, errors="coerce").dropna().to_numpy())
        return pd.Series({"mean_abs": np.mean(a), "p95_abs": np.percentile(a, 95), "max_abs": np.max(a)})
    return data.groupby(by).apply(one, include_groups=False).reset_index()


def pde_balance(data: pd.DataFrame, tolerance: float = 1e-6) -> tuple[pd.DataFrame, float, bool]:
    required = ["dlogN_dt", "advective", "mu", "dg_dw"]
    if not set(required).issubset(data):
        raise ValueError("PDE balance needs " + ", ".join(required))
    out = data.copy(); out["term_sum"] = out[required].sum(axis=1)
    difference = np.abs(out.term_sum - out.residual_log) if "residual_log" in out else pd.Series(dtype=float)
    maximum = float(difference.max()) if not difference.empty else np.nan
    scale = max(1.0, float(np.nanmax(np.abs(out.residual_log))) if "residual_log" in out else 1.0)
    return out, maximum, bool(np.isfinite(maximum) and maximum > tolerance * scale)


def cv_to_sigma(cv):
    return np.sqrt(np.log1p(np.asarray(cv, dtype=float) ** 2))


def prepare_observations(raw: pd.DataFrame, species_names: dict[int, str] | None = None) -> pd.DataFrame:
    data = raw.copy(); species_names = species_names or {}
    required = {"value", "prediction", "species_idx", "t_start", "t_end"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("data_predictions_final.csv is missing: " + ", ".join(missing))
    numeric = ["species_idx", "gear_idx", "t_start", "t_end", "w_min", "w_max", "value", "prediction", "log_residual", "cv", "sd_log", "cv_used", "sd_log_used", "loss_contribution", "value_true"]
    for column in numeric:
        if column in data: data[column] = pd.to_numeric(data[column], errors="coerce")
    if "sd_log_used" not in data and "sd_log" in data: data["sd_log_used"] = data.sd_log
    if "sd_log_used" not in data:
        raise ValueError("data_predictions_final.csv is missing sd_log_used/sd_log")
    if "log_residual" not in data: data["log_residual"] = np.log(data.value) - np.log(data.prediction)
    data["z"] = data.log_residual / data.sd_log_used
    data["half_z2"] = 0.5 * data.z**2
    data["t_mid"] = 0.5 * (data.t_start + data.t_end)
    data["observation_id"] = np.arange(len(data))
    if "species" not in data:
        data["species"] = data.species_idx.map(lambda i: species_names.get(int(i), f"species_{int(i)}") if np.isfinite(i) else "all")
    for column in ("obs_type", "dataset"):
        if column not in data: data[column] = "unspecified"
        data[column] = data[column].fillna("unspecified").astype(str)
    valid = np.isfinite(data.value) & np.isfinite(data.prediction) & np.isfinite(data.sd_log_used) & (data.value > 0) & (data.prediction > 0) & (data.sd_log_used > 0)
    return data[valid].copy()


def denoising_metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = data[np.isfinite(data.value_true) & (data.value_true > 0)].copy()
    out["e_obs"] = np.abs(np.log(out.value) - np.log(out.value_true))
    out["e_PINN"] = np.abs(np.log(out.prediction) - np.log(out.value_true))
    return out, {"fraction_below": float((out.e_PINN < out.e_obs).mean()), "mean_e_obs": float(out.e_obs.mean()), "mean_e_PINN": float(out.e_PINN.mean())}


def rank_misfits(data: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    return data.nlargest(min(n, len(data)), "half_z2")


def discrepancy_q(z: Iterable[float]) -> float:
    values = np.asarray(list(z), dtype=float); values = values[np.isfinite(values)]
    return float(np.sum(values**2))


def discrepancy_action(q: float, q95: float, loss_data: float, *, enabled: bool) -> tuple[float, bool]:
    """Mirror the repository gate comparison using an already-saved threshold."""
    active = (not enabled) or q > q95
    return (float(loss_data) if active else 0.0), active


def normal_quantiles(n: int) -> np.ndarray:
    return np.asarray([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])


def interval_seconds_per_step(history: pd.DataFrame) -> pd.DataFrame:
    out = history[["step", "seconds_elapsed"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("step")
    out["seconds_per_step"] = out.seconds_elapsed.diff() / out.step.diff()
    return out


def available_columns(data: pd.DataFrame, candidates: Iterable[str]) -> tuple[list[str], list[str]]:
    present = [name for name in candidates if name in data and pd.to_numeric(data[name], errors="coerce").notna().any()]
    return present, [name for name in candidates if name not in present]
