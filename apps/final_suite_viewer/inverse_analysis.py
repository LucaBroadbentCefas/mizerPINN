"""Pure calculations for saved inverse-parameter outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmax_recovery(estimated: pd.DataFrame, truth: np.ndarray) -> pd.DataFrame:
    out = estimated.copy()
    out["true_r_max"] = out.species_idx.astype(int).map(dict(enumerate(np.asarray(truth, dtype=float))))
    out["ratio_to_truth"] = out.estimated_r_max / out.true_r_max
    out["abs_log_error"] = np.abs(np.log(out.ratio_to_truth))
    return out


def rmse_log_ratio(estimated, truth) -> float:
    estimated, truth = np.asarray(estimated, dtype=float), np.asarray(truth, dtype=float)
    valid = np.isfinite(estimated) & np.isfinite(truth) & (estimated > 0) & (truth > 0)
    return float(np.sqrt(np.mean(np.log(estimated[valid] / truth[valid]) ** 2))) if valid.any() else np.nan


def effort_errors(effort: pd.DataFrame) -> pd.DataFrame:
    out = effort.copy()
    positive = out.true_reference_effort > 0
    out["effort_ratio"] = np.where(positive, out.estimated_effort / out.true_reference_effort, np.nan)
    out["positive_truth_log_error"] = np.where(positive, np.abs(np.log(out.effort_ratio)), np.nan)
    out["zero_truth_absolute_error"] = np.where(~positive, np.abs(out.estimated_effort - out.true_reference_effort), np.nan)
    return out


def effort_recovery_summary(effort: pd.DataFrame, gear_idx: int | None = None) -> dict[str, float]:
    data = effort_errors(effort)
    if gear_idx is not None: data = data[data.gear_idx == gear_idx]
    log_error = data.positive_truth_log_error.dropna().to_numpy()
    zero_error = data.zero_truth_absolute_error.dropna().to_numpy()
    return {
        "RMSE_logE_positive_truth": float(np.sqrt(np.mean(log_error**2))) if len(log_error) else np.nan,
        "mean_absolute_error_zero_truth": float(np.mean(zero_error)) if len(zero_error) else np.nan,
        "n_positive_truth": int(len(log_error)), "n_zero_truth": int(len(zero_error)),
    }


def reshape_selectivity(raw: np.ndarray, n_gear: int, n_species: int, n_weight: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (n_species * n_gear, n_weight):
        raise ValueError(f"selectivity shape {raw.shape}; expected {(n_species*n_gear, n_weight)}")
    return raw.reshape(n_species, n_gear, n_weight).transpose(1, 0, 2)


def fishing_mortality(effort_by_gear, catchability, selectivity, species_idx: int, gear_idx: int | None = None) -> np.ndarray:
    """Exact numpy equivalent of E_g * q_gi * s_gi(w) on the saved grid."""
    effort = np.asarray(effort_by_gear, dtype=float)
    catchability = np.asarray(catchability, dtype=float)
    selectivity = np.asarray(selectivity, dtype=float)
    terms = effort[:, None] * catchability[:, species_idx, None] * selectivity[:, species_idx, :]
    return terms.sum(axis=0) if gear_idx is None else terms[int(gear_idx)]
