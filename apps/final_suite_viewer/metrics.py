"""Shared final-suite state error metrics."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def state_rmse(errors: Iterable[float]) -> float:
    values = np.asarray(list(errors), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def fold_error(errors: Iterable[float]) -> float:
    values = np.asarray(list(errors), dtype=float)
    values = values[np.isfinite(values)]
    return float(10.0 ** np.mean(np.abs(values))) if values.size else float("nan")


def aggregate_state_metrics(aligned: pd.DataFrame, *, group_by: str | list[str] | None = None, time_range: tuple[float, float] | None = None, weight_range: tuple[float, float] | None = None, withheld_mask: Iterable[bool] | None = None) -> tuple[pd.DataFrame, str | None]:
    data = aligned.copy()
    if time_range is not None:
        data = data[data.time.between(*time_range)]
    if weight_range is not None:
        data = data[data.w.between(*weight_range)]
    if withheld_mask is not None:
        mask = np.asarray(list(withheld_mask), dtype=bool)
        if mask.size != len(aligned):
            raise ValueError("withheld_mask length must equal the unfiltered aligned table")
        data = data.loc[aligned.index[mask].intersection(data.index)]
    data = data[np.isfinite(data.error_log10_N)]
    columns = ([group_by] if isinstance(group_by, str) else list(group_by or []))
    if data.empty:
        return pd.DataFrame(columns=columns + ["n_cells", "RMSE_log10N", "fold_error"]), "No valid comparison cells after selection."
    def summarise(frame: pd.DataFrame) -> pd.Series:
        return pd.Series({"n_cells": len(frame), "RMSE_log10N": state_rmse(frame.error_log10_N), "fold_error": fold_error(frame.error_log10_N)})
    if columns:
        result = data.groupby(columns, dropna=False).apply(summarise, include_groups=False).reset_index()
    else:
        result = summarise(data).to_frame().T
    return result, None
