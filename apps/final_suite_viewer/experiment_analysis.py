"""Pure experimental-design mappings and suite-level calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogue import CATALOGUE, SuiteTask
from .metrics import fold_error, state_rmse

CVS = (0.1, 0.2, 0.3, 0.4, 0.5)
NN_SCENARIOS = {
    "Perfect data": (13, 50),
    "CV 0.3, rep1": (24, 51),
    "Gap years 30–40": (46, 52),
}
BASELINE_PAIR = (12, 13)


def tasks_by_id(catalogue=CATALOGUE) -> dict[int, SuiteTask]:
    return {task.task_id: task for task in catalogue}


def validate_nn_pairings(catalogue=CATALOGUE) -> dict[str, tuple[int, int]]:
    by_id = tasks_by_id(catalogue)
    for scenario, (pinn, nn) in NN_SCENARIOS.items():
        if pinn not in by_id or nn not in by_id or by_id[nn].family != "No-PDE NN":
            raise ValueError(f"Invalid catalogue pairing for {scenario}")
    return dict(NN_SCENARIOS)


def noise_design(catalogue=CATALOGUE) -> pd.DataFrame:
    rows = [task for task in catalogue if task.family == "Noise + discrepancy gate"]
    return pd.DataFrame({
        "task_id": [task.task_id for task in rows], "cv": [task.cv for task in rows],
        "replicate": [task.replicate for task in rows], "noise_seed": [task.noise_seed for task in rows],
        "gate": [task.gate for task in rows],
    })


def noise_summary(values: pd.DataFrame, metric: str = "value") -> pd.DataFrame:
    gate_on = values[values.gate].copy()
    return gate_on.groupby("cv", as_index=False)[metric].agg(
        mean="mean", sd=lambda x: x.std(ddof=1), minimum="min", maximum="max", n="count"
    )


def paired_cv_differences(values: pd.DataFrame, metric: str = "value") -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = values[values.gate].pivot(index="noise_seed", columns="cv", values=metric)
    rows = []
    for c1 in wide.columns:
        for c2 in wide.columns:
            difference = (wide[c2] - wide[c1]).dropna()
            rows.append({"cv_from": c1, "cv_to": c2, "mean_difference": difference.mean(), "sd_difference": difference.std(ddof=1), "n": len(difference)})
    return pd.DataFrame(rows), wide


def gap_mask(data: pd.DataFrame, start: float, end: float) -> pd.Series:
    """Mask the actual withheld interval [start, end); observations resume at end."""
    return data.time.between(start, end, inclusive="left")


def missing_species_mask(data: pd.DataFrame, species_idx: int) -> pd.Series:
    return data.species_idx.astype(int).eq(int(species_idx))


def retained_omitted_years(perfect: pd.DataFrame, reduced: pd.DataFrame) -> tuple[list[float], list[float]]:
    all_years = set(pd.to_numeric(perfect.t_start, errors="coerce").dropna().unique())
    retained = set(pd.to_numeric(reduced.t_start, errors="coerce").dropna().unique())
    return sorted(retained), sorted(all_years - retained)


def year_location_mask(data: pd.DataFrame, years: list[float]) -> pd.Series:
    if not years:
        return pd.Series(False, index=data.index)
    return pd.Series(np.isclose(data.time.to_numpy()[:, None], np.asarray(years)[None, :]).any(axis=1), index=data.index)


def missing_seen_metrics(data: pd.DataFrame, missing: pd.Series) -> dict[str, float]:
    missing_errors = data.loc[missing, "error_log10_N"]
    seen_errors = data.loc[~missing, "error_log10_N"]
    miss_rmse, seen_rmse = state_rmse(missing_errors), state_rmse(seen_errors)
    return {
        "RMSE_missing": miss_rmse, "RMSE_seen": seen_rmse,
        "generalisation_penalty": miss_rmse - seen_rmse,
        "fold_missing": fold_error(missing_errors), "fold_seen": fold_error(seen_errors),
        "n_missing": int(np.isfinite(missing_errors).sum()), "n_seen": int(np.isfinite(seen_errors).sum()),
    }


def ablation_task_matrix(catalogue=CATALOGUE) -> pd.DataFrame:
    tasks = [task for task in catalogue if task.family == "Single-species main/ablations"]
    labels = {("log-u", "fourier"): "log-u Fourier", ("log-u", "mlp"): "log-u MLP", ("log-n", "fourier"): "log-n Fourier", ("log-n", "mlp"): "log-n MLP"}
    records = [{"species": task.species, "variant": labels[(task.state, task.architecture)], "task_id": task.task_id} for task in tasks]
    return pd.DataFrame(records).pivot(index="species", columns="variant", values="task_id").reindex(index=["sp_3", "sp_7", "sp_11"], columns=list(labels.values()))


def validate_baseline_pair(catalogue=CATALOGUE) -> tuple[int, int]:
    by_id = tasks_by_id(catalogue)
    if by_id[12].run_label != "ms_no_data" or by_id[13].run_label != "ms_perfect":
        raise ValueError("Baseline catalogue mapping changed")
    return BASELINE_PAIR
