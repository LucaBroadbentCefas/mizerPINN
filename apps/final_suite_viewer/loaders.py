"""Cached, read-only loaders for final-suite output schemas."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .state import normalise_state


@st.cache_data(show_spinner=False)
def read_csv(run_dir: str, name: str) -> pd.DataFrame | None:
    path = Path(run_dir) / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(run_dir: str, name: str = "config.json") -> dict:
    try:
        value = json.loads((Path(run_dir) / name).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _fixed_csv_path(run_dir: Path) -> Path | None:
    for path in (run_dir / "fixed_grid_fields.csv", run_dir / "fixed_grid_diagnostics/fixed_grid_fields.csv"):
        if path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_fixed_fields(run_dir: str) -> pd.DataFrame | None:
    """Load the actual single- or multispecies fixed-field CSV into one schema."""
    path = _fixed_csv_path(Path(run_dir))
    if path is None:
        return None
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    data = data.rename(columns={"t_eval": "time", "x_eval": "x", "w_eval": "w", "g_eval": "g", "mu_eval": "mu"})
    if "species_idx" not in data:
        data["species_idx"] = 0
    if "species" not in data:
        data["species"] = data.species_idx.map(lambda value: f"species_{int(value)}")
    if "w" not in data and "x" in data:
        data["w"] = np.exp(pd.to_numeric(data.x, errors="coerce"))
    for column in data.columns.difference(["species"]):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "advective" not in data and {"g", "dlogN_dw"}.issubset(data):
        data["advective"] = data.g * data.dlogN_dw
    return _restore_single_species_identity(data, run_dir)


@st.cache_data(show_spinner=False)
def load_prediction_state(run_dir: str) -> pd.DataFrame | None:
    """Prefer the denser fixed diagnostic state, then the final prediction grid."""
    fixed = load_fixed_fields(run_dir)
    if fixed is not None and not fixed.empty and "log10_N" in fixed:
        raw = fixed[[c for c in ("time", "species_idx", "species", "w", "x", "log10_N") if c in fixed]].copy()
        raw["N"] = np.power(10.0, raw.log10_N)
        state = normalise_state(raw, "fixed_grid_fields.csv")
        return _restore_single_species_identity(state, run_dir)
    final = read_csv(run_dir, "final_predictions_grid.csv")
    if final is None or final.empty:
        return None
    return _restore_single_species_identity(normalise_state(final, "final_predictions_grid.csv"), run_dir)


def _restore_single_species_identity(state: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Restore the original fixture species index omitted by single-species CSVs."""
    if state.species_idx.nunique() != 1:
        return state
    input_dir = str(read_json(run_dir).get("input_dir", ""))
    match = re.search(r"(?:^|[/\\])sp_(\d+)(?:[/\\]|$)", input_dir)
    if not match:
        return state
    species_idx = int(match.group(1))
    out = state.copy()
    out["species_idx"] = species_idx
    out["species"] = f"sp_{species_idx}"
    return out


def _input_candidates(run_dir: str, filename: str) -> list[Path]:
    run = Path(run_dir)
    config = read_json(run_dir)
    candidates = [run / filename]
    configured = config.get("input_dir")
    if configured:
        raw = Path(str(configured))
        candidates.extend([raw / filename, run / raw / filename, run.parent / raw / filename])
        # Copied HPC runs commonly retain a project-relative input path. Resolve it
        # relative to the repository root rather than inferring biology from PINN output.
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / raw / filename)
    return candidates


@st.cache_data(show_spinner=False)
def load_w_max(run_dir: str) -> dict[int, float]:
    """Load biological species w_max only from explicit model inputs.

    Never infer w_max from the saved PINN support: the network is evaluated on the
    shared weight grid, including species-inactive cells above w_max.
    """
    for path in _input_candidates(run_dir, "w_max.csv"):
        if path.is_file():
            values = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").to_numpy().reshape(-1)
            if not (np.isfinite(values) & (values > 0)).all():
                continue
            if values.size == 1:
                state = load_prediction_state(run_dir)
                if state is not None and state.species_idx.nunique() == 1:
                    return {int(state.species_idx.iloc[0]): float(values[0])}
            return {i: float(value) for i, value in enumerate(values)}
    return {}
