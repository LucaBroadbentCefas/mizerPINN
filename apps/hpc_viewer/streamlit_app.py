"""Streamlit entrypoint with species-specific active-weight masking."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_IMPL_PATH = Path(__file__).with_name("_streamlit_app_impl.py")
_SPEC = importlib.util.spec_from_file_location("pinnmizer_hpc_viewer_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load Streamlit app implementation from {_IMPL_PATH}")

_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)

_DATA_LOSS_EXTENSION_PATH = Path(__file__).with_name("_data_loss_extension.py")
_DATA_LOSS_SPEC = importlib.util.spec_from_file_location(
    "pinnmizer_hpc_viewer_data_loss",
    _DATA_LOSS_EXTENSION_PATH,
)
if _DATA_LOSS_SPEC is None or _DATA_LOSS_SPEC.loader is None:
    raise ImportError(
        f"Could not load Streamlit data-loss extension from {_DATA_LOSS_EXTENSION_PATH}"
    )

_data_loss_extension = importlib.util.module_from_spec(_DATA_LOSS_SPEC)
_DATA_LOSS_SPEC.loader.exec_module(_data_loss_extension)


_original_normalise_mizer_dataframe = _impl.normalise_mizer_dataframe
_original_load_fixed_fields = _impl.load_fixed_fields


def _normalise_mizer_dataframe_without_masked_bins(df, source_name):
    out = _original_normalise_mizer_dataframe(df, source_name)
    masked = out["N"].notna() & (out["N"] <= 0)
    return out.loc[~masked].copy()


def _candidate_input_dirs(run_dir: Path, input_dir: Any) -> list[Path]:
    if input_dir in (None, ""):
        return []

    raw = Path(str(input_dir)).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([
            run_dir / raw,
            run_dir.parent / raw,
            Path.cwd() / raw,
        ])

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _load_species_w_max(run_dir: str | Path, species_idx: int) -> float | None:
    run_path = Path(run_dir).expanduser()
    config = _impl.safe_read_json(run_path / "config.json")

    for input_path in _candidate_input_dirs(run_path, config.get("input_dir")):
        w_max_path = input_path / "w_max.csv"
        if not w_max_path.exists():
            continue

        try:
            values = (
                pd.read_csv(w_max_path)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
                .reshape(-1)
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            idx = int(np.clip(species_idx, 0, values.size - 1))
            w_max = float(values[idx])
            if w_max > 0:
                return w_max
        except Exception:
            continue

    return None


def _crop_fixed_fields_to_w_max(
    fields: dict[str, Any],
    w_max: float | None,
) -> dict[str, Any]:
    if w_max is None or "w_eval" not in fields:
        return fields

    w = np.asarray(fields["w_eval"], dtype=float).reshape(-1)
    n_x = w.size
    active = np.isfinite(w) & (w <= w_max)

    if active.size != n_x or not active.any() or active.all():
        return fields

    out = dict(fields)
    out["w_eval"] = w[active]

    x = np.asarray(fields.get("x_eval", []))
    if x.shape == (n_x,):
        out["x_eval"] = x[active]

    field_names = {
        "log10_N",
        "log_N",
        "residual_log",
        "dlogN_dt",
        "dlogN_dw",
        "advective",
        "mu",
        "dg_dw",
        "g_eval",
    }
    for name in field_names:
        if name not in fields:
            continue
        values = np.asarray(fields[name])
        if values.ndim >= 1 and values.shape[-1] == n_x:
            out[name] = values[..., active]

    out["w_max"] = w_max
    out["active_weight_mask"] = active
    return out


def _load_fixed_fields_with_species_mask(
    run_dir: str | Path,
    species_idx: int = 0,
):
    fields = _original_load_fixed_fields(run_dir, species_idx=species_idx)
    if fields is None:
        return None

    selected_idx = int(fields.get("selected_species_idx", species_idx))
    w_max = _load_species_w_max(run_dir, selected_idx)
    return _crop_fixed_fields_to_w_max(fields, w_max)


_impl.normalise_mizer_dataframe = _normalise_mizer_dataframe_without_masked_bins
_impl.load_fixed_fields = _load_fixed_fields_with_species_mask
_data_loss_extension.install(_impl)


if __name__ == "__main__":
    _impl.main()
