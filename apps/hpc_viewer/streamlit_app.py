"""Streamlit entrypoint with plotting-time masking for structural zero mizer bins."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPL_PATH = Path(__file__).with_name("_streamlit_app_impl.py")
_SPEC = importlib.util.spec_from_file_location("pinnmizer_hpc_viewer_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load Streamlit app implementation from {_IMPL_PATH}")

_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_impl)
_original_normalise_mizer_dataframe = _impl.normalise_mizer_dataframe


def _normalise_mizer_dataframe_without_masked_bins(df, source_name):
    out = _original_normalise_mizer_dataframe(df, source_name)
    masked = out["N"].notna() & (out["N"] <= 0)
    return out.loc[~masked].copy()


_impl.normalise_mizer_dataframe = _normalise_mizer_dataframe_without_masked_bins


if __name__ == "__main__":
    _impl.main()
