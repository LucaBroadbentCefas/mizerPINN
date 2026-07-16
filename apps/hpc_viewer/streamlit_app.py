"""Streamlit entrypoint with species-specific active-weight masking."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


_IMPL_PATH = Path(__file__).with_name("_streamlit_app_impl.py")
_SPEC = importlib.util.spec_from_file_location("pinnmizer_hpc_viewer_impl", _IMPL_PATH)
