"""Small training-page extension for observation-loss and gradient diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _has_finite(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns and pd.to_numeric(df[name], errors="coerce").notna().any()


def install(impl) -> None:
    """Add data to the existing raw-loss plot and expose per-term weighting gradients."""
    original_history_page = impl.history_page
    original_make_multi_line_plot = impl.make_multi_line_plot

    def make_multi_line_plot_with_data(df, x, ys, title, y_label, log_y, markers):
        if title == "Unscaled loss terms" and _has_finite(df, "loss_data") and "loss_data" not in ys:
            ys = [*ys, "loss_data"]
        return original_make_multi_line_plot(df, x, ys, title, y_label, log_y, markers)

    def history_page(run_dir, fixed, log_y, markers):
        if fixed:
            return original_history_page(run_dir, fixed, log_y, markers)

        # Reuse the base page so every existing training plot remains unchanged,
        # but intercept the raw-loss plot to include loss_data when available.
        old_make = impl.make_multi_line_plot
        impl.make_multi_line_plot = make_multi_line_plot_with_data
        try:
            original_history_page(run_dir, fixed, log_y, markers)
        finally:
            impl.make_multi_line_plot = old_make

        df = impl.load_loss_history(run_dir)
        if df is None or df.empty:
            return

        expert = [
            "grad_norm_pde_for_weighting",
            "grad_norm_ic_for_weighting",
            "grad_norm_bc_for_weighting",
            "grad_norm_data_for_weighting",
        ]
        legacy = [
            "grad_pde_max_for_weighting",
            "grad_ic_mean_for_weighting",
            "grad_bc_mean_for_weighting",
            "grad_data_mean_for_weighting",
        ]

        expert_present = [name for name in expert if _has_finite(df, name)]
        legacy_present = [name for name in legacy if _has_finite(df, name)]
        gradient_terms = expert_present if expert_present else legacy_present

        if not gradient_terms:
            return

        impl.plot_with_desc(
            lambda: original_make_multi_line_plot(
                df,
                "step",
                gradient_terms,
                "Gradient terms",
                "gradient",
                log_y,
                markers,
            ),
            (
                "Per-term neural-network gradient diagnostics used by adaptive loss weighting. "
                "PDE, IC, BC, and observation-data terms are shown together when present; "
                "these are optimisation diagnostics rather than biological quantities."
            ),
            impl.line_desc(
                "loss_history.csv",
                ["step", *gradient_terms],
                "expert gradient norms when available; otherwise legacy Wang gradient summaries",
                log_y,
                False,
            ),
        )

    impl.PLOT_INTERPRETATIONS["Unscaled loss terms"] = (
        "Raw constraint losses before objective/adaptive weighting. `loss_pde`, `loss_ic`, "
        "`loss_bc`, and `loss_data` show PDE, initial-condition, boundary-condition, and "
        "observation-data mismatch respectively; `loss_timestep` is shown only when present "
        "in the base history plot."
    )
    impl.PLOT_INTERPRETATIONS["Gradient terms"] = (
        "Per-term gradient diagnostics used by the adaptive weighting scheme, including the "
        "observation-data gradient when it was recorded."
    )
    impl.history_page = history_page
