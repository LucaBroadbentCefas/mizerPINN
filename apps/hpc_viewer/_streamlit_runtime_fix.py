"""Streamlit API compatibility and Plotly render-stability fixes."""
from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _translate_width(kwargs: dict[str, Any]) -> None:
    """Translate Streamlit's removed use_container_width argument in place."""
    if "use_container_width" not in kwargs:
        return
    use_container_width = kwargs.pop("use_container_width")
    if "width" not in kwargs and use_container_width is not None:
        kwargs["width"] = "stretch" if bool(use_container_width) else "content"


def _force_svg_scatter_traces(figure: Any) -> Any:
    """Avoid browser WebGL-context exhaustion from Plotly Express auto mode."""
    if not isinstance(figure, go.Figure):
        return figure
    if not any(getattr(trace, "type", "") == "scattergl" for trace in figure.data):
        return figure

    try:
        payload = figure.to_plotly_json()
        for trace in payload.get("data", []):
            if trace.get("type") == "scattergl":
                trace["type"] = "scatter"
        return go.Figure(payload)
    except Exception:
        return figure


def _patch_streamlit_function(name: str, *, force_svg: bool = False) -> None:
    original = getattr(st, name, None)
    if original is None or getattr(original, "_pinnmizer_runtime_patched", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        _translate_width(kwargs)
        if force_svg:
            if args:
                args = (_force_svg_scatter_traces(args[0]), *args[1:])
            elif "figure_or_data" in kwargs:
                kwargs["figure_or_data"] = _force_svg_scatter_traces(
                    kwargs["figure_or_data"]
                )
        return original(*args, **kwargs)

    wrapped._pinnmizer_runtime_patched = True
    setattr(st, name, wrapped)


def _patch_streamlit_api() -> None:
    _patch_streamlit_function("plotly_chart", force_svg=True)
    _patch_streamlit_function("dataframe")


def _values_equal(left: Any, right: Any, impl) -> bool:
    try:
        return bool(
            np.isclose(
                float(left),
                float(right),
                rtol=1e-6,
                atol=1e-12,
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return impl._value_group_key(left) == impl._value_group_key(right)


def _display_value(value: Any) -> str:
    return "NA" if pd.isna(value) else str(value)


def _show_architecture_differences(impl, run_df, selected_runs: list[str]) -> None:
    """Render the comparison table with Arrow-safe homogeneous columns."""
    st.subheader("Selected-run architecture/config differences")
    fields = [
        "model_arch", "hidden_width", "hidden_layers", "fourier_num_features",
        "fourier_scale", "fourier_include_raw_input", "weight_factorization",
        "rwf_mu", "rwf_sigma", "rwf_apply_to", "rwf_base_init",
        "residual_form", "boundary_loss_form", "time_sampling", "causal_loss",
        "loss_weighting", "collocation_strategy", "r3_population_size", "seed",
        "fourier_seed", "lr", "n_steps", "n_time", "n_eval", "lambda_pde",
        "lambda_ic", "lambda_bc", "lambda_timestep",
    ]
    selected = run_df[run_df.run_id.isin(selected_runs)].set_index("run_id")
    display_rows: list[dict[str, str]] = []
    style_rows: list[dict[str, str]] = []

    for field in fields:
        values = [
            selected.at[run_id, field]
            if field in selected.columns and run_id in selected.index
            else np.nan
            for run_id in selected_runs
        ]
        groups: list[list[Any]] = []
        for value in values:
            for group in groups:
                if _values_equal(value, group[0], impl):
                    group.append(value)
                    break
            else:
                groups.append([value])

        if len(groups) <= 1:
            continue

        mode = max(groups, key=len)[0]
        display_rows.append(
            {
                "field": field,
                **{
                    run_id: _display_value(value)
                    for run_id, value in zip(selected_runs, values)
                },
            }
        )
        style_rows.append(
            {
                "field": field,
                **{
                    run_id: (
                        ""
                        if _values_equal(value, mode, impl)
                        else "background-color: #ffe08a; font-weight: 600"
                    )
                    for run_id, value in zip(selected_runs, values)
                },
            }
        )

    if not display_rows:
        st.info("No differing architecture/config fields among the selected runs.")
        return

    display = pd.DataFrame(display_rows).set_index("field").astype(str)
    styles = pd.DataFrame(style_rows).set_index("field").reindex_like(display)
    styled = display.style.apply(lambda _: styles, axis=None)
    st.dataframe(styled, use_container_width=True)


def install(impl) -> None:
    _patch_streamlit_api()
    impl.show_architecture_differences = (
        lambda run_df, selected_runs: _show_architecture_differences(
            impl, run_df, selected_runs
        )
    )
