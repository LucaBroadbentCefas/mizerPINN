"""Reusable UI components for scientifically explicit plots."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def plot_explanation(st, *, interpretation: str, calculation: str, inputs: Iterable[str], selection: str, alignment: str) -> None:
    """Render a collapsed scientific explanation directly below a plot."""
    with st.expander("Interpretation", expanded=False):
        st.markdown(interpretation)
        st.markdown("**Calculation**")
        st.latex(calculation)
        st.markdown("**Inputs** — " + ", ".join(f"`{value}`" for value in inputs))
        st.markdown(f"**Selection** — {selection}")
        st.markdown(f"**Alignment/transformation** — {alignment}")


def _trace_y_values(fig) -> np.ndarray:
    values = []
    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is None:
            continue
        try:
            numeric = np.asarray(y, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size:
            values.append(numeric)
    return np.concatenate(values) if values else np.array([], dtype=float)


def _is_raw_loss_figure(fig) -> bool:
    """Identify raw loss-component plots without treating total/objective loss as raw loss."""
    names = [str(getattr(trace, "name", "")) for trace in fig.data]
    return bool(names) and any(name.startswith("loss_") for name in names) and not any(name.startswith("objective_loss_") for name in names)


def _symlog_transform(values, linthresh: float):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)


def _symlog_linthresh(values: np.ndarray) -> float:
    """Choose a stable near-zero linear scale from the lower tail of non-zero magnitudes."""
    magnitudes = np.abs(values[np.isfinite(values) & (values != 0)])
    if magnitudes.size == 0:
        return 1.0
    lower = float(np.percentile(magnitudes, 5))
    exponent = np.floor(np.log10(lower))
    return float(max(np.finfo(float).tiny, 10.0 ** exponent))


def _format_loss_tick(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e4:
        return f"{value:g}"
    return f"{value:.0e}".replace("e+", "e")


def _apply_symlog_y(fig, linthresh: float):
    """Transform raw loss coordinates while keeping hover/ticks in original loss units."""
    all_values = _trace_y_values(fig)
    if all_values.size == 0:
        return fig

    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is None:
            continue
        try:
            raw = np.asarray(y, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        trace.y = _symlog_transform(raw, linthresh)
        trace.customdata = raw.reshape(-1, 1)
        trace.hovertemplate = "optimization step=%{x}<br>loss=%{customdata[0]:.6g}<extra>%{fullData.name}</extra>"

    max_abs = float(np.max(np.abs(all_values)))
    min_exp = int(np.floor(np.log10(linthresh)))
    max_exp = int(np.ceil(np.log10(max(max_abs, linthresh))))
    exponents = np.arange(min_exp, max_exp + 1, dtype=int)
    if exponents.size > 7:
        exponents = np.unique(np.rint(np.linspace(min_exp, max_exp, 7)).astype(int))
    magnitudes = np.power(10.0, exponents.astype(float))

    ticks = []
    if np.any(all_values < 0):
        ticks.extend((-magnitudes[::-1]).tolist())
    ticks.append(0.0)
    if np.any(all_values > 0):
        ticks.extend(magnitudes.tolist())

    tick_values = _symlog_transform(np.asarray(ticks, dtype=float), linthresh)
    fig.update_yaxes(
        type="linear",
        tickmode="array",
        tickvals=tick_values.tolist(),
        ticktext=[_format_loss_tick(value) for value in ticks],
        title_text="Raw loss (symmetric log scale)",
    )
    return fig


def plot_scale_controls(st, fig, *, key: str, log_y: bool = False, log_x: bool = False, default_log_x: bool = False):
    """Place mathematically valid axis-scale toggles directly above a plot.

    Raw ``loss_*`` component plots use a signed symmetric-log transform because
    their likelihood contribution may legitimately cross zero. Total/objective
    plots are deliberately not transformed this way.
    """
    controls = []
    raw_loss = bool(log_y and _is_raw_loss_figure(fig))
    if log_y:
        controls.append("y")
    if log_x:
        controls.append("x")
    if not controls:
        return fig

    columns = st.columns(len(controls))
    column = 0
    if log_y:
        if raw_loss:
            use_symlog_y = columns[column].toggle("Symmetric log Y", value=True, key=f"{key}_symlog_y")
            if use_symlog_y:
                values = _trace_y_values(fig)
                linthresh = _symlog_linthresh(values)
                fig = _apply_symlog_y(fig, linthresh)
                st.caption(f"Symmetric-log linear region: |loss| ≤ {linthresh:.3g}. Axis ticks and hover use the original loss values.")
            else:
                fig.update_yaxes(type="linear")
        else:
            use_log_y = columns[column].toggle("Log Y", value=False, key=f"{key}_log_y")
            fig.update_yaxes(type="log" if use_log_y else "linear")
        column += 1
    if log_x:
        use_log_x = columns[column].toggle("Log X", value=default_log_x, key=f"{key}_log_x")
        fig.update_xaxes(type="log" if use_log_x else "linear")
    return fig
