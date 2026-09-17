"""Reusable UI components for scientifically explicit plots."""
from __future__ import annotations

from typing import Iterable


def plot_explanation(st, *, interpretation: str, calculation: str, inputs: Iterable[str], selection: str, alignment: str) -> None:
    """Render a collapsed scientific explanation directly below a plot."""
    with st.expander("Interpretation", expanded=False):
        st.markdown(interpretation)
        st.markdown("**Calculation**")
        st.latex(calculation)
        st.markdown("**Inputs** — " + ", ".join(f"`{value}`" for value in inputs))
        st.markdown(f"**Selection** — {selection}")
        st.markdown(f"**Alignment/transformation** — {alignment}")


def plot_scale_controls(st, fig, *, key: str, log_y: bool = False, log_x: bool = False, default_log_x: bool = False):
    """Place mathematically valid axis-scale toggles directly above a plot."""
    controls = []
    if log_y:
        controls.append("y")
    if log_x:
        controls.append("x")
    if not controls:
        return fig

    columns = st.columns(len(controls))
    column = 0
    if log_y:
        use_log_y = columns[column].toggle("Log Y", value=False, key=f"{key}_log_y")
        fig.update_yaxes(type="log" if use_log_y else "linear")
        column += 1
    if log_x:
        use_log_x = columns[column].toggle("Log X", value=default_log_x, key=f"{key}_log_x")
        fig.update_xaxes(type="log" if use_log_x else "linear")
    return fig
