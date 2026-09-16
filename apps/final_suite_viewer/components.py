"""Reusable UI components for scientifically explicit plot descriptions."""
from __future__ import annotations

from typing import Iterable


def plot_explanation(st, *, interpretation: str, calculation: str, inputs: Iterable[str], selection: str, alignment: str) -> None:
    """Render the mandatory explanation directly below a scientific plot."""
    with st.container(border=True):
        st.markdown(f"**Interpretation** — {interpretation}")
        st.markdown("**Calculation**")
        st.latex(calculation)
        st.markdown("**Inputs** — " + ", ".join(f"`{value}`" for value in inputs))
        st.markdown(f"**Selection** — {selection}")
        st.markdown(f"**Alignment/transformation** — {alignment}")
