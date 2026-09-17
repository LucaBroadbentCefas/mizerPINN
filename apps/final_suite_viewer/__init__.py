"""Read-only analysis helpers for the 70-run final PINNmizer suite.

The viewer uses a single plotting policy:
- line plots are rendered as lines only;
- Plotly charts use a stable transparent background that follows Streamlit's theme;
- scroll-wheel zoom and box/lasso selection are disabled to reduce accidental
  navigation into empty plot regions;
- double-click resets/autoscales the axes.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


_original_px_line = px.line
_original_scatter_init = go.Scatter.__init__
_original_plotly_chart = st.plotly_chart


def _line_without_markers(*args, **kwargs):
    kwargs["markers"] = False
    return _original_px_line(*args, **kwargs)


def _scatter_without_line_markers(self, *args, **kwargs):
    if kwargs.get("mode") == "lines+markers":
        kwargs["mode"] = "lines"
    return _original_scatter_init(self, *args, **kwargs)


def _stable_plotly_chart(figure_or_data, *args, **kwargs):
    """Apply viewer-wide Plotly defaults without altering plotted data."""
    try:
        figure_or_data.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            dragmode="zoom",
        )
    except AttributeError:
        pass
    config = dict(kwargs.pop("config", {}) or {})
    config.setdefault("scrollZoom", False)
    config.setdefault("doubleClick", "reset+autosize")
    config.setdefault("displaylogo", False)
    config.setdefault("modeBarButtonsToRemove", ["select2d", "lasso2d", "pan2d"])
    return _original_plotly_chart(figure_or_data, *args, config=config, **kwargs)


px.line = _line_without_markers
go.Scatter.__init__ = _scatter_without_line_markers
st.plotly_chart = _stable_plotly_chart
