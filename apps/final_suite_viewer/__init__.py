"""Read-only analysis helpers for the 70-run final PINNmizer suite.

The viewer uses a single plotting policy:
- line plots are rendered as lines only;
- Plotly scatter/scattergl traces are rendered with SVG rather than WebGL to
  avoid browser WebGL-context exhaustion when many long plots share a page;
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
    # Plotly Express otherwise switches to scattergl automatically for long
    # series. Multiple long charts can then exhaust browser WebGL contexts.
    kwargs["render_mode"] = "svg"
    return _original_px_line(*args, **kwargs)


def _scatter_without_line_markers(self, *args, **kwargs):
    if kwargs.get("mode") == "lines+markers":
        kwargs["mode"] = "lines"
    return _original_scatter_init(self, *args, **kwargs)


def _force_svg_scatter_traces(figure_or_data):
    """Convert any auto-generated scattergl traces to ordinary SVG scatter."""
    if not isinstance(figure_or_data, go.Figure):
        return figure_or_data
    if not any(getattr(trace, "type", "") == "scattergl" for trace in figure_or_data.data):
        return figure_or_data
    try:
        payload = figure_or_data.to_plotly_json()
        for trace in payload.get("data", []):
            if trace.get("type") == "scattergl":
                trace["type"] = "scatter"
        return go.Figure(payload)
    except Exception:
        return figure_or_data


def _stable_plotly_chart(figure_or_data, *args, **kwargs):
    """Apply viewer-wide Plotly rendering and interaction defaults."""
    figure_or_data = _force_svg_scatter_traces(figure_or_data)
    try:
        figure_or_data.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            dragmode="zoom",
            autosize=True,
        )
    except AttributeError:
        pass

    # Current Streamlit uses width="stretch"; translating here keeps all old
    # page calls compatible while avoiding the deprecated sizing path.
    use_container_width = kwargs.pop("use_container_width", None)
    if "width" not in kwargs and use_container_width is not None:
        kwargs["width"] = "stretch" if bool(use_container_width) else "content"

    config = dict(kwargs.pop("config", {}) or {})
    config.setdefault("scrollZoom", False)
    config.setdefault("doubleClick", "reset+autosize")
    config.setdefault("responsive", True)
    config.setdefault("displaylogo", False)
    config.setdefault("modeBarButtonsToRemove", ["select2d", "lasso2d", "pan2d"])
    return _original_plotly_chart(figure_or_data, *args, config=config, **kwargs)


px.line = _line_without_markers
go.Scatter.__init__ = _scatter_without_line_markers
st.plotly_chart = _stable_plotly_chart
