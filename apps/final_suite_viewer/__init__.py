"""Read-only analysis helpers for the 70-run final PINNmizer suite.

The viewer uses a single plotting policy: line plots are rendered as lines only.
Point markers remain available for genuine scatter/strip plots, but are removed
whenever a trace is a line series.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


# Keep the plotting rule in one place so individual pages cannot accidentally
# re-introduce point markers with ``markers=True`` or ``mode='lines+markers'``.
_original_px_line = px.line
_original_scatter_init = go.Scatter.__init__


def _line_without_markers(*args, **kwargs):
    kwargs["markers"] = False
    return _original_px_line(*args, **kwargs)


def _scatter_without_line_markers(self, *args, **kwargs):
    if kwargs.get("mode") == "lines+markers":
        kwargs["mode"] = "lines"
    return _original_scatter_init(self, *args, **kwargs)


px.line = _line_without_markers
go.Scatter.__init__ = _scatter_without_line_markers
