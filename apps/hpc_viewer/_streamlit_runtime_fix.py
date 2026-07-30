"""Streamlit API compatibility and Plotly render-stability fixes."""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st


_PAGE_NAMES = [
    "Run browser",
    "Single run: training",
    "Single run: data",
    "Single run: fixed diagnostics",
    "Single run: fields",
    "Compare runs",
    "Mizer comparison",
    "File/config view",
]


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
        # Rendering the original figure is preferable to breaking the page.
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
    # These are the width-aware calls used by the viewer. Keeping the translation
    # central prevents warnings from both the base app and its data-loss extension.
    _patch_streamlit_function("plotly_chart", force_svg=True)
    for name in (
        "dataframe",
        "data_editor",
        "altair_chart",
        "vega_lite_chart",
        "line_chart",
        "area_chart",
        "bar_chart",
        "scatter_chart",
        "pyplot",
        "image",
    ):
        _patch_streamlit_function(name)


def _render_selected_page(
    impl,
    page: str,
    run_df,
    selected_run: str | None,
    selected_runs: list[str],
    run_dir: Path,
    *,
    log_y: bool,
    markers: bool,
    clip: bool,
    heat_mode: str,
    mizer_paths: str,
    uploads,
) -> None:
    if page == "Run browser":
        impl.run_browser_page(run_df, selected_runs, log_y, markers)
    elif page == "Single run: training":
        impl.history_page(run_dir, False, log_y, markers) if selected_run else st.info(
            "Select a run."
        )
    elif page == "Single run: data":
        impl.data_page(run_dir, markers) if selected_run else st.info("Select a run.")
    elif page == "Single run: fixed diagnostics":
        impl.history_page(run_dir, True, log_y, markers) if selected_run else st.info(
            "Select a run."
        )
    elif page == "Single run: fields":
        impl.fields_page(run_dir, clip, heat_mode, markers) if selected_run else st.info(
            "Select a run."
        )
    elif page == "Compare runs":
        impl.compare_page(run_df, selected_runs, clip, heat_mode, markers)
    elif page == "Mizer comparison":
        if selected_run:
            mizers = impl.load_mizer_sources(mizer_paths, uploads)
            impl.mizer_page(
                run_df,
                selected_run,
                selected_runs,
                mizers,
                clip,
                heat_mode,
                markers,
            )
        else:
            st.info("Select a PINN run.")
    elif page == "File/config view":
        impl.file_view_page(run_dir) if selected_run else st.info("Select a run.")


def _main(impl) -> None:
    st.set_page_config(page_title="PINNmizer HPC Viewer", layout="wide")
    st.title("PINNmizer HPC run viewer")

    with st.sidebar:
        run_root = Path(
            st.text_input("Run root path", str(impl.DEFAULT_RUN_ROOT))
        ).expanduser()
        if st.button("Refresh / re-scan"):
            st.cache_data.clear()

        label_mode = st.selectbox(
            "Run label mode",
            [
                "folder name",
                "short folder name",
                "model_arch + seed",
                "custom label assembled from selected config fields",
            ],
        )
        if label_mode.startswith("custom"):
            st.multiselect(
                "Custom label config fields",
                impl.RUN_COLUMNS,
                default=["model_arch", "seed"],
            )

        log_y = st.checkbox("Log y-axis where relevant", True)
        markers = st.checkbox("Show points", True)
        clip = st.checkbox("Quantile clipping for heatmaps", True)
        heat_mode = st.selectbox(
            "Heatmap colour range",
            ["auto", "symmetric around zero", "percentile clipped"],
            index=2,
        )
        mizer_paths = st.text_area("Mizer CSV local paths (one per line)")
        uploads = st.file_uploader(
            "Upload mizer CSVs", type="csv", accept_multiple_files=True
        )

    run_df = impl.scan_runs(run_root)
    if run_df.empty:
        st.warning(f"No run folders found under {run_root}")

    run_ids = run_df.run_id.tolist()
    if run_ids and st.session_state.get("selected_run") not in run_ids:
        st.session_state["selected_run"] = run_ids[0]
    if not run_ids:
        st.session_state["selected_run"] = None

    with st.sidebar:
        selected_run = (
            st.selectbox(
                "Single selected run",
                run_ids,
                index=(
                    run_ids.index(st.session_state["selected_run"])
                    if run_ids
                    and st.session_state.get("selected_run") in run_ids
                    else 0
                ),
            )
            if run_ids
            else None
        )
        if selected_run:
            st.session_state["selected_run"] = selected_run

        selected_runs = st.multiselect(
            "Runs for comparison",
            run_ids,
            default=run_ids[: min(3, len(run_ids))],
        )
        page = st.radio("Page", _PAGE_NAMES, key="hpc_viewer_page")

    selected_run = st.session_state.get("selected_run")
    run_dir = (
        Path(run_df.loc[run_df.run_id == selected_run, "run_dir"].iloc[0])
        if selected_run
        else Path("")
    )

    _render_selected_page(
        impl,
        page,
        run_df,
        selected_run,
        selected_runs,
        run_dir,
        log_y=log_y,
        markers=markers,
        clip=clip,
        heat_mode=heat_mode,
        mizer_paths=mizer_paths,
        uploads=uploads,
    )


def install(impl) -> None:
    _patch_streamlit_api()
    impl.main = lambda: _main(impl)
