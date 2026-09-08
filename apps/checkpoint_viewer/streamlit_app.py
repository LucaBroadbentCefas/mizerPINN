"""Read-only Streamlit browser for PINNmizer checkpoint snapshots."""
from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HPC_VIEWER_PATH = PROJECT_ROOT / "apps" / "hpc_viewer" / "streamlit_app.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PINNmizer.diagnostics.checkpoint_snapshots import (  # noqa: E402
    discover_checkpoints,
    discover_snapshots,
)


def _load_hpc_viewer():
    spec = importlib.util.spec_from_file_location("pinnmizer_checkpoint_hpc_viewer", HPC_VIEWER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HPC viewer from {HPC_VIEWER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._impl


impl = _load_hpc_viewer()
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "checkpoint_views"


def _run_label(run_id: str, run_dir: Path) -> str:
    label_file = run_dir / "final_suite_label.txt"
    if label_file.is_file():
        try:
            label = label_file.read_text(encoding="utf-8").strip()
        except OSError:
            label = ""
        if label:
            return f"{label} | {run_id}"
    return run_id


def _final_step(run_dir: Path) -> int | None:
    summary = impl.load_final_summary(run_dir)
    for key in ("n_steps_completed", "n_steps"):
        value = summary.get(key)
        if value is not None:
            try:
                if np.isfinite(float(value)):
                    return int(float(value))
            except (TypeError, ValueError):
                pass
    config = impl.load_config(run_dir)
    value = config.get("n_steps")
    try:
        if value is not None and np.isfinite(float(value)):
            return int(float(value))
    except (TypeError, ValueError):
        pass
    history = impl.load_loss_history(run_dir)
    if history is not None and not history.empty and "step" in history:
        steps = pd.to_numeric(history["step"], errors="coerce").dropna()
        if not steps.empty:
            return int(steps.max())
    return None


def _as_comparison_frame(entries: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for label, run_dir in entries.items():
        frame = impl.scan_runs(run_dir)
        if frame.empty:
            continue
        row = frame.iloc[[0]].copy()
        row.loc[:, "run_id"] = label
        row.loc[:, "run_dir"] = str(run_dir)
        frames.append(row)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=impl.RUN_COLUMNS)


def _default_comparison(labels: list[str]) -> list[str]:
    if len(labels) <= 4:
        return labels
    return [labels[0], labels[len(labels) // 2], labels[-1]]


def main() -> None:
    st.set_page_config(page_title="PINNmizer checkpoint viewer", layout="wide")
    st.title("PINNmizer checkpoint viewer")

    with st.sidebar:
        run_root = Path(st.text_input("Source run root", str(DEFAULT_RUN_ROOT))).expanduser()
        snapshot_root = Path(st.text_input("Checkpoint snapshot root", str(DEFAULT_SNAPSHOT_ROOT))).expanduser()
        if st.button("Refresh / re-scan"):
            st.cache_data.clear()
        log_y = st.checkbox("Log y-axis where relevant", True)
        markers = st.checkbox("Show points", True)
        clip = st.checkbox("Quantile clipping for heatmaps", True)
        heat_mode = st.selectbox(
            "Heatmap colour range",
            ["auto", "symmetric around zero", "percentile clipped"],
            index=2,
        )
        mizer_paths = st.text_area("Mizer CSV local paths (one per line)")
        uploads = st.file_uploader("Upload mizer CSVs", type="csv", accept_multiple_files=True)

    run_df = impl.scan_runs(run_root)
    if run_df.empty:
        st.warning(f"No source runs found under {run_root}")
        return

    run_options = []
    run_lookup: dict[str, tuple[str, Path]] = {}
    for row in run_df.itertuples(index=False):
        run_id = str(row.run_id)
        run_dir = Path(row.run_dir)
        display = _run_label(run_id, run_dir)
        if display in run_lookup:
            display = f"{display} | {run_dir}"
        run_options.append(display)
        run_lookup[display] = (run_id, run_dir)

    with st.sidebar:
        selected_display = st.selectbox("Source run", run_options)
    source_run_id, source_run = run_lookup[selected_display]

    raw_checkpoints = discover_checkpoints(source_run)
    snapshots = discover_snapshots(
        source_run,
        snapshot_root,
        project_root=PROJECT_ROOT,
    )

    available: dict[str, Path] = {
        f"step {step:,}": path for step, path in snapshots.items()
    }
    final_step = _final_step(source_run)
    if impl.load_fixed_fields(source_run, species_idx=0) is not None:
        final_label = f"final ({final_step:,})" if final_step is not None else "final"
        available[final_label] = source_run

    status = pd.DataFrame([
        {
            "step": step,
            "checkpoint": str(path),
            "snapshot_ready": step in snapshots,
            "snapshot": str(snapshots[step]) if step in snapshots else "",
        }
        for step, path in raw_checkpoints.items()
    ])

    st.subheader("Checkpoint availability")
    if status.empty:
        st.info("This run has no saved model_step checkpoints. Only the final state can be viewed if final HPC outputs exist.")
    else:
        st.dataframe(status, use_container_width=True, hide_index=True)

    missing_steps = [step for step in raw_checkpoints if step not in snapshots]
    if missing_steps:
        command = [
            sys.executable,
            "-m",
            "scripts.materialize_checkpoint_outputs",
            str(source_run),
            "--steps",
            *[str(step) for step in missing_steps],
            "--device",
            "cpu",
        ]
        st.info(
            "These checkpoints need one zero-step evaluation before the viewer can plot them. "
            "Run this outside Streamlit; it does not retrain the model."
        )
        st.code(shlex.join(command), language="bash")

    if not available:
        st.warning("No materialized checkpoint snapshots or final fixed-grid outputs are available for this run.")
        return

    labels = list(available)
    state_key = f"checkpoint_selected::{source_run.resolve()}"
    if st.session_state.get(state_key) not in labels:
        st.session_state[state_key] = labels[-1]

    with st.sidebar:
        selected_label = st.select_slider(
            "Checkpoint / stopping step",
            options=labels,
            key=state_key,
        )
        compare_labels = st.multiselect(
            "Checkpoints to compare",
            labels,
            default=_default_comparison(labels),
            key=f"checkpoint_compare::{source_run.resolve()}",
        )

    selected_dir = available[selected_label]
    comparison_entries = {label: available[label] for label in compare_labels}
    comparison_df = _as_comparison_frame(comparison_entries)
    mizers = impl.load_mizer_sources(mizer_paths, uploads)

    st.caption(f"Source run: `{source_run_id}` | showing `{selected_label}`")
    tabs = st.tabs([
        "Checkpoint fields",
        "Training to checkpoint",
        "Data at checkpoint",
        "Compare checkpoints",
        "Mizer comparison",
        "Files/config",
    ])
    with tabs[0]:
        impl.fields_page(selected_dir, clip, heat_mode, markers)
    with tabs[1]:
        impl.history_page(selected_dir, False, log_y, markers)
    with tabs[2]:
        impl.data_page(selected_dir, markers)
    with tabs[3]:
        if len(comparison_entries) < 2:
            st.info("Select at least two checkpoints in the sidebar.")
        else:
            impl.compare_page(comparison_df, compare_labels, clip, heat_mode, markers)
    with tabs[4]:
        if not mizers:
            st.info("Provide at least one mizer CSV in the sidebar.")
        elif comparison_df.empty:
            st.info("Select checkpoint snapshots for comparison.")
        else:
            selected_for_mizer = selected_label if selected_label in comparison_entries else compare_labels[0]
            impl.mizer_page(
                comparison_df,
                selected_for_mizer,
                compare_labels,
                mizers,
                clip,
                heat_mode,
                markers,
            )
    with tabs[5]:
        impl.file_view_page(selected_dir)


if __name__ == "__main__":
    main()
