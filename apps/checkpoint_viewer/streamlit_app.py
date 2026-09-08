"""Streamlit browser and materializer for PINNmizer checkpoint snapshots."""
from __future__ import annotations

import importlib.util
import subprocess
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


def _run_collections() -> dict[str, Path]:
    roots = {"All runs": DEFAULT_RUN_ROOT}
    if DEFAULT_RUN_ROOT.is_dir():
        for path in sorted(p for p in DEFAULT_RUN_ROOT.iterdir() if p.is_dir()):
            roots[str(path.relative_to(PROJECT_ROOT))] = path
    roots["Custom path"] = Path("")
    return roots


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


def _materialize_command(
    run_dir: Path,
    *,
    snapshot_root: Path,
    device: str | None,
    steps: list[int] | None,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.materialize_checkpoint_outputs",
        str(run_dir),
        "--snapshot-root",
        str(snapshot_root),
    ]
    if device is not None:
        command.extend(["--device", device])
    if steps is not None:
        command.extend(["--steps", *[str(step) for step in steps]])
    if overwrite:
        command.append("--overwrite")
    return command


def _materialize_runs(
    selected: list[tuple[str, Path]],
    *,
    snapshot_root: Path,
    device: str | None,
    per_run_steps: dict[str, list[int] | None],
    overwrite: bool,
) -> tuple[bool, list[tuple[str, subprocess.CompletedProcess[str]]]]:
    results: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    ok = True
    progress = st.progress(0.0, text="Preparing checkpoint views...")
    total = max(1, len(selected))

    for idx, (display, run_dir) in enumerate(selected, start=1):
        progress.progress(
            (idx - 1) / total,
            text=f"Preparing {idx}/{total}: {display}",
        )
        command = _materialize_command(
            run_dir,
            snapshot_root=snapshot_root,
            device=device,
            steps=per_run_steps.get(display),
            overwrite=overwrite,
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        results.append((display, completed))
        if completed.returncode != 0:
            ok = False
            break

    progress.progress(1.0, text="Checkpoint preparation finished." if ok else "Checkpoint preparation failed.")
    return ok, results


def main() -> None:
    st.set_page_config(page_title="PINNmizer checkpoint viewer", layout="wide")
    st.title("PINNmizer checkpoint viewer")

    collections = _run_collections()
    with st.sidebar:
        collection_name = st.selectbox("Run folder", list(collections))
        if collection_name == "Custom path":
            run_root = Path(st.text_input("Custom run folder", str(DEFAULT_RUN_ROOT))).expanduser()
        else:
            run_root = collections[collection_name]
        snapshot_root = Path(st.text_input("Checkpoint snapshot folder", str(DEFAULT_SNAPSHOT_ROOT))).expanduser()
        if st.button("Refresh / re-scan"):
            st.cache_data.clear()
            st.rerun()
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

    if message := st.session_state.pop("checkpoint_materialize_message", None):
        st.success(message)

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
        selected_display = st.selectbox("Run to view", run_options)
    source_run_id, source_run = run_lookup[selected_display]

    st.subheader("Prepare checkpoint views")
    st.caption(
        "Select runs and generate the intermediate plotting outputs directly from the app. "
        "This loads saved checkpoints and evaluates them with zero optimisation steps."
    )

    prep_displays = st.multiselect(
        "Runs to prepare",
        run_options,
        default=[selected_display],
    )
    prep_selected = [(display, run_lookup[display][1]) for display in prep_displays]

    prep_rows = []
    for display, run_dir in prep_selected:
        checkpoints = discover_checkpoints(run_dir)
        snapshots = discover_snapshots(run_dir, snapshot_root, project_root=PROJECT_ROOT)
        prep_rows.append({
            "run": display,
            "saved_checkpoints": len(checkpoints),
            "ready_views": len(snapshots),
            "missing_views": len([step for step in checkpoints if step not in snapshots]),
            "steps": ", ".join(f"{step:,}" for step in checkpoints),
        })
    if prep_rows:
        st.dataframe(pd.DataFrame(prep_rows), use_container_width=True, hide_index=True)

    per_run_steps: dict[str, list[int] | None] = {display: None for display in prep_displays}
    step_mode = "All saved checkpoints"
    if len(prep_selected) == 1:
        display, run_dir = prep_selected[0]
        saved = discover_checkpoints(run_dir)
        existing = discover_snapshots(run_dir, snapshot_root, project_root=PROJECT_ROOT)
        step_mode = st.radio(
            "Checkpoint selection",
            ["All saved checkpoints", "Choose checkpoint steps"],
            horizontal=True,
        )
        if step_mode == "Choose checkpoint steps":
            step_options = list(saved)
            default_steps = [step for step in step_options if step not in existing] or step_options
            per_run_steps[display] = st.multiselect(
                "Checkpoint steps",
                step_options,
                default=default_steps,
                format_func=lambda x: f"{x:,}",
            )

    c1, c2 = st.columns(2)
    device_label = c1.selectbox("Evaluation device", ["CPU", "Original run device", "CUDA"])
    device = {"CPU": "cpu", "Original run device": None, "CUDA": "cuda"}[device_label]
    overwrite = c2.checkbox("Regenerate existing checkpoint views", False)

    can_generate = bool(prep_selected)
    if len(prep_selected) == 1 and step_mode == "Choose checkpoint steps":
        can_generate = bool(per_run_steps[prep_selected[0][0]])

    if st.button("Generate checkpoint views", type="primary", disabled=not can_generate):
        ok, results = _materialize_runs(
            prep_selected,
            snapshot_root=snapshot_root.resolve(),
            device=device,
            per_run_steps=per_run_steps,
            overwrite=overwrite,
        )
        if ok:
            st.cache_data.clear()
            total = len(results)
            st.session_state["checkpoint_materialize_message"] = (
                f"Checkpoint views prepared for {total} run{'s' if total != 1 else ''}."
            )
            st.rerun()
        else:
            display, completed = results[-1]
            st.error(f"Checkpoint preparation failed for {display}.")
            output = "\n".join(x for x in [completed.stdout, completed.stderr] if x).strip()
            if output:
                st.code(output)

    st.divider()

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
            "view_ready": step in snapshots,
        }
        for step, path in raw_checkpoints.items()
    ])

    st.subheader("Checkpoint availability")
    if status.empty:
        st.info("This run has no saved model_step checkpoints. Only the final state can be viewed if final HPC outputs exist.")
    else:
        st.dataframe(status, use_container_width=True, hide_index=True)

    if not available:
        st.warning("No checkpoint views or final fixed-grid outputs are available for this run. Generate them above.")
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
