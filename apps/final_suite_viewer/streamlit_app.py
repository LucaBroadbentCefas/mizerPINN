"""Streamlit entry point for the purpose-built final-suite viewer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from apps.final_suite_viewer.catalogue import CATALOGUE
from apps.final_suite_viewer.discovery import RUN_ROOTS, discover_local_runs, match_catalogue
from apps.final_suite_viewer.pages_data_training import data_page, training_page
from apps.final_suite_viewer.pages_experiments import experiment_page
from apps.final_suite_viewer.pages_inverse import inverse_page
from apps.final_suite_viewer.pages_state_pde import pde_page, state_page
from apps.final_suite_viewer.state import find_truth_source, load_truth_state

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "HPC_clone"
PAGES = (
    "Suite map", "Selected run: State", "Selected run: PDE", "Selected run: Data",
    "Selected run: Training", "Experiment analyses", "Inverse parameter analyses", "Technical details",
)
STATUS_ICON = {"found/complete": "✅", "missing": "⬜", "duplicate": "⚠️", "incomplete/error": "❌"}


@st.cache_data(show_spinner=False)
def cached_discovery(project_root: str, roots: tuple[str, ...], refresh_token: int):
    del refresh_token
    return match_catalogue(discover_local_runs(Path(project_root), roots))


@st.cache_data(show_spinner=False)
def cached_truth(path: str) -> pd.DataFrame:
    return load_truth_state(path)


def _technical_help(row: dict) -> str:
    selected = row["selected_instance"]
    bits = [f"Task {row['task_id']}", f"suite label: {row['run_label']}", f"status: {row['status']}"]
    if selected:
        bits.append(f"local folder: {selected.run_dir}")
        for key in ("slurm_job_id", "slurm_array_job_id", "git_commit"):
            if selected.metadata.get(key):
                bits.append(f"{key}: {selected.metadata[key]}")
    return "\n".join(bits)


def _select(task_id: int) -> None:
    """Button callback runs before widgets are recreated on the next rerun."""
    st.session_state.selected_task_id = task_id
    st.session_state.page = "Selected run: State"


def _button(row: dict, label: str | None = None, key: str | None = None) -> None:
    label = label or row["display_label"]
    st.button(
        f"{STATUS_ICON[row['status']]} {label}",
        key=key or f"task-{row['task_id']}",
        help=_technical_help(row),
        use_container_width=True,
        on_click=_select,
        args=(row["task_id"],),
    )


def _suite_map(rows: list[dict]) -> None:
    st.header("Final suite map")
    st.caption("Click an experiment to select it and open its state-analysis foundation. ✅ complete · ⚠️ duplicate · ❌ incomplete/error · ⬜ missing")
    by_id = {row["task_id"]: row for row in rows}

    st.subheader("1. Single-species main / ablations")
    header = st.columns([0.7, 1, 1, 1, 1])
    for col, title in zip(header, ("Species", "log-u Fourier", "log-u MLP", "log-n Fourier", "log-n MLP")):
        col.markdown(f"**{title}**")
    for i, species in enumerate(("sp_3", "sp_7", "sp_11")):
        cols = st.columns([0.7, 1, 1, 1, 1]); cols[0].markdown(f"**{species}**")
        ids = (i, 3 + i * 3, 4 + i * 3, 5 + i * 3)
        for col, task_id in zip(cols[1:], ids):
            with col: _button(by_id[task_id], by_id[task_id]["display_label"])

    st.subheader("2. Multispecies baseline")
    cols = st.columns(2)
    for col, task_id in zip(cols, (12, 13)):
        with col: _button(by_id[task_id])

    st.subheader("3. Noise + discrepancy gate")
    cols = st.columns([0.8, 1, 1, 1, 1, 1]); cols[0].markdown("**Design**")
    for col, cv in zip(cols[1:], (0.1, 0.2, 0.3, 0.4, 0.5)): col.markdown(f"**CV {cv:.1f}**")
    for rep in range(1, 6):
        cols = st.columns([0.8, 1, 1, 1, 1, 1]); cols[0].markdown(f"**gate ON rep{rep}**")
        for cv_i, col in enumerate(cols[1:]):
            with col: _button(by_id[14 + cv_i * 5 + rep - 1], f"rep{rep}")
    cols = st.columns([0.8, 1, 1, 1, 1, 1]); cols[0].markdown("**gate OFF rep1**")
    for cv_i, col in enumerate(cols[1:]):
        with col: _button(by_id[39 + cv_i], "rep1 only")
    st.caption("Gate-OFF has one replicate per CV; the application will not present it as a five-replicate uncertainty estimate.")

    sections = (
        ("4. Missing / sparse data", range(44, 50)), ("5. No-PDE NN", range(50, 53)),
        ("6. Rmax recovery", range(53, 62)), ("7. CV recovery", range(62, 66)),
        ("8. Effort recovery", range(66, 70)),
    )
    for title, ids in sections:
        st.subheader(title)
        ids = list(ids)
        for start in range(0, len(ids), 5):
            for col, task_id in zip(st.columns(min(5, len(ids) - start)), ids[start:start + 5]):
                with col: _button(by_id[task_id])


def _selected(rows: list[dict]) -> dict | None:
    task_id = st.session_state.get("selected_task_id")
    return next((row for row in rows if row["task_id"] == task_id), None)


def _placeholder(page: str, row: dict | None, truth_path: str) -> None:
    st.header(page)
    if row is None:
        st.info("Select an experiment from the Suite map.")
        return
    st.subheader(f"Task {row['task_id']}: {row['display_label']}")
    st.caption(row["run_label"])
    if not row["selected_instance"]:
        st.warning("This expected suite task has no discovered local run.")
    if page == "Selected run: State" and not truth_path:
        st.error("Truth source unavailable: no canonical full time × species × weight truth export was found. Configure an explicit long-form truth CSV under Technical details; no PINN run is substituted as truth.")
    st.info("This scientific page is intentionally reserved for a later stage; this change provides navigation, discovery, truth loading, alignment, metrics, and explanation foundations only.")


def _show_file(path: Path) -> None:
    if path.suffix == ".json":
        try: st.json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError): st.warning("Unreadable JSON")
    elif path.suffix == ".csv":
        try: st.dataframe(pd.read_csv(path), use_container_width=True)
        except Exception as exc: st.warning(f"Unreadable CSV: {exc}")
    else:
        try: st.code(path.read_text(encoding="utf-8"))
        except OSError: st.warning("Unreadable file")


def _technical(rows: list[dict], project_root: Path, truth_path: str) -> None:
    st.header("Technical details")
    st.markdown("Normal scientific navigation uses suite labels. Timestamp folders and infrastructure metadata are kept here.")
    row = _selected(rows)
    if row:
        instances = row["instances"]
        if len(instances) > 1:
            choices = {f"{run.run_dir} (modified {run.modified:.0f})": run for run in instances}
            key = f"duplicate_instance_{row['task_id']}"
            chosen_label = st.selectbox("Duplicate instance", list(choices), key=key, help="The most recently modified instance is selected by default and listed first. This choice is retained for this browser session.")
            row["selected_instance"] = choices[chosen_label]
            st.warning("Duplicate suite identity. Selection defaults to the most recently modified local instance.")
        run = row["selected_instance"]
        st.json({"suite": {key: value for key, value in row.items() if key not in {"instances", "selected_instance"}}, "manifest": run.metadata if run else {}})
        if run:
            st.code(str(run.run_dir), language=None)
            files = sorted(path for path in run.run_dir.rglob("*") if path.is_file())
            chosen = st.selectbox("Actual output file", [str(path.relative_to(run.run_dir)) for path in files]) if files else None
            if chosen: _show_file(run.run_dir / chosen)
    else:
        st.info("Select a task to inspect its configuration, command, summary, files, and metadata.")
    st.subheader("Truth state")
    if truth_path:
        try:
            truth = cached_truth(truth_path)
            st.success(f"Loaded canonical truth from {truth_path}: {len(truth):,} positive active rows.")
            st.dataframe(truth.head(100), use_container_width=True)
        except Exception as exc: st.error(f"Truth source could not be loaded: {exc}")
    else:
        st.warning("No canonical full truth export was discovered. The fixture n.csv/n_init_full.csv files are initial-state inputs, and perfect.csv is observation data; neither is treated as full truth.")


def main() -> None:
    st.set_page_config(page_title="PINNmizer final suite", layout="wide")
    st.title("PINNmizer final HPC suite")
    st.session_state.setdefault("page", "Suite map")
    st.session_state.setdefault("refresh_token", 0)
    automatic_truth = find_truth_source(PROJECT_ROOT)
    with st.sidebar:
        st.selectbox("Page", PAGES, key="page")
        selected = st.session_state.get("selected_task_id")
        st.caption(f"Selected task: {selected if selected is not None else 'none'}")
        with st.expander("Technical settings"):
            project_root = Path(st.text_input("Project root", str(PROJECT_ROOT))).expanduser()
            roots_text = st.text_area("Run roots (one per line)", "\n".join(RUN_ROOTS))
            truth_path = st.text_input("Canonical truth CSV", str(automatic_truth or ""), help="Explicit long-form mizer truth only; never a PINN prediction.")
            if st.button("Refresh disk index"):
                st.session_state.refresh_token += 1
                st.cache_data.clear()
                st.rerun()
    roots = tuple(line.strip() for line in roots_text.splitlines() if line.strip())
    rows = cached_discovery(str(project_root), roots, st.session_state.refresh_token)
    # Reapply a duplicate choice on every rerun.  Mutating the cached row only
    # on the Technical details page previously lost the choice on navigation.
    for row in rows:
        key = f"duplicate_instance_{row['task_id']}"
        if key in st.session_state and len(row["instances"]) > 1:
            chosen = st.session_state[key]
            match = next((run for run in row["instances"] if chosen.startswith(str(run.run_dir) + " ")), None)
            if match is not None:
                row["selected_instance"] = match
    page = st.session_state.page
    if page == "Suite map": _suite_map(rows)
    elif page == "Selected run: State": state_page(rows, truth_path)
    elif page == "Selected run: PDE": pde_page(rows)
    elif page == "Selected run: Data": data_page(rows)
    elif page == "Selected run: Training": training_page(rows)
    elif page == "Experiment analyses": experiment_page(rows, truth_path)
    elif page == "Inverse parameter analyses": inverse_page(rows)
    elif page == "Technical details": _technical(rows, project_root, truth_path)
    else: _placeholder(page, _selected(rows), truth_path)


if __name__ == "__main__":
    main()
