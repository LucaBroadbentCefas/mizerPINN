"""File-only run and manifest discovery for the final suite."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .catalogue import BY_TASK_ID, CATALOGUE, SuiteTask

RUN_ROOTS = ("runs/pde_only_single_species", "runs/pde_multispecies")
MANIFEST_GLOBS = ("final_runs/final_suite_manifest/**/*", "**/*final*suite*manifest*")


@dataclass
class RunInstance:
    run_dir: Path
    run_label: str
    task_id: int | None = None
    source: str = "label file"
    metadata: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    error: str | None = None
    modified: float = 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        pairs: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                pairs[key.strip()] = value.strip()
        return [pairs] if pairs else []
    except (OSError, ValueError, csv.Error):
        return []


def discover_manifests(project_root: Path) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for pattern in MANIFEST_GLOBS:
        for path in project_root.glob(pattern):
            if not path.is_file() or path in seen or path.name == "final_suite_label.txt":
                continue
            seen.add(path)
            for record in _read_manifest(path):
                if "run_label" in record or "task_id" in record:
                    records.append({**record, "manifest_file": str(path)})
    return records


def _fallback_identity(run_dir: Path) -> tuple[str | None, int | None, str]:
    config = _read_json(run_dir / "config.json")
    label = config.get("run_label") or config.get("final_suite_label")
    task = config.get("task_id") or config.get("slurm_array_task_id")
    try:
        task = int(task) if task is not None else None
    except (TypeError, ValueError):
        task = None
    if label:
        return str(label), task, "config metadata"

    # Final-suite HPC run directories are created with a terminal `_taskNN`
    # suffix.  Older/copied runs may be missing final_suite_label.txt and may
    # not carry suite metadata in config.json, but the task number is still an
    # authoritative part of the final-suite run identity.  Map it back through
    # the fixed 0-69 catalogue rather than inferring the experimental label.
    match = re.search(r"_task(\d+)$", run_dir.name)
    if match:
        task = int(match.group(1))
        suite_task = BY_TASK_ID.get(task)
        if suite_task is not None:
            return suite_task.run_label, task, "run-folder task id"

    return None, task, "config metadata"


def _is_complete(run_dir: Path) -> bool:
    prediction = run_dir / "final_predictions_grid.csv"
    summary = (run_dir / "final_summary.json").exists() or (run_dir / "final_summary.csv").exists()
    return prediction.is_file() and prediction.stat().st_size > 0 and summary


def discover_local_runs(project_root: Path, run_roots: Iterable[str | Path] = RUN_ROOTS) -> list[RunInstance]:
    manifests = discover_manifests(project_root)
    manifests_by_label: dict[str, list[dict[str, Any]]] = {}
    for record in manifests:
        manifests_by_label.setdefault(str(record.get("run_label", "")), []).append(record)
    found: list[RunInstance] = []
    for root in run_roots:
        path = Path(root)
        path = path if path.is_absolute() else project_root / path
        if not path.is_dir():
            continue
        for run_dir in (item for item in path.iterdir() if item.is_dir()):
            label_file = run_dir / "final_suite_label.txt"
            label = label_file.read_text(encoding="utf-8").strip() if label_file.is_file() else None
            task_id, source = None, "label file"
            metadata: dict[str, Any] = {}
            if label and label in manifests_by_label:
                metadata = manifests_by_label[label][0].copy()
                try:
                    task_id = int(metadata.get("task_id"))
                except (TypeError, ValueError):
                    pass
            if not label:
                # An obsolete absolute manifest run_dir is only a hint. Match its
                # basename locally, otherwise use configuration metadata and,
                # finally, the authoritative `_taskNN` folder suffix.
                candidates = [m for m in manifests if Path(str(m.get("run_dir", ""))).name == run_dir.name]
                if candidates:
                    metadata = candidates[0].copy()
                    label = str(metadata.get("run_label") or "") or None
                    try:
                        task_id = int(metadata.get("task_id"))
                    except (TypeError, ValueError):
                        pass
                    source = "manifest basename"
                else:
                    label, task_id, source = _fallback_identity(run_dir)
            if not label:
                continue
            error_files = [p.name for p in run_dir.glob("*error*") if p.is_file() and p.stat().st_size]
            found.append(RunInstance(run_dir, label, task_id, source, metadata, _is_complete(run_dir), ", ".join(error_files) or None, run_dir.stat().st_mtime))
    return found


def match_catalogue(instances: Iterable[RunInstance], catalogue: Iterable[SuiteTask] = CATALOGUE) -> list[dict[str, Any]]:
    instances = list(instances)
    rows: list[dict[str, Any]] = []
    for task in catalogue:
        matches = [run for run in instances if run.run_label == task.run_label or run.task_id == task.task_id]
        matches.sort(key=lambda run: run.modified, reverse=True)
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "duplicate"
        elif matches[0].error:
            status = "incomplete/error"
        elif matches[0].complete:
            status = "found/complete"
        else:
            status = "incomplete/error"
        rows.append({**asdict(task), "status": status, "instances": matches, "selected_instance": matches[0] if matches else None})
    return rows
