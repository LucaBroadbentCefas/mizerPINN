"""Backfill final-suite bookkeeping after post-training failures.

This script does not train or modify model checkpoints. It repairs the files
that ``record_run()`` in ``scripts/final_model_suite.sh`` would have written
after a successful trainer exit:

* ``<run_dir>/final_suite_label.txt``
* ``final_runs/final_suite_manifest/job_<ARRAY_ID>/task_XX_<label>.txt``
* ``<run_dir>/r_max_true.csv`` for Rmax-recovery tasks, when applicable

A run is eligible only when the original directory contains a non-empty
``model_final.pt``. Candidate directories are matched to the final-suite SLURM
log start time so unrelated historical ``_taskNN`` runs are not silently
claimed as final-suite runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.final_suite_viewer.catalogue import CATALOGUE, SuiteTask

RUN_ROOTS = (
    Path("runs/pde_only_single_species"),
    Path("runs/pde_multispecies"),
)
RUN_RE = re.compile(r"_job(?P<job_id>\d+)_task(?P<task_id>\d+)$")
RUN_TIME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?:_\d+)?_job")
GIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
FINAL_SUITE_N_STEPS = 15000
FINAL_SUITE_START = datetime(2026, 9, 8)
MAX_LOG_TIME_DELTA_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class Candidate:
    run_dir: Path
    task_id: int
    job_id: str
    started_at: datetime | None


@dataclass(frozen=True)
class LogAttempt:
    task_id: int
    run_label: str | None
    started_at: datetime | None
    git_commit: str | None


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    # Folder stamps are local wall-clock timestamps. Compare like with like.
    return parsed.replace(tzinfo=None)


def _folder_started_at(name: str) -> datetime | None:
    match = RUN_TIME_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        )
    except ValueError:
        return None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _looks_like_final_suite_run(candidate: Candidate, task: SuiteTask) -> bool:
    run_dir = candidate.run_dir
    model_final = run_dir / "model_final.pt"
    if not model_final.is_file() or model_final.stat().st_size == 0:
        return False

    config = _read_json(run_dir / "config.json")
    n_steps = config.get("n_steps")
    if n_steps is not None:
        try:
            if int(n_steps) != FINAL_SUITE_N_STEPS:
                return False
        except (TypeError, ValueError):
            return False

    if task.state and config.get("state_parameterization") not in (None, task.state):
        return False
    if task.architecture and config.get("model_arch") not in (None, task.architecture):
        return False
    if task.species:
        input_dir = str(config.get("input_dir", ""))
        if input_dir and task.species not in input_dir.replace("\\", "/"):
            return False

    return True


def _discover_candidates(project_root: Path) -> dict[int, list[Candidate]]:
    by_task: dict[int, list[Candidate]] = {}
    for relative_root in RUN_ROOTS:
        root = project_root / relative_root
        if not root.is_dir():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            match = RUN_RE.search(run_dir.name)
            if not match:
                continue
            task_id = int(match.group("task_id"))
            by_task.setdefault(task_id, []).append(
                Candidate(
                    run_dir=run_dir,
                    task_id=task_id,
                    job_id=match.group("job_id"),
                    started_at=_folder_started_at(run_dir.name),
                )
            )
    return by_task


def _parse_log_attempts(path: Path) -> list[LogAttempt]:
    if not path.is_file():
        return []

    attempts: list[LogAttempt] = []
    current: dict[str, str] = {}

    def finish() -> None:
        if "task" not in current:
            return
        try:
            task_id = int(current["task"])
        except ValueError:
            return
        git_commit = current.get("git_commit")
        if git_commit and not GIT_RE.fullmatch(git_commit):
            git_commit = None
        attempts.append(
            LogAttempt(
                task_id=task_id,
                run_label=current.get("run_label"),
                started_at=_parse_iso_datetime(current["start_time"])
                if current.get("start_time")
                else None,
                git_commit=git_commit,
            )
        )

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("task="):
            finish()
            current = {"task": line.split("=", 1)[1].strip()}
            continue
        if "=" not in line or not current:
            continue
        key, value = line.split("=", 1)
        if key in {"run_label", "start_time", "git_commit"}:
            current[key] = value.strip()
    finish()
    return attempts


def _attempts_for_task(
    project_root: Path, array_job_id: str, task: SuiteTask
) -> list[LogAttempt]:
    path = project_root / "slurm_logs" / f"pinn_final_suite_{array_job_id}_{task.task_id}.out"
    attempts = [a for a in _parse_log_attempts(path) if a.task_id == task.task_id]
    labelled = [a for a in attempts if a.run_label == task.run_label]
    return labelled or attempts


def _select_candidate(
    task: SuiteTask,
    candidates: list[Candidate],
    attempts: list[LogAttempt],
) -> Candidate | None:
    valid = [c for c in candidates if _looks_like_final_suite_run(c, task)]
    if not valid:
        return None

    attempt_times = [a.started_at for a in attempts if a.started_at is not None]
    if attempt_times:
        ranked: list[tuple[float, float, Candidate]] = []
        for candidate in valid:
            if candidate.started_at is None:
                continue
            delta = min(abs((candidate.started_at - t).total_seconds()) for t in attempt_times)
            ranked.append((delta, -candidate.run_dir.stat().st_mtime, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if ranked and ranked[0][0] <= MAX_LOG_TIME_DELTA_SECONDS:
            return ranked[0][2]

    recent = [
        c for c in valid
        if c.started_at is not None and c.started_at >= FINAL_SUITE_START
    ]
    if len(recent) == 1:
        return recent[0]
    if recent:
        return max(recent, key=lambda c: c.run_dir.stat().st_mtime)
    return None


def _matching_attempt(candidate: Candidate, attempts: list[LogAttempt]) -> LogAttempt | None:
    if candidate.started_at is None:
        return attempts[-1] if attempts else None
    timed = [a for a in attempts if a.started_at is not None]
    if not timed:
        return attempts[-1] if attempts else None
    return min(timed, key=lambda a: abs((candidate.started_at - a.started_at).total_seconds()))


def _atomic_write(path: Path, text: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _restore_rmax_true(
    project_root: Path,
    array_job_id: str,
    task: SuiteTask,
    run_dir: Path,
    *,
    dry_run: bool,
) -> str | None:
    if task.task_id < 53 or task.task_id > 61:
        return None
    source_root = (
        project_root
        / "final_runs"
        / "final_suite_inputs"
        / f"job_{array_job_id}"
        / f"task_{task.task_id}"
    )
    sources = list(source_root.rglob("r_max_true.csv")) if source_root.is_dir() else []
    if len(sources) != 1:
        raise RuntimeError(
            f"Task {task.task_id}: expected exactly one r_max_true.csv under {source_root}, found {len(sources)}"
        )
    destination = run_dir / "r_max_true.csv"
    if not dry_run:
        shutil.copy2(sources[0], destination)
    return str(destination)


def _viewer_outputs_present(run_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    prediction = run_dir / "final_predictions_grid.csv"
    if not prediction.is_file() or prediction.stat().st_size == 0:
        missing.append("final_predictions_grid.csv")
    if not ((run_dir / "final_summary.json").is_file() or (run_dir / "final_summary.csv").is_file()):
        missing.append("final_summary.json/csv")
    return not missing, missing


def repair_task(
    project_root: Path,
    array_job_id: str,
    task: SuiteTask,
    candidates: list[Candidate],
    *,
    dry_run: bool,
    overwrite: bool,
) -> tuple[str, str]:
    manifest = (
        project_root
        / "final_runs"
        / "final_suite_manifest"
        / f"job_{array_job_id}"
        / f"task_{task.task_id:02d}_{task.run_label}.txt"
    )
    if manifest.is_file() and not overwrite:
        existing = manifest.read_text(encoding="utf-8", errors="replace")
        expected_identity = f"task_id={task.task_id}\nrun_label={task.run_label}\n"
        if expected_identity not in existing:
            return "error", f"existing manifest conflicts: {manifest}"
        return "already_recorded", str(manifest)

    attempts = _attempts_for_task(project_root, array_job_id, task)
    candidate = _select_candidate(task, candidates, attempts)
    if candidate is None:
        return "not_repaired", "no completed final-suite model_final.pt could be identified"

    label_path = candidate.run_dir / "final_suite_label.txt"
    if label_path.is_file():
        existing_label = label_path.read_text(encoding="utf-8").strip()
        if existing_label != task.run_label and not overwrite:
            return "error", f"existing label is {existing_label!r}, expected {task.run_label!r}"

    matched_attempt = _matching_attempt(candidate, attempts)
    git_commit = matched_attempt.git_commit if matched_attempt and matched_attempt.git_commit else "unknown"

    _atomic_write(label_path, task.run_label + "\n", dry_run=dry_run)
    _restore_rmax_true(
        project_root,
        array_job_id,
        task,
        candidate.run_dir,
        dry_run=dry_run,
    )
    manifest_text = (
        f"task_id={task.task_id}\n"
        f"run_label={task.run_label}\n"
        f"run_dir={candidate.run_dir.resolve()}\n"
        f"git_commit={git_commit}\n"
        f"slurm_job_id={candidate.job_id}\n"
        f"slurm_array_job_id={array_job_id}\n"
    )
    _atomic_write(manifest, manifest_text, dry_run=dry_run)

    viewer_complete, missing = _viewer_outputs_present(candidate.run_dir)
    suffix = "viewer outputs complete" if viewer_complete else "viewer outputs still missing: " + ", ".join(missing)
    action = "would_repair" if dry_run else "repaired"
    return action, f"{candidate.run_dir} ({suffix})"


def _parse_tasks(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid task range {part!r}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    invalid = sorted(task for task in result if task < 0 or task > 69)
    if invalid:
        raise ValueError(f"Task IDs outside 0-69: {invalid}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill final_suite_label.txt and final-suite manifest records for "
            "completed final-suite models whose post-training process exited before record_run()."
        )
    )
    parser.add_argument("--array-job-id", default="4091267")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional task selection, e.g. 2,9-69. Default: all 70 tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting existing label/manifest files. Normally unnecessary.",
    )
    args = parser.parse_args()

    project_root = REPO_ROOT
    selected = _parse_tasks(args.tasks)
    candidates = _discover_candidates(project_root)

    counts: dict[str, int] = {}
    for task in CATALOGUE:
        if selected is not None and task.task_id not in selected:
            continue
        try:
            status, detail = repair_task(
                project_root,
                str(args.array_job_id),
                task,
                candidates.get(task.task_id, []),
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            status, detail = "error", str(exc)
        counts[status] = counts.get(status, 0) + 1
        print(f"task {task.task_id:02d} {task.run_label}: {status} - {detail}")

    print("\nSummary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
