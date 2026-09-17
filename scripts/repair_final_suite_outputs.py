"""Repair post-training outputs for the final 70-run suite without retraining.

For each final-suite task whose original run contains ``model_final.pt`` but is
missing final viewer artifacts, this script:

1. replays the saved command with the original checkpoint and zero training steps;
2. lets the normal trainer regenerate final/HPC output artifacts in a temporary run;
3. copies only missing post-processing artifacts into the original run directory;
4. repairs ``final_summary`` so it describes the original 15,000-step run;
5. removes the temporary run; and
6. restores ``final_suite_label.txt`` and the final-suite manifest record.

The original checkpoint, config, command, loss history and training diagnostics
are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.final_suite_viewer.catalogue import CATALOGUE
from scripts.recover_hpc_outputs import build_recovery_command
import scripts.repair_final_suite_records as records

_REPLAY_PROVENANCE_FILES = {
    "config.json",
    "run_command.txt",
    "loss_history.csv",
    "fixed_diagnostic_history.csv",
    "timing_summary.csv",
    "latest_metrics.csv",
    "model_final.pt",
}
_MODEL_FILE_RE = re.compile(r"model_step_\d+\.pt$")


def _run_root_for(run_dir: Path) -> Path:
    if run_dir.parent.name not in {"pde_only_single_species", "pde_multispecies"}:
        raise ValueError(f"Unexpected final-suite run parent: {run_dir}")
    return run_dir.parent


def _snapshot_dirs(root: Path) -> set[Path]:
    return {p.resolve() for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()


def _run_recovery(run_dir: Path, device: str | None) -> Path:
    root = _run_root_for(run_dir)
    before = _snapshot_dirs(root)
    command = build_recovery_command(run_dir, device=device)

    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_ARRAY_JOB_ID", None)
    env.pop("SLURM_ARRAY_TASK_ID", None)

    print("  replay:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    after = _snapshot_dirs(root)
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected exactly one temporary recovery directory under {root}; found {len(created)}: {created}"
        )
    return created[0]


def _copy_missing_tree(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for src in source.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        name = rel.name
        if name in _REPLAY_PROVENANCE_FILES or _MODEL_FILE_RE.fullmatch(name):
            continue
        if name in {"final_summary.json", "final_summary.csv"}:
            continue
        dst = destination / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel))
    return copied


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _coerce_number(value: str):
    if value is None or value == "":
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_step_*.pt"):
        match = re.fullmatch(r"model_step_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _build_repaired_summary(original: Path, replay: Path) -> dict:
    summary = _read_json(original / "final_summary.json")
    if not summary:
        summary = _read_json(replay / "final_summary.json")
    if not summary:
        summary = {}

    config = _read_json(original / "config.json")
    history = _read_last_csv_row(original / "loss_history.csv")
    fixed = _read_last_csv_row(original / "fixed_diagnostic_history.csv")

    n_steps = config.get("n_steps", 15000)
    try:
        n_steps = int(n_steps)
    except (TypeError, ValueError):
        n_steps = 15000
    if history.get("step") not in (None, ""):
        try:
            n_steps = int(float(history["step"]))
        except ValueError:
            pass

    summary.update(
        {
            "run_id": str(original),
            "run_dir": str(original),
            "n_steps_completed": n_steps,
            "status": "completed",
            "error_message": "",
            "final_model_path": str(original / "model_final.pt"),
        }
    )
    checkpoint = _latest_checkpoint(original)
    if checkpoint is not None:
        summary["final_checkpoint_path"] = str(checkpoint)

    for key, value in history.items():
        target = f"final_{key}"
        if target in summary or key in {
            "loss", "loss_unweighted", "loss_pde", "loss_ic", "loss_bc",
            "loss_data", "objective_loss_data", "weighted_loss_data",
            "n_data_obs", "data_log_residual_abs_mean", "data_log_residual_abs_max",
            "loss_pde_ungated", "rmax_raw_grad_norm", "rmax_mean", "rmax_ratio_mean",
        }:
            summary[target] = _coerce_number(value)
    for key, value in fixed.items():
        if key.startswith("fixed_"):
            summary[f"final_{key}"] = _coerce_number(value)

    return summary


def _write_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _needs_output_repair(run_dir: Path) -> bool:
    complete, _ = records._viewer_outputs_present(run_dir)
    return not complete


def _parse_tasks(value: str | None) -> set[int] | None:
    return records._parse_tasks(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate missing final-suite post-training outputs from model_final.pt without retraining."
    )
    parser.add_argument("--array-job-id", default="4091267")
    parser.add_argument("--tasks", default=None, help="Optional selection, e.g. 2,9-69. Default: all tasks.")
    parser.add_argument("--device", default="cpu", help="Device for zero-step replay; default: cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    selected = _parse_tasks(args.tasks)
    candidates_by_task = records._discover_candidates(REPO_ROOT)
    failures = 0

    for task in CATALOGUE:
        if selected is not None and task.task_id not in selected:
            continue

        attempts = records._attempts_for_task(REPO_ROOT, str(args.array_job_id), task)
        candidate = records._select_candidate(task, candidates_by_task.get(task.task_id, []), attempts)
        if candidate is None:
            print(f"task {task.task_id:02d} {task.run_label}: SKIP - no completed model_final.pt")
            continue

        run_dir = candidate.run_dir
        if args.dry_run:
            state = "needs replay" if _needs_output_repair(run_dir) else "outputs already complete"
            print(f"task {task.task_id:02d} {task.run_label}: {state} - {run_dir}")
            continue

        temp_dir: Path | None = None
        try:
            if _needs_output_repair(run_dir):
                print(f"task {task.task_id:02d} {task.run_label}: regenerating final outputs")
                temp_dir = _run_recovery(run_dir, args.device)
                copied = _copy_missing_tree(temp_dir, run_dir)
                summary = _build_repaired_summary(run_dir, temp_dir)
                _write_summary(run_dir, summary)
                print(f"  restored {len(copied)} missing artifact(s)")
            else:
                print(f"task {task.task_id:02d} {task.run_label}: final outputs already present")

            status, detail = records.repair_task(
                REPO_ROOT,
                str(args.array_job_id),
                task,
                candidates_by_task.get(task.task_id, []),
                dry_run=False,
                overwrite=False,
            )
            complete, missing = records._viewer_outputs_present(run_dir)
            if not complete:
                raise RuntimeError("still missing required viewer outputs: " + ", ".join(missing))
            print(f"  bookkeeping: {status} - {detail}")
        except Exception as exc:
            failures += 1
            print(f"task {task.task_id:02d} {task.run_label}: ERROR - {exc}", file=sys.stderr)
        finally:
            if temp_dir is not None and temp_dir.exists() and not args.keep_temp:
                shutil.rmtree(temp_dir)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
