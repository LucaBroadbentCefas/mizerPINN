"""Repair final-suite outputs stored in the local HPC_clone.

Local layout expected by this utility::

    mizerPINN2/                 <- code root (PINNmizer, apps, scripts)
    mizerPINN2/HPC_clone/       <- final-suite data root
        runs/
        final_runs/
        validation/
        slurm_logs/
        scripts/repair_final_suite_outputs.py

The normal ``mizerPINN2/runs`` tree is never searched for final-suite runs.
Training code is imported/executed from the parent ``mizerPINN2`` repository,
while all original inputs and recovered outputs belong to ``HPC_clone``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = DATA_ROOT.parent

if not (CODE_ROOT / "PINNmizer").is_dir():
    raise RuntimeError(
        "Expected the main code repository one level above HPC_clone. "
        f"Could not find {CODE_ROOT / 'PINNmizer'}"
    )

# Import only catalogue metadata from the main repository. Never use its runs.
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from apps.final_suite_viewer.catalogue import CATALOGUE

RUN_ROOT_NAMES = ("pde_only_single_species", "pde_multispecies")
RUN_RE = re.compile(r"_job(?P<job_id>\d+)_task(?P<task_id>\d+)$")
RUN_TIME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})(?:_\d+)?_job")
MODEL_STEP_RE = re.compile(r"model_step_(\d+)\.pt$")
FINAL_SUITE_START = datetime(2026, 9, 8)
FINAL_N_STEPS = 15000
PATH_FLAGS = {"--input-dir", "--data-csv", "--diag-grid-csv"}

# Files from a zero-step replay that must never replace the original training
# provenance/history/checkpoints.
NEVER_COPY = {
    "config.json",
    "run_command.txt",
    "loss_history.csv",
    "fixed_diagnostic_history.csv",
    "timing_summary.csv",
    "latest_metrics.csv",
    "model_final.pt",
    "final_summary.json",
    "final_summary.csv",
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _folder_time(path: Path) -> datetime | None:
    match = RUN_TIME_RE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_log_start(array_job_id: str, task_id: int, expected_label: str) -> tuple[datetime | None, str | None]:
    path = DATA_ROOT / "slurm_logs" / f"pinn_final_suite_{array_job_id}_{task_id}.out"
    if not path.is_file():
        return None, None

    current: dict[str, str] = {}
    attempts: list[dict[str, str]] = []

    def finish() -> None:
        if current.get("task") == str(task_id):
            attempts.append(dict(current))

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("task="):
            finish()
            current = {"task": line.split("=", 1)[1].strip()}
            continue
        if current and "=" in line:
            key, value = line.split("=", 1)
            if key in {"run_label", "start_time", "git_commit"}:
                current[key] = value.strip()
    finish()

    matching = [x for x in attempts if x.get("run_label") == expected_label] or attempts
    if not matching:
        return None, None
    attempt = matching[-1]
    start = None
    if attempt.get("start_time"):
        try:
            start = datetime.fromisoformat(attempt["start_time"]).replace(tzinfo=None)
        except ValueError:
            pass
    commit = attempt.get("git_commit")
    return start, commit


def _candidate_valid(run_dir: Path, task) -> bool:
    final = run_dir / "model_final.pt"
    if not final.is_file() or final.stat().st_size == 0:
        return False

    config = _read_json(run_dir / "config.json")
    if config.get("n_steps") is not None:
        try:
            if int(config["n_steps"]) != FINAL_N_STEPS:
                return False
        except (TypeError, ValueError):
            return False

    if task.state and config.get("state_parameterization") not in (None, task.state):
        return False
    if task.architecture and config.get("model_arch") not in (None, task.architecture):
        return False
    if task.species:
        input_dir = str(config.get("input_dir", "")).replace("\\", "/")
        if input_dir and task.species not in input_dir:
            return False
    return True


def _find_original_run(task, array_job_id: str) -> tuple[Path, str] | None:
    candidates: list[tuple[Path, str]] = []
    for root_name in RUN_ROOT_NAMES:
        root = DATA_ROOT / "runs" / root_name
        if not root.is_dir():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            match = RUN_RE.search(run_dir.name)
            if not match or int(match.group("task_id")) != task.task_id:
                continue
            started = _folder_time(run_dir)
            if started is None or started < FINAL_SUITE_START:
                continue
            if _candidate_valid(run_dir, task):
                candidates.append((run_dir, match.group("job_id")))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    log_start, _ = _parse_log_start(array_job_id, task.task_id, task.run_label)
    if log_start is not None:
        timed = [(abs((_folder_time(p) - log_start).total_seconds()), p, j) for p, j in candidates if _folder_time(p)]
        if timed:
            _, path, job = min(timed, key=lambda x: x[0])
            return path, job

    # All candidates are already restricted to the final-suite date and task id.
    return max(candidates, key=lambda x: x[0].stat().st_mtime)


def _parse_saved_command(run_dir: Path) -> tuple[str, list[str]]:
    path = run_dir / "run_command.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding="utf-8").replace("^", " ").replace("\\\n", " ")
    tokens = shlex.split(text, posix=True)
    try:
        idx = tokens.index("-m")
        module = tokens[idx + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse python -m command in {path}") from exc
    if module not in {"scripts.train_pde_only_single_species", "scripts.train_pde_multispecies"}:
        raise ValueError(f"Unsupported training module {module!r}")
    return module, tokens[idx + 2 :]


def _strip_replay_overrides(args: list[str]) -> list[str]:
    value_flags = {"--load-weights", "--n-steps", "--start-step", "--device"}
    bool_flags = {"--hpc", "--HPC", "--load-optimizer-state"}
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if name in bool_flags:
            i += 1
            continue
        if name in value_flags:
            i += 1 if "=" in token else 2
            continue
        out.append(token)
        i += 1
    return out


def _absolute_data_paths(args: list[str]) -> list[str]:
    out = list(args)
    i = 0
    while i < len(out):
        token = out[i]
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in PATH_FLAGS and value and not Path(value).is_absolute():
                out[i] = f"{flag}={DATA_ROOT / Path(value)}"
            i += 1
            continue
        if token in PATH_FLAGS and i + 1 < len(out):
            value = out[i + 1]
            if value and not Path(value).is_absolute():
                out[i + 1] = str(DATA_ROOT / Path(value))
            i += 2
            continue
        i += 1
    return out


def _build_replay_command(run_dir: Path, device: str) -> list[str]:
    module, original = _parse_saved_command(run_dir)
    args = _strip_replay_overrides(original)
    args = _absolute_data_paths(args)
    args.extend(
        [
            "--n-steps", "0",
            "--start-step", "0",
            "--load-weights", str((run_dir / "model_final.pt").resolve()),
            "--device", device,
            "--hpc",
        ]
    )
    return [sys.executable, "-m", module, *args]


def _temp_run_root(original: Path) -> Path:
    return CODE_ROOT / "runs" / original.parent.name


def _snapshot(root: Path) -> set[Path]:
    return {p.resolve() for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()


def _run_zero_step_replay(original: Path, device: str) -> Path:
    temp_root = _temp_run_root(original)
    temp_root.mkdir(parents=True, exist_ok=True)
    before = _snapshot(temp_root)
    command = _build_replay_command(original, device)

    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_ARRAY_JOB_ID", None)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(CODE_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    print("  replay:", shlex.join(command))
    subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)

    created = sorted(_snapshot(temp_root) - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if len(created) != 1:
        raise RuntimeError(f"Expected one temporary replay directory in {temp_root}; found {created}")
    return created[0]


def _copy_missing_outputs(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for src in source.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        if rel.name in NEVER_COPY or MODEL_STEP_RE.fullmatch(rel.name):
            continue
        dst = destination / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel))
    return copied


def _last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _number(value):
    if value in (None, ""):
        return value
    try:
        x = float(value)
        return int(x) if x.is_integer() else x
    except (TypeError, ValueError):
        return value


def _latest_checkpoint(run_dir: Path) -> Path | None:
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_step_*.pt"):
        match = MODEL_STEP_RE.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return max(found, default=(0, None), key=lambda x: x[0])[1]


def _write_repaired_summary(original: Path, replay: Path) -> None:
    summary = _read_json(original / "final_summary.json") or _read_json(replay / "final_summary.json")
    config = _read_json(original / "config.json")
    history = _last_csv_row(original / "loss_history.csv")
    fixed = _last_csv_row(original / "fixed_diagnostic_history.csv")

    completed = FINAL_N_STEPS
    if history.get("step") not in (None, ""):
        try:
            completed = int(float(history["step"]))
        except ValueError:
            pass
    elif config.get("n_steps") is not None:
        try:
            completed = int(config["n_steps"])
        except (TypeError, ValueError):
            pass

    summary.update(
        {
            "run_id": str(original),
            "run_dir": str(original),
            "n_steps_completed": completed,
            "status": "completed",
            "error_message": "",
            "final_model_path": str(original / "model_final.pt"),
        }
    )
    checkpoint = _latest_checkpoint(original)
    if checkpoint is not None:
        summary["final_checkpoint_path"] = str(checkpoint)

    important_history = {
        "loss", "loss_unweighted", "loss_pde", "loss_ic", "loss_bc",
        "loss_data", "objective_loss_data", "weighted_loss_data", "n_data_obs",
        "data_log_residual_abs_mean", "data_log_residual_abs_max",
        "loss_pde_ungated", "rmax_raw_grad_norm", "rmax_mean", "rmax_ratio_mean",
    }
    for key, value in history.items():
        if key in important_history or f"final_{key}" in summary:
            summary[f"final_{key}"] = _number(value)
    for key, value in fixed.items():
        if key.startswith("fixed_"):
            summary[f"final_{key}"] = _number(value)

    (original / "final_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    with (original / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _required_outputs(run_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    pred = run_dir / "final_predictions_grid.csv"
    if not pred.is_file() or pred.stat().st_size == 0:
        missing.append("final_predictions_grid.csv")
    if not ((run_dir / "final_summary.json").is_file() or (run_dir / "final_summary.csv").is_file()):
        missing.append("final_summary.json/csv")
    return not missing, missing


def _restore_rmax_truth(task_id: int, array_job_id: str, run_dir: Path) -> None:
    if not 53 <= task_id <= 61 or (run_dir / "r_max_true.csv").is_file():
        return
    root = DATA_ROOT / "final_runs" / "final_suite_inputs" / f"job_{array_job_id}" / f"task_{task_id}"
    matches = list(root.rglob("r_max_true.csv")) if root.is_dir() else []
    if len(matches) == 1:
        shutil.copy2(matches[0], run_dir / "r_max_true.csv")
    elif len(matches) != 0:
        raise RuntimeError(f"Task {task_id}: multiple r_max_true.csv files under {root}")


def _write_bookkeeping(task, array_job_id: str, run_dir: Path, job_id: str) -> None:
    (run_dir / "final_suite_label.txt").write_text(task.run_label + "\n", encoding="utf-8")
    _, git_commit = _parse_log_start(array_job_id, task.task_id, task.run_label)
    git_commit = git_commit or "unknown"

    manifest = DATA_ROOT / "final_runs" / "final_suite_manifest" / f"job_{array_job_id}" / f"task_{task.task_id:02d}_{task.run_label}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"task_id={task.task_id}\n"
        f"run_label={task.run_label}\n"
        f"run_dir={run_dir.resolve()}\n"
        f"git_commit={git_commit}\n"
        f"slurm_job_id={job_id}\n"
        f"slurm_array_job_id={array_job_id}\n"
    )
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(manifest)


def _parse_tasks(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for bit in value.split(","):
        bit = bit.strip()
        if "-" in bit:
            a, b = map(int, bit.split("-", 1))
            result.update(range(a, b + 1))
        elif bit:
            result.add(int(bit))
    bad = sorted(x for x in result if x < 0 or x > 69)
    if bad:
        raise ValueError(f"Task IDs outside 0-69: {bad}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair final-suite post-training outputs in HPC_clone without retraining.")
    parser.add_argument("--array-job-id", default="4091267")
    parser.add_argument("--tasks", default=None, help="Optional selection, e.g. 2,9-69")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    selected = _parse_tasks(args.tasks)
    failures = 0

    print(f"code root: {CODE_ROOT}")
    print(f"data root: {DATA_ROOT}")

    for task in CATALOGUE:
        if selected is not None and task.task_id not in selected:
            continue
        found = _find_original_run(task, str(args.array_job_id))
        if found is None:
            print(f"task {task.task_id:02d} {task.run_label}: SKIP - no final-suite model_final.pt in HPC_clone")
            continue
        run_dir, job_id = found
        complete, missing = _required_outputs(run_dir)

        if args.dry_run:
            state = "complete" if complete else "needs " + ", ".join(missing)
            print(f"task {task.task_id:02d} {task.run_label}: {state} - {run_dir}")
            continue

        temp: Path | None = None
        try:
            if not complete:
                print(f"task {task.task_id:02d} {task.run_label}: regenerating {', '.join(missing)}")
                temp = _run_zero_step_replay(run_dir, args.device)
                copied = _copy_missing_outputs(temp, run_dir)
                _write_repaired_summary(run_dir, temp)
                print(f"  restored {len(copied)} post-processing artifact(s)")
            else:
                print(f"task {task.task_id:02d} {task.run_label}: final outputs already present")

            _restore_rmax_truth(task.task_id, str(args.array_job_id), run_dir)
            _write_bookkeeping(task, str(args.array_job_id), run_dir, job_id)

            complete, missing = _required_outputs(run_dir)
            if not complete:
                raise RuntimeError("still missing: " + ", ".join(missing))
            print("  repaired in HPC_clone")
        except Exception as exc:
            failures += 1
            print(f"task {task.task_id:02d} {task.run_label}: ERROR - {exc}", file=sys.stderr)
        finally:
            if temp is not None and temp.exists() and not args.keep_temp:
                shutil.rmtree(temp)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
