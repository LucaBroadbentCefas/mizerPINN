"""Evaluate saved PINN checkpoints into read-only viewer snapshots.

Each checkpoint is replayed with the source run's saved command, zero training
steps, and HPC output enabled. The generated CSV/JSON/NPZ diagnostics are moved
under ``checkpoint_views/`` so the normal run browser does not confuse them
with independently trained runs.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from PINNmizer.diagnostics.checkpoint_snapshots import (
    discover_checkpoints,
    parse_emitted_run_dir,
    snapshot_dir,
)
from scripts.recover_hpc_outputs import parse_saved_command, recovery_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "checkpoint_views"
_INPUT_RELATED_PATH_FLAGS = ("--data-csv", "--diag-grid-csv")


def _arg_value(args: list[str], flag: str) -> str | None:
    for idx, token in enumerate(args):
        if token == flag:
            return args[idx + 1] if idx + 1 < len(args) else None
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _replace_value_arg(args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    found = False
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == flag:
            out.extend([flag, value])
            found = True
            idx += 2
            continue
        if token.startswith(f"{flag}="):
            out.extend([flag, value])
            found = True
            idx += 1
            continue
        out.append(token)
        idx += 1
    if not found:
        out.extend([flag, value])
    return out


def override_input_dir_args(original: list[str], input_dir: Path) -> list[str]:
    """Replace the replay input bundle and remap child CSV paths safely.

    Only path arguments that were lexically inside the original ``--input-dir``
    are remapped. Unrelated external paths are left unchanged.
    """
    input_dir = input_dir.expanduser().resolve()
    if not (input_dir / "n_init_full.csv").is_file():
        raise FileNotFoundError(
            f"Replacement input bundle is not a valid Mizer export: {input_dir}. "
            "Expected n_init_full.csv."
        )

    old_input_raw = _arg_value(original, "--input-dir")
    out = _replace_value_arg(original, "--input-dir", str(input_dir))
    if old_input_raw is None:
        return out

    old_input = Path(old_input_raw)
    for flag in _INPUT_RELATED_PATH_FLAGS:
        raw = _arg_value(original, flag)
        if raw is None:
            continue
        try:
            relative = Path(raw).relative_to(old_input)
        except ValueError:
            continue
        mapped = input_dir / relative
        out = _replace_value_arg(out, flag, str(mapped))
    return out


def _truncate_histories(source_run: Path, snapshot: Path, step: int) -> None:
    for source in source_run.glob("*history.csv"):
        try:
            frame = pd.read_csv(source)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if "step" not in frame.columns:
            continue
        numeric_step = pd.to_numeric(frame["step"], errors="coerce")
        frame.loc[numeric_step <= step].to_csv(snapshot / source.name, index=False)


def _last_row(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    return {} if frame.empty else frame.iloc[-1].dropna().to_dict()


def _patch_snapshot_metadata(
    *,
    snapshot: Path,
    source_run: Path,
    checkpoint: Path,
    checkpoint_step: int,
    command: list[str],
    input_dir_override: Path | None,
) -> None:
    generated_config_path = snapshot / "config.json"
    config = {}
    if generated_config_path.is_file():
        try:
            config = json.loads(generated_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    source_config_path = source_run / "config.json"
    source_config = {}
    if source_config_path.is_file():
        try:
            source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_config = {}

    config.update({
        "snapshot_checkpoint_step": checkpoint_step,
        "snapshot_source_run_dir": str(source_run),
        "snapshot_source_n_steps": source_config.get("n_steps"),
        "snapshot_evaluation_n_steps": 0,
        "snapshot_input_dir_override": str(input_dir_override) if input_dir_override else None,
        "n_steps": checkpoint_step,
    })
    generated_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    training = _last_row(snapshot / "loss_history.csv")
    fixed = _last_row(snapshot / "fixed_diagnostic_history.csv")
    summary = {}
    summary_json = snapshot / "final_summary.json"
    if summary_json.is_file():
        try:
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    summary.update({
        "run_id": str(snapshot),
        "run_dir": str(snapshot),
        "status": "checkpoint_snapshot",
        "n_steps_completed": checkpoint_step,
        "snapshot_checkpoint_step": checkpoint_step,
        "snapshot_source_run_dir": str(source_run),
        "snapshot_input_dir_override": str(input_dir_override) if input_dir_override else None,
        "final_checkpoint_path": str(checkpoint),
        "final_model_path": None,
    })
    for key, value in training.items():
        summary[f"final_{key}"] = value
    for key, value in fixed.items():
        summary[f"final_{key}"] = value

    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(snapshot / "final_summary.csv", index=False)

    metadata = {
        "source_run_dir": str(source_run),
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "evaluation_n_steps": 0,
        "input_dir_override": str(input_dir_override) if input_dir_override else None,
        "replay_command": shlex.join(command),
    }
    (snapshot / "checkpoint_snapshot.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    source_command = source_run / "run_command.txt"
    if source_command.is_file():
        shutil.copy2(source_command, snapshot / "source_run_command.txt")

    duplicate_model = snapshot / "model_final.pt"
    if duplicate_model.exists():
        duplicate_model.unlink()


def _materialize_one(
    *,
    source_run: Path,
    checkpoint: Path,
    step: int,
    snapshot_root: Path,
    device: str | None,
    input_dir_override: Path | None,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    destination = snapshot_dir(
        source_run,
        snapshot_root,
        step,
        project_root=PROJECT_ROOT,
    )
    if destination.exists() and not overwrite:
        print(f"Skip existing snapshot: {destination}")
        return destination

    module, original = parse_saved_command(source_run)
    resolved_override = input_dir_override.expanduser().resolve() if input_dir_override else None
    if resolved_override is not None:
        original = override_input_dir_args(original, resolved_override)

    command = [
        sys.executable,
        "-m",
        module,
        *recovery_args(original, checkpoint, device),
    ]
    print(f"Checkpoint {step}: {checkpoint}")
    if resolved_override is not None:
        print(f"Input bundle override: {resolved_override}")
    print(f"Replay: {shlex.join(command)}")
    print(f"Snapshot: {destination}")
    if dry_run:
        return destination

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    generated = parse_emitted_run_dir(completed.stdout, repo_root=PROJECT_ROOT)
    if not generated.is_dir():
        raise RuntimeError(f"Trainer reported missing run directory: {generated}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), str(destination))

    _truncate_histories(source_run, destination, step)
    _patch_snapshot_metadata(
        snapshot=destination,
        source_run=source_run,
        checkpoint=checkpoint.resolve(),
        checkpoint_step=step,
        command=command,
        input_dir_override=resolved_override,
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints into lightweight HPC-viewer snapshots."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed PINN run directories")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Only materialize these checkpoint steps. Default: every saved model_step checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Optional evaluation device override, e.g. cpu")
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--input-dir-override",
        type=Path,
        default=None,
        help=(
            "Exact replacement Mizer input bundle for a legacy run whose saved --input-dir no longer exists. "
            "Child --data-csv/--diag-grid-csv paths are remapped when they lived inside the original input bundle."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = set(args.steps) if args.steps is not None else None
    if args.input_dir_override is not None and len(args.run_dirs) != 1:
        raise ValueError("--input-dir-override can only be used with one run directory at a time.")

    for raw_run in args.run_dirs:
        source_run = raw_run.expanduser().resolve()
        checkpoints = discover_checkpoints(source_run)
        if not checkpoints:
            raise FileNotFoundError(
                f"No model_step_<N>.pt checkpoints found in {source_run} or {source_run / 'checkpoints'}"
            )

        chosen = checkpoints
        if requested is not None:
            missing = sorted(requested.difference(checkpoints))
            if missing:
                raise ValueError(
                    f"Requested checkpoint steps are absent from {source_run}: {missing}. "
                    f"Available: {list(checkpoints)}"
                )
            chosen = {step: checkpoints[step] for step in sorted(requested)}

        for step, checkpoint in chosen.items():
            _materialize_one(
                source_run=source_run,
                checkpoint=checkpoint,
                step=step,
                snapshot_root=args.snapshot_root.expanduser().resolve(),
                device=args.device,
                input_dir_override=args.input_dir_override,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
