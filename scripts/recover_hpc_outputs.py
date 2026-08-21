"""Regenerate HPC-viewer outputs from completed non-HPC PINN runs without training.

The original run is never modified.  For each supplied run directory this
script replays its saved run command with the final/latest checkpoint, forces
zero training steps, and enables HPC output mode.  The trainer therefore
creates a new run directory containing fixed-grid and final prediction outputs
for the viewer while preserving the loaded model weights exactly.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


_VALUE_OVERRIDES = {"--n-steps": "0", "--start-step": "0"}
_VALUE_FLAGS_TO_REMOVE = {"--load-weights", "--n-steps", "--start-step", "--device"}
_BOOLEAN_FLAGS_TO_REMOVE = {"--hpc", "--HPC", "--load-optimizer-state"}


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_step_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(run_dir: Path) -> Path:
    final = run_dir / "model_final.pt"
    if final.is_file():
        return final
    stepped = [p for p in run_dir.glob("model_step_*.pt") if _checkpoint_step(p) >= 0]
    if not stepped:
        raise FileNotFoundError(
            f"No model_final.pt or model_step_<N>.pt checkpoint found in {run_dir}"
        )
    return max(stepped, key=_checkpoint_step)


def parse_saved_command(run_dir: Path) -> tuple[str, list[str]]:
    command_path = run_dir / "run_command.txt"
    if not command_path.is_file():
        raise FileNotFoundError(
            f"{command_path} is required so the original architecture and model settings can be replayed safely."
        )
    text = command_path.read_text(encoding="utf-8").replace("^", " ").replace("\\\n", " ")
    tokens = shlex.split(text, posix=True)
    try:
        module_idx = tokens.index("-m") + 1
        module = tokens[module_idx]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not identify `python -m <module>` in {command_path}") from exc
    if module not in {"scripts.train_pde_only_single_species", "scripts.train_pde_multispecies"}:
        raise ValueError(
            f"Unsupported training module {module!r} in {command_path}; recovery is limited to the PINNmizer single- and multispecies trainers."
        )
    return module, tokens[module_idx + 1 :]


def recovery_args(original: list[str], checkpoint: Path, device: str | None) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(original):
        token = original[i]
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if name in _BOOLEAN_FLAGS_TO_REMOVE:
            i += 1
            continue
        if name in _VALUE_FLAGS_TO_REMOVE:
            if "=" in token:
                i += 1
            else:
                i += 2
            continue
        out.append(token)
        i += 1

    out.extend(["--n-steps", "0", "--start-step", "0", "--load-weights", str(checkpoint.resolve())])
    if device is not None:
        out.extend(["--device", device])
    else:
        # Preserve the original device when one was explicitly supplied.
        for i, token in enumerate(original):
            if token == "--device" and i + 1 < len(original):
                out.extend(["--device", original[i + 1]])
                break
            if token.startswith("--device="):
                out.extend(["--device", token.split("=", 1)[1]])
                break
    out.append("--hpc")
    return out


def build_recovery_command(run_dir: Path, device: str | None = None) -> list[str]:
    run_dir = run_dir.expanduser().resolve()
    module, original = parse_saved_command(run_dir)
    checkpoint = find_latest_checkpoint(run_dir)
    return [sys.executable, "-m", module, *recovery_args(original, checkpoint, device)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create HPC-viewer outputs from completed non-HPC PINNmizer runs without changing model weights."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="One or more completed legacy run directories")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print recovery commands without executing them")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for run_dir in args.run_dirs:
        command = build_recovery_command(run_dir, device=args.device)
        print("Recovery command:")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
