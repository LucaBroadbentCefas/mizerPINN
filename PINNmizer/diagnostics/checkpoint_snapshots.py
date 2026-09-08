from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_CHECKPOINT_RE = re.compile(r"model_step_(\d+)\.pt$")
_SNAPSHOT_RE = re.compile(r"step_(\d+)$")


def checkpoint_step(path: str | Path) -> int | None:
    match = _CHECKPOINT_RE.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


def discover_checkpoints(run_dir: str | Path) -> dict[int, Path]:
    """Return saved training checkpoints indexed by step.

    Current single-species HPC runs save under ``checkpoints/`` while
    multispecies and older runs may save directly in the run directory.
    """
    run_dir = Path(run_dir).expanduser()
    found: dict[int, Path] = {}
    for parent in (run_dir, run_dir / "checkpoints"):
        if not parent.is_dir():
            continue
        for path in parent.glob("model_step_*.pt"):
            step = checkpoint_step(path)
            if step is not None:
                found[step] = path
    return dict(sorted(found.items()))


def snapshot_run_root(
    source_run_dir: str | Path,
    snapshot_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    source = Path(source_run_dir).expanduser().resolve()
    root = Path(snapshot_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve() if project_root is not None else None

    if project is not None:
        try:
            relative = source.relative_to(project)
        except ValueError:
            relative = None
        if relative is not None:
            return root / relative

    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name) or "run"
    return root / "external" / f"{safe_name}_{digest}"


def snapshot_dir(
    source_run_dir: str | Path,
    snapshot_root: str | Path,
    step: int,
    *,
    project_root: str | Path | None = None,
) -> Path:
    return snapshot_run_root(
        source_run_dir,
        snapshot_root,
        project_root=project_root,
    ) / f"step_{int(step):08d}"


def load_snapshot_metadata(path: str | Path) -> dict:
    metadata_path = Path(path) / "checkpoint_snapshot.json"
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def discover_snapshots(
    source_run_dir: str | Path,
    snapshot_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[int, Path]:
    parent = snapshot_run_root(
        source_run_dir,
        snapshot_root,
        project_root=project_root,
    )
    if not parent.is_dir():
        return {}

    source = str(Path(source_run_dir).expanduser().resolve())
    found: dict[int, Path] = {}
    for path in parent.glob("step_*"):
        if not path.is_dir():
            continue
        match = _SNAPSHOT_RE.fullmatch(path.name)
        if not match:
            continue
        metadata = load_snapshot_metadata(path)
        if metadata.get("source_run_dir") != source:
            continue
        step = int(metadata.get("checkpoint_step", match.group(1)))
        found[step] = path
    return dict(sorted(found.items()))


def parse_emitted_run_dir(stdout: str, *, repo_root: str | Path) -> Path:
    matches = re.findall(r"^Run directory:\s*(.+?)\s*$", stdout, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError("Trainer completed without printing `Run directory: ...`.")
    path = Path(matches[-1]).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()
