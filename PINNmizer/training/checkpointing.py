from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    scheduler=None,
    latest_history_row: dict | None = None,
    latest_fixed_diagnostic_row: dict | None = None,
    subdir: str | None = None,
) -> Path:
    outdir = run_dir if subdir is None else run_dir / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if latest_history_row is not None:
        checkpoint["latest_training_history_row"] = latest_history_row
    if latest_fixed_diagnostic_row is not None:
        checkpoint["latest_fixed_diagnostic_row"] = latest_fixed_diagnostic_row

    path = outdir / f"model_step_{step}.pt"
    torch.save(checkpoint, path)
    return path
