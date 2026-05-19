from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(*, run_dir: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, config: dict) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        run_dir / f"model_step_{step}.pt",
    )
