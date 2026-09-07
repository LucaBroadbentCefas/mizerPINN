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
    inverse_rmax=None,
    inverse_data_cv=None,
    inverse_effort=None,
    effort_time=None,
) -> Path:
    outdir = run_dir if subdir is None else run_dir / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if inverse_rmax is not None:
        checkpoint.update({
            "inverse_parameter_state_dict": inverse_rmax.state_dict(),
            "inverse_parameter_config": inverse_rmax.config(),
            "initial_r_max": inverse_rmax.initial_r_max.detach().cpu(),
            "initial_log_r_max": inverse_rmax.initial_log_r_max.detach().cpu(),
            "current_r_max": inverse_rmax.current_r_max().detach().cpu(),
            "current_log_r_max": inverse_rmax.current_log_r_max().detach().cpu(),
        })
    if inverse_data_cv is not None:
        checkpoint.update({
            "data_cv_state_dict": inverse_data_cv.state_dict(),
            "data_cv_config": inverse_data_cv.config(),
            "initial_data_cv": inverse_data_cv.initial_cv.detach().cpu(),
            "current_data_cv": inverse_data_cv.current_cv().detach().cpu(),
            "current_data_sd_log": inverse_data_cv.current_sd_log().detach().cpu(),
        })
    if inverse_effort is not None:
        checkpoint.update({
            "inverse_effort_state_dict": inverse_effort.state_dict(),
            "inverse_effort_config": inverse_effort.config(),
            "fishing_effort_time": effort_time.detach().cpu(),
            "initial_inverse_effort": inverse_effort.initial_effort.detach().cpu(),
            "current_estimated_effort": inverse_effort.current_effort().detach().cpu(),
        })
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if latest_history_row is not None:
        checkpoint["latest_training_history_row"] = latest_history_row
    if latest_fixed_diagnostic_row is not None:
        checkpoint["latest_fixed_diagnostic_row"] = latest_fixed_diagnostic_row

    path = outdir / f"model_step_{step}.pt"
    torch.save(checkpoint, path)
    return path
