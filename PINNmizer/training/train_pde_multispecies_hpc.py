from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import PINNmizer.training.train_pde_multispecies as base
from PINNmizer.training.outputs import (
    HPC_FIXED_DIAGNOSTIC_COLUMNS,
    HPC_HISTORY_COLUMNS,
    filter_hpc_fixed_diagnostic_row,
    save_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_MAKE_RUN_DIR = base.make_run_dir
_ORIGINAL_SAVE_HISTORY = base.save_history
_ORIGINAL_APPEND_DIAGNOSTIC_ROW = base.append_diagnostic_row
_ORIGINAL_SAVE_LATEST_METRICS_TABLE = base.save_latest_metrics_table
_ORIGINAL_SAVE_TRAINING_DIAGNOSTIC_PLOTS = base.save_training_diagnostic_plots
_ORIGINAL_SAVE_FINAL_RESIDUAL_SAMPLE = base.save_final_residual_sample_multispecies
_ORIGINAL_SAVE_CHECKPOINT = base.save_checkpoint

_RUN_DIR: Path | None = None
_LATEST_CHECKPOINT_PATH: str | None = None


def make_hpc_run_dir() -> Path:
    """Match the single-species HPC run-directory naming convention."""
    global _RUN_DIR

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_job_id is not None and slurm_array_task_id is not None:
        name = f"{stamp}_job{slurm_job_id}_task{slurm_array_task_id}"
    elif slurm_job_id is not None:
        name = f"{stamp}_job{slurm_job_id}"
    else:
        name = stamp

    _RUN_DIR = PROJECT_ROOT / "runs" / "pde_multispecies" / name
    _RUN_DIR.mkdir(parents=True, exist_ok=False)
    return _RUN_DIR


def parse_hpc_args():
    """Parse multispecies args while accepting --hpc as an active mode flag."""
    original_argv = list(sys.argv)
    try:
        sys.argv = [arg for arg in sys.argv if arg != "--hpc"]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.hpc = True
    args.print_every = 2000
    args.diag_every = 2000
    args.diag_grad_every = 2000
    args.checkpoint_every = 4000
    return args


def save_hpc_history(history: list[dict], run_dir: Path, columns: list[str] | None = None) -> None:
    """Always write the filtered HPC loss-history schema."""
    _ORIGINAL_SAVE_HISTORY(history, run_dir, columns=HPC_HISTORY_COLUMNS)


def append_hpc_diagnostic_row(row: dict, path: str | Path) -> None:
    """Always write the filtered HPC fixed-diagnostic schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = filter_hpc_fixed_diagnostic_row(row)
    pd.DataFrame([filtered], columns=HPC_FIXED_DIAGNOSTIC_COLUMNS).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def save_hpc_checkpoint(*args, **kwargs):
    """Record the latest checkpoint path while preserving the existing save routine."""
    global _LATEST_CHECKPOINT_PATH
    path = _ORIGINAL_SAVE_CHECKPOINT(*args, **kwargs)
    _LATEST_CHECKPOINT_PATH = str(path)
    return path


def _read_last_csv_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _metric(row: dict | None, key: str) -> float:
    if row is None:
        return math.nan
    value = row.get(key, math.nan)
    return float(value) if value is not None else math.nan


def save_hpc_final_summary(*, status: str, error_message: str | None = None) -> None:
    """Create the same final-summary files used by the single-species HPC path."""
    if _RUN_DIR is None:
        return

    history_row = _read_last_csv_row(_RUN_DIR / "loss_history.csv")
    fixed_row = _read_last_csv_row(_RUN_DIR / "fixed_diagnostic_history.csv")
    timing_row = _read_last_csv_row(_RUN_DIR / "timing_summary.csv") or {}
    config = _read_json(_RUN_DIR / "config.json")

    n_steps_completed = int(_metric(history_row, "step")) if history_row is not None else 0
    actual_total_seconds = float(timing_row.get("actual_total_seconds", math.nan))
    seconds_per_step = float(timing_row.get("seconds_per_step", math.nan))
    if not math.isfinite(seconds_per_step) and n_steps_completed > 0 and math.isfinite(actual_total_seconds):
        seconds_per_step = actual_total_seconds / n_steps_completed

    final_model_path = _RUN_DIR / "model_final.pt"
    row = {
        "run_id": str(_RUN_DIR),
        "run_dir": str(_RUN_DIR),
        "seed": config.get("seed", math.nan),
        "fourier_seed": config.get("fourier_seed", math.nan),
        "model_arch": config.get("model_arch", ""),
        "hidden_width": config.get("hidden_width", math.nan),
        "hidden_layers": config.get("hidden_layers", math.nan),
        "fourier_num_features": config.get("fourier_num_features", math.nan),
        "fourier_scale": config.get("fourier_scale", math.nan),
        "fourier_include_raw_input": config.get("fourier_include_raw_input", ""),
        "weight_factorization": config.get("weight_factorization", ""),
        "rwf_mu": config.get("rwf_mu", math.nan),
        "rwf_sigma": config.get("rwf_sigma", math.nan),
        "rwf_apply_to": config.get("rwf_apply_to", ""),
        "rwf_base_init": config.get("rwf_base_init", ""),
        "n_species": config.get("n_species", math.nan),
        "species_mode": config.get("species_mode", ""),
        "lambda_data": config.get("lambda_data", math.nan),
        "data_csv": config.get("data_csv", ""),
        "n_steps_completed": n_steps_completed,
        "status": status,
        "error_message": error_message or "",
        "seconds_per_step": seconds_per_step,
        "actual_total_seconds": actual_total_seconds,
        "final_loss": _metric(history_row, "loss"),
        "final_loss_unweighted": _metric(history_row, "loss_unweighted"),
        "final_loss_pde": _metric(history_row, "loss_pde"),
        "final_loss_ic": _metric(history_row, "loss_ic"),
        "final_loss_bc": _metric(history_row, "loss_bc"),
        "final_loss_data": _metric(history_row, "loss_data"),
        "final_objective_loss_data": _metric(history_row, "objective_loss_data"),
        "final_weighted_loss_data": _metric(history_row, "weighted_loss_data"),
        "final_n_data_obs": _metric(history_row, "n_data_obs"),
        "final_data_log_residual_abs_mean": _metric(history_row, "data_log_residual_abs_mean"),
        "final_data_log_residual_abs_max": _metric(history_row, "data_log_residual_abs_max"),
        "final_loss_pde_ungated": _metric(history_row, "loss_pde_ungated"),
        "final_fixed_loss": _metric(fixed_row, "fixed_loss"),
        "final_fixed_loss_unweighted": _metric(fixed_row, "fixed_loss_unweighted"),
        "final_fixed_loss_pde": _metric(fixed_row, "fixed_loss_pde"),
        "final_fixed_loss_ic": _metric(fixed_row, "fixed_loss_ic"),
        "final_fixed_loss_bc": _metric(fixed_row, "fixed_loss_bc"),
        "final_fixed_residual_log_abs_p95": _metric(fixed_row, "fixed_residual_log_abs_p95"),
        "final_checkpoint_path": _LATEST_CHECKPOINT_PATH,
        "final_model_path": str(final_model_path) if final_model_path.exists() else None,
    }
    pd.DataFrame([row]).to_csv(_RUN_DIR / "final_summary.csv", index=False)
    save_json(row, _RUN_DIR / "final_summary.json")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def install_hpc_mode() -> None:
    """Patch only the multispecies module-level hooks needed for HPC output mode."""
    base.parse_args = parse_hpc_args
    base.make_run_dir = make_hpc_run_dir
    base.save_history = save_hpc_history
    base.append_diagnostic_row = append_hpc_diagnostic_row
    base.save_latest_metrics_table = lambda *args, **kwargs: None
    base.save_training_diagnostic_plots = lambda *args, **kwargs: None
    base.save_final_residual_sample_multispecies = lambda *args, **kwargs: None
    base.save_checkpoint = save_hpc_checkpoint


def main() -> None:
    install_hpc_mode()
    try:
        base.main()
    except Exception as exc:
        save_hpc_final_summary(status="failed", error_message=str(exc))
        raise
    else:
        save_hpc_final_summary(status="completed")


if __name__ == "__main__":
    main()
