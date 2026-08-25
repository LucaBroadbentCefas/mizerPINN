from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import sys

from PINNmizer.params import scale_t, scale_x
from PINNmizer.pinn.residual import compute_pde_residual
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.sampling import sample_pde_batch


def save_json(x: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)

def save_run_command(args_path: Path, module: str = "scripts.train_pde_only_single_species") -> None:
    tokens = sys.argv[1:]
    parts = []
    i = 0

    while i < len(tokens):
        if tokens[i].startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            parts.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            parts.append(tokens[i])
            i += 1

    if not parts:
        text = f"python -m {module}\n"
    else:
        lines = [f"python -m {module} ^"]
        lines.extend(
            f"  {part}{' ^' if j < len(parts) - 1 else ''}"
            for j, part in enumerate(parts)
        )
        text = "\n".join(lines) + "\n"

    args_path.write_text(text, encoding="utf-8")

HPC_HISTORY_COLUMNS = [
    "step",
    "seconds_elapsed",
    "lr",
    "rmax_lr",
    "data_cv_lr",
    "data_cv_grad_norm",
    "loss",
    "loss_unweighted",
    "loss_pde",
    "loss_ic",
    "loss_bc",
    "loss_timestep",
    "loss_data",
    "loss_data_effective",
    "data_discrepancy_q",
    "data_discrepancy_q95",
    "data_loss_active",
    "loss_pde_ungated",
    "loss_pde_gated",
    "objective_loss_pde",
    "objective_loss_ic",
    "objective_loss_bc",
    "objective_loss_timestep",
    "objective_loss_data",
    "weighted_loss_data",
    "w_pde",
    "w_ic",
    "w_bc",
    "w_timestep",
    "w_data",
    "wang_scaled_loss_data",
    "n_data_obs",
    "data_pred_min",
    "data_pred_max",
    "data_obs_min",
    "data_obs_max",
    "data_log_residual_abs_mean",
    "data_log_residual_abs_max",
    "grad_norm",
    "rmax_raw_grad_norm",
    "rmax_min",
    "rmax_mean",
    "rmax_max",
    "log_rmax_min",
    "log_rmax_mean",
    "log_rmax_max",
    "rmax_ratio_min",
    "rmax_ratio_mean",
    "rmax_ratio_max",
    "data_cv_min",
    "data_cv_mean",
    "data_cv_max",
    "data_sd_log_min",
    "data_sd_log_mean",
    "data_sd_log_max",
    "causal_fraction",
    "t_max_current",
    "pde_causal_weight_first",
    "pde_causal_weight_mean",
    "pde_causal_weight_last",
    "pde_causal_chunk_loss_mean",
    "pde_causal_chunk_loss_max",
]

HPC_FIXED_DIAGNOSTIC_COLUMNS = [
    "step",
    "fixed_loss",
    "fixed_loss_unweighted",
    "fixed_loss_pde",
    "fixed_loss_ic",
    "fixed_loss_bc",
    "fixed_residual_log_rms",
    "fixed_residual_log_abs_mean",
    "fixed_residual_log_abs_p95",
    "fixed_residual_log_abs_max",
    "rms_dlogN_dt",
    "rms_advective",
    "rms_mu",
    "rms_dg_dw",
]


def filter_row(row: dict, columns: list[str]) -> dict:
    return {key: row.get(key, float("nan")) for key in columns}


def filter_hpc_history_row(row: dict) -> dict:
    return filter_row(row, HPC_HISTORY_COLUMNS)


def filter_hpc_fixed_diagnostic_row(row: dict) -> dict:
    return filter_row(row, HPC_FIXED_DIAGNOSTIC_COLUMNS)


def save_history(history: list[dict], run_dir: Path, columns: list[str] | None = None) -> None:
    if columns is None:
        pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    else:
        pd.DataFrame([filter_row(row, columns) for row in history], columns=columns).to_csv(
            run_dir / "loss_history.csv",
            index=False,
        )


def save_final_predictions(*, run_dir: Path, model: nn.Module, params, n_times: int = 50) -> None:
    model.eval()
    dtype = params.w.dtype
    device = params.w.device
    t_min = torch.as_tensor(params.t_min, dtype=dtype, device=device)
    t_max = torch.as_tensor(params.t_max, dtype=dtype, device=device)
    t_grid = torch.linspace(t_min, t_max, n_times, dtype=dtype, device=device)
    x_grid = torch.log(params.w)
    t_scaled = scale_t(t_grid, params)
    x_scaled = scale_x(x_grid, params)
    with torch.no_grad():
        out = evaluate_log_model_on_points(model=model, x_scaled=x_scaled, t_scaled=t_scaled, params=params)
    log_N = out["log_N"][:, 0, :].detach().cpu()
    N = out["N"][:, 0, :].detach().cpu()
    log_U = out.get("log_U", torch.zeros_like(out["log_N"]))[:, 0, :].detach().cpu()
    U = out.get("U", torch.ones_like(out["N"]))[:, 0, :].detach().cpu()
    log_S = out.get("log_S", torch.zeros_like(out["log_N"]))[:, 0, :].detach().cpu()
    S = out.get("S", torch.ones_like(out["N"]))[:, 0, :].detach().cpu()
    tt = t_grid.detach().cpu()[:, None].expand(-1, params.w.numel())
    ww = params.w.detach().cpu()[None, :].expand(n_times, -1)
    xx = x_grid.detach().cpu()[None, :].expand(n_times, -1)
    xs = x_scaled.detach().cpu()[None, :].expand(n_times, -1)
    ts = t_scaled.detach().cpu()[:, None].expand(-1, params.w.numel())
    pd.DataFrame({"t": tt.reshape(-1).numpy(), "w": ww.reshape(-1).numpy(), "x": xx.reshape(-1).numpy(), "x_scaled": xs.reshape(-1).numpy(), "t_scaled": ts.reshape(-1).numpy(), "log_N": log_N.reshape(-1).numpy(), "N": N.reshape(-1).numpy(), "log_U": log_U.reshape(-1).numpy(), "U": U.reshape(-1).numpy(), "log_S": log_S.reshape(-1).numpy(), "S": S.reshape(-1).numpy()}).to_csv(run_dir / "final_predictions_grid.csv", index=False)


def save_final_residual_sample(*, run_dir: Path, model: nn.Module, params, n_pp: torch.Tensor, n_time: int, n_eval: int) -> None:
    model.eval()
    batch = sample_pde_batch(params=params, n_time=n_time, n_eval=n_eval)
    out = compute_pde_residual(model=model, batch=batch, params=params, n_pp=n_pp)
    tt = batch["t_eval"].detach().cpu()[:, None].expand(n_time, n_eval)
    ww = batch["w_eval"].detach().cpu()[None, :].expand(n_time, n_eval)
    def flat(name: str):
        return out[name][:, 0, :].detach().cpu().reshape(-1).numpy()
    pd.DataFrame({"t_eval": tt.reshape(-1).numpy(), "w_eval": ww.reshape(-1).numpy(), "residual_log": flat("residual_log"), "residual": flat("residual"), "residual_scaled": flat("residual_scaled"), "log_N_eval": flat("log_N_eval"), "N_eval": flat("N_eval"), "g_eval": flat("g_eval"), "dg_dw": flat("dg_dw"), "mu_eval": flat("mu_eval")}).to_csv(run_dir / "final_residual_sample.csv", index=False)
