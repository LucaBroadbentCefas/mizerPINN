from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PINNmizer.params import scale_t, scale_x
from PINNmizer.pinn.residual import compute_pde_residual
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.sampling import sample_pde_batch


def save_json(x: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)


def save_history(history: list[dict], run_dir: Path) -> None:
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)


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
    tt = t_grid.detach().cpu()[:, None].expand(-1, params.w.numel())
    ww = params.w.detach().cpu()[None, :].expand(n_times, -1)
    xx = x_grid.detach().cpu()[None, :].expand(n_times, -1)
    xs = x_scaled.detach().cpu()[None, :].expand(n_times, -1)
    ts = t_scaled.detach().cpu()[:, None].expand(-1, params.w.numel())
    pd.DataFrame({"t": tt.reshape(-1).numpy(), "w": ww.reshape(-1).numpy(), "x": xx.reshape(-1).numpy(), "x_scaled": xs.reshape(-1).numpy(), "t_scaled": ts.reshape(-1).numpy(), "log_N": log_N.reshape(-1).numpy(), "N": N.reshape(-1).numpy()}).to_csv(run_dir / "final_predictions_grid.csv", index=False)


def save_final_residual_sample(*, run_dir: Path, model: nn.Module, params, n_pp: torch.Tensor, n_time: int, n_eval: int) -> None:
    model.eval()
    batch = sample_pde_batch(params=params, n_time=n_time, n_eval=n_eval)
    out = compute_pde_residual(model=model, batch=batch, params=params, n_pp=n_pp)
    tt = batch["t_eval"].detach().cpu()[:, None].expand(n_time, n_eval)
    ww = batch["w_eval"].detach().cpu()[None, :].expand(n_time, n_eval)
    def flat(name: str):
        return out[name][:, 0, :].detach().cpu().reshape(-1).numpy()
    pd.DataFrame({"t_eval": tt.reshape(-1).numpy(), "w_eval": ww.reshape(-1).numpy(), "residual_log": flat("residual_log"), "residual": flat("residual"), "log_N_eval": flat("log_N_eval"), "N_eval": flat("N_eval"), "g_eval": flat("g_eval"), "dg_dw": flat("dg_dw"), "mu_eval": flat("mu_eval")}).to_csv(run_dir / "final_residual_sample.csv", index=False)
