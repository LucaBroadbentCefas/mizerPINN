from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PINNmizer.params import scale_t, scale_x
from PINNmizer.pinn.residual import compute_pde_residual
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.sampling import sample_pde_batch


def _species_name(params, idx: int) -> str:
    species = getattr(params, "species", None)
    if species is not None and idx < len(species):
        return str(species[idx])
    return f"species_{idx}"


def save_final_predictions_multispecies(*, run_dir: Path, model: nn.Module, params, n_times: int = 50) -> None:
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

    log_N = out["log_N"].detach().cpu()
    N = out["N"].detach().cpu()
    log_U = out.get("log_U", torch.zeros_like(out["log_N"])).detach().cpu()
    U = out.get("U", torch.ones_like(out["N"])).detach().cpu()
    log_S = out.get("log_S", torch.zeros_like(out["log_N"])).detach().cpu()
    S = out.get("S", torch.ones_like(out["N"])).detach().cpu()
    n_species = log_N.shape[1]
    n_w = params.w.numel()

    rows = []
    tt = t_grid.detach().cpu()[:, None].expand(-1, n_w)
    ww = params.w.detach().cpu()[None, :].expand(n_times, -1)
    xx = x_grid.detach().cpu()[None, :].expand(n_times, -1)
    xs = x_scaled.detach().cpu()[None, :].expand(n_times, -1)
    ts = t_scaled.detach().cpu()[:, None].expand(-1, n_w)
    for s in range(n_species):
        rows.append(pd.DataFrame({
            "species_idx": s,
            "species": _species_name(params, s),
            "t": tt.reshape(-1).numpy(),
            "w": ww.reshape(-1).numpy(),
            "x": xx.reshape(-1).numpy(),
            "x_scaled": xs.reshape(-1).numpy(),
            "t_scaled": ts.reshape(-1).numpy(),
            "log_N": log_N[:, s, :].reshape(-1).numpy(),
            "N": N[:, s, :].reshape(-1).numpy(),
            "log_U": log_U[:, s, :].reshape(-1).numpy(),
            "U": U[:, s, :].reshape(-1).numpy(),
            "log_S": log_S[:, s, :].reshape(-1).numpy(),
            "S": S[:, s, :].reshape(-1).numpy(),
        }))
    pd.concat(rows, ignore_index=True).to_csv(run_dir / "final_predictions_grid.csv", index=False)


def save_final_residual_sample_multispecies(*, run_dir: Path, model: nn.Module, params, n_pp: torch.Tensor, n_time: int, n_eval: int) -> None:
    model.eval()
    batch = sample_pde_batch(params=params, n_time=n_time, n_eval=n_eval)
    out = compute_pde_residual(model=model, batch=batch, params=params, n_pp=n_pp)
    n_species = out["residual_log"].shape[1]
    tt = batch["t_eval"].detach().cpu()[:, None].expand(n_time, n_eval)
    ww = batch["w_eval"].detach().cpu()[None, :].expand(n_time, n_eval)

    rows = []
    for s in range(n_species):
        def flat(name: str):
            return out[name][:, s, :].detach().cpu().reshape(-1).numpy()
        rows.append(pd.DataFrame({
            "species_idx": s,
            "species": _species_name(params, s),
            "t_eval": tt.reshape(-1).numpy(),
            "w_eval": ww.reshape(-1).numpy(),
            "residual_log": flat("residual_log"),
            "residual": flat("residual"),
            "residual_scaled": flat("residual_scaled"),
            "log_N_eval": flat("log_N_eval"),
            "N_eval": flat("N_eval"),
            "g_eval": flat("g_eval"),
            "dg_dw": flat("dg_dw"),
            "mu_eval": flat("mu_eval"),
        }))
    pd.concat(rows, ignore_index=True).to_csv(run_dir / "final_residual_sample.csv", index=False)
