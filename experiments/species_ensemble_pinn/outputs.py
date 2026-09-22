from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.params import active_grid_mask

from .model_eval import evaluate_log_model_on_points
from .pde_state import compute_pde_state
from .residual import compute_pde_residual_from_state
from .residual_scale import grid_residual_scale


def save_json(data: dict, path: str | Path) -> None:
    def convert(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist() if value.numel() != 1 else value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")


def save_run_command(path: str | Path) -> None:
    command = "python -m experiments.species_ensemble_pinn.train_species " + " ".join(sys.argv[1:]) + "\n"
    Path(path).write_text(command, encoding="utf-8")


def save_history(rows: list[dict], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def scalar(value) -> float:
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def fixed_diagnostics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    rref = losses["residual_reference_scaled"].detach().reshape(-1)
    rlog = losses["residual_log"].detach().reshape(-1)
    scale = losses["reference_scale_eval"].detach().reshape(-1)
    ratio = losses["N_over_reference_scale"].detach().reshape(-1)
    rms = lambda x: float(torch.sqrt(torch.mean(x.square())).cpu())
    p95 = lambda x: float(torch.quantile(torch.abs(x), 0.95).cpu())
    return {
        "fixed_loss_pde_reference_scaled": scalar(losses["loss_pde"]),
        "fixed_loss_ic": scalar(losses["loss_ic"]),
        "fixed_loss_bc": scalar(losses["loss_bc"]),
        "fixed_residual_reference_scaled_rms": rms(rref),
        "fixed_residual_reference_scaled_abs_mean": scalar(rref.abs().mean()),
        "fixed_residual_reference_scaled_abs_p95": p95(rref),
        "fixed_residual_reference_scaled_abs_max": scalar(rref.abs().max()),
        "fixed_residual_log_rms": rms(rlog),
        "fixed_residual_log_abs_mean": scalar(rlog.abs().mean()),
        "fixed_residual_log_abs_p95": p95(rlog),
        "fixed_residual_log_abs_max": scalar(rlog.abs().max()),
        "fixed_reference_scale_min": scalar(scale.min()),
        "fixed_reference_scale_max": scalar(scale.max()),
        "fixed_N_over_reference_scale_min": scalar(ratio.min()),
        "fixed_N_over_reference_scale_mean": scalar(ratio.mean()),
        "fixed_N_over_reference_scale_abs_p95": p95(ratio),
        "fixed_N_over_reference_scale_max": scalar(ratio.max()),
        "rms_dlogN_dt": rms(losses["dlogN_dt"]),
        "rms_advective": rms(losses["g_eval"] * losses["dlogN_dw"]),
        "rms_mu": rms(losses["mu_eval"]),
        "rms_dg_dw": rms(losses["dg_dw"]),
    }


def save_final_outputs(*, run_dir: Path, model, params, n_init, n_pp, known_state,
                       species_idx: int, species_name: str, batch: dict) -> None:
    state = compute_pde_state(model, batch, params, n_init, n_pp, known_state, species_idx=species_idx)
    residual = compute_pde_residual_from_state(state, params, species_idx=species_idx)
    t, w = batch["t_eval"], batch["w_eval"]
    tt = t[:, None].expand(-1, w.numel()).detach().cpu().reshape(-1).numpy()
    ww = w[None, :].expand(t.numel(), -1).detach().cpu().reshape(-1).numpy()
    flat = lambda value: value[:, 0, :].detach().cpu().reshape(-1).numpy()
    known_grid = known_state.at(t)[:, species_idx:species_idx + 1, :]
    grid_out = evaluate_log_model_on_points(model, batch["x_grid_scaled"], batch["t_scaled"], params)
    log_scale, scale = grid_residual_scale(params)
    log_scale_t = log_scale[species_idx][None, :].expand(t.numel(), -1)
    scale_t = scale[species_idx][None, :].expand(t.numel(), -1)
    n_pred = grid_out["N"][:, 0, :]
    floor = torch.finfo(n_pred.dtype).tiny
    log_error = grid_out["log_N"][:, 0, :] - torch.log(torch.clamp(known_grid[:, 0, :], min=floor))
    relative_error = (n_pred - known_grid[:, 0, :]) / torch.clamp(known_grid[:, 0, :].abs(), min=floor)
    grid_tt = t[:, None].expand(-1, params.w.numel()).detach().cpu().reshape(-1).numpy()
    grid_ww = params.w[None, :].expand(t.numel(), -1).detach().cpu().reshape(-1).numpy()
    active_grid = active_grid_mask(params)[species_idx].detach().cpu().numpy()[None, :].repeat(t.numel(), 0).reshape(-1)
    pd.DataFrame({
        "species_idx": species_idx, "species": species_name, "time": grid_tt, "weight": grid_ww,
        "log_N_pred": grid_out["log_N"][:, 0, :].detach().cpu().reshape(-1).numpy(),
        "N_pred": n_pred.detach().cpu().reshape(-1).numpy(),
        "N_known": known_grid[:, 0, :].detach().cpu().reshape(-1).numpy(),
        "log_S_reference": log_scale_t.detach().cpu().reshape(-1).numpy(),
        "S_reference": scale_t.detach().cpu().reshape(-1).numpy(),
        "N_over_S_reference": (n_pred / scale_t).detach().cpu().reshape(-1).numpy(),
        "log_error": log_error.detach().cpu().reshape(-1).numpy(),
        "relative_error": relative_error.detach().cpu().reshape(-1).numpy(),
        "active": active_grid,
    }).to_csv(run_dir / "predictions_final.csv", index=False)
    pd.DataFrame({
        "species_idx": species_idx, "species": species_name, "time": tt, "weight": ww,
        "dlogN_dt": flat(residual["dlogN_dt"]), "g": flat(residual["g_eval"]),
        "dlogN_dw": flat(residual["dlogN_dw"]),
        "advective": flat(residual["g_eval"] * residual["dlogN_dw"]),
        "mu_background": flat(residual["mu_b_eval"]),
        "mu_predation": flat(residual["pred_mort_eval"]),
        "mu_fishing": flat(residual["f_mort_eval"]), "mu_total": flat(residual["mu_eval"]),
        "dg_dw": flat(residual["dg_dw"]), "residual_log": flat(residual["residual_log"]),
        "residual_physical": flat(residual["residual_physical"]),
        "residual_physical_check": flat(residual["residual_physical_check"]),
        "residual_reference_scaled": flat(residual["residual_reference_scaled"]),
        "log_S_reference": flat(residual["log_reference_scale_eval"]),
        "S_reference": flat(residual["reference_scale_eval"]),
        "N_over_S_reference": flat(residual["N_over_reference_scale"]),
        "active": (w <= params.w_max[species_idx]).detach().cpu().numpy()[None, :].repeat(t.numel(), 0).reshape(-1),
    }).to_csv(run_dir / "residuals_final.csv", index=False)
    pd.DataFrame({
        "time": tt, "weight": ww,
        "growth": flat(state["growth_eval"]["e_growth_eval"]),
        "dg_dw": flat(state["growth_eval"]["dg_dw"]),
        "mu_total": flat(state["mortality"]["mu_eval"]),
    }).to_csv(run_dir / "biology_sample_final.csv", index=False)
