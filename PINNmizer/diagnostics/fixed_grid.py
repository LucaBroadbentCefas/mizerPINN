from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PINNmizer.diagnostics.metrics import _as_float,_rms,_abs_mean,_abs_p95,_abs_max,_min,_max,_grad_norm,_grad_abs_stats
from PINNmizer.params import _params_dtype_device,_t_limits,_x_grid,scale_t,scale_x
from PINNmizer.pinn.losses import compute_pde_loss

def make_fixed_pde_batch(
    *,
    params,
    n_time: int = 31,
    n_eval: int = 100,
    use_mizer_x_grid: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Deterministic Cartesian diagnostic grid.

    Current PDE code expects separate 1D t_eval and x_eval vectors and then
    evaluates the Cartesian product. This is a fixed grid, not a paired Sobol cloud.
    """
    dtype, device = _params_dtype_device(params)

    x_native = _x_grid(params)
    t_min, t_max = _t_limits(params)

    t_eval = torch.linspace(
        t_min,
        t_max,
        n_time,
        dtype=dtype,
        device=device,
    )

    if use_mizer_x_grid:
        x_eval = x_native
    else:
        x_eval = torch.linspace(
            x_native[0],
            x_native[-1],
            n_eval,
            dtype=dtype,
            device=device,
        )

    w_eval = torch.exp(x_eval)

    return {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": w_eval,
        "x_grid": x_native,
        "x_grid_scaled": scale_x(x_native, params),
        "w_grid": params.w,
    }

def make_fixed_pde_batch_from_csv(
    *,
    params,
    path: str | Path,
    t_col: str = "t_eval",
    x_col: str = "x_eval",
) -> dict[str, torch.Tensor]:
    """
    Load a deterministic diagnostic grid from CSV.

    The CSV should contain physical time values in `t_col` and log-weight values
    in `x_col`. Repeated rows are allowed; unique sorted values are used to create
    the Cartesian grid expected by the existing PDE code.
    """
    df = pd.read_csv(path)

    if t_col not in df.columns:
        raise ValueError(f"Missing column {t_col!r} in {path}")
    if x_col not in df.columns:
        raise ValueError(f"Missing column {x_col!r} in {path}")

    dtype, device = _params_dtype_device(params)

    t_eval = torch.as_tensor(
        np.sort(df[t_col].dropna().unique()),
        dtype=dtype,
        device=device,
    )
    x_eval = torch.as_tensor(
        np.sort(df[x_col].dropna().unique()),
        dtype=dtype,
        device=device,
    )
    w_eval = torch.exp(x_eval)
    x_native = _x_grid(params)

    return {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": w_eval,
        "x_grid": x_native,
        "x_grid_scaled": scale_x(x_native, params),
        "w_grid": params.w,
    }

def compute_fixed_diagnostics(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    n_init: torch.Tensor | None,
    fixed_batch: dict[str, torch.Tensor],
    residual_form: str = "log",
    boundary_loss_form: str = "log",
    species_idx: int | None = 0,
    compute_grad_norms: bool = True,
    eps: float = 1e-30,
    loss_weights: dict[str, float] | None = None,
    bc_eps: float | None = None,
    bc_g_min: float = 1e-12,    
    bc_use_constant_r: bool = False,
    bc_constant_r: float | None = None,
) -> dict[str, float]:
    """
    Deterministic diagnostics on a fixed grid.

    This recomputes the graph independently of the training batch. It does not
    call optimizer.step() and does not populate parameter .grad fields.
    """
    model_was_training = model.training
    model.train()

    loss, out = compute_pde_loss(
        model=model,
        batch=fixed_batch,
        params=params,
        n_pp=n_pp,
        residual_form=residual_form,
        n_init=n_init,
        lambda_pde=1.0,
        lambda_ic=1.0 if n_init is not None else 0.0,
        lambda_bc=1.0,
        boundary_loss_form=boundary_loss_form,
        species_idx=species_idx,
        eps=eps,
        bc_eps=bc_eps,
        bc_g_min=bc_g_min,
        use_constant_recruitment_r=bc_use_constant_r,
        constant_recruitment_r=bc_constant_r,
        )

    advective = out["g_eval"] * out["dlogN_dw"]

    row = {
        "fixed_loss": _as_float(loss),
        "fixed_loss_pde": _as_float(out["loss_pde"]),
        "fixed_loss_ic": _as_float(out["loss_ic"]),
        "fixed_loss_bc": _as_float(out["loss_bc"]),
        "fixed_residual_log_rms": _rms(out["residual_log"]),
        "fixed_residual_log_abs_mean": _abs_mean(out["residual_log"]),
        "fixed_residual_log_abs_p95": _abs_p95(out["residual_log"]),
        "fixed_residual_log_abs_max": _abs_max(out["residual_log"]),
        "rms_dlogN_dt": _rms(out["dlogN_dt"]),
        "rms_advective": _rms(advective),
        "rms_mu": _rms(out["mu_eval"]),
        "rms_dg_dw": _rms(out["dg_dw"]),
        "log_N_eval_min": _min(out["log_N_eval"]),
        "log_N_eval_max": _max(out["log_N_eval"]),
        "N_eval_min": _min(out["N_eval"]),
        "N_eval_max": _max(out["N_eval"]),
        "g_eval_min": _min(out["g_eval"]),
        "g_eval_max": _max(out["g_eval"]),
        "mu_eval_min": _min(out["mu_eval"]),
        "mu_eval_max": _max(out["mu_eval"]),
        "dg_dw_min": _min(out["dg_dw"]),
        "dg_dw_max": _max(out["dg_dw"]),
    }

    row["fixed_loss_unweighted"] = (
        row["fixed_loss_pde"]
        + row["fixed_loss_ic"]
        + row["fixed_loss_bc"]
    )
    row["bc_use_constant_r"] = 1.0 if bc_use_constant_r else 0.0
    row["bc_constant_r"] = float(bc_constant_r) if bc_constant_r is not None else math.nan
    
    if loss_weights is not None:
        row["fixed_w_pde"] = float(loss_weights["pde"])
        row["fixed_w_ic"] = float(loss_weights["ic"])
        row["fixed_w_bc"] = float(loss_weights["bc"])
    
        row["fixed_weighted_loss_pde"] = row["fixed_w_pde"] * row["fixed_loss_pde"]
        row["fixed_weighted_loss_ic"] = row["fixed_w_ic"] * row["fixed_loss_ic"]
        row["fixed_weighted_loss_bc"] = row["fixed_w_bc"] * row["fixed_loss_bc"]
    
        row["fixed_loss_weighted"] = (
            row["fixed_weighted_loss_pde"]
            + row["fixed_weighted_loss_ic"]
            + row["fixed_weighted_loss_bc"]
        )
    else:
        row["fixed_w_pde"] = math.nan
        row["fixed_w_ic"] = math.nan
        row["fixed_w_bc"] = math.nan
        row["fixed_weighted_loss_pde"] = math.nan
        row["fixed_weighted_loss_ic"] = math.nan
        row["fixed_weighted_loss_bc"] = math.nan
        row["fixed_loss_weighted"] = math.nan

    if "flux_left" in out and "recruitment_flux" in out:
        flux_left = out["flux_left"]
        recruitment_flux = out["recruitment_flux"]
        flux_mismatch = flux_left - recruitment_flux
        target_log_N = out["bc_target_log_N"]
        target_N = out["bc_target_N"]
        density_mismatch = out["N_left"] - target_N
        log_density_mismatch = out["log_N_left"] - target_log_N

        for key in [
            "bc_eps",
            "flux_left_min",
            "flux_left_max",
            "recruitment_flux_min",
            "recruitment_flux_max",
            "frac_flux_left_clamped",
            "frac_recruitment_flux_clamped",
            "boundary_residual_abs_p95",
            "boundary_residual_abs_max",
            "bc_g_min",
            "bc_valid_count",
            "bc_total_count",
            "bc_valid_fraction",
            "bc_invalid_fraction",
            "bc_invalid_g_fraction",
            "bc_invalid_recruitment_fraction",
            "bc_nonfinite_fraction",
            "bc_target_log_N_min",
            "bc_target_log_N_max",
            "bc_target_N_min",
            "bc_target_N_max",
            "bc_use_constant_recruitment_r",
            "bc_constant_recruitment_r",
        ]:
            if key in out:
                row[key] = _as_float(out[key])

        row.update(
            {
                "flux_left_mean": _as_float(torch.mean(flux_left.detach())),
                "recruitment_flux_mean": _as_float(torch.mean(recruitment_flux.detach())),
                "flux_mismatch_rms": _rms(flux_mismatch),
                "flux_mismatch_abs_mean": _abs_mean(flux_mismatch),
                "flux_mismatch_abs_p95": _abs_p95(flux_mismatch),

                "bc_density_mismatch_rms": _rms(density_mismatch),
                "bc_density_mismatch_abs_mean": _abs_mean(density_mismatch),
                "bc_log_density_mismatch_rms": _rms(log_density_mismatch),
                "bc_log_density_mismatch_abs_mean": _abs_mean(log_density_mismatch),
                "boundary_residual_rms": _rms(out["boundary_residual"]),
            }
        )

    if compute_grad_norms:
        row.update(
            {
                "grad_norm_pde": _grad_norm(out["loss_pde"], model),
                "grad_norm_ic": _grad_norm(out["loss_ic"], model),
                "grad_norm_bc": _grad_norm(out["loss_bc"], model),
            }
        )

        pde_abs_max, pde_abs_mean = _grad_abs_stats(out["loss_pde"], model)
        ic_abs_max, ic_abs_mean = _grad_abs_stats(out["loss_ic"], model)
        bc_abs_max, bc_abs_mean = _grad_abs_stats(out["loss_bc"], model)
        
        row.update(
            {
                "grad_abs_max_pde": pde_abs_max,
                "grad_abs_mean_pde": pde_abs_mean,
                "grad_abs_max_ic": ic_abs_max,
                "grad_abs_mean_ic": ic_abs_mean,
                "grad_abs_max_bc": bc_abs_max,
                "grad_abs_mean_bc": bc_abs_mean,
            }
        )

        grad_values = [
            row["grad_norm_pde"],
            row["grad_norm_ic"],
            row["grad_norm_bc"],
        ]
        finite_positive = [
            x for x in grad_values
            if isinstance(x, float) and math.isfinite(x) and x > 0.0
        ]
        row["grad_norm_max_min_ratio"] = (
            max(finite_positive) / min(finite_positive)
            if len(finite_positive) >= 2
            else math.nan
        )
    else:
        row.update(
            {
                "grad_norm_pde": math.nan,
                "grad_norm_ic": math.nan,
                "grad_norm_bc": math.nan,
                "grad_norm_max_min_ratio": math.nan,
            }
        )

    if model_was_training:
        model.train()
    else:
        model.eval()

    return row


__all__=["make_fixed_pde_batch","make_fixed_pde_batch_from_csv","compute_fixed_diagnostics"]
