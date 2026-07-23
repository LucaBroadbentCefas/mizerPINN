from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PINNmizer.diagnostics.fixed_grid import make_fixed_pde_batch
from PINNmizer.diagnostics.plots import _plot_heatmap
from PINNmizer.pinn.losses import compute_pde_loss


def _species_name(params, idx: int) -> str:
    species = getattr(params, "species", None)
    if species is not None and idx < len(species):
        return str(species[idx])
    return f"species_{idx}"


def save_fixed_grid_fields_and_plots_multispecies(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    n_init: torch.Tensor | None,
    outdir: str | Path,
    residual_form: str = "log",
    boundary_loss_form: str = "log",
    n_time: int = 61,
    n_eval: int = 160,
    bc_g_min: float = 1e-12,
    fixed_batch: dict[str, torch.Tensor] | None = None,
    bc_use_constant_r: bool = False,
    bc_constant_r: float | None = None,
    boundary_target_gradient_mode: str = "detached",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if fixed_batch is None:
        fixed_batch = make_fixed_pde_batch(
            params=params,
            n_time=n_time,
            n_eval=n_eval,
            use_mizer_x_grid=False,
        )

    _, out = compute_pde_loss(
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
        species_idx=None,
        bc_g_min=bc_g_min,
        use_constant_recruitment_r=bc_use_constant_r,
        constant_recruitment_r=bc_constant_r,
        boundary_target_gradient_mode=boundary_target_gradient_mode,
    )

    t = fixed_batch["t_eval"].detach().cpu().numpy()
    x = fixed_batch["x_eval"].detach().cpu().numpy()
    w = fixed_batch["w_eval"].detach().cpu().numpy()
    n_species = out["log_N_eval"].shape[1]

    arrays = {
        "t": t,
        "x": x,
        "w": w,
        "log_N_eval": out["log_N_eval"].detach().cpu().numpy(),
        "N_eval": out["N_eval"].detach().cpu().numpy(),
        "residual_log": out["residual_log"].detach().cpu().numpy(),
        "residual": out["residual"].detach().cpu().numpy(),
        "dlogN_dt": out["dlogN_dt"].detach().cpu().numpy(),
        "dlogN_dw": out["dlogN_dw"].detach().cpu().numpy(),
        "g_eval": out["g_eval"].detach().cpu().numpy(),
        "dg_dw": out["dg_dw"].detach().cpu().numpy(),
        "mu_eval": out["mu_eval"].detach().cpu().numpy(),
    }
    np.savez(outdir / "fixed_grid_fields.npz", **arrays)

    rows = []
    tt = np.broadcast_to(t[:, None], (len(t), len(w)))
    xx = np.broadcast_to(x[None, :], (len(t), len(w)))
    ww = np.broadcast_to(w[None, :], (len(t), len(w)))
    for s in range(n_species):
        log_N = arrays["log_N_eval"][:, s, :]
        rows.append(pd.DataFrame({
            "species_idx": s,
            "species": _species_name(params, s),
            "t": tt.reshape(-1),
            "x": xx.reshape(-1),
            "w": ww.reshape(-1),
            "log_N": log_N.reshape(-1),
            "log10_N": (log_N / math.log(10.0)).reshape(-1),
            "N": arrays["N_eval"][:, s, :].reshape(-1),
            "residual_log": arrays["residual_log"][:, s, :].reshape(-1),
            "residual": arrays["residual"][:, s, :].reshape(-1),
            "dlogN_dt": arrays["dlogN_dt"][:, s, :].reshape(-1),
            "dlogN_dw": arrays["dlogN_dw"][:, s, :].reshape(-1),
            "advective": (arrays["g_eval"][:, s, :] * arrays["dlogN_dw"][:, s, :]).reshape(-1),
            "mu": arrays["mu_eval"][:, s, :].reshape(-1),
            "dg_dw": arrays["dg_dw"][:, s, :].reshape(-1),
            "g_eval": arrays["g_eval"][:, s, :].reshape(-1),
        }))

        log10_N = log_N / math.log(10.0)
        residual_log = arrays["residual_log"][:, s, :]
        _plot_heatmap(
            values=log10_N,
            t=t,
            x=x,
            path=outdir / f"surface_log10_N_species_{s}.png",
            title=f"Predicted log10 N on fixed grid ({_species_name(params, s)})",
            colorbar_label="log10(N)",
        )
        _plot_heatmap(
            values=residual_log,
            t=t,
            x=x,
            path=outdir / f"surface_residual_log_species_{s}.png",
            title=f"Log-form PDE residual on fixed grid ({_species_name(params, s)})",
            colorbar_label="residual_log",
        )
        plt.figure()
        for idx in np.linspace(0, len(t) - 1, min(6, len(t))).astype(int):
            plt.plot(x, log10_N[idx, :], label=f"t={t[idx]:.2f}")
        plt.xlabel("log weight")
        plt.ylabel("log10(N)")
        plt.title(f"log10(N) profiles through time ({_species_name(params, s)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"log10_N_profiles_by_time_species_{s}.png", dpi=200)
        plt.close()

    pd.concat(rows, ignore_index=True).to_csv(outdir / "fixed_grid_fields.csv", index=False)

    if "flux_left" in out and "recruitment_flux" in out:
        bc_rows = []
        for s in range(n_species):
            bc_rows.append(pd.DataFrame({
                "species_idx": s,
                "species": _species_name(params, s),
                "t_eval": t,
                "flux_left": out["flux_left"][:, s].detach().cpu().numpy(),
                "recruitment_flux": out["recruitment_flux"][:, s].detach().cpu().numpy(),
                "g_left": out["g_left"][:, s].detach().cpu().numpy(),
                "N_left": out["N_left"][:, s].detach().cpu().numpy(),
                "log_N_left": out["log_N_left"][:, s].detach().cpu().numpy(),
                "bc_target_N": out["bc_target_N"][:, s].detach().cpu().numpy(),
                "bc_target_log_N": out["bc_target_log_N"][:, s].detach().cpu().numpy(),
                "bc_valid": out["bc_valid_mask"][:, s].detach().cpu().numpy().astype(bool),
                "bc_use_constant_r": np.full_like(t, 1.0 if bc_use_constant_r else 0.0, dtype=float),
                "bc_constant_r": np.full_like(t, float(bc_constant_r) if bc_constant_r is not None else np.nan, dtype=float),
            }))
        pd.concat(bc_rows, ignore_index=True).to_csv(outdir / "boundary_flux_diagnostics.csv", index=False)


__all__ = ["save_fixed_grid_fields_and_plots_multispecies"]
