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

def save_fixed_grid_fields_and_plots(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    n_init: torch.Tensor | None,
    outdir: str | Path,
    residual_form: str = "log",
    boundary_loss_form: str = "log",
    species_idx: int = 0,
    n_time: int = 61,
    n_eval: int = 160,
    fixed_batch: dict[str, torch.Tensor] | None = None,
) -> None:
    """
    Final after-run diagnostic fields and heatmaps on one deterministic grid.
    """
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
        species_idx=species_idx,
    )

    t = fixed_batch["t_eval"].detach().cpu().numpy()
    x = fixed_batch["x_eval"].detach().cpu().numpy()
    w = fixed_batch["w_eval"].detach().cpu().numpy()

    def field(name: str) -> np.ndarray:
        return out[name][:, species_idx, :].detach().cpu().numpy()

    log_N = field("log_N_eval")
    log10_N = log_N / math.log(10.0)
    residual_log = field("residual_log")
    dlogN_dt = field("dlogN_dt")
    advective = (out["g_eval"] * out["dlogN_dw"])[:, species_idx, :].detach().cpu().numpy()
    mu = field("mu_eval")
    dg_dw = field("dg_dw")
    g_eval = field("g_eval")

    np.savez(
        outdir / "fixed_grid_fields.npz",
        t_eval=t,
        x_eval=x,
        w_eval=w,
        log10_N=log10_N,
        residual_log=residual_log,
        dlogN_dt=dlogN_dt,
        advective=advective,
        mu=mu,
        dg_dw=dg_dw,
        g_eval=g_eval,
    )

    tt = np.repeat(t[:, None], len(x), axis=1)
    xx = np.repeat(x[None, :], len(t), axis=0)
    ww = np.repeat(w[None, :], len(t), axis=0)

    pd.DataFrame(
        {
            "t_eval": tt.reshape(-1),
            "x_eval": xx.reshape(-1),
            "w_eval": ww.reshape(-1),
            "log10_N": log10_N.reshape(-1),
            "residual_log": residual_log.reshape(-1),
            "dlogN_dt": dlogN_dt.reshape(-1),
            "advective": advective.reshape(-1),
            "mu": mu.reshape(-1),
            "dg_dw": dg_dw.reshape(-1),
            "g_eval": g_eval.reshape(-1),
        }
    ).to_csv(outdir / "fixed_grid_fields.csv", index=False)

    _plot_heatmap(
        values=log10_N,
        t=t,
        x=x,
        path=outdir / "surface_log10_N.png",
        title="Predicted log10 N on fixed grid",
        colorbar_label="log10(N)",
    )

    _plot_heatmap(
        values=residual_log,
        t=t,
        x=x,
        path=outdir / "surface_residual_log.png",
        title="Log-form PDE residual on fixed grid",
        colorbar_label="residual_log",
    )

    _plot_heatmap(
        values=np.abs(residual_log),
        t=t,
        x=x,
        path=outdir / "surface_abs_residual_log.png",
        title="Absolute log-form PDE residual on fixed grid",
        colorbar_label="abs(residual_log)",
    )

    for name, values in {
        "dlogN_dt": dlogN_dt,
        "advective": advective,
        "mu": mu,
        "dg_dw": dg_dw,
    }.items():
        _plot_heatmap(
            values=values,
            t=t,
            x=x,
            path=outdir / f"surface_{name}.png",
            title=f"{name} on fixed grid",
            colorbar_label=name,
        )

    plt.figure()
    for idx in np.linspace(0, len(t) - 1, 6).astype(int):
        plt.plot(x, log10_N[idx, :], label=f"t={t[idx]:.2f}")
    plt.xlabel("log weight")
    plt.ylabel("log10(N)")
    plt.title("log10(N) profiles through time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "log10_N_profiles_by_time.png", dpi=200)
    plt.close()

    if "flux_left" in out and "recruitment_flux" in out:
        flux_left = out["flux_left"][:, species_idx].detach().cpu().numpy()
        recruitment_flux = out["recruitment_flux"][:, species_idx].detach().cpu().numpy()
        mismatch = flux_left - recruitment_flux

        pd.DataFrame(
            {
                "t_eval": t,
                "flux_left": flux_left,
                "recruitment_flux": recruitment_flux,
                "flux_mismatch": mismatch,
            }
        ).to_csv(outdir / "boundary_flux_diagnostics.csv", index=False)

        plt.figure()
        plt.plot(t, flux_left, label="g(w_min,t) * N(w_min,t)")
        plt.plot(t, recruitment_flux, label="recruitment_flux")
        plt.xlabel("time")
        plt.ylabel("flux")
        plt.title("Recruitment boundary flux check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "boundary_flux_check.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(t, mismatch)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("time")
        plt.ylabel("flux_left - recruitment_flux")
        plt.title("Recruitment boundary flux mismatch")
        plt.tight_layout()
        plt.savefig(outdir / "boundary_flux_mismatch_timeseries.png", dpi=200)
        plt.close()

__all__=["save_fixed_grid_fields_and_plots"]
