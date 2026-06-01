from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PINNmizer.mizer_grid_ops import get_encounter, feeding_level, e_repro_and_growth
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
        g_left = out["g_left"][:, species_idx].detach().cpu().numpy()
        N_left = out["N_left"][:, species_idx].detach().cpu().numpy()
        log_N_left = out["log_N_left"][:, species_idx].detach().cpu().numpy()
    
        g_left = out["g_left"][:, species_idx].detach().cpu().numpy()
        N_left = out["N_left"][:, species_idx].detach().cpu().numpy()
        erepog_left = out["erepog_left"][:, species_idx].detach().cpu().numpy()
        pos_erepog_left = out["pos_erepog_left"][:, species_idx].detach().cpu().numpy()
        e_repro_left = out["e_repro_left"][:, species_idx].detach().cpu().numpy()
        psi_left = out["psi_left"][:, species_idx].detach().cpu().numpy()
        encounter_left = out["encounter_left"][:, species_idx].detach().cpu().numpy()
        feeding_left = out["feeding_left"][:, species_idx].detach().cpu().numpy()
        h_left = out["h_left"][:, species_idx].detach().cpu().numpy()
        metab_left = out["metab_left"][:, species_idx].detach().cpu().numpy()
        reconstructed_g_left = pos_erepog_left * (1.0 - psi_left)
    
        with torch.no_grad():
            N_grid = out["N_grid"][:, species_idx, :].detach()
            egg_idx = params.w_min_idx.to(torch.long) - 1
            fixed_encounter = []
            fixed_feeding = []
            fixed_erepog = []
        
            for tt_i in range(N_grid.shape[0]):
                n_t = out["N_grid"][tt_i].detach()
                enc_t = get_encounter(n_pp.detach(), n_t, params)
                feed_t = feeding_level(enc_t, params.intake_max)
                erepog_t = e_repro_and_growth(feed_t, enc_t, params.alpha, params.metab)
        
                fixed_encounter.append(enc_t[species_idx, egg_idx[species_idx]].detach().cpu().item())
                fixed_feeding.append(feed_t[species_idx, egg_idx[species_idx]].detach().cpu().item())
                fixed_erepog.append(erepog_t[species_idx, egg_idx[species_idx]].detach().cpu().item())
        
            fixed_encounter = np.asarray(fixed_encounter)
            fixed_feeding = np.asarray(fixed_feeding)
            fixed_erepog = np.asarray(fixed_erepog)
    
        mismatch = flux_left - recruitment_flux
        tiny = np.finfo(float).tiny
        pd.DataFrame(
            {
                "t_eval": t,
                "flux_left": flux_left,
                "recruitment_flux": recruitment_flux,
                "flux_mismatch": mismatch,
                "g_left": g_left,
                "N_left": N_left,
                "log_N_left": log_N_left,
                "log10_flux_left": np.log10(np.maximum(flux_left, tiny)),
                "log10_recruitment_flux": np.log10(np.maximum(recruitment_flux, tiny)),
                "log10_g_left": np.log10(np.maximum(g_left, tiny)),
                "log10_N_left": log_N_left / math.log(10.0),
                "flux_left_is_zero_or_tiny": flux_left <= tiny,
                "g_left_is_zero_or_tiny": g_left <= tiny,
                "N_left_is_zero_or_tiny": N_left <= tiny,
                "g_left": g_left,
                "N_left": N_left,
                "erepog_left": erepog_left,
                "pos_erepog_left": pos_erepog_left,
                "e_repro_left": e_repro_left,
                "psi_left": psi_left,
                "encounter_left": encounter_left,
                "feeding_left": feeding_left,
                "h_left": h_left,
                "metab_left": metab_left,
                "reconstructed_g_left": reconstructed_g_left,
                "g_left_minus_reconstructed": g_left - reconstructed_g_left,
                "fixed_encounter_left": fixed_encounter,
                "fixed_feeding_left": fixed_feeding,
                "fixed_erepog_left": fixed_erepog,
                "encounter_left_minus_fixed": encounter_left - fixed_encounter,
                "feeding_left_minus_fixed": feeding_left - fixed_feeding,
                "erepog_left_minus_fixed": erepog_left - fixed_erepog,
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

        plt.figure()
        plt.plot(t, g_left, label="g_left")
        plt.plot(t, N_left, label="N_left")
        plt.plot(t, flux_left, label="g_left * N_left")
        plt.xlabel("time")
        plt.ylabel("raw value")
        plt.title("Recruitment boundary components")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "boundary_flux_components.png", dpi=200)
        plt.close()
        
        plt.figure()
        plt.plot(t, np.log10(np.maximum(flux_left, tiny)), label="log10(g_left * N_left)")
        plt.plot(t, np.log10(np.maximum(recruitment_flux, tiny)), label="log10(recruitment_flux)")
        plt.plot(t, np.log10(np.maximum(g_left, tiny)), label="log10(g_left)")
        plt.plot(t, log_N_left / math.log(10.0), label="log10(N_left)")
        plt.xlabel("time")
        plt.ylabel("log10 value")
        plt.title("Recruitment boundary components on log scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "boundary_flux_components_log10.png", dpi=200)
        plt.close()

__all__=["save_fixed_grid_fields_and_plots"]
