from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PINNmizer.params import (
    _params_dtype_device,
    _t_limits,
    _x_grid,
    scale_t,
    scale_x,
)
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


def _as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu())


def _rms(x: torch.Tensor) -> float:
    x = x.detach()
    return _as_float(torch.sqrt(torch.mean(x ** 2)))


def _abs_mean(x: torch.Tensor) -> float:
    return _as_float(torch.mean(torch.abs(x.detach())))


def _abs_p95(x: torch.Tensor) -> float:
    z = torch.abs(x.detach()).reshape(-1)
    return _as_float(torch.quantile(z, 0.95))


def _abs_max(x: torch.Tensor) -> float:
    return _as_float(torch.max(torch.abs(x.detach())))


def _min(x: torch.Tensor) -> float:
    return _as_float(torch.min(x.detach()))


def _max(x: torch.Tensor) -> float:
    return _as_float(torch.max(x.detach()))


def _grad_norm(loss: torch.Tensor, model) -> float:
    if not loss.requires_grad:
        return math.nan

    parameters = [p for p in model.parameters() if p.requires_grad]

    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    total = None
    for grad in grads:
        if grad is None:
            continue

        if not torch.isfinite(grad).all():
            return math.nan

        value = (grad.detach() ** 2).sum()
        total = value if total is None else total + value

    if total is None:
        return math.nan

    return float(torch.sqrt(total).cpu())

def _grad_abs_stats(loss: torch.Tensor, model) -> tuple[float, float]:
    if not loss.requires_grad:
        return math.nan, math.nan

    parameters = [p for p in model.parameters() if p.requires_grad]

    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    flat = [
        grad.detach().reshape(-1)
        for grad in grads
        if grad is not None and torch.isfinite(grad).all()
    ]

    if not flat:
        return math.nan, math.nan

    values = torch.cat(flat).abs()

    return (
        float(values.max().cpu()),
        float(values.mean().cpu()),
    )

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
        relative_mismatch = flux_mismatch / torch.clamp(
            torch.abs(recruitment_flux),
            min=eps,
        )

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
                "flux_mismatch_rel_abs_mean": _abs_mean(relative_mismatch),
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


def append_diagnostic_row(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _plot_lines(
    *,
    df: pd.DataFrame,
    columns: list[str],
    path: Path,
    title: str,
    ylabel: str,
    yscale: str | None = None,
) -> None:
    available = [
        col for col in columns
        if col in df.columns and df[col].notna().any()
    ]

    if not available:
        return

    plt.figure()
    for col in available:
        plt.plot(df["step"], df[col], label=col)

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_training_diagnostic_plots(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "diagnostic_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    loss_path = run_dir / "loss_history.csv"
    if loss_path.exists():
        loss_df = pd.read_csv(loss_path)

        _plot_lines(
            df=loss_df,
            columns=["loss", "loss_pde", "loss_ic", "loss_bc"],
            path=plot_dir / "training_losses.png",
            title="Training losses",
            ylabel="loss",
            yscale="log",
        )

        _plot_lines(
            df=loss_df,
            columns=["grad_norm"],
            path=plot_dir / "training_total_grad_norm.png",
            title="Total gradient norm",
            ylabel="gradient norm",
            yscale="log",
        )

        _plot_lines(
            df=loss_df,
            columns=[
                "residual_log_abs_mean",
                "residual_log_abs_max",
            ],
            path=plot_dir / "training_sampled_residuals.png",
            title="Sampled training residual summaries",
            ylabel="absolute residual",
            yscale="log",
        )

    diag_path = run_dir / "fixed_diagnostic_history.csv"
    if not diag_path.exists():
        return

    diag_df = pd.read_csv(diag_path)

    _plot_lines(
        df=diag_df,
        columns=[
            "fixed_loss",
            "fixed_loss_pde",
            "fixed_loss_ic",
            "fixed_loss_bc",
        ],
        path=plot_dir / "fixed_grid_losses.png",
        title="Fixed-grid diagnostic losses",
        ylabel="loss",
        yscale="log",
    )

    _plot_lines(
        df=diag_df,
        columns=[
            "fixed_residual_log_rms",
            "fixed_residual_log_abs_mean",
            "fixed_residual_log_abs_p95",
            "fixed_residual_log_abs_max",
        ],
        path=plot_dir / "fixed_grid_residual_metrics.png",
        title="Fixed-grid residual diagnostics",
        ylabel="residual",
        yscale="log",
    )

    _plot_lines(
        df=diag_df,
        columns=[
            "grad_norm_pde",
            "grad_norm_ic",
            "grad_norm_bc",
        ],
        path=plot_dir / "component_gradient_norms.png",
        title="Component-wise gradient norms",
        ylabel="gradient norm",
        yscale="log",
    )

    _plot_lines(
        df=diag_df,
        columns=[
            "rms_dlogN_dt",
            "rms_advective",
            "rms_mu",
            "rms_dg_dw",
        ],
        path=plot_dir / "pde_term_rms.png",
        title="RMS magnitude of additive log-PDE terms",
        ylabel="RMS magnitude",
        yscale="log",
    )

    _plot_lines(
        df=diag_df,
        columns=[
            "flux_mismatch_rms",
            "flux_mismatch_abs_mean",
            "flux_mismatch_abs_p95",
            "flux_mismatch_rel_abs_mean",
        ],
        path=plot_dir / "boundary_flux_mismatch.png",
        title="Recruitment boundary flux mismatch",
        ylabel="mismatch",
        yscale="log",
    )

    _plot_lines(
        df=diag_df,
        columns=[
            "N_eval_min",
            "N_eval_max",
        ],
        path=plot_dir / "fixed_grid_N_range.png",
        title="Predicted abundance range on fixed grid",
        ylabel="N",
        yscale="log",
    )


def save_latest_metrics_table(
    *,
    metrics: dict[str, float],
    outdir: str | Path,
    csv_name: str = "latest_fixed_diagnostics.csv",
    png_name: str = "latest_fixed_diagnostics_table.png",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(outdir / csv_name, index=False)

    preferred = [
        "step",
        "fixed_loss_pde",
        "fixed_loss_ic",
        "fixed_loss_bc",
        "fixed_residual_log_rms",
        "fixed_residual_log_abs_p95",
        "grad_norm_pde",
        "grad_norm_ic",
        "grad_norm_bc",
        "grad_norm_max_min_ratio",
        "rms_dlogN_dt",
        "rms_advective",
        "rms_mu",
        "rms_dg_dw",
        "flux_mismatch_abs_mean",
        "flux_mismatch_abs_p95",
        "flux_mismatch_rel_abs_mean",
        "N_eval_min",
        "N_eval_max",
    ]

    rows = []
    for key in preferred:
        if key not in metrics:
            continue

        value = metrics[key]
        if isinstance(value, float):
            value_str = f"{value:.6e}" if math.isfinite(value) else str(value)
        else:
            value_str = str(value)

        rows.append((key, value_str))

    fig_height = max(4.0, 0.32 * len(rows))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    plt.tight_layout()
    plt.savefig(outdir / png_name, dpi=200)
    plt.close(fig)


def _plot_heatmap(
    *,
    values: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    plt.figure()
    plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("log weight")
    plt.ylabel("time")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


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
