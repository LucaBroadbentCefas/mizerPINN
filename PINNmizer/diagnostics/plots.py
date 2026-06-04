from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _plot_lines(
    *,
    df: pd.DataFrame,
    columns: list[str],
    path: Path,
    title: str,
    ylabel: str,
    yscale: str | None = None,
    alpha: float = 1.0,
) -> None:
    available = [
        col for col in columns
        if col in df.columns and df[col].notna().any()
    ]

    if not available:
        return

    plot_df = df[["step", *available]].copy()
    if yscale == "log":
        for col in available:
            plot_df.loc[plot_df[col] <= 0.0, col] = np.nan
        available = [
            col for col in available
            if plot_df[col].notna().any()
        ]
        if not available:
            return

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for col in available:
        plt.plot(plot_df["step"], plot_df[col], label=col, alpha=alpha)

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def _plot_unscaled_loss_terms(*, loss_df: pd.DataFrame, plot_dir: Path) -> None:
    loss_term_dir = plot_dir / "unscaled_loss_terms"
    loss_terms = [
        ("loss_pde", "Unscaled PDE loss"),
        ("loss_ic", "Unscaled initial-condition loss"),
        ("loss_bc", "Unscaled boundary-condition loss"),
        ("loss_timestep", "Unscaled timestep-consistency loss"),
    ]

    for column, title in loss_terms:
        _plot_lines(
            df=loss_df,
            columns=[column],
            path=loss_term_dir / f"{column}.png",
            title=title,
            ylabel="unscaled loss",
            yscale="log",
        )

def save_training_diagnostic_plots(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "diagnostic_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    loss_path = run_dir / "loss_history.csv"
    if loss_path.exists():
        loss_df = pd.read_csv(loss_path)

        _plot_lines(
            df=loss_df,
            columns=[
                "loss",
                "weighted_loss_pde",
                "weighted_loss_ic",
                "weighted_loss_bc",
                "weighted_loss_timestep",
            ],
            path=plot_dir / "training_losses.png",
            title="Training losses",
            ylabel="loss",
            yscale="log",
            alpha=0.65,
        )

        _plot_unscaled_loss_terms(loss_df=loss_df, plot_dir=plot_dir)

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

__all__=["_plot_lines","_plot_heatmap","save_training_diagnostic_plots"]
