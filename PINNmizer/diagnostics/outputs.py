from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def append_diagnostic_row(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
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

__all__=["append_diagnostic_row","save_latest_metrics_table"]
