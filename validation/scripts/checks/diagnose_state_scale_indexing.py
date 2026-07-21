#!/usr/bin/env python3
"""
Diagnose why interpolate_log_state_scale(params, params.w) does not reproduce
grid_state_scale(params).

Run from the repository root:

    set "PYTHONPATH=%CD%" && python validation\scripts\checks\diagnose_state_scale_indexing.py ^
      --input-dir validation\fixtures\pde_multispecies

This script is checkpoint-free and uses the repository's current loading and
state-scale functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "PINNmizer").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {resolved}")


REPO_ROOT = _find_repo_root(Path(__file__))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask
from PINNmizer.pinn.state_scale import (
    grid_state_scale,
    interpolate_log_state_scale,
    set_state_scale_from_initial_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <input-dir>/state_scale_indexing_diagnostic.csv",
    )
    args = parser.parse_args()

    params, n_init, _ = load_mizer_inputs(
        args.input_dir,
        dtype=torch.float64,
        device="cpu",
    )
    params.state_parameterization = "log-u"
    set_state_scale_from_initial_condition(params, n_init)

    w = params.w
    x_grid = torch.log(w)
    x = torch.log(w.clone())

    expected_idx = torch.arange(w.numel(), dtype=torch.long)
    left_idx_raw = torch.searchsorted(x_grid, x, right=False)
    right_idx_raw = torch.searchsorted(x_grid, x, right=True)

    left_idx = left_idx_raw.clamp(1, w.numel() - 1)
    x0 = x_grid[left_idx - 1]
    x1 = x_grid[left_idx]
    frac = (x - x0) / (x1 - x0)

    log_s_grid, s_grid = grid_state_scale(params)
    log_s_interp, s_interp, dlog_s_dw = interpolate_log_state_scale(params, w)
    active = active_grid_mask(params)

    log_diff = log_s_interp - log_s_grid
    rel_s_diff = torch.abs(s_interp - s_grid) / torch.clamp(
        torch.maximum(torch.abs(s_interp), torch.abs(s_grid)),
        min=1e-300,
    )

    print("\nGRID PROPERTIES")
    print(f"w dtype: {w.dtype}")
    print(f"w shape: {tuple(w.shape)}")
    print(f"strictly increasing w: {bool(torch.all(w[1:] > w[:-1]))}")
    print(f"strictly increasing log(w): {bool(torch.all(x_grid[1:] > x_grid[:-1]))}")
    print(f"minimum log-grid spacing: {float((x_grid[1:] - x_grid[:-1]).min()):.17e}")
    print(f"maximum |log(w) - log(w.clone())|: {float(torch.max(torch.abs(x_grid - x))):.17e}")
    print(f"x_grid contiguous: {x_grid.is_contiguous()}")
    print(f"x contiguous: {x.is_contiguous()}")

    left_mismatch = left_idx_raw != expected_idx
    # At the first point, left insertion index 0 is expected.
    print("\nSEARCHSORTED")
    print(f"left-index mismatches vs arange: {int(left_mismatch.sum())}")
    print(
        "left raw indices:",
        left_idx_raw.detach().cpu().tolist(),
    )
    print(
        "right raw indices:",
        right_idx_raw.detach().cpu().tolist(),
    )
    print(
        f"fraction range after clamping: "
        f"[{float(frac.min()):.17e}, {float(frac.max()):.17e}]"
    )

    print("\nSTATE-SCALE DIFFERENCES")
    print(f"max |grid log_S - interpolated log_S|: {float(torch.abs(log_diff).max()):.17e}")
    print(f"max relative S difference: {float(rel_s_diff.max()):.17e}")
    active_log_diff = torch.abs(log_diff)[active]
    active_rel_diff = rel_s_diff[active]
    print(f"max active |log_S difference|: {float(active_log_diff.max()):.17e}")
    print(f"max active relative S difference: {float(active_rel_diff.max()):.17e}")

    rows = []
    n_species, n_w = log_s_grid.shape
    species_names = getattr(params, "species", None)
    for sp in range(n_species):
        sp_name = (
            str(species_names[sp])
            if species_names is not None and sp < len(species_names)
            else f"species_{sp}"
        )
        for j in range(n_w):
            rows.append(
                {
                    "species_idx": sp,
                    "species": sp_name,
                    "weight_idx": j,
                    "w": float(w[j]),
                    "x": float(x[j]),
                    "expected_left_index": j,
                    "searchsorted_left_raw": int(left_idx_raw[j]),
                    "searchsorted_right_raw": int(right_idx_raw[j]),
                    "interval_left_index": int(left_idx[j] - 1),
                    "interval_right_index": int(left_idx[j]),
                    "fraction": float(frac[j]),
                    "active": bool(active[sp, j]),
                    "grid_log_S": float(log_s_grid[sp, j]),
                    "interpolated_log_S": float(log_s_interp[sp, j]),
                    "log_S_difference": float(log_diff[sp, j]),
                    "grid_S": float(s_grid[sp, j]),
                    "interpolated_S": float(s_interp[sp, j]),
                    "relative_S_difference": float(rel_s_diff[sp, j]),
                    "dlogS_dw": float(dlog_s_dw[sp, j]),
                }
            )

    df = pd.DataFrame(rows)
    output = (
        Path(args.output)
        if args.output
        else Path(args.input_dir) / "state_scale_indexing_diagnostic.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\nTOP 20 ACTIVE MISMATCHES")
    cols = [
        "species_idx",
        "species",
        "weight_idx",
        "w",
        "searchsorted_left_raw",
        "searchsorted_right_raw",
        "fraction",
        "grid_log_S",
        "interpolated_log_S",
        "log_S_difference",
        "relative_S_difference",
    ]
    top = (
        df[df["active"]]
        .assign(abs_log_difference=lambda z: z["log_S_difference"].abs())
        .nlargest(20, "abs_log_difference")
    )
    print(top[cols].to_string(index=False))

    print(f"\nFull diagnostic written to: {output.resolve()}")


if __name__ == "__main__":
    main()
