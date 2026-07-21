#!/usr/bin/env python3
"""
Verification diagnostics for the log-U grid/off-grid PINN pathway.

Run from the repository root, for example:

    python validation/scripts/checks/verify_log_u_grid_offgrid_v2.py \
        --input-dir validation/fixtures/pde_multispecies \
        --checkpoint runs/pde_multispecies/<RUN>/model_final.pt \
        --observations validation/fixtures/pde_multispecies/observations.csv

The script deliberately reuses the current repository implementations for:
- loading mizer inputs and observations;
- constructing the trained architecture;
- log-U reconstruction and state-scale interpolation;
- grid and off-grid model evaluation;
- autograd derivatives;
- PDE-state construction and biological operators;
- physical, log, and scaled PDE residuals;
- observation operators and lognormal likelihood.

It writes CSV diagnostics plus summary.json. Critical algebraic/pathway checks
produce a non-zero exit code unless --report-only is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

def _find_repo_root(script_path: Path) -> Path:
    """Find the nearest parent containing the local PINNmizer package."""
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "PINNmizer").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the repository root containing a PINNmizer directory. "
        f"Script location: {resolved}"
    )


# Direct execution normally puts only validation/scripts/checks on sys.path.
# Discover and prepend the repository root before importing PINNmizer.
REPO_ROOT = _find_repo_root(Path(__file__))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.io_observations import load_observation_csv
from PINNmizer.params import (
    active_eval_mask,
    active_grid_mask,
    scale_t,
    scale_x,
)
from PINNmizer.pinn.data_losses import lognormal_nll
from PINNmizer.pinn.derivatives import evaluate_log_model_with_derivatives_at_eval
from PINNmizer.pinn.model_eval import _make_model_inputs, evaluate_log_model_on_points
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.observation_operators import predict_observations
from PINNmizer.pinn.pde_state import compute_pde_state
from PINNmizer.pinn.residual import compute_pde_residual_from_state
from PINNmizer.pinn.state_scale import (
    grid_state_scale,
    interpolate_log_state_scale,
    reconstruct_from_model_output,
    set_state_scale_from_initial_condition,
)


DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the trained log-U grid/off-grid reconstruction and residual pathways."
    )
    parser.add_argument("--input-dir", required=True, help="Mizer fixture/export directory.")
    parser.add_argument("--checkpoint", required=True, help="model_final.pt or model_step_*.pt.")
    parser.add_argument(
        "--observations",
        default=None,
        help="Optional observation CSV. Predictions are recomputed through the repository observation operators.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <checkpoint directory>/grid_offgrid_verification.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--times",
        default=None,
        help="Comma-separated physical times. If omitted, --n-times evenly spaced times are used.",
    )
    parser.add_argument("--n-times", type=int, default=5)
    parser.add_argument(
        "--profile-fractions",
        default="0,0.25,0.5,0.75,1",
        help="Fractions within each log-weight interval for the phase/profile diagnostic.",
    )
    parser.add_argument(
        "--knot-epsilon-fraction",
        type=float,
        default=1e-4,
        help="Offset from each knot as a fraction of the adjacent log-weight spacing.",
    )
    parser.add_argument(
        "--fd-x-fraction",
        type=float,
        default=1e-4,
        help="Finite-difference half-step as a fraction of each log-weight interval.",
    )
    parser.add_argument(
        "--data-time-quadrature-points",
        type=int,
        default=None,
        help="Defaults to the checkpoint configuration, otherwise 1.",
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--fd-atol", type=float, default=1e-6)
    parser.add_argument("--fd-rtol", type=float, default=5e-4)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write diagnostics but do not return exit code 1 when a critical check fails.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def species_names(params, n_species: int) -> list[str]:
    names = getattr(params, "species", None)
    if names is None:
        return [f"species_{i}" for i in range(n_species)]
    return [str(names[i]) if i < len(names) else f"species_{i}" for i in range(n_species)]


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None and default is not None else value


def load_trained_model(
    *,
    input_dir: str,
    checkpoint_path: Path,
    device: str,
):
    params, n_init, n_pp = load_mizer_inputs(
        input_dir,
        dtype=DTYPE,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=params.w.device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a repository training checkpoint with model_state_dict."
        )

    config = checkpoint.get("config") or {}
    state_parameterization = config_value(
        config, "state_parameterization", "log-u"
    )
    if state_parameterization != "log-u":
        raise ValueError(
            "This verification targets the log-U pathway, but the checkpoint uses "
            f"state_parameterization={state_parameterization!r}."
        )

    params.state_parameterization = state_parameterization
    state_scale_eps = float(config_value(config, "state_scale_eps", 1e-30))
    set_state_scale_from_initial_condition(
        params,
        n_init,
        eps=state_scale_eps,
    )

    n_species = int(params.interaction.shape[0])
    model = build_pinn_model(
        model_arch=config_value(config, "model_arch", "mlp"),
        in_dim=2,
        out_dim=n_species,
        hidden_width=int(config_value(config, "hidden_width", 64)),
        hidden_layers=int(config_value(config, "hidden_layers", 3)),
        fourier_num_features=int(config_value(config, "fourier_num_features", 64)),
        fourier_scale=float(config_value(config, "fourier_scale", 1.0)),
        fourier_include_raw_input=bool(
            config_value(config, "fourier_include_raw_input", False)
        ),
        fourier_seed=config_value(config, "fourier_seed", None),
        weight_factorization=config_value(config, "weight_factorization", "none"),
        rwf_mu=float(config_value(config, "rwf_mu", 1.0)),
        rwf_sigma=float(config_value(config, "rwf_sigma", 0.1)),
        rwf_apply_to=config_value(config, "rwf_apply_to", "all"),
        rwf_base_init=config_value(config, "rwf_base_init", "pytorch"),
    ).to(dtype=DTYPE, device=params.w.device)

    model.state_parameterization = state_parameterization
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, params, n_init, n_pp, checkpoint, config


def make_times(args: argparse.Namespace, params) -> torch.Tensor:
    t_min = as_float(params.t_min)
    t_max = as_float(params.t_max)

    if args.times:
        values = [float(x.strip()) for x in args.times.split(",") if x.strip()]
        if not values:
            raise ValueError("--times did not contain any numeric values.")
        times = torch.tensor(values, dtype=DTYPE, device=params.w.device)
    else:
        if args.n_times < 1:
            raise ValueError("--n-times must be positive.")
        times = torch.linspace(
            t_min,
            t_max,
            args.n_times,
            dtype=DTYPE,
            device=params.w.device,
        )

    if bool(((times < t_min) | (times > t_max)).any().detach().cpu()):
        raise ValueError(
            f"All requested times must lie in [{t_min}, {t_max}]. Got {times.tolist()}."
        )
    return torch.unique(times).sort().values


def parse_fractions(text: str) -> torch.Tensor:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 3:
        raise ValueError("--profile-fractions must contain at least three values.")
    if any(x < 0.0 or x > 1.0 for x in values):
        raise ValueError("Profile fractions must lie in [0, 1].")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError("Profile fractions must include 0 and 1.")
    values = sorted(set(values))
    return torch.tensor(values, dtype=DTYPE)


def make_eval_batch(params, times: torch.Tensor, x_eval: torch.Tensor) -> dict[str, torch.Tensor]:
    x_eval = x_eval.to(dtype=DTYPE, device=params.w.device)
    times = times.to(dtype=DTYPE, device=params.w.device)
    x_grid = torch.log(params.w)
    return {
        "t_eval": times,
        "t_scaled": scale_t(times, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": torch.exp(x_eval),
        "x_grid": x_grid,
        "x_grid_scaled": scale_x(x_grid, params),
        "w_grid": params.w,
    }


def evaluate_offgrid_values(
    *,
    model,
    params,
    x_eval: torch.Tensor,
    times: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Value-only off-grid evaluation.

    This intentionally uses the repository input builder and state reconstruction,
    but calls reconstruction with grid=False. evaluate_log_model_on_points() always
    uses grid=True and is therefore reserved here for exact params.w evaluations.
    """
    x_eval = x_eval.to(dtype=DTYPE, device=params.w.device)
    times = times.to(dtype=DTYPE, device=params.w.device)
    inputs = _make_model_inputs(
        scale_x(x_eval, params),
        scale_t(times, params),
    )
    raw_flat = model(inputs)

    n_time = times.numel()
    n_x = x_eval.numel()
    n_species = int(params.interaction.shape[0])
    if raw_flat.shape != (n_time * n_x, n_species):
        raise ValueError(
            f"Model returned {tuple(raw_flat.shape)}, expected {(n_time * n_x, n_species)}."
        )

    raw = (
        raw_flat.reshape(n_time, n_x, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )
    return reconstruct_from_model_output(
        raw,
        params,
        w=torch.exp(x_eval),
        grid=False,
    )


def expanded_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=target.device)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(0)
    return mask.expand_as(target)


def comparison_stats(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    mask_full = expanded_mask(mask, actual)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    selected = mask_full & finite

    n_active = int(mask_full.sum().detach().cpu())
    n_finite = int(selected.sum().detach().cpu())
    n_nonfinite = n_active - n_finite

    if n_finite == 0:
        return {
            "name": name,
            "passed": False,
            "n_active": n_active,
            "n_finite": n_finite,
            "n_nonfinite": n_nonfinite,
            "max_abs": math.nan,
            "mean_abs": math.nan,
            "p95_abs": math.nan,
            "max_relative": math.nan,
            "max_tolerance_ratio": math.inf,
        }

    a = actual[selected].detach()
    b = expected[selected].detach()
    abs_diff = torch.abs(a - b)
    scale = torch.maximum(torch.abs(a), torch.abs(b))
    tolerance = atol + rtol * scale
    tolerance_ratio = abs_diff / torch.clamp(tolerance, min=torch.finfo(abs_diff.dtype).tiny)
    relative = abs_diff / torch.clamp(scale, min=atol)

    passed = n_nonfinite == 0 and bool((abs_diff <= tolerance).all().detach().cpu())
    return {
        "name": name,
        "passed": passed,
        "n_active": n_active,
        "n_finite": n_finite,
        "n_nonfinite": n_nonfinite,
        "max_abs": float(abs_diff.max().cpu()),
        "mean_abs": float(abs_diff.mean().cpu()),
        "p95_abs": float(torch.quantile(abs_diff, 0.95).cpu()),
        "max_relative": float(relative.max().cpu()),
        "max_tolerance_ratio": float(tolerance_ratio.max().cpu()),
    }


def run_grid_offgrid_equivalence(
    *,
    model,
    params,
    times: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    x_grid = torch.log(params.w)
    x_scaled = scale_x(x_grid, params)
    t_scaled = scale_t(times, params)

    with torch.no_grad():
        grid = evaluate_log_model_on_points(
            model=model,
            x_scaled=x_scaled,
            t_scaled=t_scaled,
            params=params,
        )

    offgrid = evaluate_log_model_with_derivatives_at_eval(
        model=model,
        x_eval_scaled=x_scaled,
        t_scaled=t_scaled,
        w_eval=params.w,
        params=params,
    )

    active = active_grid_mask(params)
    checks = [
        comparison_stats(
            name="grid_vs_offgrid_log_N",
            actual=grid["log_N"],
            expected=offgrid["log_N_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="grid_vs_offgrid_N",
            actual=grid["N"],
            expected=offgrid["N_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="grid_vs_offgrid_log_U",
            actual=grid["log_U"],
            expected=offgrid["log_U_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="grid_vs_offgrid_U",
            actual=grid["U"],
            expected=offgrid["U_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="grid_vs_offgrid_log_S",
            actual=grid["log_S"],
            expected=offgrid["log_S_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="grid_vs_offgrid_S",
            actual=grid["S"],
            expected=offgrid["S_eval"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
    ]

    names = species_names(params, int(params.interaction.shape[0]))
    rows: list[dict[str, Any]] = []
    for ti, t in enumerate(times.detach().cpu().tolist()):
        for si, name in enumerate(names):
            for wi, w in enumerate(params.w.detach().cpu().tolist()):
                if not bool(active[si, wi].detach().cpu()):
                    continue
                rows.append(
                    {
                        "t": t,
                        "species_idx": si,
                        "species": name,
                        "weight_idx": wi,
                        "w": w,
                        "grid_N": float(grid["N"][ti, si, wi].detach().cpu()),
                        "offgrid_N": float(offgrid["N_eval"][ti, si, wi].detach().cpu()),
                        "N_difference": float(
                            (grid["N"][ti, si, wi] - offgrid["N_eval"][ti, si, wi])
                            .detach()
                            .cpu()
                        ),
                        "grid_U": float(grid["U"][ti, si, wi].detach().cpu()),
                        "offgrid_U": float(offgrid["U_eval"][ti, si, wi].detach().cpu()),
                        "U_difference": float(
                            (grid["U"][ti, si, wi] - offgrid["U_eval"][ti, si, wi])
                            .detach()
                            .cpu()
                        ),
                        "grid_S": float(grid["S"][ti, si, wi].detach().cpu()),
                        "offgrid_S": float(offgrid["S_eval"][ti, si, wi].detach().cpu()),
                        "S_difference": float(
                            (grid["S"][ti, si, wi] - offgrid["S_eval"][ti, si, wi])
                            .detach()
                            .cpu()
                        ),
                    }
                )
    return checks, pd.DataFrame(rows)


def run_state_scale_fd_check(
    *,
    params,
    fd_x_fraction: float,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x_grid = torch.log(params.w)
    dx = x_grid[1:] - x_grid[:-1]
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    h_x = fd_x_fraction * dx
    w_mid = torch.exp(x_mid)
    w_plus = torch.exp(x_mid + h_x)
    w_minus = torch.exp(x_mid - h_x)

    log_s_mid, _, analytic = interpolate_log_state_scale(params, w_mid)
    log_s_plus, _, _ = interpolate_log_state_scale(params, w_plus)
    log_s_minus, _, _ = interpolate_log_state_scale(params, w_minus)
    fd = (log_s_plus - log_s_minus) / (w_plus - w_minus)[None, :]

    active = active_eval_mask(w_mid, params)
    check = comparison_stats(
        name="state_scale_dlogS_dw_vs_finite_difference",
        actual=analytic,
        expected=fd,
        mask=active,
        atol=atol,
        rtol=rtol,
    )

    names = species_names(params, int(params.interaction.shape[0]))
    rows: list[dict[str, Any]] = []
    for si, name in enumerate(names):
        for interval in range(x_mid.numel()):
            if not bool(active[si, interval].detach().cpu()):
                continue
            rows.append(
                {
                    "species_idx": si,
                    "species": name,
                    "interval_idx": interval,
                    "x_left": float(x_grid[interval].detach().cpu()),
                    "x_mid": float(x_mid[interval].detach().cpu()),
                    "x_right": float(x_grid[interval + 1].detach().cpu()),
                    "w_mid": float(w_mid[interval].detach().cpu()),
                    "log_S_mid": float(log_s_mid[si, interval].detach().cpu()),
                    "dlogS_dw_analytic": float(analytic[si, interval].detach().cpu()),
                    "dlogS_dw_fd": float(fd[si, interval].detach().cpu()),
                    "difference": float((analytic[si, interval] - fd[si, interval]).detach().cpu()),
                }
            )
    return check, pd.DataFrame(rows)


def run_reconstructed_derivative_fd_check(
    *,
    model,
    params,
    times: torch.Tensor,
    fd_x_fraction: float,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    x_grid = torch.log(params.w)
    dx = x_grid[1:] - x_grid[:-1]
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    h_x = fd_x_fraction * dx
    x_plus = x_mid + h_x
    x_minus = x_mid - h_x
    w_mid = torch.exp(x_mid)
    w_plus = torch.exp(x_plus)
    w_minus = torch.exp(x_minus)

    analytic = evaluate_log_model_with_derivatives_at_eval(
        model=model,
        x_eval_scaled=scale_x(x_mid, params),
        t_scaled=scale_t(times, params),
        w_eval=w_mid,
        params=params,
    )

    with torch.no_grad():
        plus = evaluate_offgrid_values(
            model=model,
            params=params,
            x_eval=x_plus,
            times=times,
        )
        minus = evaluate_offgrid_values(
            model=model,
            params=params,
            x_eval=x_minus,
            times=times,
        )

    denominator = (w_plus - w_minus)[None, None, :]
    fd_log_n = (plus["log_N"] - minus["log_N"]) / denominator
    fd_n = (plus["N"] - minus["N"]) / denominator

    active = active_eval_mask(w_mid, params)
    checks = [
        comparison_stats(
            name="dlogN_dw_autograd_plus_scale_vs_finite_difference",
            actual=analytic["dlogN_dw"],
            expected=fd_log_n,
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="dN_dw_autograd_plus_scale_vs_finite_difference",
            actual=analytic["dN_dw"],
            expected=fd_n,
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
    ]

    names = species_names(params, int(params.interaction.shape[0]))
    rows: list[dict[str, Any]] = []
    for ti, t in enumerate(times.detach().cpu().tolist()):
        for si, name in enumerate(names):
            for interval in range(x_mid.numel()):
                if not bool(active[si, interval].detach().cpu()):
                    continue
                rows.append(
                    {
                        "t": t,
                        "species_idx": si,
                        "species": name,
                        "interval_idx": interval,
                        "w_mid": float(w_mid[interval].detach().cpu()),
                        "dlogN_dw_analytic": float(
                            analytic["dlogN_dw"][ti, si, interval].detach().cpu()
                        ),
                        "dlogN_dw_fd": float(fd_log_n[ti, si, interval].detach().cpu()),
                        "dN_dw_analytic": float(
                            analytic["dN_dw"][ti, si, interval].detach().cpu()
                        ),
                        "dN_dw_fd": float(fd_n[ti, si, interval].detach().cpu()),
                    }
                )
    return checks, pd.DataFrame(rows)


def run_residual_identity_checks(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    times: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    x_grid = torch.log(params.w)
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    batch = make_eval_batch(params, times, x_mid)

    state = compute_pde_state(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=False,
    )
    residual = compute_pde_residual_from_state(state)
    derivs = state["eval_derivs"]

    active = active_eval_mask(batch["w_eval"], params)
    checks = [
        comparison_stats(
            name="physical_residual_N_times_log_residual_vs_direct",
            actual=residual["residual"],
            expected=residual["residual_physical_check"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="physical_residual_vs_N_times_log_residual",
            actual=residual["residual"],
            expected=residual["N_eval"] * residual["residual_log"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
        comparison_stats(
            name="scaled_residual_vs_U_times_log_residual",
            actual=residual["residual_scaled"],
            expected=derivs["U_eval"] * residual["residual_log"],
            mask=active,
            atol=atol,
            rtol=rtol,
        ),
    ]

    names = species_names(params, int(params.interaction.shape[0]))
    rows: list[dict[str, Any]] = []
    for ti, t in enumerate(times.detach().cpu().tolist()):
        for si, name in enumerate(names):
            for interval in range(x_mid.numel()):
                if not bool(active[si, interval].detach().cpu()):
                    continue
                rows.append(
                    {
                        "t": t,
                        "species_idx": si,
                        "species": name,
                        "interval_idx": interval,
                        "w_mid": float(batch["w_eval"][interval].detach().cpu()),
                        "N": float(residual["N_eval"][ti, si, interval].detach().cpu()),
                        "U": float(derivs["U_eval"][ti, si, interval].detach().cpu()),
                        "residual_log": float(
                            residual["residual_log"][ti, si, interval].detach().cpu()
                        ),
                        "residual_physical_N_log": float(
                            residual["residual"][ti, si, interval].detach().cpu()
                        ),
                        "residual_physical_direct": float(
                            residual["residual_physical_check"][ti, si, interval]
                            .detach()
                            .cpu()
                        ),
                        "residual_scaled": float(
                            residual["residual_scaled"][ti, si, interval].detach().cpu()
                        ),
                    }
                )
    return checks, pd.DataFrame(rows)


def run_knot_side_diagnostics(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    times: torch.Tensor,
    epsilon_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if epsilon_fraction <= 0.0 or epsilon_fraction >= 0.25:
        raise ValueError("--knot-epsilon-fraction must lie in (0, 0.25).")

    x_grid = torch.log(params.w)
    interior = x_grid[1:-1]
    left_dx = interior - x_grid[:-2]
    right_dx = x_grid[2:] - interior
    h = epsilon_fraction * torch.minimum(left_dx, right_dx)

    x_left = interior - h
    x_exact = interior
    x_right = interior + h
    x_probe = torch.stack([x_left, x_exact, x_right], dim=1).reshape(-1)

    knot_idx = torch.arange(1, params.w.numel() - 1, device=params.w.device)
    knot_idx_probe = knot_idx[:, None].expand(-1, 3).reshape(-1)
    side_probe = ["left", "exact", "right"] * knot_idx.numel()

    batch = make_eval_batch(params, times, x_probe)
    state = compute_pde_state(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=False,
    )
    residual = compute_pde_residual_from_state(state)
    derivs = state["eval_derivs"]
    active = active_eval_mask(batch["w_eval"], params)

    names = species_names(params, int(params.interaction.shape[0]))
    rows: list[dict[str, Any]] = []
    for ti, t in enumerate(times.detach().cpu().tolist()):
        for si, name in enumerate(names):
            for pi in range(x_probe.numel()):
                if not bool(active[si, pi].detach().cpu()):
                    continue
                rows.append(
                    {
                        "t": t,
                        "species_idx": si,
                        "species": name,
                        "knot_idx": int(knot_idx_probe[pi].detach().cpu()),
                        "side": side_probe[pi],
                        "x": float(x_probe[pi].detach().cpu()),
                        "w": float(batch["w_eval"][pi].detach().cpu()),
                        "log_S": float(derivs["log_S_eval"][ti, si, pi].detach().cpu()),
                        "S": float(derivs["S_eval"][ti, si, pi].detach().cpu()),
                        "dlogS_dw": float(derivs["dlogS_dw"][ti, si, pi].detach().cpu()),
                        "log_U": float(derivs["log_U_eval"][ti, si, pi].detach().cpu()),
                        "U": float(derivs["U_eval"][ti, si, pi].detach().cpu()),
                        "log_N": float(derivs["log_N_eval"][ti, si, pi].detach().cpu()),
                        "N": float(derivs["N_eval"][ti, si, pi].detach().cpu()),
                        "dlogU_dw": float(derivs["dlogU_dw"][ti, si, pi].detach().cpu()),
                        "dlogN_dw": float(derivs["dlogN_dw"][ti, si, pi].detach().cpu()),
                        "dN_dw": float(derivs["dN_dw"][ti, si, pi].detach().cpu()),
                        "g": float(residual["g_eval"][ti, si, pi].detach().cpu()),
                        "dg_dw": float(residual["dg_dw"][ti, si, pi].detach().cpu()),
                        "mu": float(residual["mu_eval"][ti, si, pi].detach().cpu()),
                        "residual_log": float(
                            residual["residual_log"][ti, si, pi].detach().cpu()
                        ),
                        "residual_scaled": float(
                            residual["residual_scaled"][ti, si, pi].detach().cpu()
                        ),
                        "residual_physical": float(
                            residual["residual"][ti, si, pi].detach().cpu()
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "max_abs_dlogS_jump": math.nan,
            "median_abs_dlogS_jump": math.nan,
            "n_active_knot_sides": 0,
        }

    side = df.pivot_table(
        index=["t", "species_idx", "species", "knot_idx"],
        columns="side",
        values="dlogS_dw",
        aggfunc="first",
    ).reset_index()
    valid = side.dropna(subset=["left", "right"]).copy()
    valid["dlogS_jump"] = valid["right"] - valid["left"]

    summary = {
        "max_abs_dlogS_jump": (
            float(valid["dlogS_jump"].abs().max()) if not valid.empty else math.nan
        ),
        "median_abs_dlogS_jump": (
            float(valid["dlogS_jump"].abs().median()) if not valid.empty else math.nan
        ),
        "n_active_knot_sides": int(len(valid)),
    }
    return df, summary


def run_interval_profile(
    *,
    model,
    params,
    n_pp: torch.Tensor,
    times: torch.Tensor,
    fractions: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fractions = fractions.to(dtype=DTYPE, device=params.w.device)
    x_grid = torch.log(params.w)
    x_left = x_grid[:-1]
    dx = x_grid[1:] - x_grid[:-1]

    x_profile = (
        x_left[:, None] + dx[:, None] * fractions[None, :]
    ).reshape(-1)
    batch = make_eval_batch(params, times, x_profile)

    state = compute_pde_state(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=False,
    )
    residual = compute_pde_residual_from_state(state)
    derivs = state["eval_derivs"]

    n_time = times.numel()
    n_species = int(params.interaction.shape[0])
    n_interval = x_left.numel()
    n_fraction = fractions.numel()

    def shaped(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(n_time, n_species, n_interval, n_fraction)

    log_n = shaped(derivs["log_N_eval"])
    log_u = shaped(derivs["log_U_eval"])
    n_value = shaped(derivs["N_eval"])
    u_value = shaped(derivs["U_eval"])
    s_value = shaped(derivs["S_eval"])
    dlog_s = shaped(derivs["dlogS_dw"])
    residual_log = shaped(residual["residual_log"])
    residual_scaled = shaped(residual["residual_scaled"])

    f = fractions[None, None, None, :]
    log_n_linear = log_n[..., :1] * (1.0 - f) + log_n[..., -1:] * f
    log_u_linear = log_u[..., :1] * (1.0 - f) + log_u[..., -1:] * f
    log_n_deviation = log_n - log_n_linear
    log_u_deviation = log_u - log_u_linear

    active_flat = active_eval_mask(batch["w_eval"], params)
    active = active_flat.reshape(n_species, n_interval, n_fraction)
    names = species_names(params, n_species)

    rows: list[dict[str, Any]] = []
    for ti, t in enumerate(times.detach().cpu().tolist()):
        for si, name in enumerate(names):
            for interval in range(n_interval):
                for fi, fraction in enumerate(fractions.detach().cpu().tolist()):
                    if not bool(active[si, interval, fi].detach().cpu()):
                        continue
                    rows.append(
                        {
                            "t": t,
                            "species_idx": si,
                            "species": name,
                            "interval_idx": interval,
                            "fraction": fraction,
                            "x": float(
                                (x_left[interval] + dx[interval] * fractions[fi])
                                .detach()
                                .cpu()
                            ),
                            "w": float(
                                torch.exp(
                                    x_left[interval] + dx[interval] * fractions[fi]
                                )
                                .detach()
                                .cpu()
                            ),
                            "N": float(n_value[ti, si, interval, fi].detach().cpu()),
                            "U": float(u_value[ti, si, interval, fi].detach().cpu()),
                            "S": float(s_value[ti, si, interval, fi].detach().cpu()),
                            "log_N": float(log_n[ti, si, interval, fi].detach().cpu()),
                            "log_U": float(log_u[ti, si, interval, fi].detach().cpu()),
                            "log_N_linear_between_knots": float(
                                log_n_linear[ti, si, interval, fi].detach().cpu()
                            ),
                            "log_U_linear_between_knots": float(
                                log_u_linear[ti, si, interval, fi].detach().cpu()
                            ),
                            "log_N_deviation_from_knot_line": float(
                                log_n_deviation[ti, si, interval, fi].detach().cpu()
                            ),
                            "log_U_deviation_from_knot_line": float(
                                log_u_deviation[ti, si, interval, fi].detach().cpu()
                            ),
                            "dlogS_dw": float(dlog_s[ti, si, interval, fi].detach().cpu()),
                            "residual_log": float(
                                residual_log[ti, si, interval, fi].detach().cpu()
                            ),
                            "residual_scaled": float(
                                residual_scaled[ti, si, interval, fi].detach().cpu()
                            ),
                        }
                    )

    df = pd.DataFrame(rows)
    interior_df = df[(df["fraction"] > 0.0) & (df["fraction"] < 1.0)].copy()
    if interior_df.empty:
        fraction_summary = pd.DataFrame()
        summary = {
            "minimum_log_N_deviation_from_knot_line": math.nan,
            "fraction_interior_points_below_knot_line": math.nan,
            "median_log_N_deviation_from_knot_line": math.nan,
        }
    else:
        fraction_summary = (
            interior_df.groupby(["species_idx", "species", "fraction"], as_index=False)
            .agg(
                n=("log_N_deviation_from_knot_line", "size"),
                mean_log_N_deviation=("log_N_deviation_from_knot_line", "mean"),
                median_log_N_deviation=("log_N_deviation_from_knot_line", "median"),
                min_log_N_deviation=("log_N_deviation_from_knot_line", "min"),
                mean_log_U_deviation=("log_U_deviation_from_knot_line", "mean"),
                median_residual_log_abs=("residual_log", lambda x: x.abs().median()),
                median_residual_scaled_abs=("residual_scaled", lambda x: x.abs().median()),
            )
        )
        summary = {
            "minimum_log_N_deviation_from_knot_line": float(
                interior_df["log_N_deviation_from_knot_line"].min()
            ),
            "fraction_interior_points_below_knot_line": float(
                (interior_df["log_N_deviation_from_knot_line"] < 0.0).mean()
            ),
            "median_log_N_deviation_from_knot_line": float(
                interior_df["log_N_deviation_from_knot_line"].median()
            ),
        }

    return df, fraction_summary, summary


def run_observation_diagnostics(
    *,
    model,
    params,
    observations_path: str,
    default_cv: float,
    data_loss_eps: float,
    quadrature_points: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observation_batch = load_observation_csv(
        observations_path,
        params,
        default_cv=default_cv,
    )

    obs_times = torch.cat(
        [observation_batch["t_start"], observation_batch["t_end"]]
    )
    if quadrature_points > 1:
        quadrature = [
            torch.linspace(
                a,
                b,
                quadrature_points,
                dtype=obs_times.dtype,
                device=obs_times.device,
            )
            for a, b in zip(
                observation_batch["t_start"],
                observation_batch["t_end"],
            )
        ]
        obs_times = torch.cat([obs_times, torch.cat(quadrature)])

    t_grid = torch.unique(obs_times).sort().values
    with torch.no_grad():
        grid_eval = evaluate_log_model_on_points(
            model=model,
            x_scaled=scale_x(torch.log(params.w), params),
            t_scaled=scale_t(t_grid, params),
            params=params,
        )
        prediction = predict_observations(
            {"N_grid": grid_eval["N"], "t_grid": t_grid},
            observation_batch,
            params,
        )
        nll = lognormal_nll(
            prediction,
            observation_batch["value"],
            observation_batch["sd_log"],
            eps=data_loss_eps,
        )

    rows: list[dict[str, Any]] = []
    n_obs = observation_batch["value"].numel()
    for j in range(n_obs):
        rows.append(
            {
                "obs_type": observation_batch["obs_type"][j],
                "dataset": observation_batch["dataset"][j],
                "species_idx": int(observation_batch["species_idx"][j].detach().cpu()),
                "gear_idx": int(observation_batch["gear_idx"][j].detach().cpu()),
                "t_start": float(observation_batch["t_start"][j].detach().cpu()),
                "t_end": float(observation_batch["t_end"][j].detach().cpu()),
                "w_min_loaded": float(observation_batch["w_min"][j].detach().cpu()),
                "w_max_loaded": float(observation_batch["w_max"][j].detach().cpu()),
                "value": float(observation_batch["value"][j].detach().cpu()),
                "prediction": float(prediction[j].detach().cpu()),
                "sd_log": float(observation_batch["sd_log"][j].detach().cpu()),
                "log_residual": float(nll["log_residual"][j].detach().cpu()),
                "loss_contribution": float(
                    nll["loss_contribution"][j].detach().cpu()
                ),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n_observations": int(len(df)),
        "observation_types": sorted(df["obs_type"].unique().tolist()),
        "n_unique_species": int(df["species_idx"].nunique()),
        "n_unique_gears": int(df["gear_idx"].nunique()),
        "loaded_w_min_min": float(df["w_min_loaded"].min()),
        "loaded_w_max_max": float(df["w_max_loaded"].max()),
        "loss_data": float(nll["loss_data"].detach().cpu()),
        "median_abs_log_residual": float(df["log_residual"].abs().median()),
        "max_abs_log_residual": float(df["log_residual"].abs().max()),
    }
    return df, summary


def write_json(path: Path, value: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if torch.is_tensor(obj):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(value), indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "grid_offgrid_verification"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, params, n_init, n_pp, checkpoint, config = load_trained_model(
        input_dir=args.input_dir,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    times = make_times(args, params)
    fractions = parse_fractions(args.profile_fractions)

    critical_checks: list[dict[str, Any]] = []
    diagnostic_summary: dict[str, Any] = {}

    grid_checks, grid_df = run_grid_offgrid_equivalence(
        model=model,
        params=params,
        times=times,
        atol=args.atol,
        rtol=args.rtol,
    )
    critical_checks.extend(grid_checks)
    grid_df.to_csv(output_dir / "grid_vs_offgrid_exact_knots.csv", index=False)

    scale_fd_check, scale_fd_df = run_state_scale_fd_check(
        params=params,
        fd_x_fraction=args.fd_x_fraction,
        atol=args.fd_atol,
        rtol=args.fd_rtol,
    )
    critical_checks.append(scale_fd_check)
    scale_fd_df.to_csv(output_dir / "state_scale_derivative_fd.csv", index=False)

    derivative_checks, derivative_df = run_reconstructed_derivative_fd_check(
        model=model,
        params=params,
        times=times,
        fd_x_fraction=args.fd_x_fraction,
        atol=args.fd_atol,
        rtol=args.fd_rtol,
    )
    critical_checks.extend(derivative_checks)
    derivative_df.to_csv(
        output_dir / "reconstructed_state_derivative_fd.csv",
        index=False,
    )

    residual_checks, residual_df = run_residual_identity_checks(
        model=model,
        params=params,
        n_pp=n_pp,
        times=times,
        atol=args.atol,
        rtol=args.rtol,
    )
    critical_checks.extend(residual_checks)
    residual_df.to_csv(output_dir / "residual_identity_midpoints.csv", index=False)

    knot_df, knot_summary = run_knot_side_diagnostics(
        model=model,
        params=params,
        n_pp=n_pp,
        times=times,
        epsilon_fraction=args.knot_epsilon_fraction,
    )
    knot_df.to_csv(output_dir / "knot_left_exact_right_diagnostics.csv", index=False)
    diagnostic_summary["knot_sides"] = knot_summary

    profile_df, profile_fraction_df, profile_summary = run_interval_profile(
        model=model,
        params=params,
        n_pp=n_pp,
        times=times,
        fractions=fractions,
    )
    profile_df.to_csv(output_dir / "interval_phase_profile.csv", index=False)
    profile_fraction_df.to_csv(
        output_dir / "interval_phase_summary_by_species_fraction.csv",
        index=False,
    )
    diagnostic_summary["interval_profile"] = profile_summary

    if args.observations is not None:
        default_cv = float(config_value(config, "data_default_cv", 0.3))
        data_loss_eps = float(config_value(config, "data_loss_eps", 1e-30))
        quadrature_points = (
            args.data_time_quadrature_points
            if args.data_time_quadrature_points is not None
            else int(config_value(config, "data_time_quadrature_points", 1))
        )
        obs_df, obs_summary = run_observation_diagnostics(
            model=model,
            params=params,
            observations_path=args.observations,
            default_cv=default_cv,
            data_loss_eps=data_loss_eps,
            quadrature_points=quadrature_points,
        )
        obs_df.to_csv(output_dir / "observation_predictions_verified.csv", index=False)
        diagnostic_summary["observations"] = obs_summary

    checks_df = pd.DataFrame(critical_checks)
    checks_df.to_csv(output_dir / "critical_checks.csv", index=False)

    passed = bool(checks_df["passed"].all()) if not checks_df.empty else False
    summary = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint.get("step"),
        "input_dir": Path(args.input_dir).resolve(),
        "output_dir": output_dir,
        "device": str(params.w.device),
        "dtype": str(params.w.dtype),
        "state_parameterization": getattr(params, "state_parameterization", None),
        "state_scale_interpolation": getattr(
            params, "state_scale_interpolation", None
        ),
        "times": times,
        "profile_fractions": fractions,
        "critical_checks": critical_checks,
        "diagnostics": diagnostic_summary,
    }
    write_json(output_dir / "summary.json", summary)

    print(checks_df.to_string(index=False))
    print()
    print(f"Status: {summary['status']}")
    print(f"Diagnostics: {output_dir}")

    if passed or args.report_only:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
