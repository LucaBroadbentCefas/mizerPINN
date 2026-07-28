from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from PINNmizer.io import load_mizer_inputs
from PINNmizer.mizer_grid_ops import mizer_operators, step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the true R_max is stationary under the discrete "
            "mizer finite-volume one-step update using an R-mizer target state."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--target-n",
        default=None,
        help=(
            "R-mizer abundance after exactly one exported dt. "
            "Defaults to <input-dir>/n_after_one_step.csv."
        ),
    )
    parser.add_argument(
        "--target-n-pp",
        default=None,
        help=(
            "R-mizer resource state after exactly one exported dt. "
            "Defaults to <input-dir>/n_pp_after_one_step.csv."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="validation/outputs/discrete_rmax_gradient_check.csv",
    )
    parser.add_argument("--finite-difference-step", type=float, default=1e-5)
    parser.add_argument("--gradient-zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--state-rtol", type=float, default=1e-7)
    parser.add_argument("--state-atol", type=float, default=1e-10)
    parser.add_argument("--fd-rtol", type=float, default=1e-4)
    parser.add_argument("--fd-atol", type=float, default=1e-9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_matrix(path: Path, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing target file: {path}")
    values = pd.read_csv(path).to_numpy().copy()
    return torch.as_tensor(values, dtype=dtype, device=device)


def read_vector(path: Path, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return read_matrix(path, dtype=dtype, device=device).reshape(-1)


def predict_one_step(
    log_rmax: torch.Tensor,
    *,
    params,
    n_before: torch.Tensor,
    n_pp_before: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    params.r_max = torch.exp(log_rmax)
    return step(
        n_pp=n_pp_before,
        n=n_before,
        params=params,
        dt=dt,
        effort=None,  # Uses the exported initial_effort/fishing parameters.
    )


def species_egg_loss(
    log_rmax: torch.Tensor,
    *,
    species_idx: int,
    params,
    n_before: torch.Tensor,
    n_pp_before: torch.Tensor,
    n_after_target: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, n_after_pred, _ = predict_one_step(
        log_rmax,
        params=params,
        n_before=n_before,
        n_pp_before=n_pp_before,
        dt=dt,
    )

    egg_idx = params.w_min_idx.to(torch.long) - 1
    egg = int(egg_idx[species_idx])

    pred = n_after_pred[species_idx, egg]
    target = n_after_target[species_idx, egg]

    if not torch.isfinite(target) or target <= 0.0:
        raise ValueError(
            f"Species {species_idx} has invalid target egg-bin abundance: {float(target)}"
        )

    residual = pred / target - 1.0
    return residual.square(), pred


def finite_difference_gradient(
    log_rmax_true: torch.Tensor,
    *,
    species_idx: int,
    step_size: float,
    params,
    n_before: torch.Tensor,
    n_pp_before: torch.Tensor,
    n_after_target: torch.Tensor,
    dt: torch.Tensor,
) -> float:
    def evaluate(offset: float) -> float:
        candidate = log_rmax_true.detach().clone()
        candidate[species_idx] += offset
        loss, _ = species_egg_loss(
            candidate,
            species_idx=species_idx,
            params=params,
            n_before=n_before,
            n_pp_before=n_pp_before,
            n_after_target=n_after_target,
            dt=dt,
        )
        return float(loss.detach())

    return (
        evaluate(step_size) - evaluate(-step_size)
    ) / (2.0 * step_size)


def implied_rmax_from_discrete_update(
    *,
    n_before_egg: float,
    n_after_egg: float,
    growth_egg: float,
    mortality_egg: float,
    dw_egg: float,
    dt: float,
    rdi: float,
) -> tuple[float, float, str]:
    """
    Mizer egg-bin update:

        n_next = (n_now + R_DD * dt / dw) /
                 (1 + growth * dt / dw + mortality * dt)

    Solve first for R_DD, then for R_max in:

        R_DD = R_DI / (1 + R_DI / R_max)
    """
    b = 1.0 + growth_egg * dt / dw_egg + mortality_egg * dt
    implied_rdd = (n_after_egg * b - n_before_egg) * dw_egg / dt

    if not math.isfinite(implied_rdd) or not math.isfinite(rdi):
        return implied_rdd, math.nan, "non-finite"
    if implied_rdd <= 0.0:
        return implied_rdd, math.nan, "non-positive implied R_DD"
    if implied_rdd >= rdi:
        return implied_rdd, math.inf, "no finite R_max: implied R_DD >= R_DI"

    implied_rmax = implied_rdd * rdi / (rdi - implied_rdd)
    return implied_rdd, implied_rmax, "finite"


def main() -> None:
    args = parse_args()

    dtype = torch.float64
    device = torch.device(args.device)
    input_dir = Path(args.input_dir)

    params, n_before, n_pp_before = load_mizer_inputs(
        input_dir,
        dtype=dtype,
        device=device,
    )

    if params.dt is None:
        raise ValueError(
            "The fixture has no dt.csv. The discrete test requires the exact "
            "mizer timestep used to generate the target state."
        )

    target_n_path = (
        Path(args.target_n)
        if args.target_n is not None
        else input_dir / "n_after_one_step.csv"
    )
    target_n_pp_path = (
        Path(args.target_n_pp)
        if args.target_n_pp is not None
        else input_dir / "n_pp_after_one_step.csv"
    )

    n_after_target = read_matrix(
        target_n_path,
        dtype=dtype,
        device=device,
    )
    n_pp_after_target = read_vector(
        target_n_pp_path,
        dtype=dtype,
        device=device,
    )

    if n_after_target.shape != n_before.shape:
        raise ValueError(
            f"Target N shape {tuple(n_after_target.shape)} does not match "
            f"fixture N shape {tuple(n_before.shape)}."
        )
    if n_pp_after_target.shape != n_pp_before.shape:
        raise ValueError(
            f"Target N_pp shape {tuple(n_pp_after_target.shape)} does not match "
            f"fixture N_pp shape {tuple(n_pp_before.shape)}."
        )

    true_rmax = params.r_max.detach().clone()
    true_log_rmax = torch.log(true_rmax)
    live_log_rmax = true_log_rmax.detach().clone().requires_grad_(True)

    n_pp_pred, n_after_pred, operators = predict_one_step(
        live_log_rmax,
        params=params,
        n_before=n_before,
        n_pp_before=n_pp_before,
        dt=params.dt,
    )

    n_matches = torch.allclose(
        n_after_pred.detach(),
        n_after_target,
        rtol=args.state_rtol,
        atol=args.state_atol,
    )
    n_pp_matches = torch.allclose(
        n_pp_pred.detach(),
        n_pp_after_target,
        rtol=args.state_rtol,
        atol=args.state_atol,
    )

    egg_idx = params.w_min_idx.to(torch.long) - 1
    n_species = int(n_before.shape[0])
    dt_float = float(params.dt.detach())

    rows: list[dict[str, object]] = []
    fd_failures: list[int] = []

    for species_idx in range(n_species):
        egg = int(egg_idx[species_idx])

        loss, pred_egg = species_egg_loss(
            live_log_rmax,
            species_idx=species_idx,
            params=params,
            n_before=n_before,
            n_pp_before=n_pp_before,
            n_after_target=n_after_target,
            dt=params.dt,
        )

        gradient = torch.autograd.grad(
            loss,
            live_log_rmax,
            retain_graph=True,
            allow_unused=False,
        )[0]

        gradient_value = float(gradient[species_idx].detach())
        off_species = torch.cat(
            [gradient[:species_idx], gradient[species_idx + 1 :]]
        )
        max_cross_gradient = (
            float(off_species.abs().max().detach())
            if off_species.numel() > 0
            else 0.0
        )

        fd_gradient = finite_difference_gradient(
            true_log_rmax,
            species_idx=species_idx,
            step_size=args.finite_difference_step,
            params=params,
            n_before=n_before,
            n_pp_before=n_pp_before,
            n_after_target=n_after_target,
            dt=params.dt,
        )

        fd_matches = math.isclose(
            gradient_value,
            fd_gradient,
            rel_tol=args.fd_rtol,
            abs_tol=args.fd_atol,
        )
        if not fd_matches:
            fd_failures.append(species_idx)

        n_before_egg = float(n_before[species_idx, egg].detach())
        n_after_egg = float(n_after_target[species_idx, egg].detach())
        growth_egg = float(operators["e_growth"][species_idx, egg].detach())
        mortality_egg = float(operators["mort"][species_idx, egg].detach())
        dw_egg = float(params.dw[egg].detach())
        rdi = float(operators["rdi"][species_idx].detach())

        implied_rdd, implied_rmax, implied_status = (
            implied_rmax_from_discrete_update(
                n_before_egg=n_before_egg,
                n_after_egg=n_after_egg,
                growth_egg=growth_egg,
                mortality_egg=mortality_egg,
                dw_egg=dw_egg,
                dt=dt_float,
                rdi=rdi,
            )
        )

        implied_log_rmax = (
            math.log(implied_rmax)
            if math.isfinite(implied_rmax) and implied_rmax > 0.0
            else implied_rmax
        )

        target_egg = float(n_after_target[species_idx, egg].detach())
        prediction_relative_error = (
            float(pred_egg.detach()) / target_egg - 1.0
        )

        rows.append(
            {
                "species_idx": species_idx,
                "egg_idx": egg,
                "true_rmax": float(true_rmax[species_idx]),
                "true_log_rmax": float(true_log_rmax[species_idx]),
                "n_before_egg": n_before_egg,
                "n_after_target_egg": n_after_egg,
                "n_after_predicted_egg": float(pred_egg.detach()),
                "egg_prediction_relative_error": prediction_relative_error,
                "rdi": rdi,
                "rdd_at_true_rmax": float(operators["rdd"][species_idx].detach()),
                "implied_rdd_from_r_target": implied_rdd,
                "implied_rmax_from_r_target": implied_rmax,
                "implied_log_rmax_from_r_target": implied_log_rmax,
                "implied_log_rmax_minus_truth": (
                    implied_log_rmax - float(true_log_rmax[species_idx])
                    if math.isfinite(implied_log_rmax)
                    else implied_log_rmax
                ),
                "implied_status": implied_status,
                "egg_loss_at_true_rmax": float(loss.detach()),
                "d_egg_loss_d_log_rmax_autograd": gradient_value,
                "d_egg_loss_d_log_rmax_finite_difference": fd_gradient,
                "finite_difference_matches": fd_matches,
                "truth_is_stationary": (
                    abs(gradient_value) <= args.gradient_zero_tolerance
                ),
                "max_cross_species_gradient": max_cross_gradient,
            }
        )

    result = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(
        result[
            [
                "species_idx",
                "true_log_rmax",
                "implied_log_rmax_from_r_target",
                "implied_log_rmax_minus_truth",
                "egg_prediction_relative_error",
                "egg_loss_at_true_rmax",
                "d_egg_loss_d_log_rmax_autograd",
                "d_egg_loss_d_log_rmax_finite_difference",
                "truth_is_stationary",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Full R-mizer N state matches Python step: {n_matches}")
    print(f"R-mizer resource state matches Python step: {n_pp_matches}")
    print(
        "True R_max stationary under discrete egg update for "
        f"{int(result['truth_is_stationary'].sum())}/{n_species} species."
    )
    print(
        "Autograd agrees with finite differences for "
        f"{int(result['finite_difference_matches'].sum())}/{n_species} species."
    )
    print(f"Saved: {output_path}")

    if fd_failures:
        raise AssertionError(
            "Autograd and finite-difference gradients disagree for species "
            f"{fd_failures}."
        )


if __name__ == "__main__":
    main()
