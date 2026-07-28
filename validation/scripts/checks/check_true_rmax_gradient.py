from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.biology.growth import compute_growth_direct_at_eval
from PINNmizer.biology.recruitment import compute_recruitment_direct_from_growth_grid
from PINNmizer.io import load_mizer_inputs
from PINNmizer.mizer_grid_ops import mizer_operators, rdi as rdi_grid_weights
from PINNmizer.params import active_grid_mask
from PINNmizer.pinn.losses import compute_recruitment_boundary_loss_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the R_max values loaded from a synthetic mizer fixture "
            "are stationary under the current PINN recruitment-boundary objective."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--loss-form",
        choices=["log", "physical", "relative"],
        default="relative",
    )
    parser.add_argument("--bc-g-min", type=float, default=1e-12)
    parser.add_argument("--finite-difference-step", type=float, default=1e-5)
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--fd-rtol", type=float, default=1e-4)
    parser.add_argument("--fd-atol", type=float, default=1e-8)
    parser.add_argument(
        "--output-csv",
        default="validation/outputs/true_rmax_gradient_check.csv",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _with_time_axis(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.unsqueeze(0) for name, value in values.items()}


def _boundary_state(
    *,
    n_grid: torch.Tensor,
    growth_grid: dict[str, torch.Tensor],
    recruitment: dict[str, torch.Tensor],
) -> dict[str, object]:
    tiny = torch.finfo(n_grid.dtype).tiny
    return {
        "N_grid": n_grid.unsqueeze(0),
        "log_N_grid": torch.log(torch.clamp(n_grid, min=tiny)).unsqueeze(0),
        "growth_grid": _with_time_axis(growth_grid),
        "recruitment": _with_time_axis(recruitment),
    }


def _manual_species_loss(
    log_rmax: torch.Tensor,
    *,
    rdi: torch.Tensor,
    g_left: torch.Tensor,
    n_left: torch.Tensor,
    loss_form: str,
) -> torch.Tensor:
    rmax = torch.exp(log_rmax)
    rdd = rdi / (1.0 + rdi / rmax)

    if loss_form == "log":
        residual = torch.log(n_left) - (torch.log(rdd) - torch.log(g_left))
    elif loss_form == "physical":
        residual = n_left - rdd / g_left
    else:
        residual = n_left * g_left / rdd - 1.0

    return residual.square()


def _finite_difference_gradient(
    log_rmax_value: float,
    *,
    rdi: float,
    g_left: float,
    n_left: float,
    loss_form: str,
    step: float,
    dtype: torch.dtype,
) -> float:
    def evaluate(value: float) -> float:
        loss = _manual_species_loss(
            torch.tensor(value, dtype=dtype),
            rdi=torch.tensor(rdi, dtype=dtype),
            g_left=torch.tensor(g_left, dtype=dtype),
            n_left=torch.tensor(n_left, dtype=dtype),
            loss_form=loss_form,
        )
        return float(loss)

    return (
        evaluate(log_rmax_value + step)
        - evaluate(log_rmax_value - step)
    ) / (2.0 * step)


def _implied_rmax(rdi: float, boundary_flux: float) -> tuple[float, str]:
    """
    Solve boundary_flux = rdi / (1 + rdi / rmax) for rmax.

    A finite positive solution exists only when 0 < boundary_flux < rdi.
    """
    if not math.isfinite(rdi) or not math.isfinite(boundary_flux):
        return math.nan, "non-finite"
    if boundary_flux <= 0.0:
        return math.nan, "non-positive boundary flux"
    if boundary_flux >= rdi:
        return math.inf, "no finite solution: boundary flux >= R_DI"

    return boundary_flux * rdi / (rdi - boundary_flux), "finite"


def main() -> None:
    args = parse_args()

    dtype = torch.float64
    params, n_init, n_pp = load_mizer_inputs(
        args.input_dir,
        dtype=dtype,
        device=args.device,
    )

    active = active_grid_mask(params).to(dtype=dtype, device=n_init.device)
    n_state = n_init * active

    true_rmax = params.r_max.detach().clone()
    true_log_rmax = torch.log(true_rmax)

    # Direct leaf parameter: this tests dL / d log(R_max), independently of
    # the bounded-logit parameterisation used during optimisation.
    log_rmax = true_log_rmax.detach().clone().requires_grad_(True)
    params.r_max = torch.exp(log_rmax)

    # Current continuous PINN biology and recruitment operator.
    growth_direct = compute_growth_direct_at_eval(
        n_pp=n_pp,
        n_grid=n_state,
        w_eval=params.w,
        params=params,
    )
    recruitment_direct = compute_recruitment_direct_from_growth_grid(
        N_grid=n_state,
        params=params,
        growth_grid=growth_direct,
    )
    state = _boundary_state(
        n_grid=n_state,
        growth_grid=growth_direct,
        recruitment=recruitment_direct,
    )

    # Two reference calculations:
    # 1. Same direct e_repro, but mizer's exported dw quadrature.
    # 2. Full fixed-grid FFT/mizer operator.
    rdi_direct_with_dw = rdi_grid_weights(
        growth_direct["e_repro_eval"],
        n_state,
        params,
    )
    grid_ops = mizer_operators(n_pp, n_state, params)

    egg_idx = params.w_min_idx.to(torch.long) - 1
    species_count = int(n_state.shape[0])

    # Algebraic control state. It satisfies the current PINN boundary equation
    # exactly while keeping R_DI and growth fixed.
    n_control = n_state.detach().clone()
    direct_rdd_detached = recruitment_direct["rdd_flux"].detach()
    direct_g_detached = growth_direct["e_growth_eval"].detach()

    for species_idx in range(species_count):
        egg = int(egg_idx[species_idx])
        g_left = direct_g_detached[species_idx, egg]
        rdd = direct_rdd_detached[species_idx]
        if (
            torch.isfinite(g_left)
            and torch.isfinite(rdd)
            and g_left > args.bc_g_min
            and rdd > 0.0
        ):
            n_control[species_idx, egg] = rdd / g_left

    control_state = _boundary_state(
        n_grid=n_control,
        growth_grid=growth_direct,
        recruitment=recruitment_direct,
    )

    rows: list[dict[str, object]] = []
    fd_failures: list[int] = []
    control_failures: list[int] = []

    for species_idx in range(species_count):
        egg = int(egg_idx[species_idx])

        out = compute_recruitment_boundary_loss_from_state(
            state,
            params,
            species_idx=species_idx,
            loss_form=args.loss_form,
            bc_g_min=args.bc_g_min,
            boundary_target_gradient_mode="rmax-only",
        )
        grad_vector = torch.autograd.grad(
            out["loss_bc"],
            log_rmax,
            retain_graph=True,
            allow_unused=False,
        )[0]

        control_out = compute_recruitment_boundary_loss_from_state(
            control_state,
            params,
            species_idx=species_idx,
            loss_form=args.loss_form,
            bc_g_min=args.bc_g_min,
            boundary_target_gradient_mode="rmax-only",
        )
        control_grad_vector = torch.autograd.grad(
            control_out["loss_bc"],
            log_rmax,
            retain_graph=True,
            allow_unused=False,
        )[0]

        n_left = float(n_state[species_idx, egg].detach())
        g_left_direct = float(
            growth_direct["e_growth_eval"][species_idx, egg].detach()
        )
        g_left_grid = float(grid_ops["e_growth"][species_idx, egg].detach())

        rdi_trapz = float(recruitment_direct["rdi_flux"][species_idx].detach())
        rdi_dw_same_growth = float(rdi_direct_with_dw[species_idx].detach())
        rdi_fft_grid = float(grid_ops["rdi"][species_idx].detach())

        rdd_direct = float(recruitment_direct["rdd_flux"][species_idx].detach())
        rdd_fft_grid = float(grid_ops["rdd"][species_idx].detach())

        boundary_flux_direct = n_left * g_left_direct
        boundary_flux_grid = n_left * g_left_grid

        implied_rmax, implied_status = _implied_rmax(
            rdi_trapz,
            boundary_flux_direct,
        )
        implied_log_rmax = (
            math.log(implied_rmax)
            if math.isfinite(implied_rmax) and implied_rmax > 0.0
            else implied_rmax
        )

        autograd_value = float(grad_vector[species_idx].detach())
        control_grad_value = float(control_grad_vector[species_idx].detach())

        finite_difference_value = _finite_difference_gradient(
            float(true_log_rmax[species_idx]),
            rdi=rdi_trapz,
            g_left=g_left_direct,
            n_left=n_left,
            loss_form=args.loss_form,
            step=args.finite_difference_step,
            dtype=dtype,
        )

        fd_matches = math.isclose(
            autograd_value,
            finite_difference_value,
            rel_tol=args.fd_rtol,
            abs_tol=args.fd_atol,
        )
        truth_is_stationary = abs(autograd_value) <= args.zero_tolerance
        control_is_stationary = abs(control_grad_value) <= args.zero_tolerance

        if not fd_matches:
            fd_failures.append(species_idx)
        if not control_is_stationary:
            control_failures.append(species_idx)

        off_species = torch.cat(
            [grad_vector[:species_idx], grad_vector[species_idx + 1 :]]
        )
        max_cross_species_gradient = (
            float(off_species.abs().max().detach())
            if off_species.numel() > 0
            else 0.0
        )

        rows.append(
            {
                "species_idx": species_idx,
                "egg_idx": egg,
                "true_rmax": float(true_rmax[species_idx]),
                "true_log_rmax": float(true_log_rmax[species_idx]),
                "n_left": n_left,
                "g_left_direct": g_left_direct,
                "g_left_grid": g_left_grid,
                "boundary_flux_direct_gN": boundary_flux_direct,
                "boundary_flux_grid_gN": boundary_flux_grid,
                "rdi_direct_trapz": rdi_trapz,
                "rdi_direct_dw_same_growth": rdi_dw_same_growth,
                "rdi_mizer_fft_grid": rdi_fft_grid,
                "trapz_to_dw_rdi_ratio": (
                    rdi_trapz / rdi_dw_same_growth
                    if rdi_dw_same_growth != 0.0
                    else math.nan
                ),
                "direct_to_fft_rdi_ratio": (
                    rdi_trapz / rdi_fft_grid
                    if rdi_fft_grid != 0.0
                    else math.nan
                ),
                "rdd_direct_at_truth": rdd_direct,
                "rdd_mizer_grid_at_truth": rdd_fft_grid,
                "direct_flux_ratio_gN_over_RDD": (
                    boundary_flux_direct / rdd_direct
                    if rdd_direct != 0.0
                    else math.nan
                ),
                "grid_flux_ratio_gN_over_RDD": (
                    boundary_flux_grid / rdd_fft_grid
                    if rdd_fft_grid != 0.0
                    else math.nan
                ),
                "implied_rmax_under_current_bc": implied_rmax,
                "implied_log_rmax_under_current_bc": implied_log_rmax,
                "implied_rmax_status": implied_status,
                "bc_loss_at_true_rmax": float(out["loss_bc"].detach()),
                "d_bc_loss_d_log_rmax_autograd": autograd_value,
                "d_bc_loss_d_log_rmax_finite_difference": finite_difference_value,
                "finite_difference_matches": fd_matches,
                "truth_is_stationary": truth_is_stationary,
                "control_bc_loss": float(control_out["loss_bc"].detach()),
                "control_gradient": control_grad_value,
                "control_is_stationary": control_is_stationary,
                "max_cross_species_gradient": max_cross_species_gradient,
                "bc_valid_fraction": float(out["bc_valid_fraction"].detach()),
            }
        )

    result = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    display_columns = [
        "species_idx",
        "true_log_rmax",
        "implied_log_rmax_under_current_bc",
        "direct_flux_ratio_gN_over_RDD",
        "trapz_to_dw_rdi_ratio",
        "direct_to_fft_rdi_ratio",
        "bc_loss_at_true_rmax",
        "d_bc_loss_d_log_rmax_autograd",
        "d_bc_loss_d_log_rmax_finite_difference",
        "truth_is_stationary",
        "control_is_stationary",
    ]

    print(result[display_columns].to_string(index=False))
    print()
    print(f"Saved: {output_path}")
    print(
        "Truth stationary for "
        f"{int(result['truth_is_stationary'].sum())}/{species_count} species."
    )
    print(
        "Finite-difference agreement for "
        f"{int(result['finite_difference_matches'].sum())}/{species_count} species."
    )
    print(
        "Self-consistent boundary control stationary for "
        f"{int(result['control_is_stationary'].sum())}/{species_count} species."
    )

    if fd_failures:
        raise AssertionError(
            "Autograd and finite-difference gradients disagree for species "
            f"{fd_failures}."
        )
    if control_failures:
        raise AssertionError(
            "The algebraically self-consistent boundary control did not have "
            f"zero gradient for species {control_failures}."
        )


if __name__ == "__main__":
    main()
