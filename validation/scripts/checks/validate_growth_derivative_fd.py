#checking d(g)/d(w) works.
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.biology.growth import compute_growth_direct_at_eval


OUT_DIR = Path("validation/fixtures/growth_derivative_fd")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_mat(x: torch.Tensor, name: str) -> None:
    pd.DataFrame(x.detach().cpu().numpy()).to_csv(
        OUT_DIR / f"{name}.csv",
        index=False,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    return torch.abs(a - b) / torch.clamp(
        torch.maximum(torch.abs(a), torch.abs(b)),
        min=eps,
    )


def summarise(name: str, err: torch.Tensor, eps_scale: float) -> dict:
    x = err[torch.isfinite(err)].reshape(-1)

    q = torch.quantile(
        x,
        torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=x.dtype, device=x.device),
    )

    return {
        "quantity": name,
        "eps_scale": eps_scale,
        "q0": float(q[0].cpu()),
        "q50": float(q[1].cpu()),
        "q90": float(q[2].cpu()),
        "q99": float(q[3].cpu()),
        "q100": float(q[4].cpu()),
    }


def central_fd(
    params,
    n,
    n_pp,
    w_eval: torch.Tensor,
    quantity: str,
    eps_scale: float,
) -> torch.Tensor:
    eps = w_eval * eps_scale

    plus = compute_growth_direct_at_eval(
        n_pp=n_pp,
        n_grid=n,
        w_eval=w_eval + eps,
        params=params,
    )

    minus = compute_growth_direct_at_eval(
        n_pp=n_pp,
        n_grid=n,
        w_eval=w_eval - eps,
        params=params,
    )

    return (plus[quantity] - minus[quantity]) / (2.0 * eps[None, :])


def main() -> None:
    params, n, n_pp = load_mizer_inputs(
        "validation/fixtures/pde_single_species",
        dtype=torch.float64,
        device="cpu",
    )

    # Use geometric midpoints, not params.w directly.
    # Reason: at exact grid points, phi has a hard active switch at w_pred == w_prey.
    # Midpoints avoid testing exactly on that discontinuity.
    w_eval = torch.sqrt(params.w[:-1] * params.w[1:])

    base = compute_growth_direct_at_eval(
        n_pp=n_pp,
        n_grid=n,
        w_eval=w_eval,
        params=params,
    )

    derivative_pairs = {
        "encounter_eval": "dencounter_dw",
        "feeding_eval": "dfeeding_dw",
        "erepog_eval": "derepog_dw",
        "pos_erepog": "dpos_erepog_dw",
        "e_repro_eval": "de_repro_dw",
        "e_growth_eval": "dg_dw",
    }

    rows = []

    for eps_scale in [1e-3, 1e-4, 1e-5, 1e-6]:
        for quantity, derivative_name in derivative_pairs.items():
            fd = central_fd(
                params=params,
                n=n,
                n_pp=n_pp,
                w_eval=w_eval,
                quantity=quantity,
                eps_scale=eps_scale,
            )

            manual = base[derivative_name]

            err = rel_err(manual, fd)

            rows.append(
                summarise(
                    name=derivative_name,
                    err=err,
                    eps_scale=eps_scale,
                )
            )

            if eps_scale == 1e-5:
                write_mat(fd, f"{derivative_name}_fd_eps_1e5")
                write_mat(manual, f"{derivative_name}_manual")
                write_mat(err, f"{derivative_name}_rel_err_eps_1e5")

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "finite_difference_summary.csv",
        index=False,
    )

    print(f"Wrote {OUT_DIR}/finite_difference_summary.csv")


if __name__ == "__main__":
    main()
