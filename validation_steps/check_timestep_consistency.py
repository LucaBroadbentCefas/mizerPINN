from __future__ import annotations

import argparse

import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.pinn.models import MLP
from PINNmizer.timestep_consistency import compute_timestep_consistency_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="validation/fixtures/pde_single_species")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dt", type=float, default=1e-2)
    p.add_argument("--n-times", type=int, default=1)
    p.add_argument("--loss-form", choices=["physical", "log", "relative"], default="physical")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params, n_init, n_pp = load_mizer_inputs(args.input_dir, dtype=torch.float64, device=args.device)
    n_species = params.interaction.shape[0]
    model = MLP(in_dim=2, out_dim=n_species, hidden_width=64, hidden_layers=3).to(dtype=torch.float64, device=params.w.device)

    t_min = float(params.t_min)
    t_max = float(params.t_max)
    t0_max = t_max - args.dt
    if t0_max < t_min:
        raise ValueError("No valid t0: dt is larger than time domain")
    t0 = torch.linspace(t_min, t0_max, args.n_times, dtype=params.w.dtype, device=params.w.device)

    for detach in [True, False]:
        model.zero_grad(set_to_none=True)
        loss, diag = compute_timestep_consistency_loss(
            model=model,
            params=params,
            n_pp=n_pp,
            t0=t0,
            dt=args.dt,
            loss_form=args.loss_form,
            detach_step_target=detach,
            species_idx=0,
        )
        assert loss.ndim == 0
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        print(f"detach={detach} loss={float(loss.detach().cpu()):.6e} grad_ok={has_grad}")
        print(
            "shape N0/N1/N1step=",
            tuple(diag["N0_pred"].shape),
            tuple(diag["N1_pred"].shape),
            tuple(diag["N1_step"].shape),
        )
        print(
            "abs means:",
            float(diag["physical_abs_mean"].detach().cpu()),
            float(diag["log_abs_mean"].detach().cpu()),
            float(diag["relative_abs_mean"].detach().cpu()),
        )


if __name__ == "__main__":
    main()
