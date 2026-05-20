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
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--n-pairs", type=int, default=1)
    p.add_argument("--loss-form", choices=["physical", "log", "relative"], default="physical")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params, _n_init, n_pp = load_mizer_inputs(args.input_dir, dtype=torch.float64, device=args.device)
    n_species = params.interaction.shape[0]
    model = MLP(in_dim=2, out_dim=n_species, hidden_width=args.hidden_width, hidden_layers=args.hidden_layers).to(dtype=torch.float64, device=params.w.device)

    dt_val = args.dt if args.dt is not None else float(getattr(params, "dt"))
    t_min = torch.as_tensor(params.t_min, dtype=params.w.dtype, device=params.w.device)
    t_max = torch.as_tensor(params.t_max, dtype=params.w.dtype, device=params.w.device)
    t0 = t_min + torch.linspace(0.1, 0.8, steps=args.n_pairs, dtype=params.w.dtype, device=params.w.device) * ((t_max - dt_val) - t_min)

    for detach in (True, False):
        model.zero_grad(set_to_none=True)
        loss, diag = compute_timestep_consistency_loss(
            model=model,
            params=params,
            n_pp=n_pp,
            t0=t0,
            dt=dt_val,
            loss_form=args.loss_form,
            detach_step_target=detach,
        )
        loss.backward()
        print(f"detach_step_target={detach} loss={float(loss.detach().cpu()):.6e}")
        print(f"shape N0={tuple(diag['N0_pred'].shape)} N1={tuple(diag['N1_pred'].shape)} N1_step={tuple(diag['N1_step'].shape)}")
        print(f"phys_abs_mean={float(diag['physical_abs_mean'].detach().cpu()):.6e} phys_abs_max={float(diag['physical_abs_max'].detach().cpu()):.6e}")
        print(f"nan_loss={bool(diag['has_nan_loss'])} inf_loss={bool(diag['has_inf_loss'])}")


if __name__ == "__main__":
    main()
