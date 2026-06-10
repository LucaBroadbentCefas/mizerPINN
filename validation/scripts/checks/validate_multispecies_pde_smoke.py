from __future__ import annotations

import argparse

import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import validate_params_shapes
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.losses import compute_pde_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="validation/fixtures/pde_multispecies")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-single-species", action="store_true")
    parser.add_argument("--n-time", type=int, default=3)
    parser.add_argument("--n-eval", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params, n_init, n_pp = load_mizer_inputs(args.input_dir, dtype=torch.float64, device=args.device)
    validate_params_shapes(params)

    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    if n_species < 2 and not args.allow_single_species:
        raise AssertionError(f"Expected at least two species, got {n_species}.")

    model = build_pinn_model(
        model_arch="mlp",
        in_dim=2,
        out_dim=n_species,
        hidden_width=8,
        hidden_layers=1,
    ).to(dtype=torch.float64, device=params.w.device)

    batch = sample_pde_batch(params=params, n_time=args.n_time, n_eval=args.n_eval)
    loss, out = compute_pde_loss(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        residual_form="log",
        n_init=n_init,
        lambda_pde=1.0,
        lambda_ic=1.0,
        lambda_bc=0.0,
        species_idx=None,
    )

    assert out["residual_log"].shape == (args.n_time, n_species, args.n_eval), out["residual_log"].shape
    assert out["N_grid"].shape == (args.n_time, n_species, n_w), out["N_grid"].shape
    assert torch.isfinite(loss), loss

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        raise AssertionError("No model gradients were produced.")
    if not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError("Non-finite model gradient detected.")

    print("multi-species PDE smoke passed")
    print(f"n_species={n_species} n_w={n_w}")
    print(f"residual_log_shape={tuple(out['residual_log'].shape)}")
    print(f"N_grid_shape={tuple(out['N_grid'].shape)}")
    print(f"loss={float(loss.detach().cpu()):.6e}")


if __name__ == "__main__":
    main()
