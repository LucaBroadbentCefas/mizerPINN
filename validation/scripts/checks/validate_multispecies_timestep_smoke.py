from __future__ import annotations

import argparse

import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask, validate_params_shapes
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.timestep_consistency_multispecies import compute_timestep_consistency_loss_multispecies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="validation/fixtures/pde_multispecies")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-single-species", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params, _n_init, n_pp = load_mizer_inputs(args.input_dir, dtype=torch.float64, device=args.device)
    validate_params_shapes(params)

    n_species = params.interaction.shape[0]
    n_w = params.w.numel()
    if n_species < 2 and not args.allow_single_species:
        raise AssertionError(f"Expected at least two species, got {n_species}.")

    active = active_grid_mask(params)
    if active.shape != (n_species, n_w):
        raise AssertionError(f"Bad active mask shape: {tuple(active.shape)}")
    if not bool(active.any(dim=1).all().detach().cpu()):
        raise AssertionError("At least one species has no active grid cells.")

    model = build_pinn_model(
        model_arch="mlp",
        in_dim=2,
        out_dim=n_species,
        hidden_width=8,
        hidden_layers=1,
    ).to(dtype=torch.float64, device=params.w.device)

    dt = params.dt if params.dt is not None else torch.as_tensor(0.01, dtype=params.w.dtype, device=params.w.device)
    t_min = torch.as_tensor(params.t_min, dtype=params.w.dtype, device=params.w.device)
    t_max = torch.as_tensor(params.t_max, dtype=params.w.dtype, device=params.w.device)
    t0_hi = t_max - torch.as_tensor(dt, dtype=params.w.dtype, device=params.w.device)
    if not bool((t0_hi > t_min).detach().cpu()):
        raise AssertionError("Fixture time range is too short for one timestep.")
    t0 = torch.linspace(t_min, t0_hi, args.n_pairs, dtype=params.w.dtype, device=params.w.device)

    loss, out = compute_timestep_consistency_loss_multispecies(
        model=model,
        params=params,
        n_pp=n_pp,
        t0=t0,
        dt=dt,
        loss_form="physical",
    )

    expected = (args.n_pairs, n_species, n_w)
    for name in ["N0_pred", "N1_pred", "N1_step", "residual_timestep_physical"]:
        assert out[name].shape == expected, (name, out[name].shape, expected)
    assert torch.isfinite(loss), loss
    assert float(out["active_count"].detach().cpu()) == float(active.sum().detach().cpu())

    print("multi-species timestep smoke passed")
    print(f"n_species={n_species} n_w={n_w} active_count={int(active.sum().detach().cpu())}")
    print(f"N0_pred_shape={tuple(out['N0_pred'].shape)}")
    print(f"loss={float(loss.detach().cpu()):.6e}")


if __name__ == "__main__":
    main()
