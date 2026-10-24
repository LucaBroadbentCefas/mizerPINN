from __future__ import annotations

import argparse

import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask
from experiments.species_ensemble_pinn.known_state import KnownStateProvider
from experiments.species_ensemble_pinn.losses import compute_composite_loss
from experiments.species_ensemble_pinn.models import build_scalar_model
from experiments.species_ensemble_pinn.pde_state import compute_pde_state
from experiments.species_ensemble_pinn.residual import compute_pde_residual_from_state
from experiments.species_ensemble_pinn.residual_scale import grid_residual_scale, set_residual_scale_from_initial_condition
from experiments.species_ensemble_pinn.sampling import sample_pde_batch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-state-csv", required=True)
    parser.add_argument("--species-idx", type=int, default=0)
    args = parser.parse_args(argv)
    params, n_init, n_pp = load_mizer_inputs(args.input_dir)
    known = KnownStateProvider(args.known_state_csv, params, n_init, mode="dynamic-known")
    set_residual_scale_from_initial_condition(params, n_init)
    model = build_scalar_model().to(dtype=params.w.dtype, device=params.w.device)
    batch = sample_pde_batch(params, 64, 24, causal_n_chunks=32)
    state = compute_pde_state(model, batch, params, n_init, n_pp, known,
                              species_idx=args.species_idx)
    residual = compute_pde_residual_from_state(state, params, species_idx=args.species_idx)
    assert residual["residual_reference_scaled"].shape == (64, 1, 24)
    assert torch.allclose(residual["residual_physical"], residual["residual_physical_check"],
                          rtol=1e-8, atol=1e-10)
    log_scale, scale = grid_residual_scale(params)
    assert not log_scale.requires_grad
    assert torch.isfinite(scale).all() and (scale > 0).all()
    loss, _ = compute_composite_loss(
        state=state, residual_out=residual, params=params, species_idx=args.species_idx,
        batch=batch, lambda_pde=1, lambda_ic=1, lambda_bc=1,
        loss_weights={"pde": 1, "ic": 1, "bc": 0.1}, causal_n_chunks=32,
        causal_epsilon=1, eps=1e-30, bc_g_min=1e-12)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert active_grid_mask(params)[args.species_idx].any()
    print("PASS validate_tranche1_core")


if __name__ == "__main__":
    main()
