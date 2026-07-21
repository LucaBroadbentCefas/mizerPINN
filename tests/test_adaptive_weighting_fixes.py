from types import SimpleNamespace

import torch

from PINNmizer.pinn.losses import compute_recruitment_boundary_loss_from_state
from PINNmizer.training.weighting import (
    rescale_fixed_calibration_batch,
    update_expert_gradient_norm_weights_,
)


def test_relative_bc_detaches_biological_target():
    N = torch.tensor([[[4.0]]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[[2.0]]], dtype=torch.float64, requires_grad=True)
    recruitment = torch.tensor([[4.0]], dtype=torch.float64, requires_grad=True)
    state = {
        "log_N_grid": torch.log(N),
        "N_grid": N,
        "growth_grid": {"e_growth_eval": g},
        "recruitment": {"rdd_flux": recruitment},
    }
    params = SimpleNamespace(w_min_idx=torch.tensor([1]))

    out = compute_recruitment_boundary_loss_from_state(
        state,
        params,
        loss_form="relative",
    )
    grad_N, grad_g, grad_recruitment = torch.autograd.grad(
        out["loss_bc"],
        (N, g, recruitment),
        allow_unused=True,
    )

    expected = (N.detach() * g.detach() / recruitment.detach() - 1.0).square().mean()
    assert torch.allclose(out["loss_bc"], expected)
    assert grad_N is not None and torch.isfinite(grad_N).all()
    assert grad_g is None
    assert grad_recruitment is None


def test_expert_weighting_uses_mean_gradient_norm():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    theta = model.weight.reshape(())
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}

    stats = update_expert_gradient_norm_weights_(
        model=model,
        losses={"pde": theta, "ic": 2.0 * theta, "bc": 4.0 * theta},
        weights=weights,
        alpha=0.9,
        min_weight=0.0,
        max_weight=100.0,
        hard_set=True,
    )

    mean_norm = (1.0 + 2.0 + 4.0) / 3.0
    assert weights["pde"] == mean_norm
    assert weights["ic"] == mean_norm / 2.0
    assert weights["bc"] == mean_norm / 4.0
    assert weights["bc"] < 1.0
    assert stats["expert_weight_total_grad_norm"] == 7.0


def test_fixed_calibration_batch_rescales_to_current_horizon():
    params = SimpleNamespace(t_min=0.0, t_max=30.0)
    batch = {
        "t_eval": torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64),
        "t_scaled": torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        "t_chunk_idx": torch.tensor([0, 1, 2]),
    }

    rescaled = rescale_fixed_calibration_batch(
        batch,
        params=params,
        t_max_current=6.0,
    )

    assert torch.allclose(rescaled["t_eval"], torch.tensor([0.0, 3.0, 6.0], dtype=torch.float64))
    assert torch.allclose(rescaled["t_scaled"], torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64))
    assert torch.equal(rescaled["t_chunk_idx"], batch["t_chunk_idx"])
    assert torch.allclose(batch["t_eval"], torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64))
