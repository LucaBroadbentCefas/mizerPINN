import math

import torch
import torch.nn as nn

from PINNmizer.io import load_mizer_inputs
from PINNmizer.pinn.losses import compute_expert_causal_pde_loss
from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.training.weighting import update_expert_gradient_norm_weights_


def test_expert_causal_weights_and_detach():
    chunk_losses = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    residual = torch.sqrt(chunk_losses).reshape(3, 1, 1)
    mask = torch.ones(1, 1, 1, dtype=torch.float64)
    loss, diag = compute_expert_causal_pde_loss(
        residual=residual,
        active_mask=mask,
        t_chunk_idx=torch.tensor([0, 1, 2]),
        n_chunks=3,
        epsilon=1.0,
    )
    expected = torch.tensor([1.0, math.exp(-0.1), math.exp(-0.3)], dtype=torch.float64)
    assert torch.allclose(diag["pde_causal_weights"], expected)
    assert diag["pde_causal_weights"].requires_grad is False
    assert loss.requires_grad is True


def test_expert_causal_epsilon_zero_matches_ungated_mean():
    chunk_losses = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    residual = torch.sqrt(chunk_losses).reshape(3, 1, 1)
    loss, diag = compute_expert_causal_pde_loss(
        residual=residual,
        active_mask=torch.ones(1, 1, 1, dtype=torch.float64),
        t_chunk_idx=torch.tensor([0, 1, 2]),
        n_chunks=3,
        epsilon=0.0,
    )
    assert torch.allclose(diag["pde_causal_weights"], torch.ones(3, dtype=torch.float64))
    assert torch.allclose(loss, diag["loss_pde_ungated"])


def test_stratified_sampling_fills_each_chunk_and_bounds_times():
    params, _, _ = load_mizer_inputs("validation/fixtures/pde_single_species", dtype=torch.float64, device="cpu")
    n_chunks = 4
    batch = sample_pde_batch(
        params=params,
        n_time=5,
        n_eval=3,
        time_sampling="stratified",
        causal_n_chunks=n_chunks,
    )
    t = batch["t_eval"]
    idx = batch["t_chunk_idx"]
    assert idx.numel() == t.numel() == batch["effective_n_time"].item()
    assert set(idx.tolist()) == set(range(n_chunks))

    edges = torch.linspace(float(params.t_min), float(params.t_max), n_chunks + 1, dtype=t.dtype)
    for i in range(n_chunks):
        ti = t[idx == i]
        assert ti.numel() >= 1
        assert torch.all(ti >= edges[i])
        assert torch.all(ti <= edges[i + 1])


def test_expert_gradient_norm_weights_exclude_inactive_and_use_paper_ema():
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    y = model(torch.ones(1, 1, dtype=torch.float64)).sum()
    losses = {
        "pde": (2.0 * y) ** 2,
        "ic": y ** 2,
        "bc": 0.0 * y,
        "timestep": torch.zeros((), dtype=torch.float64),
    }
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0, "timestep": 1.0}
    stats = update_expert_gradient_norm_weights_(
        model=model,
        losses=losses,
        weights=weights,
        alpha=0.9,
        min_weight=1e-6,
        max_weight=1e6,
        hard_set=False,
    )
    assert math.isnan(stats["grad_norm_bc_for_weighting"])
    assert math.isnan(stats["grad_norm_timestep_for_weighting"])
    assert stats["grad_norm_pde_for_weighting"] > stats["grad_norm_ic_for_weighting"]
    assert stats["target_w_pde"] < stats["target_w_ic"]
    assert math.isclose(weights["pde"], 0.9 * 1.0 + 0.1 * stats["target_w_pde"])
    assert math.isclose(weights["ic"], 0.9 * 1.0 + 0.1 * stats["target_w_ic"])
