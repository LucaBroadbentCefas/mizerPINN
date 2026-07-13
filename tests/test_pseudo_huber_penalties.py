import math
from types import SimpleNamespace

import pytest
import torch

from PINNmizer.pinn import losses
from PINNmizer.pinn.data_losses import lognormal_nll


def test_pointwise_squared_exact():
    r = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    assert torch.equal(losses._pointwise_penalty(r, penalty="squared", delta=1.0), r.square())


def test_pseudo_huber_basic_properties_and_grad():
    r = torch.tensor([-1e20, -2.0, 0.0, 2.0, 1e20], dtype=torch.float64, requires_grad=True)
    ph = losses._pointwise_penalty(r, penalty="pseudo-huber", delta=1.5)
    assert ph[2].item() == 0.0
    assert torch.isfinite(ph).all()
    assert torch.allclose(ph[1], ph[3])
    ph.sum().backward()
    assert torch.isfinite(r.grad).all()


@pytest.mark.parametrize("delta", [0.0, -1.0])
def test_pseudo_huber_delta_must_be_positive(delta):
    with pytest.raises(ValueError, match="strictly positive"):
        losses._pointwise_penalty(torch.ones(1), penalty="pseudo-huber", delta=delta)


def test_masked_penalty_mean_uses_active_entries_only():
    r = torch.tensor([1.0, 10.0, 3.0], dtype=torch.float64)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    got = losses._masked_penalty_mean(r, mask, penalty="squared", delta=1.0)
    assert torch.allclose(got, torch.tensor((1.0 + 9.0) / 2.0, dtype=torch.float64))


def test_expert_causal_chunk_losses_use_pseudo_huber():
    residual = torch.tensor([1.0, 10.0], dtype=torch.float64).reshape(2, 1, 1)
    loss, diag = losses.compute_expert_causal_pde_loss(
        residual=residual,
        active_mask=torch.ones(1, 1, 1, dtype=torch.float64),
        t_chunk_idx=torch.tensor([0, 1]),
        n_chunks=2,
        epsilon=0.0,
        pde_penalty="pseudo-huber",
        pde_pseudo_huber_delta=1.0,
    )
    expected_chunks = torch.sqrt(1 + residual.reshape(-1).square()) - 1
    assert torch.allclose(diag["pde_causal_chunk_losses"], expected_chunks)
    assert torch.allclose(loss, expected_chunks.mean())


def test_ordinary_pde_pathway_uses_pseudo_huber(monkeypatch):
    residual = torch.tensor([[[1.0, 10.0]]], dtype=torch.float64)
    state = {"dummy": torch.tensor(0.0)}
    monkeypatch.setattr(losses, "compute_pde_state", lambda **kwargs: state)
    monkeypatch.setattr(losses, "compute_pde_residual_from_state", lambda s: {"residual_log": residual, "residual": residual, "residual_scaled": residual})
    monkeypatch.setattr(losses, "active_eval_mask", lambda w, params: torch.ones(1, 2, dtype=torch.bool))
    params = SimpleNamespace()
    batch = {"w_eval": torch.ones(2, dtype=torch.float64)}
    _, out = losses.compute_pde_loss(
        model=None, batch=batch, params=params, n_pp=torch.ones(1), residual_form="log",
        pde_penalty="pseudo-huber", pde_pseudo_huber_delta=1.0,
    )
    expected = (torch.sqrt(1 + residual.square()) - 1).mean()
    assert torch.allclose(out["loss_pde"], expected)


def test_paired_and_r3_pde_pathways_accept_pseudo_huber(monkeypatch):
    residual_pair = torch.tensor([[1.0, 10.0]], dtype=torch.float64)
    residual_r3 = torch.tensor([[[1.0, 10.0]]], dtype=torch.float64)
    monkeypatch.setattr(losses, "compute_pde_state_paired", lambda **kwargs: {})
    monkeypatch.setattr(losses, "compute_pde_state_r3_slabbed", lambda **kwargs: {})
    monkeypatch.setattr(losses, "compute_pde_residual_from_state", lambda s: {"residual_log": residual_pair if s.get('paired') else residual_r3, "residual": residual_pair if s.get('paired') else residual_r3, "residual_scaled": residual_pair if s.get('paired') else residual_r3})
    monkeypatch.setattr(losses, "compute_pde_state_paired", lambda **kwargs: {"paired": True})
    monkeypatch.setattr(losses, "active_eval_mask", lambda w, params: torch.ones(1, 2, dtype=torch.bool))
    monkeypatch.setattr(losses, "_active_eval_mask_for_slab", lambda w, params: torch.ones(1, 1, 2, dtype=torch.bool))
    monkeypatch.setattr(losses, "_params_dtype_device", lambda params: (torch.float64, torch.device("cpu")))
    params = SimpleNamespace()
    _, paired = losses.compute_pde_loss_paired(None, {"w_pair": torch.ones(2)}, params, torch.ones(1), pde_penalty="pseudo-huber")
    monkeypatch.setattr(losses, "compute_pde_state_r3_slabbed", lambda **kwargs: {})
    _, r3 = losses.compute_pde_loss_r3_slabbed(None, {"w_slab": torch.ones(1, 2)}, params, torch.ones(1), pde_penalty="pseudo-huber")
    expected_pair = (torch.sqrt(1 + residual_pair.square()) - 1).mean()
    expected_r3 = (torch.sqrt(1 + residual_r3.square()) - 1).mean()
    assert torch.allclose(paired["loss_pde"], expected_pair)
    assert torch.allclose(r3["loss_pde"], expected_r3)


def test_log_bc_uses_pseudo_huber_and_squared_default_matches_previous():
    state = {
        "log_N_grid": torch.log(torch.tensor([[[4.0]]], dtype=torch.float64)),
        "N_grid": torch.tensor([[[4.0]]], dtype=torch.float64),
        "growth_grid": {"e_growth_eval": torch.tensor([[[2.0]]], dtype=torch.float64)},
        "recruitment": {"rdd_flux": torch.tensor([[4.0]], dtype=torch.float64)},
    }
    params = SimpleNamespace(w_min_idx=torch.tensor([1]))
    squared = losses.compute_recruitment_boundary_loss_from_state(state, params)["loss_bc"]
    pseudo = losses.compute_recruitment_boundary_loss_from_state(state, params, bc_penalty="pseudo-huber", bc_pseudo_huber_delta=1.0)["loss_bc"]
    residual = math.log(4.0) - (math.log(4.0) - math.log(2.0))
    assert torch.allclose(squared, torch.tensor(residual**2, dtype=torch.float64))
    assert torch.allclose(pseudo, torch.tensor(math.sqrt(1 + residual**2) - 1, dtype=torch.float64))


def test_data_loss_unchanged_for_identical_predictions_when_penalty_options_exist():
    pred = torch.tensor([2.0, 3.0], dtype=torch.float64)
    value = torch.tensor([2.5, 2.5], dtype=torch.float64)
    sd = torch.tensor([0.2, 0.3], dtype=torch.float64)
    before = lognormal_nll(pred, value, sd)["loss_data"]
    _ = losses._pointwise_penalty(torch.tensor([1.0]), penalty="pseudo-huber", delta=1.0)
    after = lognormal_nll(pred, value, sd)["loss_data"]
    assert torch.equal(before, after)
