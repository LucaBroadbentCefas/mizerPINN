import sys

import torch

from PINNmizer.biology.fishing import compute_fishing_mortality_grid, evaluate_effort_at_time
from PINNmizer.inverse_parameters import LogFishingEffort
from PINNmizer.io import load_mizer_inputs
from PINNmizer.pinn.observation_operators import _gear_fishing
from PINNmizer.training.checkpointing import save_checkpoint
from PINNmizer.training.train_pde_multispecies import annual_effort_time_grid, load_checkpoint_weights, parse_args

FIXTURE = "validation/fixtures/pde_multispecies"


def inverse_fixture():
    params, _, _ = load_mizer_inputs(FIXTURE, dtype=torch.float64, device="cpu")
    truth = params.initial_effort.detach().clone()
    times = annual_effort_time_grid(params)
    inv = LogFishingEffort(torch.ones((times.numel(), truth.numel()), dtype=torch.float64))
    params.fishing_effort_time = times
    params.fishing_effort = inv.current_effort()
    params.initial_effort = params.fishing_effort[0]
    return params, inv, times, truth


def test_log_fishing_effort_initialises_exact_ones_and_zero_truth_is_separate():
    params, inv, times, truth = inverse_fixture()
    assert inv.current_effort().shape == (41, 5)
    assert torch.equal(inv.current_effort(), torch.ones((41, 5), dtype=torch.float64))
    assert torch.equal(inv.log_effort, torch.zeros_like(inv.log_effort))
    assert torch.equal(times, torch.arange(41, dtype=torch.float64))
    assert torch.equal(truth, torch.tensor([0, .6, .5, 3, .02], dtype=torch.float64))
    assert torch.isfinite(inv.log_effort).all()  # no log of reference truth
    assert not torch.equal(params.initial_effort, truth)


def test_mortality_and_gear_catch_paths_reach_log_effort():
    params, inv, _, _ = inverse_fixture()
    mortality = compute_fishing_mortality_grid(params)
    gear_f = _gear_fishing(params, 1, 0, torch.tensor(12.25, dtype=torch.float64))
    loss = mortality.square().sum() + gear_f.square().sum()
    loss.backward()
    assert inv.log_effort.grad is not None
    assert torch.isfinite(inv.log_effort.grad).all()
    assert inv.log_effort.grad.norm() > 0


def test_interpolation_gradients_only_neighbouring_rows():
    params, inv, _, _ = inverse_fixture()
    evaluate_effort_at_time(torch.tensor(4.25), params)[2].backward()
    nonzero = torch.nonzero(inv.log_effort.grad[:, 2]).flatten().tolist()
    assert nonzero == [4, 5]


def test_optimizer_step_can_reduce_effort_without_nonfinite_values():
    inv = LogFishingEffort(torch.ones((2, 1), dtype=torch.float64))
    optimizer = torch.optim.Adam(inv.parameters(), lr=.1)
    for _ in range(10):
        optimizer.zero_grad(); loss = inv.current_effort().sum(); loss.backward(); optimizer.step()
    assert 0 < inv.current_effort().max() < 1
    assert torch.isfinite(inv.log_effort.grad).all()


def test_checkpoint_round_trip_inverse_effort(tmp_path):
    _, inv, times, _ = inverse_fixture()
    model = torch.nn.Linear(1, 1).double()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "name": "network"},
                                  {"params": inv.parameters(), "name": "effort"}])
    inv.log_effort.data[3, 2] = -2
    path = save_checkpoint(run_dir=tmp_path, step=2, model=model, optimizer=optimizer,
                           config={}, inverse_effort=inv, effort_time=times)
    restored = LogFishingEffort(torch.ones_like(inv.initial_effort))
    optimizer2 = torch.optim.Adam([{"params": model.parameters(), "name": "network"},
                                   {"params": restored.parameters(), "name": "effort"}])
    loaded = load_checkpoint_weights(model=model, optimizer=optimizer2, checkpoint_path=path,
                                     device="cpu", inverse_effort=restored)
    assert loaded["effort_loaded"]
    assert torch.equal(restored.log_effort, inv.log_effort)


def test_cli_effort_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--estimate-effort", "--effort-lr", "0.025"])
    args = parse_args()
    assert args.estimate_effort and args.effort_lr == .025 and args.effort_init == 1.0
