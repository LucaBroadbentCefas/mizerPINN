import sys

import torch

from PINNmizer.inverse_parameters import BoundedDataCV, BoundedLogRMax
from PINNmizer.training.train_pde_only_single_species import load_checkpoint_weights, parse_args


def test_single_species_data_and_inverse_cli_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trainer", "--data-csv", "obs.csv", "--lambda-data", "2", "--estimate-rmax", "--estimate-data-cv", "--data-cv-scope", "global"])
    args = parse_args()
    assert args.data_csv == "obs.csv"
    assert args.lambda_data == 2.0
    assert args.estimate_rmax and args.estimate_data_cv
    assert args.data_cv_scope == "global"


def test_single_species_inverse_and_cv_checkpoint_resume(tmp_path):
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    rmax = BoundedLogRMax(torch.tensor([10.0], dtype=torch.float64), lower=0.0, upper=50.0)
    cv = BoundedDataCV(torch.tensor([0.3], dtype=torch.float64), lower=0.02, upper=1.5, scope="species")
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3, "name": "network"},
        {"params": rmax.parameters(), "lr": 2e-3, "name": "rmax"},
        {"params": cv.parameters(), "lr": 3e-3, "name": "data_cv"},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with torch.no_grad():
        rmax.raw_logit.add_(0.4)
        cv.raw_parameter.sub_(0.2)
    expected_rmax = rmax.current_r_max().clone()
    expected_cv = cv.current_cv().clone()
    path = tmp_path / "checkpoint.pt"
    torch.save({
        "step": 7,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "inverse_parameter_state_dict": rmax.state_dict(),
        "data_cv_state_dict": cv.state_dict(),
        "config": {"state_parameterization": "log-n"},
    }, path)

    resumed_model = torch.nn.Linear(2, 1, dtype=torch.float64)
    resumed_model.state_parameterization = "log-n"
    resumed_rmax = BoundedLogRMax(torch.tensor([5.0], dtype=torch.float64), lower=0.0, upper=50.0)
    resumed_cv = BoundedDataCV(torch.tensor([0.5], dtype=torch.float64), lower=0.02, upper=1.5, scope="species")
    resumed_optimizer = torch.optim.Adam([
        {"params": resumed_model.parameters(), "lr": 1e-3, "name": "network"},
        {"params": resumed_rmax.parameters(), "lr": 2e-3, "name": "rmax"},
        {"params": resumed_cv.parameters(), "lr": 3e-3, "name": "data_cv"},
    ])
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1)
    result = load_checkpoint_weights(model=resumed_model, optimizer=resumed_optimizer, checkpoint_path=path, device="cpu", load_optimizer_state=True, inverse_rmax=resumed_rmax, inverse_data_cv=resumed_cv, scheduler=resumed_scheduler)

    assert result["inverse_parameter_loaded"] and result["data_cv_loaded"] and result["scheduler_loaded"]
    assert torch.allclose(resumed_rmax.current_r_max(), expected_rmax)
    assert torch.allclose(resumed_cv.current_cv(), expected_cv)
