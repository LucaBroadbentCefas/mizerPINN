import math

import pytest
import torch

from PINNmizer.pinn.data_losses import (
    apply_data_discrepancy_gate,
    chi_square_95_quantile,
    lognormal_nll,
)
from PINNmizer.training import train_pde_multispecies


def _gate(z, *, enabled=True):
    prediction = torch.tensor([2.0] * len(z), dtype=torch.float64, requires_grad=True)
    sd_log = torch.tensor([0.2] * len(z), dtype=torch.float64)
    value = prediction.detach() * torch.exp(sd_log * torch.tensor(z, dtype=torch.float64))
    nll = lognormal_nll(prediction, value, sd_log)
    result = apply_data_discrepancy_gate(
        nll["loss_data"], nll["log_residual"], sd_log, enabled=enabled
    )
    return prediction, nll, result


def test_gate_disabled_preserves_existing_loss_and_gradient():
    prediction, nll, result = _gate([0.1], enabled=False)
    assert result["loss_data_effective"] is nll["loss_data"]
    result["loss_data_effective"].backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_below_threshold_preserves_raw_loss_but_removes_effective_loss():
    prediction, nll, result = _gate([0.5, -0.5])
    assert nll["loss_data"].item() != 0.0
    assert result["data_discrepancy_q"] < result["data_discrepancy_q95"]
    assert result["loss_data_effective"].item() == 0.0
    assert not result["loss_data_effective"].requires_grad
    assert result["data_loss_active"].item() == 0.0
    assert prediction.grad is None


def test_above_threshold_uses_raw_loss_and_existing_gradient():
    prediction, nll, result = _gate([3.0])
    assert result["data_discrepancy_q"] > result["data_discrepancy_q95"]
    assert result["loss_data_effective"] is nll["loss_data"]
    result["loss_data_effective"].backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_exact_prediction_has_zero_discrepancy_and_inactive_gate():
    _, nll, result = _gate([0.0])
    assert nll["loss_data"].item() == pytest.approx(math.log(0.2))
    assert result["data_discrepancy_q"].item() == 0.0
    assert result["loss_data_effective"].item() == 0.0
    assert result["data_loss_active"].item() == 0.0


def test_threshold_uses_number_of_active_residuals():
    _, _, one = _gate([0.0])
    _, _, two = _gate([0.0, 0.0])
    assert one["data_discrepancy_q95"].item() == pytest.approx(3.841458820694124)
    assert two["data_discrepancy_q95"].item() == pytest.approx(5.991464547107979)
    assert chi_square_95_quantile(2) > chi_square_95_quantile(1)


def test_gate_rejects_estimated_cv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train_pde_multispecies", "--data-discrepancy-gate", "--estimate-data-cv"],
    )
    with pytest.raises(ValueError, match="requires fixed known observation uncertainty"):
        train_pde_multispecies.main()
