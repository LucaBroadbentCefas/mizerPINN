from __future__ import annotations

from functools import lru_cache

import torch


def lognormal_nll(prediction: torch.Tensor, value: torch.Tensor, sd_log: torch.Tensor, *, eps: float = 1e-30) -> dict[str, torch.Tensor]:
    """Lognormal negative log likelihood without the additive Normal constant.

    ell = 0.5 * ((log(value+eps) - log(prediction+eps)) / sd_log)^2 + log(sd_log).
    All tensors are shape [n_obs]; prediction remains attached for backpropagation.
    """
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError("Observed data values must be finite and positive.")
    if not torch.isfinite(sd_log).all() or not (sd_log > 0).all():
        raise ValueError("sd_log must be finite and positive.")
    pred_eps = prediction + torch.as_tensor(eps, dtype=prediction.dtype, device=prediction.device)
    if not torch.isfinite(pred_eps).all() or not (pred_eps > 0).all():
        raise ValueError("Predicted data values must be finite and positive after epsilon.")
    log_residual = torch.log(value + eps) - torch.log(pred_eps)
    contrib = 0.5 * (log_residual / sd_log) ** 2 + torch.log(sd_log)
    return {"loss_data": contrib.mean(), "loss_contribution": contrib, "log_residual": log_residual}


@lru_cache(maxsize=None)
def chi_square_95_quantile(n: int) -> float:
    """Return the chi-square 0.95 quantile using PyTorch's regularized gamma."""
    if n <= 0:
        raise ValueError("Chi-square degrees of freedom must be positive.")
    shape = torch.tensor(0.5 * n, dtype=torch.float64)
    target = torch.tensor(0.95, dtype=torch.float64)
    low = torch.tensor(0.0, dtype=torch.float64)
    high = torch.tensor(max(float(n), 1.0), dtype=torch.float64)
    while bool((torch.special.gammainc(shape, high / 2.0) < target).item()):
        high *= 2.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if bool((torch.special.gammainc(shape, mid / 2.0) < target).item()):
            low = mid
        else:
            high = mid
    return float(((low + high) / 2.0).item())


def apply_data_discrepancy_gate(
    loss_data: torch.Tensor,
    log_residual: torch.Tensor,
    sd_log_used: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Apply an on/off 95% chi-square gate while preserving the raw NLL."""
    n = log_residual.numel()
    if n == 0:
        nan = loss_data.new_tensor(float("nan"))
        return {
            "loss_data_effective": loss_data.new_zeros(()),
            "data_discrepancy_q": nan,
            "data_discrepancy_q95": nan,
            "data_loss_active": loss_data.new_zeros(()),
        }

    q = ((log_residual / sd_log_used) ** 2).sum()
    q95 = loss_data.new_tensor(chi_square_95_quantile(n))
    active = (not enabled) or bool((q.detach() > q95).cpu())
    return {
        "loss_data_effective": loss_data if active else loss_data.new_zeros(()),
        "data_discrepancy_q": q.detach(),
        "data_discrepancy_q95": q95,
        "data_loss_active": loss_data.new_tensor(float(active)),
    }
