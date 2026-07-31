from __future__ import annotations

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
