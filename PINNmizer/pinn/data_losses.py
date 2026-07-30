from __future__ import annotations

import torch


def observation_relative_nll(
    prediction: torch.Tensor,
    value: torch.Tensor,
    cv: torch.Tensor,
    *,
    eps: float = 1e-30,
) -> dict[str, torch.Tensor]:
    """Observation-normalised relative Gaussian loss.

    The dimensionless residual is

        relative_residual = prediction / value - 1
        standardized_residual = relative_residual / cv

    and the per-observation loss, excluding constants independent of the
    prediction and CV, is

        0.5 * standardized_residual**2 + log(cv).

    All tensors are shape [n_obs]. ``prediction`` remains attached for
    backpropagation. The log(CV) term is required when CV is estimated; without
    it, increasing CV would always reduce the squared residual contribution.
    """
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError("Observed data values must be finite and positive.")
    if not torch.isfinite(cv).all() or not (cv > 0).all():
        raise ValueError("Observation CV must be finite and positive.")
    if not torch.isfinite(prediction).all() or not (prediction >= 0).all():
        raise ValueError("Predicted data values must be finite and non-negative.")

    relative_residual = prediction / value - 1.0
    standardized_residual = relative_residual / cv
    contribution = 0.5 * standardized_residual.square() + torch.log(cv)

    eps_t = torch.as_tensor(eps, dtype=prediction.dtype, device=prediction.device)
    log_residual = torch.log(value + eps_t) - torch.log(prediction + eps_t)

    return {
        "loss_data": contribution.mean(),
        "loss_contribution": contribution,
        "relative_residual": relative_residual,
        "standardized_residual": standardized_residual,
        "log_residual": log_residual,
        "cv_used": cv,
    }


def lognormal_nll(
    prediction: torch.Tensor,
    value: torch.Tensor,
    sd_log: torch.Tensor,
    *,
    eps: float = 1e-30,
) -> dict[str, torch.Tensor]:
    """Compatibility wrapper for existing training call sites.

    The training pipeline currently supplies lognormal ``sd_log`` values. They
    are converted back to CV before evaluating the observation-normalised
    relative loss. Despite the legacy function name, this no longer evaluates
    a lognormal negative log likelihood.
    """
    if not torch.isfinite(sd_log).all() or not (sd_log > 0).all():
        raise ValueError("sd_log must be finite and positive.")
    cv = torch.sqrt(torch.expm1(sd_log.square()))
    return observation_relative_nll(prediction, value, cv, eps=eps)
