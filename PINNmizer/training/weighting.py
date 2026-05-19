from __future__ import annotations

import math

import torch
import torch.nn as nn


def _flat_loss_grad(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor | None:
    """Return flattened detached gradient for a scalar loss, or None if inactive."""
    if not loss.requires_grad:
        return None

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if not flat:
        return None
    return torch.cat(flat)


def update_wang_gradient_weights_(
    *,
    model: nn.Module,
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
    alpha: float,
    min_weight: float,
    max_weight: float,
    eps: float = 1e-12,
    hard_set: bool = False,
) -> dict[str, float]:
    params = [p for p in model.parameters() if p.requires_grad]
    grad_pde = _flat_loss_grad(losses["pde"], params)

    if grad_pde is None:
        return {
            "grad_pde_max": math.nan,
            "grad_ic_mean": math.nan,
            "grad_bc_mean": math.nan,
            "target_ic": math.nan,
            "target_bc": math.nan,
            "hard_set": float(hard_set),
        }

    pde_max = grad_pde.abs().max().clamp_min(eps)
    stats = {
        "grad_pde_max": float(pde_max.cpu()),
        "grad_ic_mean": math.nan,
        "grad_bc_mean": math.nan,
        "target_ic": math.nan,
        "target_bc": math.nan,
        "hard_set": float(hard_set),
    }
    weights["pde"] = 1.0

    for name in ("ic", "bc"):
        grad = _flat_loss_grad(losses[name], params)
        if grad is None:
            continue
        grad_mean = grad.abs().mean().clamp_min(eps)
        target = float((pde_max / grad_mean).cpu())
        target = max(min_weight, min(max_weight, target))
        weights[name] = target if hard_set else (1.0 - alpha) * weights[name] + alpha * target
        stats[f"grad_{name}_mean"] = float(grad_mean.cpu())
        stats[f"target_{name}"] = target

    return stats
