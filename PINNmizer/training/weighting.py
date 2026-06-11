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

    stats = {
        "grad_pde_max": math.nan,
        "hard_set": float(hard_set),
    }
    for name in losses:
        if name == "pde":
            continue
        stats[f"grad_{name}_mean"] = math.nan
        stats[f"target_{name}"] = math.nan

    if grad_pde is None:
        return stats

    pde_max = grad_pde.abs().max().clamp_min(eps)
    stats["grad_pde_max"] = float(pde_max.cpu())
    weights["pde"] = 1.0

    for name, loss in losses.items():
        if name == "pde":
            continue

        grad = _flat_loss_grad(loss, params)
        if grad is None:
            continue

        grad_mean = grad.abs().mean().clamp_min(eps)
        target = float((pde_max / grad_mean).cpu())
        target = max(min_weight, min(max_weight, target))

        if name not in weights:
            weights[name] = 1.0

        weights[name] = target if hard_set else (1.0 - alpha) * weights[name] + alpha * target
        stats[f"grad_{name}_mean"] = float(grad_mean.cpu())
        stats[f"target_{name}"] = target

    return stats


def update_expert_gradient_norm_weights_(
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
    """Update loss weights using expert-guide inverse gradient-norm targets."""
    known = ("pde", "ic", "bc", "timestep")
    stats = {f"grad_norm_{name}_for_weighting": math.nan for name in known}
    stats.update({f"target_w_{name}": math.nan for name in known})
    stats["expert_weight_total_grad_norm"] = math.nan
    stats["expert_weight_hard_set"] = float(hard_set)

    params = [p for p in model.parameters() if p.requires_grad]
    grad_norms: dict[str, torch.Tensor] = {}

    for name in known:
        loss = losses.get(name)
        if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
            continue
        grad = _flat_loss_grad(loss, params)
        if grad is None:
            continue
        norm = torch.linalg.vector_norm(grad, ord=2)
        if not bool((norm > eps).detach().cpu()):
            continue
        grad_norms[name] = norm
        stats[f"grad_norm_{name}_for_weighting"] = float(norm.cpu())

    if not grad_norms:
        return stats

    total = torch.stack(list(grad_norms.values())).sum()
    stats["expert_weight_total_grad_norm"] = float(total.cpu())

    for name, norm in grad_norms.items():
        target = float((total / norm.clamp_min(eps)).cpu())
        target = max(min_weight, min(max_weight, target))
        if name not in weights:
            weights[name] = 1.0
        weights[name] = target if hard_set else alpha * weights[name] + (1.0 - alpha) * target
        stats[f"target_w_{name}"] = target

    return stats
