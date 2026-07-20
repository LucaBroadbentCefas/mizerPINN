from __future__ import annotations

import math
import torch


def flat_loss_grad(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor | None:
    if not loss.requires_grad:
        return None
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().reshape(-1) for gradient in gradients if gradient is not None]
    return torch.cat(values) if values else None


def update_expert_gradient_norm_weights_(model, losses: dict[str, torch.Tensor],
                                         weights: dict[str, float], *, alpha: float,
                                         min_weight: float, max_weight: float,
                                         eps: float = 1e-12) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms: dict[str, torch.Tensor] = {}
    stats = {f"grad_norm_{name}_for_weighting": math.nan for name in ("pde", "ic", "bc")}
    for name in ("pde", "ic", "bc"):
        gradient = flat_loss_grad(losses[name], parameters)
        if gradient is None:
            continue
        norm = torch.linalg.vector_norm(gradient)
        if bool((norm > eps).detach().cpu()):
            norms[name] = norm
            stats[f"grad_norm_{name}_for_weighting"] = float(norm.cpu())
    if not norms:
        return stats
    total = torch.stack(list(norms.values())).sum()
    for name, norm in norms.items():
        target = float((total / norm.clamp_min(eps)).cpu())
        target = max(min_weight, min(max_weight, target))
        weights[name] = alpha * weights.get(name, 1.0) + (1.0 - alpha) * target
        stats[f"target_w_{name}"] = target
    stats["expert_weight_total_grad_norm"] = float(total.cpu())
    return stats
