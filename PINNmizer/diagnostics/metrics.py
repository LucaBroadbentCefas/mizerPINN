from __future__ import annotations

import math

import torch

def _as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu())

def _rms(x: torch.Tensor) -> float:
    x = x.detach()
    return _as_float(torch.sqrt(torch.mean(x ** 2)))

def _abs_mean(x: torch.Tensor) -> float:
    return _as_float(torch.mean(torch.abs(x.detach())))

def _abs_p95(x: torch.Tensor) -> float:
    z = torch.abs(x.detach()).reshape(-1)
    return _as_float(torch.quantile(z, 0.95))

def _abs_max(x: torch.Tensor) -> float:
    return _as_float(torch.max(torch.abs(x.detach())))

def _min(x: torch.Tensor) -> float:
    return _as_float(torch.min(x.detach()))

def _max(x: torch.Tensor) -> float:
    return _as_float(torch.max(x.detach()))

def _grad_norm(loss: torch.Tensor, model) -> float:
    if not loss.requires_grad:
        return math.nan

    parameters = [p for p in model.parameters() if p.requires_grad]

    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    total = None
    for grad in grads:
        if grad is None:
            continue

        if not torch.isfinite(grad).all():
            return math.nan

        value = (grad.detach() ** 2).sum()
        total = value if total is None else total + value

    if total is None:
        return math.nan

    return float(torch.sqrt(total).cpu())

def _grad_abs_stats(loss: torch.Tensor, model) -> tuple[float, float]:
    if not loss.requires_grad:
        return math.nan, math.nan

    parameters = [p for p in model.parameters() if p.requires_grad]

    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    flat = [
        grad.detach().reshape(-1)
        for grad in grads
        if grad is not None and torch.isfinite(grad).all()
    ]

    if not flat:
        return math.nan, math.nan

    values = torch.cat(flat).abs()

    return (
        float(values.max().cpu()),
        float(values.mean().cpu()),
    )

__all__=["_as_float","_rms","_abs_mean","_abs_p95","_abs_max","_min","_max","_grad_norm","_grad_abs_stats"]
