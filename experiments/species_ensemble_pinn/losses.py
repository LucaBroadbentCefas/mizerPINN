from __future__ import annotations

import torch


def pde_loss(residuals: dict[str, torch.Tensor], active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """PDE loss is always mean squared U-scaled residual."""
    r = residuals["residual_scaled"]
    if active_mask is not None:
        r = r[..., active_mask]
    return torch.mean(r.square())


def initial_condition_loss(state: dict[str, torch.Tensor], log_u_target: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = state["log_U_ic"][..., active_mask]
    target = log_u_target.reshape(1,1,-1)[..., active_mask].to(pred)
    loss = torch.mean((pred - target).square())
    diag = {"log_U_ic_pred": state["log_U_ic"], "U_ic_pred": state["U_ic"], "log_U_ic_target": log_u_target, "U_ic_target": torch.exp(log_u_target), "log_N_ic_pred": state["log_N_ic"], "N_ic_pred": state["N_ic"], "log_N_ic_target": state["log_S_ic"].squeeze(0).squeeze(0)+log_u_target, "N_ic_target": torch.exp(state["log_S_ic"].squeeze(0).squeeze(0)+log_u_target)}
    return loss, diag


def relative_physical_boundary_loss(g_left: torch.Tensor, n_left: torch.Tensor, recruitment: torch.Tensor, *, bc_g_min: float = 0.0) -> torch.Tensor:
    valid = (g_left > bc_g_min) & (recruitment > 0) & torch.isfinite(g_left) & torch.isfinite(n_left) & torch.isfinite(recruitment)
    if not torch.any(valid):
        return (g_left * n_left).sum() * 0.0
    residual = 1.0 - (g_left[valid] * n_left[valid]) / recruitment[valid]
    return torch.mean(residual.square())
