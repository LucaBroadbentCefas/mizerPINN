from __future__ import annotations

import torch

from PINNmizer.params import active_eval_mask, active_grid_mask


def masked_square_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(dtype=values.dtype, device=values.device).expand_as(values)
    denominator = expanded.sum()
    if not bool((denominator > 0).detach().cpu()):
        raise ValueError("Masked loss has no active entries.")
    return (values.square() * expanded).sum() / denominator


def expert_causal_pde_loss(residual: torch.Tensor, active_mask: torch.Tensor,
                           t_chunk_idx: torch.Tensor, *, n_chunks: int,
                           epsilon: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = active_mask.to(dtype=residual.dtype, device=residual.device).expand_as(residual)
    indices = t_chunk_idx.to(device=residual.device, dtype=torch.long)
    if indices.shape != (residual.shape[0],):
        raise ValueError("t_chunk_idx must contain one chunk index per sampled time.")
    chunks = []
    for idx in range(n_chunks):
        selected = indices == idx
        if not bool(selected.any().detach().cpu()):
            raise ValueError(f"Causal chunk {idx} is empty.")
        chunk_mask = mask[selected]
        chunks.append((residual[selected].square() * chunk_mask).sum() / chunk_mask.sum())
    chunk_losses = torch.stack(chunks)
    previous = torch.cat([
        torch.zeros(1, dtype=residual.dtype, device=residual.device),
        torch.cumsum(chunk_losses[:-1], dim=0),
    ])
    weights = torch.exp(-float(epsilon) * previous).detach()
    loss = (weights * chunk_losses).mean()
    return loss, {
        "loss_pde_ungated": masked_square_mean(residual, mask),
        "pde_causal_weights": weights,
        "pde_causal_chunk_losses": chunk_losses.detach(),
        "pde_causal_weight_first": weights[0],
        "pde_causal_weight_mean": weights.mean(),
        "pde_causal_weight_last": weights[-1],
    }


def initial_condition_loss(state: dict[str, object], params, *, species_idx: int,
                           eps: float) -> dict[str, torch.Tensor]:
    log_pred = state["log_N_ic"][0]
    target = state["n_init_target"]
    log_target = torch.log(torch.clamp(target, min=eps))
    mask = active_grid_mask(params)[species_idx:species_idx + 1]
    return {
        "loss_ic": masked_square_mean(log_pred - log_target, mask),
        "log_N_ic_pred": log_pred,
        "log_N_ic_target": log_target,
    }


def relative_boundary_loss(state: dict[str, object], params, *, species_idx: int,
                           bc_g_min: float) -> dict[str, torch.Tensor]:
    egg_idx = int(params.w_min_idx[species_idx].item()) - 1
    n_left = state["N_grid"][:, :, egg_idx]
    log_n_left = state["log_N_grid"][:, :, egg_idx]
    g_left = state["growth_grid"]["e_growth_eval"][:, :, egg_idx]
    recruitment = state["recruitment"]["rdd_flux"]
    threshold = torch.as_tensor(bc_g_min, dtype=n_left.dtype, device=n_left.device)
    finite = torch.isfinite(log_n_left) & torch.isfinite(n_left) & torch.isfinite(g_left) & torch.isfinite(recruitment)
    valid = finite & (g_left > threshold) & (recruitment > 0)
    if bool(valid.any().detach().cpu()):
        residual_valid = 1.0 - (g_left[valid] * n_left[valid]) / recruitment[valid]
        loss = residual_valid.square().mean()
        residual = torch.full_like(n_left.detach(), float("nan"))
        residual[valid.detach()] = residual_valid.detach()
    else:
        loss = (log_n_left * 0.0).sum()
        residual = torch.full_like(n_left.detach(), float("nan"))
    return {
        "loss_bc": loss,
        "boundary_residual": residual,
        "bc_valid_mask": valid.detach(),
        "bc_valid_fraction": valid.to(n_left.dtype).mean().detach(),
        "bc_invalid_g_fraction": (finite & (g_left <= threshold)).to(n_left.dtype).mean().detach(),
        "bc_invalid_recruitment_fraction": (finite & (recruitment <= 0)).to(n_left.dtype).mean().detach(),
        "bc_nonfinite_fraction": (~finite).to(n_left.dtype).mean().detach(),
        "N_left": n_left,
        "g_left": g_left,
        "recruitment_flux": recruitment,
    }


def compute_composite_loss(*, state: dict[str, object], residual_out: dict[str, torch.Tensor],
                           params, species_idx: int, batch: dict[str, torch.Tensor],
                           lambda_pde: float, lambda_ic: float, lambda_bc: float,
                           loss_weights: dict[str, float], causal_n_chunks: int,
                           causal_epsilon: float, eps: float, bc_g_min: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = active_eval_mask(batch["w_eval"], params)[species_idx:species_idx + 1][None, :, :]
    loss_pde, causal = expert_causal_pde_loss(
        residual_out["residual_reference_scaled"], active, batch["t_chunk_idx"],
        n_chunks=causal_n_chunks, epsilon=causal_epsilon,
    )
    ic = initial_condition_loss(state, params, species_idx=species_idx, eps=eps)
    bc = relative_boundary_loss(state, params, species_idx=species_idx, bc_g_min=bc_g_min)
    raw = lambda_pde * loss_pde + lambda_ic * ic["loss_ic"] + lambda_bc * bc["loss_bc"]
    weighted = (
        loss_weights["pde"] * lambda_pde * loss_pde
        + loss_weights["ic"] * lambda_ic * ic["loss_ic"]
        + loss_weights["bc"] * lambda_bc * bc["loss_bc"]
    )
    return weighted, {
        **residual_out, **causal, **ic, **bc,
        "loss": weighted,
        "loss_unweighted": raw,
        "loss_pde": loss_pde,
        "loss_ic": ic["loss_ic"],
        "loss_bc": bc["loss_bc"],
    }
