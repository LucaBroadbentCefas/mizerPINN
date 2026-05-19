from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.losses import compute_pde_loss
from PINNmizer.training.weighting import update_wang_gradient_weights_


def scalar_min(x: torch.Tensor) -> float:
    return float(torch.min(x.detach()).cpu())

def scalar_max(x: torch.Tensor) -> float:
    return float(torch.max(x.detach()).cpu())

def scalar_mean(x: torch.Tensor) -> float:
    return float(torch.mean(x.detach()).cpu())

def total_grad_norm_and_check(model: nn.Module) -> float:
    total = None
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            raise FloatingPointError(f"Non-finite gradient in parameter: {name}")
        val = (p.grad.detach() ** 2).sum()
        total = val if total is None else total + val
    if total is None:
        return 0.0
    return float(torch.sqrt(total).cpu())

def train_one_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    params,
    n_pp: torch.Tensor,
    n_init: torch.Tensor,
    n_time: int,
    n_eval: int,
    residual_form: str,
    boundary_loss_form: str,
    eps: float,
    bc_eps: float | None,
    weight_state: dict[str, bool],
    hard_set_first_weight_update: bool,
    step: int,
    start_time: float,
    loss_weights: dict[str, float],
    weight_update_every: int,
    weight_warmup_steps: int,
    weight_alpha: float,
    weight_min: float,
    weight_max: float,
    causal_fraction: float,
    t_max_current: float,
    lambda_pde: float,
    lambda_ic: float,
    lambda_bc: float,
    disable_wang_weights: bool,
) -> dict:
    optimizer.zero_grad(set_to_none=True)

    batch = sample_pde_batch(
        params=params,
        n_time=n_time,
        n_eval=n_eval,
        t_max_current=t_max_current,
    )

    _, out = compute_pde_loss(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        residual_form=residual_form,
        n_init=n_init,
        lambda_pde=lambda_pde,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        boundary_loss_form=boundary_loss_form,
        species_idx=0,
        eps=eps,
        bc_eps=bc_eps,
    )

    raw_losses = {
        "pde": out["loss_pde"],
        "ic": out["loss_ic"],
        "bc": out["loss_bc"],
    }

    weight_stats = {
        "grad_pde_max": math.nan,
        "grad_ic_mean": math.nan,
        "grad_bc_mean": math.nan,
        "target_ic": math.nan,
        "target_bc": math.nan,
        "hard_set": 0.0,
    }

    if (
        not disable_wang_weights
        and step >= weight_warmup_steps
        and step % weight_update_every == 0
        ):
        hard_set = hard_set_first_weight_update and not weight_state["has_updated"]
    
        weight_stats = update_wang_gradient_weights_(
            model=model,
            losses=raw_losses,
            weights=loss_weights,
            alpha=weight_alpha,
            min_weight=weight_min,
            max_weight=weight_max,
            hard_set=hard_set,
        )
    
        weight_state["has_updated"] = True

    loss_unweighted = (
        out["loss_pde"]
        + out["loss_ic"]
        + out["loss_bc"]
    )

    if disable_wang_weights:
        loss = (
            lambda_pde * out["loss_pde"]
            + lambda_ic * out["loss_ic"]
            + lambda_bc * out["loss_bc"]
        )
    else:
        loss = (
            lambda_pde * loss_weights["pde"] * out["loss_pde"]
            + lambda_ic * loss_weights["ic"] * out["loss_ic"]
            + lambda_bc * loss_weights["bc"] * out["loss_bc"]
        )

    out["loss"] = loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

    loss.backward()

    grad_norm = total_grad_norm_and_check(model)

    optimizer.step()

    residual_log = out["residual_log"].detach()

    return {
        "step": step,
        "loss": float(out["loss"].detach().cpu()),
        "loss_pde": float(out["loss_pde"].detach().cpu()),
        "loss_ic": float(out["loss_ic"].detach().cpu()),
        "loss_bc": float(out["loss_bc"].detach().cpu()),
        "grad_norm": grad_norm,
        "residual_log_mean": scalar_mean(residual_log),
        "residual_log_abs_mean": scalar_mean(torch.abs(residual_log)),
        "residual_log_abs_max": scalar_max(torch.abs(residual_log)),
        "g_eval_min": scalar_min(out["g_eval"]),
        "g_eval_max": scalar_max(out["g_eval"]),
        "mu_eval_min": scalar_min(out["mu_eval"]),
        "mu_eval_max": scalar_max(out["mu_eval"]),
        "N_eval_min": scalar_min(out["N_eval"]),
        "N_eval_max": scalar_max(out["N_eval"]),
        "seconds_elapsed": time.perf_counter() - start_time,
        "w_pde": float(loss_weights["pde"]),
        "w_ic": float(loss_weights["ic"]),
        "w_bc": float(loss_weights["bc"]),
        "weighted_loss_pde": float((loss_weights["pde"] * out["loss_pde"]).detach().cpu()),
        "weighted_loss_ic": float((loss_weights["ic"] * out["loss_ic"]).detach().cpu()),
        "weighted_loss_bc": float((loss_weights["bc"] * out["loss_bc"]).detach().cpu()),
        "grad_pde_max_for_weighting": weight_stats["grad_pde_max"],
        "grad_ic_mean_for_weighting": weight_stats["grad_ic_mean"],
        "grad_bc_mean_for_weighting": weight_stats["grad_bc_mean"],
        "target_w_ic": weight_stats["target_ic"],
        "target_w_bc": weight_stats["target_bc"],
        "loss_weighted": float(loss.detach().cpu()),
        "loss_unweighted": float(loss_unweighted.detach().cpu()),
        "boundary_loss_form": boundary_loss_form,
        "bc_eps": float(bc_eps if bc_eps is not None else eps),
        "frac_flux_left_clamped": float(out.get("frac_flux_left_clamped", torch.tensor(float("nan"))).detach().cpu()),
        "frac_recruitment_flux_clamped": float(out.get("frac_recruitment_flux_clamped", torch.tensor(float("nan"))).detach().cpu()),
        "flux_left_min": float(out.get("flux_left_min", torch.tensor(float("nan"))).detach().cpu()),
        "recruitment_flux_min": float(out.get("recruitment_flux_min", torch.tensor(float("nan"))).detach().cpu()),
        "boundary_residual_abs_p95": float(out.get("boundary_residual_abs_p95", torch.tensor(float("nan"))).detach().cpu()),
        "boundary_residual_abs_max": float(out.get("boundary_residual_abs_max", torch.tensor(float("nan"))).detach().cpu()),
        "weight_update_hard_set": weight_stats["hard_set"],
        "causal_fraction": float(causal_fraction),
        "t_max_current": float(t_max_current),
    }
