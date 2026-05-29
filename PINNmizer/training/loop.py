from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.losses import compute_pde_loss, compute_pde_loss_paired
from PINNmizer.pinn.r3 import update_r3_population_
from PINNmizer.training.weighting import update_wang_gradient_weights_
from PINNmizer.pinn.timestep_consistency import compute_timestep_consistency_loss


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
    lambda_timestep: float = 0.0,
    timestep_loss_form: str = "physical",
    detach_step_target: bool = True,
    timestep_dt: float | None = None,
    timestep_n_pairs: int = 1,
    collocation_strategy: str = "uniform",
    r3_population=None,
    r3_update_every: int = 1,
    r3_warmup_steps: int = 0,
    r3_score_form: str = "abs",
    causal_r3=None,
    causal_r3_weight_pde_loss: bool = False,
    causal_r3_score: bool = True,
) -> dict:
    optimizer.zero_grad(set_to_none=True)

    if collocation_strategy == "uniform":
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

    elif collocation_strategy in {"r3", "causal-r3"}:
        if r3_population is None:
            raise ValueError("R3 collocation requires r3_population.")

        batch = r3_population.as_batch(params=params)

        pde_weights = None
        if collocation_strategy == "causal-r3" and causal_r3_weight_pde_loss:
            pde_weights = causal_r3.gate(batch["t_pair"], params)

        _, out = compute_pde_loss_paired(
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
            pde_weights=pde_weights,
        )

    else:
        raise ValueError("collocation_strategy must be 'uniform', 'r3', or 'causal-r3'.")

    loss_timestep = out["loss_pde"].new_zeros(())
    timestep_out = None
    if lambda_timestep > 0.0:
      n_pairs = max(1, timestep_n_pairs)
      
      dt_value = params.dt if timestep_dt is None else timestep_dt
      dt_tensor = torch.as_tensor(dt_value, dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)
      
      t_min = torch.as_tensor(params.t_min, dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)
      t_max = torch.as_tensor(params.t_max, dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)
      t_current = torch.as_tensor(t_max_current, dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)
      
      t0_max = torch.minimum(t_current, t_max) - dt_tensor
      
      if t0_max > t_min:
          t0 = t_min + (t0_max - t_min) * torch.rand(
              n_pairs,
              dtype=out["loss_pde"].dtype,
              device=out["loss_pde"].device,
          )
      
          loss_timestep, timestep_out = compute_timestep_consistency_loss(
              model=model,
              params=params,
              n_pp=n_pp,
              t0=t0,
              dt=timestep_dt,
              loss_form=timestep_loss_form,
              detach_step_target=detach_step_target,
              species_idx=0,
              eps=eps,
          )
    out["loss_timestep"] = loss_timestep

    raw_losses = {
        "pde": out["loss_pde"],
        "ic": out["loss_ic"],
        "bc": out["loss_bc"],
        "timestep": out["loss_timestep"],
    }

    weight_stats = {
        "grad_pde_max": math.nan,
        "grad_ic_mean": math.nan,
        "grad_bc_mean": math.nan,
        "grad_timestep_mean": math.nan,
        "target_ic": math.nan,
        "target_bc": math.nan,
        "target_timestep": math.nan,
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
        + out["loss_timestep"]
    )


    if disable_wang_weights:
        loss = (
            lambda_pde * out["loss_pde"]
            + lambda_ic * out["loss_ic"]
            + lambda_bc * out["loss_bc"]
            + lambda_timestep * out["loss_timestep"]
        )
    else:
        loss = (
            lambda_pde * loss_weights["pde"] * out["loss_pde"]
            + lambda_ic * loss_weights["ic"] * out["loss_ic"]
            + lambda_bc * loss_weights["bc"] * out["loss_bc"]
            + lambda_timestep * loss_weights["timestep"] * out["loss_timestep"]
        )


    out["loss"] = loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

    loss.backward()

    grad_norm = total_grad_norm_and_check(model)

    optimizer.step()

    r3_diag = {
        "r3_population_size": math.nan,
        "r3_retained_fraction": math.nan,
        "r3_resampled": math.nan,
        "r3_score_mean": math.nan,
        "r3_score_max": math.nan,
        "causal_r3_gamma": math.nan,
        "causal_r3_gamma_update": math.nan,
        "causal_r3_gate_mean": math.nan,
    }

    if collocation_strategy in {"r3", "causal-r3"}:
        if step > r3_warmup_steps and step % max(1, r3_update_every) == 0:
            residual_for_score = (
                out["residual_log"]
                if residual_form == "log"
                else out["residual"]
            )

            r3_diag.update(
                update_r3_population_(
                    population=r3_population,
                    residual=residual_for_score,
                    batch=batch,
                    params=params,
                    score_form=r3_score_form,
                    causal=causal_r3 if collocation_strategy == "causal-r3" else None,
                    causal_score=causal_r3_score,
                )
            )

        if collocation_strategy == "causal-r3":
            r3_diag.update(causal_r3.update_(out["loss_pde"]))
            gate = causal_r3.gate(batch["t_pair"], params).detach()
            r3_diag["causal_r3_gate_mean"] = float(gate.mean().cpu())

    residual_log = out["residual_log"].detach()

    base = {
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
        "w_timestep": float(loss_weights["timestep"]),
        "weighted_loss_pde": float((loss_weights["pde"] * out["loss_pde"]).detach().cpu()),
        "weighted_loss_ic": float((loss_weights["ic"] * out["loss_ic"]).detach().cpu()),
        "weighted_loss_bc": float((loss_weights["bc"] * out["loss_bc"]).detach().cpu()),
        "weighted_loss_timestep": float((loss_weights["timestep"] * out["loss_timestep"]).detach().cpu()),
        "grad_pde_max_for_weighting": weight_stats["grad_pde_max"],
        "grad_ic_mean_for_weighting": weight_stats["grad_ic_mean"],
        "grad_bc_mean_for_weighting": weight_stats["grad_bc_mean"],
        "target_w_ic": weight_stats["target_ic"],
        "target_w_bc": weight_stats["target_bc"],
        "grad_timestep_mean_for_weighting": weight_stats["grad_timestep_mean"],
        "target_w_timestep": weight_stats["target_timestep"],
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
        "loss_timestep": float(out["loss_timestep"].detach().cpu()),
        "lambda_timestep": float(lambda_timestep),
        "timestep_loss_form": timestep_loss_form,
        "detach_step_target": bool(detach_step_target),
        "collocation_strategy": collocation_strategy,
        **r3_diag,
        "timestep_physical_abs_mean": float((timestep_out["physical_abs_mean"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
        "timestep_physical_abs_max": float((timestep_out["physical_abs_max"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
        "timestep_log_abs_mean": float((timestep_out["log_abs_mean"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
        "timestep_log_abs_max": float((timestep_out["log_abs_max"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
        "timestep_relative_abs_mean": float((timestep_out["relative_abs_mean"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
        "timestep_relative_abs_max": float((timestep_out["relative_abs_max"] if timestep_out is not None else torch.tensor(float("nan"))).detach().cpu()),
    }
    return base
