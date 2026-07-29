from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.losses import (
    compute_pde_loss,
    compute_pde_loss_paired,
    compute_pde_loss_r3_slabbed,
)
from PINNmizer.pinn.r3 import update_r3_population_
from PINNmizer.training.weighting import (
    rescale_fixed_calibration_batch,
    update_expert_gradient_norm_weights_,
    update_wang_gradient_weights_,
)
from PINNmizer.pinn.timestep_consistency_multispecies import compute_timestep_consistency_loss_multispecies
from PINNmizer.params import scale_x, scale_t
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.observation_operators import observation_time_grid, predict_observations
from PINNmizer.pinn.data_losses import lognormal_nll


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


def inverse_rmax_grad_stats(inverse_rmax, *, require_nonzero: bool = False) -> dict:
    if inverse_rmax is None:
        return {"rmax_raw_grad_norm": math.nan, "rmax_raw_grad_min": math.nan, "rmax_raw_grad_max": math.nan, "rmax_grad_finite": math.nan}
    grad = inverse_rmax.raw_logit.grad
    if grad is None:
        raise RuntimeError("Missing r_max raw gradient; boundary graph is disconnected from inverse r_max.")
    if not torch.isfinite(grad).all():
        raise FloatingPointError("Non-finite gradient in inverse r_max raw_logit.")
    norm = float(torch.linalg.vector_norm(grad.detach()).cpu())
    if require_nonzero and norm == 0.0:
        raise RuntimeError("Zero r_max gradient on first active step; recruitment boundary graph is disconnected from inverse r_max.")
    return {
        "rmax_raw_grad_norm": norm,
        "rmax_raw_grad_min": float(grad.detach().min().cpu()),
        "rmax_raw_grad_max": float(grad.detach().max().cpu()),
        "rmax_grad_finite": 1.0,
    }

def train_one_step_multispecies(
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
    pde_penalty: str = "squared",
    pde_pseudo_huber_delta: float = 1.0,
    bc_penalty: str = "squared",
    bc_pseudo_huber_delta: float = 1.0,
    eps: float = 1e-30,
    bc_eps: float | None,
    bc_g_min: float,
    bc_use_constant_r: bool = False,
    bc_constant_r: float | None = None,
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
    lr_scheduler=None,
    lr_scheduler_name: str = "none",
    wang_weight_batch: str = "fixed",
    weight_calibration_batch: dict[str, torch.Tensor] | None = None,
    time_sampling: str = "uniform",
    causal_loss: str = "off",
    causal_n_chunks: int = 32,
    causal_epsilon: float = 1.0,
    loss_weighting: str = "legacy-wang",
    expert_weight_update_every: int = 1000,
    expert_weight_alpha: float = 0.9,
    expert_weight_min: float | None = None,
    expert_weight_max: float | None = None,
    expert_weight_batch: str = "fixed",
    observation_batch: dict[str, object] | None = None,
    lambda_data: float = 0.0,
    data_loss_eps: float = 1e-30,
    data_time_quadrature_points: int = 1,
    inverse_rmax=None,
    boundary_target_gradient_mode: str = "detached",
) -> dict:
    if wang_weight_batch not in {"fixed", "training"}:
        raise ValueError("wang_weight_batch must be 'fixed' or 'training'.")
    if expert_weight_batch not in {"fixed", "training"}:
        raise ValueError("expert_weight_batch must be 'fixed' or 'training'.")
    if loss_weighting not in {"legacy-wang", "none", "expert-grad-norm"}:
        raise ValueError("loss_weighting must be 'legacy-wang', 'none', or 'expert-grad-norm'.")
    expert_weight_min = weight_min if expert_weight_min is None else expert_weight_min
    expert_weight_max = weight_max if expert_weight_max is None else expert_weight_max

    optimizer.zero_grad(set_to_none=True)
    if inverse_rmax is not None:
        params.r_max = inverse_rmax.current_r_max()

    if collocation_strategy == "uniform":
        batch = sample_pde_batch(
            params=params,
            n_time=n_time,
            n_eval=n_eval,
            t_max_current=t_max_current,
            time_sampling=time_sampling,
            causal_n_chunks=causal_n_chunks,
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
            species_idx=None,
            eps=eps,
            bc_eps=bc_eps,
            bc_g_min=bc_g_min,
            use_constant_recruitment_r=bc_use_constant_r,
            constant_recruitment_r=bc_constant_r,
            causal_loss=causal_loss,
            causal_n_chunks=causal_n_chunks,
            causal_epsilon=causal_epsilon,
            pde_penalty=pde_penalty,
            pde_pseudo_huber_delta=pde_pseudo_huber_delta,
            bc_penalty=bc_penalty,
            bc_pseudo_huber_delta=bc_pseudo_huber_delta,
            boundary_target_gradient_mode=boundary_target_gradient_mode,
        )

    elif collocation_strategy in {"r3", "causal-r3"}:
        if r3_population is None:
            raise ValueError("R3 collocation requires r3_population.")

        r3_population.resample_time_points_(
            params=params,
            t_max_current=t_max_current,
        )

        batch = r3_population.as_batch(params=params)

        pde_weights = None
        if collocation_strategy == "causal-r3" and causal_r3_weight_pde_loss:
            if causal_r3 is None:
                raise ValueError("causal-r3 weighting requires causal_r3.")
            pde_weights = causal_r3.gate(batch["t_slab"], params)

        _, out = compute_pde_loss_r3_slabbed(
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
            species_idx=None,
            eps=eps,
            bc_eps=bc_eps,
            bc_g_min=bc_g_min,
            pde_weights=pde_weights,
            use_constant_recruitment_r=bc_use_constant_r,
            constant_recruitment_r=bc_constant_r,
            causal_loss=causal_loss,
            causal_n_chunks=causal_n_chunks,
            causal_epsilon=causal_epsilon,
            pde_penalty=pde_penalty,
            pde_pseudo_huber_delta=pde_pseudo_huber_delta,
            bc_penalty=bc_penalty,
            bc_pseudo_huber_delta=bc_pseudo_huber_delta,
            boundary_target_gradient_mode=boundary_target_gradient_mode,
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
      
          loss_timestep, timestep_out = compute_timestep_consistency_loss_multispecies(
              model=model,
              params=params,
              n_pp=n_pp,
              t0=t0,
              dt=timestep_dt,
              loss_form=timestep_loss_form,
              detach_step_target=detach_step_target,
              eps=eps,
          )
    out["loss_timestep"] = loss_timestep

    loss_data = out["loss_pde"].new_zeros(())
    data_out = {}

    if lambda_data > 0.0 and observation_batch is not None:
        keep = torch.maximum(
            observation_batch["t_start"],
            observation_batch["t_end"],
        ) <= torch.as_tensor(
            t_max_current,
            dtype=observation_batch["t_start"].dtype,
            device=observation_batch["t_start"].device,
        )

        out["n_data_obs_active"] = torch.as_tensor(
            keep.sum().item(),
            dtype=loss_data.dtype,
            device=loss_data.device,
        )

        if bool(keep.any().detach().cpu()):
            idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
            idx_list = idx.detach().cpu().tolist()
            n_obs = observation_batch["value"].numel()

            observation_batch = {
                k: (
                    v[idx]
                    if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == n_obs
                    else [v[i] for i in idx_list]
                    if isinstance(v, list) and len(v) == n_obs
                    else v
                )
                for k, v in observation_batch.items()
            }

            t_grid = observation_time_grid(
                observation_batch,
                data_time_quadrature_points=data_time_quadrature_points,
            )

            grid_eval = evaluate_log_model_on_points(
                model=model,
                x_scaled=scale_x(torch.log(params.w), params),
                t_scaled=scale_t(t_grid, params),
                params=params,
            )

            pred = predict_observations(
                {"N_grid": grid_eval["N"], "t_grid": t_grid},
                observation_batch,
                params,
                data_time_quadrature_points=data_time_quadrature_points,
            )
            nll = lognormal_nll(pred, observation_batch["value"], observation_batch["sd_log"], eps=data_loss_eps)

            loss_data = nll["loss_data"]
            data_out = {
                "loss_data": loss_data,
                "data_prediction": pred,
                "data_value": observation_batch["value"],
                "data_log_residual": nll["log_residual"],
                "data_loss_contribution": nll["loss_contribution"],
            }

    out.update(data_out)
    out["loss_data"] = loss_data

    if "n_data_obs_active" not in out:
        out["n_data_obs_active"] = loss_data.new_zeros(())
    loss_pde_for_weighting = out["loss_pde"] if loss_weighting == "expert-grad-norm" else out.get("loss_pde_ungated", out["loss_pde"])
    
    raw_losses = {
        "pde": out["loss_pde"],
        "ic": out["loss_ic"],
        "bc": out["loss_bc"],
        "timestep": out["loss_timestep"],
        "data": out["loss_data"],
    }
    
    losses_for_weighting = {
        "pde": lambda_pde * loss_pde_for_weighting,
        "ic": lambda_ic * out["loss_ic"],
        "bc": lambda_bc * out["loss_bc"],
        "timestep": lambda_timestep * out["loss_timestep"],
        "data": lambda_data * out["loss_data"],
    }

    weight_stats = {
        "grad_pde_max": math.nan,
        "grad_ic_mean": math.nan,
        "grad_bc_mean": math.nan,
        "grad_timestep_mean": math.nan,
        "grad_data_mean": math.nan,
        "target_ic": math.nan,
        "target_bc": math.nan,
        "target_timestep": math.nan,
        "target_data": math.nan,
        "hard_set": 0.0,
        "grad_norm_pde_for_weighting": math.nan,
        "grad_norm_ic_for_weighting": math.nan,
        "grad_norm_bc_for_weighting": math.nan,
        "grad_norm_timestep_for_weighting": math.nan,
        "grad_norm_data_for_weighting": math.nan,
        "target_w_pde": math.nan,
        "target_w_ic": math.nan,
        "target_w_bc": math.nan,
        "target_w_timestep": math.nan,
        "target_w_data": math.nan,
        "expert_weight_total_grad_norm": math.nan,
        "expert_weight_hard_set": 0.0,
    }

    active_weight_update_every = expert_weight_update_every if loss_weighting == "expert-grad-norm" else weight_update_every
    active_weight_batch = expert_weight_batch if loss_weighting == "expert-grad-norm" else wang_weight_batch
    weight_update_due = (
        not disable_wang_weights
        and loss_weighting in {"legacy-wang", "expert-grad-norm"}
        and step >= weight_warmup_steps
        and step % active_weight_update_every == 0
    )
    weight_update_used_fixed_batch = 0.0
    calibration_out = None
    calibration_loss_timestep = None
    losses_for_current_weight_update = losses_for_weighting

    if weight_update_due:
        if active_weight_batch == "fixed":
            if weight_calibration_batch is None:
                raise ValueError(
                    "fixed adaptive weighting requires weight_calibration_batch."
                )

            calibration_batch = rescale_fixed_calibration_batch(
                weight_calibration_batch,
                params=params,
                t_max_current=t_max_current,
            )

            _, calibration_out = compute_pde_loss(
                model=model,
                batch=calibration_batch,
                params=params,
                n_pp=n_pp,
                residual_form=residual_form,
                n_init=n_init,
                lambda_pde=lambda_pde,
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc,
                boundary_loss_form=boundary_loss_form,
                species_idx=None,
                eps=eps,
                bc_eps=bc_eps,
                bc_g_min=bc_g_min,
                use_constant_recruitment_r=bc_use_constant_r,
                constant_recruitment_r=bc_constant_r,
                causal_loss=causal_loss,
                causal_n_chunks=causal_n_chunks,
                causal_epsilon=causal_epsilon,
                pde_penalty=pde_penalty,
                pde_pseudo_huber_delta=pde_pseudo_huber_delta,
                bc_penalty=bc_penalty,
                bc_pseudo_huber_delta=bc_pseudo_huber_delta,
                boundary_target_gradient_mode=boundary_target_gradient_mode,
            )
            calibration_loss_pde_for_weighting = calibration_out["loss_pde"] if loss_weighting == "expert-grad-norm" else calibration_out.get(
                "loss_pde_ungated",
                calibration_out["loss_pde"],
            )
            calibration_loss_timestep = calibration_out["loss_pde"].new_zeros(())
            losses_for_current_weight_update = {
                "pde": lambda_pde * calibration_loss_pde_for_weighting,
                "ic": lambda_ic * calibration_out["loss_ic"],
                "bc": lambda_bc * calibration_out["loss_bc"],
                "timestep": lambda_timestep * calibration_loss_timestep,
                "data": lambda_data * loss_data,
            }
            weight_update_used_fixed_batch = 1.0
        elif active_weight_batch == "training":
            losses_for_current_weight_update = losses_for_weighting

        hard_set = hard_set_first_weight_update and not weight_state["has_updated"]

        if loss_weighting == "legacy-wang":
            weight_stats.update(update_wang_gradient_weights_(
                model=model,
                losses=losses_for_current_weight_update,
                weights=loss_weights,
                alpha=weight_alpha,
                min_weight=weight_min,
                max_weight=weight_max,
                hard_set=hard_set,
            ))
        elif loss_weighting == "expert-grad-norm":
            weight_stats.update(update_expert_gradient_norm_weights_(
                model=model,
                losses=losses_for_current_weight_update,
                weights=loss_weights,
                alpha=expert_weight_alpha,
                min_weight=expert_weight_min,
                max_weight=expert_weight_max,
                hard_set=hard_set,
            ))

        weight_state["has_updated"] = True

    loss_unweighted = (
        out["loss_pde"]
        + out["loss_ic"]
        + out["loss_bc"]
        + out["loss_timestep"]
        + out["loss_data"]
    )


    if disable_wang_weights or loss_weighting == "none":
        loss = (
            lambda_pde * out["loss_pde"]
            + lambda_ic * out["loss_ic"]
            + lambda_bc * out["loss_bc"]
            + lambda_timestep * out["loss_timestep"]
            + lambda_data * out["loss_data"]
        )
    else:
        loss = (
            lambda_pde * loss_weights["pde"] * out["loss_pde"]
            + lambda_ic * loss_weights["ic"] * out["loss_ic"]
            + lambda_bc * loss_weights["bc"] * out["loss_bc"]
            + lambda_timestep * loss_weights["timestep"] * out["loss_timestep"]
            + lambda_data * loss_weights["data"] * out["loss_data"]
        )


    out["loss"] = loss

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

    loss.backward()

    grad_norm = total_grad_norm_and_check(model)
    rmax_grad_stats = inverse_rmax_grad_stats(inverse_rmax, require_nonzero=(inverse_rmax is not None and lambda_bc != 0.0 and step == 1))

    optimizer.step()
    if inverse_rmax is not None:
        params.r_max = inverse_rmax.current_r_max()

    if lr_scheduler is not None:
        if lr_scheduler_name == "plateau":
            lr_scheduler.step(float(loss.detach().cpu()))
        else:
            lr_scheduler.step()

    lr = float(optimizer.param_groups[0]["lr"])
    rmax_lr = next((float(g["lr"]) for g in optimizer.param_groups if g.get("name") == "rmax"), math.nan)

    r3_diag = {
        "r3_population_size": math.nan,
        "r3_n_time": math.nan,
        "r3_n_eval_per_time": math.nan,
        "r3_biology_time_loops": math.nan,
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
                {
                    "r3_population_size": float(batch["w_slab"].numel()),
                    "r3_n_time": float(batch["t_slab"].numel()),
                    "r3_n_eval_per_time": float(batch["w_slab"].shape[1]),
                    "r3_biology_time_loops": float(batch["t_slab"].numel()),
                }
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
                    species_idx=None,
                )
            )

        if collocation_strategy == "causal-r3":
            r3_diag.update(causal_r3.update_(out["loss_pde"]))
            gate = causal_r3.gate(batch["t_slab"], params).detach()
            r3_diag["causal_r3_gate_mean"] = float(gate.mean().cpu())

    residual_log = out["residual_log"].detach()
    calibration_nan = torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)
    calibration_residual_log = (
        calibration_out["residual_log"].detach()
        if calibration_out is not None
        else calibration_nan
    )

    base = {
        "step": step,
        "loss": float(out["loss"].detach().cpu()),
        "lr": lr,
        "rmax_lr": rmax_lr,
        "loss_pde": float(out["loss_pde"].detach().cpu()),
        "pde_penalty": pde_penalty,
        "pde_pseudo_huber_delta": float(pde_pseudo_huber_delta),
        "bc_penalty": bc_penalty,
        "bc_pseudo_huber_delta": float(bc_pseudo_huber_delta),
        "loss_ic": float(out["loss_ic"].detach().cpu()),
        "loss_bc": float(out["loss_bc"].detach().cpu()),
        "loss_data": float(out["loss_data"].detach().cpu()),
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
        "time_sampling": time_sampling,
        "effective_n_time": int(batch.get("effective_n_time", batch["t_eval"].numel() if "t_eval" in batch else batch["t_slab"].numel()).detach().cpu() if torch.is_tensor(batch.get("effective_n_time", None)) else batch.get("effective_n_time", batch["t_eval"].numel() if "t_eval" in batch else batch["t_slab"].numel())),
        "causal_loss": causal_loss,
        "causal_n_chunks": int(causal_n_chunks),
        "causal_epsilon": float(causal_epsilon),
        "loss_weighting": "none" if disable_wang_weights else loss_weighting,
        "w_pde": float(loss_weights["pde"]),
        "w_ic": float(loss_weights["ic"]),
        "w_bc": float(loss_weights["bc"]),
        "w_timestep": float(loss_weights["timestep"]),
        "w_data": float(loss_weights.get("data", 1.0)),
        "wang_scaled_loss_pde": float((loss_weights["pde"] * out["loss_pde"]).detach().cpu()),
        "wang_scaled_loss_ic": float((loss_weights["ic"] * out["loss_ic"]).detach().cpu()),
        "wang_scaled_loss_bc": float((loss_weights["bc"] * out["loss_bc"]).detach().cpu()),
        "wang_scaled_loss_timestep": float((loss_weights["timestep"] * out["loss_timestep"]).detach().cpu()),
        "wang_scaled_loss_data": float((loss_weights.get("data", 1.0) * out["loss_data"]).detach().cpu()),
        "loss_pde_for_weighting": float(loss_pde_for_weighting.detach().cpu()),
        "loss_pde_ungated": float(out.get("loss_pde_ungated", out["loss_pde"]).detach().cpu()),
        "loss_pde_gated": float(out.get("loss_pde_gated", out["loss_pde"]).detach().cpu()),
        "loss_pde_causal": float(out.get("loss_pde_causal", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_weight_min": float(out.get("pde_causal_weight_min", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_weight_mean": float(out.get("pde_causal_weight_mean", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_weight_max": float(out.get("pde_causal_weight_max", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_weight_first": float(out.get("pde_causal_weight_first", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_weight_last": float(out.get("pde_causal_weight_last", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_chunk_loss_min": float(out.get("pde_causal_chunk_loss_min", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_chunk_loss_mean": float(out.get("pde_causal_chunk_loss_mean", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_causal_chunk_loss_max": float(out.get("pde_causal_chunk_loss_max", torch.tensor(float("nan"), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_gate_mean": float(out.get("pde_gate_mean", torch.ones((), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_gate_min": float(out.get("pde_gate_min", torch.ones((), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "pde_gate_max": float(out.get("pde_gate_max", torch.ones((), dtype=out["loss_pde"].dtype, device=out["loss_pde"].device)).detach().cpu()),
        "wang_uses_lambda_scaled_losses": True,
        "wang_pde_anchor": "loss_pde_ungated_if_available",
        "wang_weight_batch": wang_weight_batch,
        "weight_update_used_fixed_batch": weight_update_used_fixed_batch,
        "weight_calibration_loss_pde": float((calibration_out["loss_pde"] if calibration_out is not None else calibration_nan).detach().cpu()),
        "weight_calibration_loss_ic": float((calibration_out["loss_ic"] if calibration_out is not None else calibration_nan).detach().cpu()),
        "weight_calibration_loss_bc": float((calibration_out["loss_bc"] if calibration_out is not None else calibration_nan).detach().cpu()),
        "weight_calibration_loss_timestep": float((calibration_loss_timestep if calibration_loss_timestep is not None else calibration_nan).detach().cpu()),
        "weight_calibration_residual_log_abs_mean": scalar_mean(torch.abs(calibration_residual_log)),
        "objective_loss_pde": float((lambda_pde * loss_weights["pde"] * out["loss_pde"]).detach().cpu()),
        "objective_loss_ic": float((lambda_ic * loss_weights["ic"] * out["loss_ic"]).detach().cpu()),
        "objective_loss_bc": float((lambda_bc * loss_weights["bc"] * out["loss_bc"]).detach().cpu()),
        "objective_loss_timestep": float((lambda_timestep * loss_weights["timestep"] * out["loss_timestep"]).detach().cpu()),
        "objective_loss_data": float((lambda_data * loss_weights.get("data", 1.0) * out["loss_data"]).detach().cpu()),
        
        # Backward-compatible aliases for old plotting/history code.
        "weighted_loss_pde": float((lambda_pde * loss_weights["pde"] * out["loss_pde"]).detach().cpu()),
        "weighted_loss_ic": float((lambda_ic * loss_weights["ic"] * out["loss_ic"]).detach().cpu()),
        "weighted_loss_bc": float((lambda_bc * loss_weights["bc"] * out["loss_bc"]).detach().cpu()),
        "weighted_loss_timestep": float((lambda_timestep * loss_weights["timestep"] * out["loss_timestep"]).detach().cpu()),
        "weighted_loss_data": float((lambda_data * loss_weights.get("data", 1.0) * out["loss_data"]).detach().cpu()),
        "grad_pde_max_for_weighting": weight_stats["grad_pde_max"],
        "grad_ic_mean_for_weighting": weight_stats["grad_ic_mean"],
        "grad_bc_mean_for_weighting": weight_stats["grad_bc_mean"],
        "target_w_ic": weight_stats["target_ic"] if math.isfinite(weight_stats["target_ic"]) else weight_stats["target_w_ic"],
        "target_w_bc": weight_stats["target_bc"] if math.isfinite(weight_stats["target_bc"]) else weight_stats["target_w_bc"],
        "grad_timestep_mean_for_weighting": weight_stats["grad_timestep_mean"],
        "target_w_timestep": weight_stats["target_timestep"] if math.isfinite(weight_stats["target_timestep"]) else weight_stats["target_w_timestep"],
        "grad_data_mean_for_weighting": weight_stats["grad_data_mean"],
        "target_w_data": weight_stats["target_data"] if math.isfinite(weight_stats["target_data"]) else weight_stats["target_w_data"],
        "grad_norm_pde_for_weighting": weight_stats["grad_norm_pde_for_weighting"],
        "grad_norm_ic_for_weighting": weight_stats["grad_norm_ic_for_weighting"],
        "grad_norm_bc_for_weighting": weight_stats["grad_norm_bc_for_weighting"],
        "grad_norm_timestep_for_weighting": weight_stats["grad_norm_timestep_for_weighting"],
        "grad_norm_data_for_weighting": weight_stats["grad_norm_data_for_weighting"],
        "target_w_pde": weight_stats["target_w_pde"],
        "expert_weight_total_grad_norm": weight_stats["expert_weight_total_grad_norm"],
        "loss_weighted": float(loss.detach().cpu()),
        "loss_unweighted": float(loss_unweighted.detach().cpu()),
        "boundary_loss_form": boundary_loss_form,
        "bc_eps": float(bc_eps if bc_eps is not None else eps),
        "bc_g_min": float(bc_g_min),
        "bc_valid_count": float(out.get("bc_valid_count", torch.tensor(float("nan"))).detach().cpu()),
        "bc_total_count": float(out.get("bc_total_count", torch.tensor(float("nan"))).detach().cpu()),
        "bc_valid_fraction": float(out.get("bc_valid_fraction", torch.tensor(float("nan"))).detach().cpu()),
        "bc_invalid_fraction": float(out.get("bc_invalid_fraction", torch.tensor(float("nan"))).detach().cpu()),
        "bc_invalid_g_fraction": float(out.get("bc_invalid_g_fraction", torch.tensor(float("nan"))).detach().cpu()),
        "bc_invalid_recruitment_fraction": float(out.get("bc_invalid_recruitment_fraction", torch.tensor(float("nan"))).detach().cpu()),
        "bc_nonfinite_fraction": float(out.get("bc_nonfinite_fraction", torch.tensor(float("nan"))).detach().cpu()),
        "bc_target_log_N_min": float(out.get("bc_target_log_N_min", torch.tensor(float("nan"))).detach().cpu()),
        "bc_target_log_N_max": float(out.get("bc_target_log_N_max", torch.tensor(float("nan"))).detach().cpu()),
        "bc_target_N_min": float(out.get("bc_target_N_min", torch.tensor(float("nan"))).detach().cpu()),
        "bc_target_N_max": float(out.get("bc_target_N_max", torch.tensor(float("nan"))).detach().cpu()),

        # Backward-compatible diagnostic aliases.
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
        "bc_use_constant_r": 1.0 if bc_use_constant_r else 0.0,
        "bc_constant_r": float(bc_constant_r) if bc_constant_r is not None else math.nan,
        "n_data_obs": float(out.get("n_data_obs_active", torch.zeros_like(out["loss_data"])).detach().cpu()),
        "data_pred_min": scalar_min(out["data_prediction"]) if "data_prediction" in out else math.nan,
        "data_pred_max": scalar_max(out["data_prediction"]) if "data_prediction" in out else math.nan,
        "data_obs_min": scalar_min(observation_batch["value"]) if observation_batch is not None and lambda_data > 0.0 else math.nan,
        "data_obs_max": scalar_max(observation_batch["value"]) if observation_batch is not None and lambda_data > 0.0 else math.nan,
        "data_log_residual_abs_mean": scalar_mean(torch.abs(out["data_log_residual"])) if "data_log_residual" in out else math.nan,
        "data_log_residual_abs_max": scalar_max(torch.abs(out["data_log_residual"])) if "data_log_residual" in out else math.nan,
    }

    if inverse_rmax is not None:
        with torch.no_grad():
            cur_r = inverse_rmax.current_r_max().detach()
            cur_log = inverse_rmax.current_log_r_max().detach()
            ratio = cur_r / inverse_rmax.initial_r_max
        base.update(rmax_grad_stats)
        base.update({
            "rmax_min": scalar_min(cur_r), "rmax_mean": scalar_mean(cur_r), "rmax_max": scalar_max(cur_r),
            "log_rmax_min": scalar_min(cur_log), "log_rmax_mean": scalar_mean(cur_log), "log_rmax_max": scalar_max(cur_log),
            "rmax_ratio_min": scalar_min(ratio), "rmax_ratio_mean": scalar_mean(ratio), "rmax_ratio_max": scalar_max(ratio),
        })
    else:
        base.update(rmax_grad_stats)
        base.update({k: math.nan for k in ["rmax_min","rmax_mean","rmax_max","log_rmax_min","log_rmax_mean","log_rmax_max","rmax_ratio_min","rmax_ratio_mean","rmax_ratio_max"]})
    return base
