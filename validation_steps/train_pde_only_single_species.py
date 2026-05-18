from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import scale_x, scale_t
from PINNmizer.pde_residual import (
    sample_pde_batch,
    compute_pde_loss,
    compute_pde_residual,
    evaluate_log_model_on_points,
)
from validation_steps.pinn_diagnostics import (
    make_fixed_pde_batch,
    make_fixed_pde_batch_from_csv,
    compute_fixed_diagnostics,
    append_diagnostic_row,
    save_latest_metrics_table,
    save_training_diagnostic_plots,
    save_fixed_grid_fields_and_plots,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_width: int = 64,
        hidden_layers: int = 3,
    ):
        super().__init__()

        layers = []
        last_dim = in_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.Tanh())
            last_dim = hidden_width

        layers.append(nn.Linear(last_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / "pde_only_single_species" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(x: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)


def save_history(history: list[dict], run_dir: Path) -> None:
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)


def total_grad_norm_and_check(model: nn.Module) -> float:
    total = None

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        if not torch.isfinite(p.grad).all():
            raise FloatingPointError(f"Non-finite gradient in parameter: {name}")

        val = (p.grad.detach() ** 2).sum()

        if total is None:
            total = val
        else:
            total = total + val

    if total is None:
        return 0.0

    return float(torch.sqrt(total).cpu())


def scalar_min(x: torch.Tensor) -> float:
    return float(torch.min(x.detach()).cpu())


def scalar_max(x: torch.Tensor) -> float:
    return float(torch.max(x.detach()).cpu())


def scalar_mean(x: torch.Tensor) -> float:
    return float(torch.mean(x.detach()).cpu())

def _to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def parse_fraction_schedule(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]

    if not values:
        raise ValueError("Causal fraction schedule is empty.")

    if any(v <= 0.0 or v > 1.0 for v in values):
        raise ValueError(
            "All causal fraction schedule values must be in (0, 1]. "
            f"Got {values}."
        )

    if values[-1] != 1.0:
        values.append(1.0)

    return values


def causal_time_fraction(
    *,
    step: int,
    mode: str,
    start_fraction: float,
    ramp_steps: int,
    step_fractions: str,
) -> float:
    if mode == "off":
        return 1.0

    if not (0.0 < start_fraction <= 1.0):
        raise ValueError(f"start_fraction must be in (0, 1], got {start_fraction}.")

    if ramp_steps <= 0:
        return 1.0

    progress = min(1.0, max(0.0, (step - 1) / ramp_steps))

    if mode == "linear":
        return start_fraction + progress * (1.0 - start_fraction)

    if mode == "step":
        levels = parse_fraction_schedule(step_fractions)
        idx = min(len(levels) - 1, int(math.floor(progress * len(levels))))
        return levels[idx]

    raise ValueError("mode must be 'off', 'linear', or 'step'.")


def causal_t_max_current(
    *,
    params,
    step: int,
    mode: str,
    start_fraction: float,
    ramp_steps: int,
    step_fractions: str,
) -> tuple[float, float]:
    t_min = _to_float(params.t_min)
    t_max = _to_float(params.t_max)

    frac = causal_time_fraction(
        step=step,
        mode=mode,
        start_fraction=start_fraction,
        ramp_steps=ramp_steps,
        step_fractions=step_fractions,
    )

    t_current = t_min + frac * (t_max - t_min)

    return frac, t_current

def initialise_final_bias_from_ic(
    *,
    model: nn.Module,
    n_init: torch.Tensor,
    eps: float,
) -> None:
    """
    Initialise the final layer bias so random predictions start near the
    average initial log-abundance scale instead of log_N ~= 0, i.e. N ~= 1.
    """
    final_linear = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            final_linear = module
            break

    if final_linear is None:
        raise ValueError("Could not find final nn.Linear layer.")

    n_init = n_init.detach()
    if n_init.ndim == 1:
        n_init = n_init.reshape(1, -1)

    target_bias = torch.log(torch.clamp(n_init, min=eps)).mean(dim=1)

    if final_linear.bias is None:
        raise ValueError("Final nn.Linear layer has no bias.")

    if final_linear.bias.numel() != target_bias.numel():
        raise ValueError(
            f"Final bias has {final_linear.bias.numel()} entries, "
            f"but IC target has {target_bias.numel()} species."
        )

    with torch.no_grad():
        final_linear.bias.copy_(target_bias.to(
            dtype=final_linear.bias.dtype,
            device=final_linear.bias.device,
        ))

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

def _flat_loss_grad(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor | None:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    flat = [
        g.detach().reshape(-1)
        for g in grads
        if g is not None
    ]

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
    """
    Wang-style PINN gradient-statistic weighting.

    PDE is the anchor:
        w_pde = 1

    IC and BC are adjusted so their mean gradient magnitude is comparable
    to the max PDE gradient magnitude.
    """
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

        if hard_set:
            weights[name] = target
        else:
            weights[name] = (1.0 - alpha) * weights[name] + alpha * target
            
        stats[f"grad_{name}_mean"] = float(grad_mean.cpu())
        stats[f"target_{name}"] = target

    return stats

def save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        run_dir / f"model_step_{step}.pt",
    )


def save_final_predictions(
    *,
    run_dir: Path,
    model: nn.Module,
    params,
    n_times: int = 50,
) -> None:
    model.eval()

    dtype = params.w.dtype
    device = params.w.device

    t_min = torch.as_tensor(params.t_min, dtype=dtype, device=device)
    t_max = torch.as_tensor(params.t_max, dtype=dtype, device=device)

    t_grid = torch.linspace(t_min, t_max, n_times, dtype=dtype, device=device)
    x_grid = torch.log(params.w)

    t_scaled = scale_t(t_grid, params)
    x_scaled = scale_x(x_grid, params)

    with torch.no_grad():
        out = evaluate_log_model_on_points(
            model=model,
            x_scaled=x_scaled,
            t_scaled=t_scaled,
            params=params,
        )

    log_N = out["log_N"][:, 0, :].detach().cpu()
    N = out["N"][:, 0, :].detach().cpu()

    tt = t_grid.detach().cpu()[:, None].expand(-1, params.w.numel())
    ww = params.w.detach().cpu()[None, :].expand(n_times, -1)
    xx = x_grid.detach().cpu()[None, :].expand(n_times, -1)
    xs = x_scaled.detach().cpu()[None, :].expand(n_times, -1)
    ts = t_scaled.detach().cpu()[:, None].expand(-1, params.w.numel())

    df = pd.DataFrame(
        {
            "t": tt.reshape(-1).numpy(),
            "w": ww.reshape(-1).numpy(),
            "x": xx.reshape(-1).numpy(),
            "x_scaled": xs.reshape(-1).numpy(),
            "t_scaled": ts.reshape(-1).numpy(),
            "log_N": log_N.reshape(-1).numpy(),
            "N": N.reshape(-1).numpy(),
        }
    )

    df.to_csv(run_dir / "final_predictions_grid.csv", index=False)


def save_final_residual_sample(
    *,
    run_dir: Path,
    model: nn.Module,
    params,
    n_pp: torch.Tensor,
    n_time: int,
    n_eval: int,
) -> None:
    model.eval()

    batch = sample_pde_batch(
        params=params,
        n_time=n_time,
        n_eval=n_eval,
    )

    out = compute_pde_residual(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
    )

    tt = batch["t_eval"].detach().cpu()[:, None].expand(n_time, n_eval)
    ww = batch["w_eval"].detach().cpu()[None, :].expand(n_time, n_eval)

    def flat(name: str):
        return out[name][:, 0, :].detach().cpu().reshape(-1).numpy()

    df = pd.DataFrame(
        {
            "t_eval": tt.reshape(-1).numpy(),
            "w_eval": ww.reshape(-1).numpy(),
            "residual_log": flat("residual_log"),
            "residual": flat("residual"),
            "log_N_eval": flat("log_N_eval"),
            "N_eval": flat("N_eval"),
            "g_eval": flat("g_eval"),
            "dg_dw": flat("dg_dw"),
            "mu_eval": flat("mu_eval"),
        }
    )

    df.to_csv(run_dir / "final_residual_sample.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="py_inputs_ns_first_species")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-time", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--residual-form", choices=["log", "physical"], default="log")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-update-every", type=int, default=10)
    parser.add_argument("--weight-warmup-steps", type=int, default=10)
    parser.add_argument("--weight-alpha", type=float, default=0.05)
    parser.add_argument("--weight-min", type=float, default=1e-3)
    parser.add_argument("--weight-max", type=float, default=1e3)
    parser.add_argument("--diag-every", type=int, default=0)
    parser.add_argument("--diag-grad-every", type=int, default=0)
    parser.add_argument("--diag-n-time", type=int, default=31)
    parser.add_argument("--diag-n-eval", type=int, default=100)
    parser.add_argument("--diag-use-mizer-x-grid", action="store_true")
    parser.add_argument("--diag-grid-csv", default=None)
    parser.add_argument("--diag-final-n-time", type=int, default=61)
    parser.add_argument("--diag-final-n-eval", type=int, default=160)

    parser.add_argument(
        "--boundary-loss-form",
        choices=["log", "physical", "relative"],
        default="log",
    )
    
    parser.add_argument("--loss-eps", type=float, default=1e-30)
    
    parser.add_argument(
        "--bc-eps",
        type=float,
        default=None,
        help="Optional separate floor for log boundary loss. If omitted, uses --loss-eps.",
    )
    
    parser.add_argument("--initial-w-pde", type=float, default=1.0)
    parser.add_argument("--initial-w-ic", type=float, default=1.0)
    parser.add_argument("--initial-w-bc", type=float, default=1e-3)
    
    parser.add_argument(
        "--hard-set-first-weight-update",
        action="store_true",
        default=True,
    )
    
    parser.add_argument(
        "--no-hard-set-first-weight-update",
        dest="hard_set_first_weight_update",
        action="store_false",
    )
    
    parser.add_argument(
        "--init-final-bias-from-ic",
        action="store_true",
        default=True,
    )
    
    parser.add_argument(
        "--no-init-final-bias-from-ic",
        dest="init_final_bias_from_ic",
        action="store_false",
    )

    parser.add_argument(
        "--causal-curriculum",
        choices=["off", "linear", "step"],
        default="linear",
    )
    parser.add_argument("--causal-start-fraction", type=float, default=0.05)
    parser.add_argument("--causal-ramp-steps", type=int, default=1500)
    parser.add_argument(
        "--causal-step-fractions",
        default="0.05,0.10,0.20,0.40,0.70,1.0",
    )
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=0.0)
    parser.add_argument("--disable-wang-weights", action="store_true") 

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = make_run_dir()

    params, n_init, n_pp = load_mizer_inputs(
        args.input_dir,
        dtype=torch.float64,
        device=args.device,
    )

    diag_every = args.diag_every if args.diag_every > 0 else args.print_every
    diag_grad_every = args.diag_grad_every if args.diag_grad_every > 0 else diag_every
    
    if args.diag_grid_csv is not None:
        fixed_diag_batch = make_fixed_pde_batch_from_csv(
            params=params,
            path=args.diag_grid_csv,
        )
    else:
        fixed_diag_batch = make_fixed_pde_batch(
            params=params,
            n_time=args.diag_n_time,
            n_eval=args.diag_n_eval,
            use_mizer_x_grid=args.diag_use_mizer_x_grid,
        )  

    n_species = params.interaction.shape[0]

    if n_species != 1:
        raise ValueError(f"Expected one species, got {n_species}")

    model = MLP(
        in_dim=2,
        out_dim=n_species,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
    ).to(dtype=torch.float64, device=params.w.device)

    if args.init_final_bias_from_ic:
        initialise_final_bias_from_ic(
            model=model,
            n_init=n_init,
            eps=args.loss_eps,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_weights = {
        "pde": args.initial_w_pde,
        "ic": args.initial_w_ic,
        "bc": args.initial_w_bc,
    }
    
    weight_state = {
        "has_updated": False,
    }
    
    config = {
        "input_dir": args.input_dir,
        "run_dir": str(run_dir),
        "n_steps": args.n_steps,
        "n_time": args.n_time,
        "n_eval": args.n_eval,
        "residual_form": args.residual_form,
        "learning_rate": args.lr,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "dtype": "torch.float64",
        "device": str(params.w.device),
        "t_min": float(params.t_min),
        "t_max": float(params.t_max),
        "print_every": args.print_every,
        "checkpoint_every": args.checkpoint_every,
        "warmup_steps": args.warmup_steps,
        "weighting": "wang_gradient_statistics",
        "weight_update_every": args.weight_update_every,
        "weight_warmup_steps": args.weight_warmup_steps,
        "weight_alpha": args.weight_alpha,
        "weight_min": args.weight_min,
        "weight_max": args.weight_max,
        "diag_every": diag_every,
        "diag_grad_every": diag_grad_every,
        "diag_n_time": args.diag_n_time,
        "diag_n_eval": args.diag_n_eval,
        "diag_use_mizer_x_grid": args.diag_use_mizer_x_grid,
        "diag_grid_csv": args.diag_grid_csv,
        "diag_final_n_time": args.diag_final_n_time,
        "diag_final_n_eval": args.diag_final_n_eval,
        "boundary_loss_form": args.boundary_loss_form,
        "loss_eps": args.loss_eps,
        "bc_eps": args.bc_eps,
        "initial_w_pde": args.initial_w_pde,
        "initial_w_ic": args.initial_w_ic,
        "initial_w_bc": args.initial_w_bc,
        "hard_set_first_weight_update": args.hard_set_first_weight_update,
        "init_final_bias_from_ic": args.init_final_bias_from_ic,
        "causal_curriculum": args.causal_curriculum,
        "causal_start_fraction": args.causal_start_fraction,
        "causal_ramp_steps": args.causal_ramp_steps,
        "causal_step_fractions": args.causal_step_fractions,
        "lambda_pde": args.lambda_pde,
        "lambda_ic": args.lambda_ic,
        "lambda_bc": args.lambda_bc,
        "disable_wang_weights": args.disable_wang_weights,
        "note": (
            "Composite PINN loss with PDE, IC, and recruitment boundary terms. "
            "IC/BC weights are adapted using Wang-style gradient statistics."
        ),
    }

    save_json(config, run_dir / "config.json")

    history = []
    timing = {
        "warmup_steps": args.warmup_steps,
        "seconds_per_step": math.nan,
        "estimated_seconds_for_2000_steps": math.nan,
        "actual_total_seconds": math.nan,
    }

    start_time = time.perf_counter()

    try:
        for step in range(1, args.n_steps + 1):
          
            current_causal_fraction, current_t_max = causal_t_max_current(
                params=params,
                step=step,
                mode=args.causal_curriculum,
                start_fraction=args.causal_start_fraction,
                ramp_steps=args.causal_ramp_steps,
                step_fractions=args.causal_step_fractions,
            )
          
            row = train_one_step(
                model=model,
                optimizer=optimizer,
                params=params,
                n_pp=n_pp,
                n_init=n_init,
                n_time=args.n_time,
                n_eval=args.n_eval,
                residual_form=args.residual_form,
                step=step,
                start_time=start_time,
                loss_weights=loss_weights,
                weight_update_every=args.weight_update_every,
                weight_warmup_steps=args.weight_warmup_steps,
                weight_alpha=args.weight_alpha,
                weight_min=args.weight_min,
                weight_max=args.weight_max,  
                boundary_loss_form=args.boundary_loss_form,
                eps=args.loss_eps,
                bc_eps=args.bc_eps,
                weight_state=weight_state,
                hard_set_first_weight_update=args.hard_set_first_weight_update,
                causal_fraction=current_causal_fraction,
                t_max_current=current_t_max,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                lambda_bc=args.lambda_bc,
                disable_wang_weights=args.disable_wang_weights,
            )

            history.append(row)

            if step == 1 or step % diag_every == 0:
                diag_row = compute_fixed_diagnostics(
                    model=model,
                    params=params,
                    n_pp=n_pp,
                    n_init=n_init,
                    fixed_batch=fixed_diag_batch,
                    residual_form=args.residual_form,
                    boundary_loss_form=args.boundary_loss_form,
                    species_idx=0,
                    compute_grad_norms=(step == 1 or step % diag_grad_every == 0),
                )
            
                diag_row = {
                    "step": step,
                    **diag_row,
                }
            
                append_diagnostic_row(
                    diag_row,
                    run_dir / "fixed_diagnostic_history.csv",
                )
            
                save_latest_metrics_table(
                    metrics=diag_row,
                    outdir=run_dir / "diagnostics",
                )
            
                print(
                    f"diag step={step:5d} "
                    f"fixed_pde={diag_row['fixed_loss_pde']:.6e} "
                    f"fixed_ic={diag_row['fixed_loss_ic']:.6e} "
                    f"fixed_bc={diag_row['fixed_loss_bc']:.6e} "
                    f"grad_pde={diag_row['grad_norm_pde']:.6e} "
                    f"grad_ic={diag_row['grad_norm_ic']:.6e} "
                    f"grad_bc={diag_row['grad_norm_bc']:.6e} "
                    f"res_p95={diag_row['fixed_residual_log_abs_p95']:.6e}"
                )

            if step == args.warmup_steps:
                elapsed = time.perf_counter() - start_time
                seconds_per_step = elapsed / args.warmup_steps
                timing["seconds_per_step"] = seconds_per_step
                timing["estimated_seconds_for_2000_steps"] = seconds_per_step * 2000

                pd.DataFrame([timing]).to_csv(
                    run_dir / "timing_summary.csv",
                    index=False,
                )

                print(
                    "Timing:",
                    f"{seconds_per_step:.4f} sec/step;",
                    f"estimated 2000 steps = {seconds_per_step * 2000 / 60:.2f} min",
                )

            if step % args.print_every == 0 or step == 1:
                print(
                    f"step={step:5d} "
                    f"loss={row['loss']:.6e} "
                    f"loss_pde={row['loss_pde']:.6e} "
                    f"loss_ic={row['loss_ic']:.6e} "
                    f"loss_bc={row['loss_bc']:.6e} "
                    f"grad_norm={row['grad_norm']:.6e} "
                    f"res_abs_mean={row['residual_log_abs_mean']:.6e} "
                    f"res_abs_max={row['residual_log_abs_max']:.6e} "
                    f"elapsed={row['seconds_elapsed']:.1f}s "
                    f"loss_unw={row['loss_unweighted']:.3e} "
                    f"w_pde={row['w_pde']:.3e} "
                    f"w_ic={row['w_ic']:.3e} "
                    f"w_bc={row['w_bc']:.3e} "
                    f"wpde={row['weighted_loss_pde']:.3e} "
                    f"wic={row['weighted_loss_ic']:.3e} "
                    f"wbc={row['weighted_loss_bc']:.3e} "
                    f"bc_clamp_flux={row['frac_flux_left_clamped']:.3e} "
                    f"bc_clamp_rec={row['frac_recruitment_flux_clamped']:.3e} "
                )

                save_history(history, run_dir)

            if step % args.checkpoint_every == 0:
                save_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                )

        timing["actual_total_seconds"] = time.perf_counter() - start_time

        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )

        save_history(history, run_dir)

        torch.save(
            {
                "step": args.n_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            run_dir / "model_final.pt",
        )

        save_final_predictions(
            run_dir=run_dir,
            model=model,
            params=params,
            n_times=50,
        )

        save_final_residual_sample(
            run_dir=run_dir,
            model=model,
            params=params,
            n_pp=n_pp,
            n_time=args.n_time,
            n_eval=args.n_eval,
        )

        save_training_diagnostic_plots(run_dir)
        
        save_fixed_grid_fields_and_plots(
            model=model,
            params=params,
            n_pp=n_pp,
            n_init=n_init,
            outdir=run_dir / "fixed_grid_diagnostics",
            residual_form=args.residual_form,
            boundary_loss_form="log",
            species_idx=0,
            n_time=args.diag_final_n_time,
            n_eval=args.diag_final_n_eval,
        )

    except Exception:
        save_history(history, run_dir)
        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )
        raise

    print("Finished.")
    print(f"Run directory: {run_dir}")
    from validation_steps.pde_output_diagnostics import save_output_surface_diagnostics

    save_output_surface_diagnostics(
       model=model,
       params=params,
       outdir=run_dir / "output_diagnostics",
       n_t=101,
       n_x=200,
       species_idx=0,
    )


if __name__ == "__main__":
    main()


