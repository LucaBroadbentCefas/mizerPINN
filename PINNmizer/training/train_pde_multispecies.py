from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PINNmizer.pinn.models import FactorizedLinear, build_pinn_model
from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.training.checkpointing import save_checkpoint
from PINNmizer.training.config import (
    _to_float,
    causal_t_max_current,
    causal_time_fraction,
    parse_fraction_schedule,
)
from PINNmizer.training.outputs import (
    save_history,
    save_json,
    save_run_command,
)
from PINNmizer.training.outputs_multispecies import (
    save_final_predictions_multispecies,
    save_final_residual_sample_multispecies,
)
from PINNmizer.training.loop_multispecies import train_one_step_multispecies, total_grad_norm_and_check, scalar_min, scalar_max, scalar_mean
from PINNmizer.pinn.state_scale import set_state_scale_from_initial_condition, DEFAULT_STATE_SCALE_EPS
from PINNmizer.pinn.r3 import make_r3_population, CausalR3
from PINNmizer.inverse_parameters import BoundedLogRMax

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from PINNmizer.io import load_mizer_inputs
from PINNmizer.io_observations import load_observation_csv
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.observation_operators import observation_time_grid, predict_observations
from PINNmizer.pinn.data_losses import lognormal_nll
from PINNmizer.params import scale_x, scale_t, active_grid_mask
from PINNmizer.diagnostics.fixed_grid import (
    make_fixed_pde_batch,
    make_fixed_pde_batch_from_csv,
    compute_fixed_diagnostics,
)
from PINNmizer.diagnostics.outputs import (
    append_diagnostic_row,
    save_latest_metrics_table,
)
from PINNmizer.diagnostics.plots import (
    save_training_diagnostic_plots,
)
from PINNmizer.diagnostics.fields_multispecies import (
    save_fixed_grid_fields_and_plots_multispecies,
)
#


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / "pde_multispecies" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_checkpoint_weights(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str | Path,
    device,
    load_optimizer_state: bool = False,
    inverse_rmax=None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model_state_dict': {checkpoint_path}")

    checkpoint_param = (checkpoint.get("config") or {}).get("state_parameterization", "log-n")
    requested_param = getattr(model, "state_parameterization", checkpoint_param)
    if checkpoint_param != requested_param:
        raise ValueError(f"Checkpoint state_parameterization={checkpoint_param!r} does not match requested {requested_param!r}.")

    model.load_state_dict(checkpoint["model_state_dict"])
    inverse_loaded = False
    if inverse_rmax is not None and "inverse_parameter_state_dict" in checkpoint:
        inverse_rmax.load_state_dict(checkpoint["inverse_parameter_state_dict"])
        inverse_loaded = True

    optimizer_loaded = False
    if load_optimizer_state:
        if optimizer is None:
            raise ValueError("optimizer must be provided when load_optimizer_state=True.")
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint has no 'optimizer_state_dict': {checkpoint_path}")
        ck_groups = checkpoint["optimizer_state_dict"].get("param_groups", [])
        cur_names = [g.get("name") for g in optimizer.param_groups]
        ck_names = [g.get("name") for g in ck_groups]
        if cur_names != ck_names:
            raise ValueError(f"Incompatible optimizer parameter groups in checkpoint: {ck_names} vs current {cur_names}.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True

    return {
        "path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step", None),
        "optimizer_loaded": optimizer_loaded,
        "inverse_parameter_loaded": inverse_loaded,
        "checkpoint_config": checkpoint.get("config", None),
    }





def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def build_lr_scheduler(*, optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.lr_scheduler == "none":
        return None

    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.n_steps),
            eta_min=args.lr_min,
        )

    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.lr_step_size),
            gamma=args.lr_gamma,
        )

    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.lr_gamma,
            patience=max(0, args.lr_plateau_patience),
        )

    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

def initialise_final_bias_from_ic(
    *,
    model: nn.Module,
    n_init: torch.Tensor,
    params,
    eps: float,
    state_parameterization: str = "log-n",
) -> None:
    """
    Initialise the final layer bias so random predictions start near the
    average initial log-abundance scale instead of log_N ~= 0, i.e. N ~= 1.
    """
    final_linear = None
    for module in reversed(list(model.modules())):
        if isinstance(module, (nn.Linear, FactorizedLinear)):
            final_linear = module
            break

    if final_linear is None:
        raise ValueError("Could not find final linear layer.")

    n_init = n_init.detach()
    if n_init.ndim == 1:
        n_init = n_init.reshape(1, -1)

    log_init = torch.log(torch.clamp(n_init, min=eps))
    if state_parameterization == "log-u":
        log_init = log_init - torch.log(torch.clamp(n_init, min=eps))
    mask = active_grid_mask(params).to(dtype=log_init.dtype, device=log_init.device)
    
    denom = mask.sum(dim=1)
    if not bool((denom > 0).all().detach().cpu()):
        raise ValueError("Cannot initialise final bias: active-grid mask has zero active entries.")
    
    target_bias = (log_init * mask).sum(dim=1) / denom
        
    if final_linear.bias is None:
        raise ValueError("Final linear layer has no bias.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="validation/fixtures/pde_multispecies")
    parser.add_argument("--species-mode", default="all")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-time", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "step", "plateau"], default="none")
    parser.add_argument("--lr-step-size", type=int, default=500)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--lr-plateau-patience", type=int, default=50)
    parser.add_argument("--model-arch", choices=["mlp", "fourier"], default="mlp")
    parser.add_argument("--fourier-num-features", type=int, default=64)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--fourier-include-raw-input", action="store_true")
    parser.add_argument("--fourier-seed", type=int, default=None)
    parser.add_argument("--weight-factorization", choices=["none", "rwf"], default="none")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--rwf-apply-to", choices=["hidden", "all"], default="all")
    parser.add_argument("--rwf-base-init", choices=["pytorch", "xavier_uniform", "xavier_normal"], default="pytorch")
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--residual-form", choices=["log", "scaled", "physical"], default="log")
    parser.add_argument("--pde-penalty", choices=["squared", "pseudo-huber"], default="squared")
    parser.add_argument("--pde-pseudo-huber-delta", type=float, default=1.0)
    parser.add_argument("--state-parameterization", choices=["log-n", "log-u"], default="log-n")
    parser.add_argument("--state-scale-eps", type=float, default=DEFAULT_STATE_SCALE_EPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-update-every", type=int, default=10)
    parser.add_argument("--weight-warmup-steps", type=int, default=10)
    parser.add_argument("--weight-alpha", type=float, default=0.05)
    parser.add_argument("--weight-min", type=float, default=1e-3)
    parser.add_argument("--weight-max", type=float, default=1e3)
    parser.add_argument(
        "--wang-weight-batch",
        choices=["fixed", "training"],
        default="fixed",
        help=(
            "Batch source for Wang gradient statistics. 'fixed' uses the "
            "fixed diagnostic batch; 'training' preserves the previous behavior."
        ),
    )
    parser.add_argument("--time-sampling", choices=["uniform", "stratified"], default="uniform")
    parser.add_argument("--causal-loss", choices=["off", "expert"], default="off")
    parser.add_argument("--causal-n-chunks", type=int, default=32)
    parser.add_argument("--causal-epsilon", type=float, default=1.0)
    parser.add_argument("--loss-weighting", choices=["legacy-wang", "none", "expert-grad-norm"], default="legacy-wang")
    parser.add_argument("--expert-weight-update-every", type=int, default=1000)
    parser.add_argument("--expert-weight-alpha", type=float, default=0.9)
    parser.add_argument("--expert-weight-min", type=float, default=None)
    parser.add_argument("--expert-weight-max", type=float, default=None)
    parser.add_argument("--expert-weight-batch", choices=["fixed", "training"], default="fixed")
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
    parser.add_argument("--bc-penalty", choices=["squared", "pseudo-huber"], default="squared")
    parser.add_argument("--bc-pseudo-huber-delta", type=float, default=1.0)
    
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
    parser.add_argument("--initial-w-timestep", type=float, default=1.0)
    parser.add_argument("--initial-w-data", type=float, default=1.0)
    
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
    parser.add_argument("--lambda-timestep", type=float, default=0.0)
    parser.add_argument("--estimate-rmax", action="store_true", default=False)
    parser.add_argument("--rmax-lr", type=float, default=1e-3)
    parser.add_argument("--rmax-log-lower", type=float, default=0.0)
    parser.add_argument("--rmax-log-upper", type=float, default=50.0)
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--lambda-data", type=float, default=0.0)
    parser.add_argument("--data-default-cv", type=float, default=0.3)
    parser.add_argument("--data-loss-eps", type=float, default=1e-30)
    parser.add_argument("--data-time-quadrature-points", type=int, default=1)
    parser.add_argument("--timestep-loss-form", choices=["physical", "log", "relative"], default="physical")
    parser.add_argument("--detach-step-target", action="store_true", default=True)
    parser.add_argument("--no-detach-step-target", dest="detach_step_target", action="store_false")
    parser.add_argument("--timestep-dt", type=float, default=None)
    parser.add_argument("--timestep-n-pairs", type=int, default=1)

    parser.add_argument(
        "--collocation-strategy",
        choices=["uniform", "r3", "causal-r3"],
        default="uniform",
    )
    parser.add_argument("--r3-population-size", type=int, default=None)
    parser.add_argument("--r3-update-every", type=int, default=1)
    parser.add_argument("--r3-warmup-steps", type=int, default=0)
    parser.add_argument("--r3-score-form", choices=["abs", "squared"], default="abs")
    parser.add_argument("--r3-seed", type=int, default=None)

    parser.add_argument("--causal-r3-alpha", type=float, default=5.0)
    parser.add_argument("--causal-r3-gamma-init", type=float, default=-0.5)
    parser.add_argument("--causal-r3-gamma-max", type=float, default=1.5)
    parser.add_argument("--causal-r3-weight-pde-loss", action="store_true")
    parser.add_argument("--no-causal-r3-score", dest="causal_r3_score", action="store_false")
    parser.set_defaults(causal_r3_score=True)

    parser.add_argument(
        "--bc-g-min",
        type=float,
        default=1e-12,
        help=(
            "Minimum valid growth at the recruitment boundary. "
            "BC samples with g(w_min,t) <= this value are excluded rather than clamped."
        ),
    )

    parser.add_argument(
        "--bc-use-constant-r",
        action="store_true",
        default=False,
        help="Use a constant recruitment flux target in the BC loss.",
    )
    
    parser.add_argument(
        "--bc-constant-r",
        type=float,
        default=None,
        help="Constant recruitment flux target used when --bc-use-constant-r is set.",
    )

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--load-weights",
        default=None,
        help="Optional path to a saved .pt checkpoint. Loads model_state_dict before training.",
    )
    
    parser.add_argument(
        "--load-optimizer-state",
        action="store_true",
        help="Also load optimizer_state_dict from --load-weights checkpoint.",
    )
    
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Initial step offset. Use 250 when continuing from model_step_250.pt.",
    )

    return parser.parse_args()



def save_data_predictions_final(*, run_dir: Path, model: nn.Module, params, observation_batch: dict[str, object], eps: float, data_time_quadrature_points: int) -> None:
    t_grid = observation_time_grid(
        observation_batch,
        data_time_quadrature_points=data_time_quadrature_points,
    )
    with torch.no_grad():
        grid_eval = evaluate_log_model_on_points(model=model, x_scaled=scale_x(torch.log(params.w), params), t_scaled=scale_t(t_grid, params), params=params)
        pred = predict_observations(
            {"N_grid": grid_eval["N"], "t_grid": t_grid},
            observation_batch,
            params,
            data_time_quadrature_points=data_time_quadrature_points,
        )
        nll = lognormal_nll(pred, observation_batch["value"], observation_batch["sd_log"], eps=eps)
    rows = []
    n = observation_batch["value"].numel()
    for j in range(n):
        rows.append({
            "obs_type": observation_batch["obs_type"][j],
            "dataset": observation_batch["dataset"][j],
            "species_idx": int(observation_batch["species_idx"][j].cpu()),
            "gear_idx": int(observation_batch["gear_idx"][j].cpu()),
            "t_start": float(observation_batch["t_start"][j].cpu()),
            "t_end": float(observation_batch["t_end"][j].cpu()),
            "w_min": float(observation_batch["w_min"][j].cpu()),
            "w_max": float(observation_batch["w_max"][j].cpu()),
            "value": float(observation_batch["value"][j].cpu()),
            "prediction": float(pred[j].cpu()),
            "log_residual": float(nll["log_residual"][j].cpu()),
            "sd_log": float(observation_batch["sd_log"][j].cpu()),
            "loss_contribution": float(nll["loss_contribution"][j].cpu()),
        })
    pd.DataFrame(rows).to_csv(run_dir / "data_predictions_final.csv", index=False)


def rmax_rows(inverse_rmax, params, step: int) -> list[dict]:
    if inverse_rmax is None:
        return []
    with torch.no_grad():
        cur = inverse_rmax.current_r_max().detach().cpu()
        log = inverse_rmax.current_log_r_max().detach().cpu()
        init = inverse_rmax.initial_r_max.detach().cpu()
        init_log = inverse_rmax.initial_log_r_max.detach().cpu()
        ratio = (cur / init).detach().cpu()
        raw = inverse_rmax.raw_logit.detach().cpu()
        grad = inverse_rmax.raw_logit.grad.detach().cpu() if inverse_rmax.raw_logit.grad is not None else torch.full_like(raw, float("nan"))
    species = getattr(params, "species", None)
    rows=[]
    for i in range(cur.numel()):
        rows.append({"step": step, "species_idx": i, "species": str(species[i]) if species is not None and i < len(species) else "", "initial_r_max": float(init[i]), "initial_log_r_max": float(init_log[i]), "r_max": float(cur[i]), "log_r_max": float(log[i]), "ratio_to_initial": float(ratio[i]), "raw_parameter": float(raw[i]), "raw_gradient": float(grad[i])})
    return rows

def save_rmax_history(rows: list[dict], run_dir: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "rmax_history.csv", index=False)

def save_estimated_rmax(inverse_rmax, params, run_dir: Path) -> str | None:
    if inverse_rmax is None:
        return None
    rows = rmax_rows(inverse_rmax, params, step=-1)
    for row in rows:
        row["estimated_r_max"] = row.pop("r_max")
        row["estimated_log_r_max"] = row.pop("log_r_max")
        row.pop("step", None); row.pop("raw_parameter", None); row.pop("raw_gradient", None)
        row["lower_log_bound"] = inverse_rmax.lower; row["upper_log_bound"] = inverse_rmax.upper
    path = run_dir / "estimated_rmax.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)

def main() -> None:
    args = parse_args()
    if args.pde_penalty == "pseudo-huber" and args.pde_pseudo_huber_delta <= 0.0:
        raise ValueError("--pde-pseudo-huber-delta must be strictly positive when --pde-penalty=pseudo-huber.")
    if args.bc_penalty == "pseudo-huber" and args.bc_pseudo_huber_delta <= 0.0:
        raise ValueError("--bc-pseudo-huber-delta must be strictly positive when --bc-penalty=pseudo-huber.")

    if args.causal_loss == "expert" and args.time_sampling != "stratified":
        raise ValueError("--causal-loss expert requires --time-sampling stratified.")
    if args.causal_loss == "expert" and args.collocation_strategy != "uniform":
        raise ValueError("--causal-loss expert currently requires --collocation-strategy uniform.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    run_dir = make_run_dir()
    save_run_command(run_dir / "run_command.txt", module="scripts.train_pde_multispecies")

    params, n_init, n_pp = load_mizer_inputs(
        args.input_dir,
        dtype=torch.float64,
        device=args.device,
    )

    params.state_parameterization = args.state_parameterization
    set_state_scale_from_initial_condition(params, n_init, eps=args.state_scale_eps)

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

    observation_batch = None
    if args.data_csv is not None and args.lambda_data != 0.0:
        observation_batch = load_observation_csv(args.data_csv, params, default_cv=args.data_default_cv)

    if args.species_mode != "all":
        raise ValueError("Only --species-mode all is supported for multi-species training.")
    if n_species < 1:
        raise ValueError(f"Expected at least one species, got {n_species}")

    r3_population_size = (
        args.r3_population_size
        if args.r3_population_size is not None
        else args.n_time * args.n_eval
    )

    r3_population = None
    causal_r3 = None

    if args.collocation_strategy in {"r3", "causal-r3"}:
        r3_population = make_r3_population(
            params=params,
            n_pair=r3_population_size,
            n_time=args.n_time,
            species_idx=None,
            seed=args.r3_seed,
        )

    if args.collocation_strategy == "causal-r3":
        causal_r3 = CausalR3(
            gamma=args.causal_r3_gamma_init,
            gamma_max=args.causal_r3_gamma_max,
            alpha=args.causal_r3_alpha,
        )

    model = build_pinn_model(
        model_arch=args.model_arch,
        in_dim=2,
        out_dim=n_species,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_num_features=args.fourier_num_features,
        fourier_scale=args.fourier_scale,
        fourier_include_raw_input=args.fourier_include_raw_input,
        fourier_seed=args.fourier_seed,
        weight_factorization=args.weight_factorization,
        rwf_mu=args.rwf_mu,
        rwf_sigma=args.rwf_sigma,
        rwf_apply_to=args.rwf_apply_to,
        rwf_base_init=args.rwf_base_init,
    ).to(dtype=torch.float64, device=params.w.device)
    model.state_parameterization = args.state_parameterization
    
    if args.init_final_bias_from_ic:
        initialise_final_bias_from_ic(
            model=model,
            n_init=n_init,
            params=params,
            eps=args.loss_eps,
            state_parameterization=args.state_parameterization,
        )
    
    inverse_rmax = None
    if args.estimate_rmax:
        inverse_rmax = BoundedLogRMax(params.r_max, lower=args.rmax_log_lower, upper=args.rmax_log_upper).to(dtype=params.r_max.dtype, device=params.r_max.device)
        params.r_max = inverse_rmax.current_r_max()
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr, "name": "network"},
            {"params": inverse_rmax.parameters(), "lr": args.rmax_lr, "name": "rmax"},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer=optimizer, args=args)
    
    loaded_checkpoint = None
    if args.load_weights is not None:
        loaded_checkpoint = load_checkpoint_weights(
            model=model,
            optimizer=optimizer,
            checkpoint_path=args.load_weights,
            device=params.w.device,
            load_optimizer_state=args.load_optimizer_state,
            inverse_rmax=inverse_rmax,
        )
        if inverse_rmax is not None:
            params.r_max = inverse_rmax.current_r_max()
    
    loss_weights = {
        "pde": args.initial_w_pde,
        "ic": args.initial_w_ic,
        "bc": args.initial_w_bc,
        "timestep": args.initial_w_timestep,
        "data": args.initial_w_data,
    }
    
    weight_state = {
        "has_updated": False,
    }
    
    fixed_weight_batch = fixed_diag_batch
    if args.causal_loss == "expert" and (
        (args.loss_weighting == "legacy-wang" and args.wang_weight_batch == "fixed")
        or (args.loss_weighting == "expert-grad-norm" and args.expert_weight_batch == "fixed")
    ):
        fixed_weight_batch = sample_pde_batch(
            params=params,
            n_time=args.n_time,
            n_eval=args.n_eval,
            time_sampling=args.time_sampling,
            causal_n_chunks=args.causal_n_chunks,
        )

    config = {
        "input_dir": args.input_dir,
        "run_dir": str(run_dir),
        "n_species": n_species,
        "species_mode": args.species_mode,
        "multi_species_training": True,
        "n_steps": args.n_steps,
        "n_time": args.n_time,
        "n_eval": args.n_eval,
        "time_sampling": args.time_sampling,
        "effective_n_time": None,
        "causal_loss": args.causal_loss,
        "causal_n_chunks": args.causal_n_chunks,
        "causal_epsilon": args.causal_epsilon,
        "loss_weighting": "none" if args.disable_wang_weights else args.loss_weighting,
        "expert_weight_update_every": args.expert_weight_update_every,
        "expert_weight_alpha": args.expert_weight_alpha,
        "expert_weight_min": args.weight_min if args.expert_weight_min is None else args.expert_weight_min,
        "expert_weight_max": args.weight_max if args.expert_weight_max is None else args.expert_weight_max,
        "expert_weight_batch": args.expert_weight_batch,
        "residual_form": args.residual_form,
        "pde_penalty": args.pde_penalty,
        "pde_pseudo_huber_delta": args.pde_pseudo_huber_delta,
        "bc_penalty": args.bc_penalty,
        "bc_pseudo_huber_delta": args.bc_pseudo_huber_delta,
        "state_parameterization": args.state_parameterization,
        "state_scale_source": "initial_condition",
        "state_scale_eps": args.state_scale_eps,
        "state_scale_interpolation": "linear_log_weight",
        "learning_rate": args.lr,
        "initial_lr": args.lr,
        "model_arch": args.model_arch,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "activation": "tanh",
        "fourier_num_features": args.fourier_num_features,
        "fourier_scale": args.fourier_scale,
        "fourier_include_raw_input": args.fourier_include_raw_input,
        "fourier_seed": args.fourier_seed,
        "weight_factorization": args.weight_factorization,
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "rwf_apply_to": args.rwf_apply_to,
        "rwf_base_init": args.rwf_base_init,
        "lr_scheduler": args.lr_scheduler,
        "lr_step_size": args.lr_step_size,
        "lr_gamma": args.lr_gamma,
        "lr_min": args.lr_min,
        "lr_plateau_patience": args.lr_plateau_patience,
        "current_lr": current_lr(optimizer),
        "final_lr": None,
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
        "wang_weight_batch": args.wang_weight_batch,
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
        "initial_w_timestep": args.initial_w_timestep,
        "initial_w_data": args.initial_w_data,
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
        "lambda_timestep": args.lambda_timestep,
        "estimate_rmax": args.estimate_rmax,
        "rmax_lr": args.rmax_lr,
        "rmax_log_lower": args.rmax_log_lower,
        "rmax_log_upper": args.rmax_log_upper,
        "boundary_target_gradient_mode": "rmax-only" if args.estimate_rmax else "detached",
        "initial_r_max": inverse_rmax.initial_r_max.detach().cpu().tolist() if inverse_rmax is not None else None,
        "initial_log_r_max": inverse_rmax.initial_log_r_max.detach().cpu().tolist() if inverse_rmax is not None else None,
        "data_csv": args.data_csv,
        "lambda_data": args.lambda_data,
        "data_default_cv": args.data_default_cv,
        "data_loss_eps": args.data_loss_eps,
        "data_time_quadrature_points": args.data_time_quadrature_points,
        "timestep_loss_form": args.timestep_loss_form,
        "detach_step_target": args.detach_step_target,
        "timestep_dt": args.timestep_dt,
        "timestep_n_pairs": args.timestep_n_pairs,
        "collocation_strategy": args.collocation_strategy,
        "r3_population_size": r3_population_size,
        "r3_update_every": args.r3_update_every,
        "r3_warmup_steps": args.r3_warmup_steps,
        "r3_score_form": args.r3_score_form,
        "r3_seed": args.r3_seed,
        "causal_r3_alpha": args.causal_r3_alpha,
        "causal_r3_gamma_init": args.causal_r3_gamma_init,
        "causal_r3_gamma_max": args.causal_r3_gamma_max,
        "causal_r3_weight_pde_loss": args.causal_r3_weight_pde_loss,
        "causal_r3_score": args.causal_r3_score,
        "r3_effective_population_size": (
            r3_population.population_size if r3_population is not None else None
        ),
        "r3_n_time": (
            r3_population.n_time if r3_population is not None else None
        ),
        "r3_n_eval_per_time": (
            r3_population.n_eval_per_time if r3_population is not None else None
        ),
        "bc_g_min": args.bc_g_min, 
        "bc_use_constant_r": args.bc_use_constant_r,
        "bc_constant_r": args.bc_constant_r,
        "load_weights": args.load_weights,
        "load_optimizer_state": args.load_optimizer_state,
        "start_step": args.start_step,
        "loaded_checkpoint_step": (
            loaded_checkpoint["checkpoint_step"] if loaded_checkpoint is not None else None
        ),
        "loaded_checkpoint_path": (
            loaded_checkpoint["path"] if loaded_checkpoint is not None else None
        ),
        "loaded_optimizer_state": (
            loaded_checkpoint["optimizer_loaded"] if loaded_checkpoint is not None else False
        ),   
        "note": (
            "Multi-species composite PINN loss with all-species PDE, IC, and recruitment boundary terms. "
            "IC/BC weights are adapted using Wang-style gradient statistics."
        ),
    }

    save_json(config, run_dir / "config.json")

    history = []
    rmax_history = []
    timing = {
        "warmup_steps": args.warmup_steps,
        "seconds_per_step": math.nan,
        "estimated_seconds_for_2000_steps": math.nan,
        "actual_total_seconds": math.nan,
    }

    start_time = time.perf_counter()

    try:
        for step in range(args.start_step + 1, args.n_steps + 1):
          
            current_causal_fraction, current_t_max = causal_t_max_current(
                params=params,
                step=step,
                mode=args.causal_curriculum,
                start_fraction=args.causal_start_fraction,
                ramp_steps=args.causal_ramp_steps,
                step_fractions=args.causal_step_fractions,
            )
          
            row = train_one_step_multispecies(
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
                pde_penalty=args.pde_penalty,
                pde_pseudo_huber_delta=args.pde_pseudo_huber_delta,
                bc_penalty=args.bc_penalty,
                bc_pseudo_huber_delta=args.bc_pseudo_huber_delta,
                eps=args.loss_eps,
                bc_eps=args.bc_eps,
                bc_g_min=args.bc_g_min,                
                weight_state=weight_state,
                hard_set_first_weight_update=args.hard_set_first_weight_update,
                causal_fraction=current_causal_fraction,
                t_max_current=current_t_max,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                lambda_bc=args.lambda_bc,
                disable_wang_weights=args.disable_wang_weights,
                lambda_timestep=args.lambda_timestep,
                timestep_loss_form=args.timestep_loss_form,
                detach_step_target=args.detach_step_target,
                timestep_dt=args.timestep_dt,
                timestep_n_pairs=args.timestep_n_pairs,
                collocation_strategy=args.collocation_strategy,
                r3_population=r3_population,
                r3_update_every=args.r3_update_every,
                r3_warmup_steps=args.r3_warmup_steps,
                r3_score_form=args.r3_score_form,
                causal_r3=causal_r3,
                causal_r3_weight_pde_loss=args.causal_r3_weight_pde_loss,
                causal_r3_score=args.causal_r3_score,
                bc_use_constant_r=args.bc_use_constant_r,
                bc_constant_r=args.bc_constant_r,
                lr_scheduler=scheduler,
                lr_scheduler_name=args.lr_scheduler,
                wang_weight_batch=args.wang_weight_batch,
                weight_calibration_batch=fixed_weight_batch,
                time_sampling=args.time_sampling,
                causal_loss=args.causal_loss,
                causal_n_chunks=args.causal_n_chunks,
                causal_epsilon=args.causal_epsilon,
                loss_weighting=args.loss_weighting,
                expert_weight_update_every=args.expert_weight_update_every,
                expert_weight_alpha=args.expert_weight_alpha,
                expert_weight_min=args.expert_weight_min,
                expert_weight_max=args.expert_weight_max,
                expert_weight_batch=args.expert_weight_batch,
                observation_batch=observation_batch,
                lambda_data=args.lambda_data,
                data_loss_eps=args.data_loss_eps,
                data_time_quadrature_points=args.data_time_quadrature_points,
                inverse_rmax=inverse_rmax,
                boundary_target_gradient_mode="rmax-only" if args.estimate_rmax else "detached",
            )

            history.append(row)
            if inverse_rmax is not None:
                rmax_history.extend(rmax_rows(inverse_rmax, params, step))

            if step == 1 or step % diag_every == 0:
                diag_row = compute_fixed_diagnostics(
                    model=model,
                    params=params,
                    n_pp=n_pp,
                    n_init=n_init,
                    fixed_batch=fixed_diag_batch,
                    residual_form=args.residual_form,
                    boundary_loss_form=args.boundary_loss_form,
                    species_idx=None,
                    bc_g_min=args.bc_g_min,                    
                    compute_grad_norms=(step == 1 or step % diag_grad_every == 0),
                    bc_use_constant_r=args.bc_use_constant_r,
                    bc_constant_r=args.bc_constant_r,
                    boundary_target_gradient_mode="rmax-only" if args.estimate_rmax else "detached",
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

            if step == args.start_step + args.warmup_steps:
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
                    f"lr={row['lr']:.6e} "
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
                    f"obj_pde={row['objective_loss_pde']:.3e} "
                    f"obj_ic={row['objective_loss_ic']:.3e} "
                    f"obj_bc={row['objective_loss_bc']:.3e} "
                    f"loss_ts={row['loss_timestep']:.3e} "
                    f"w_ts={row['w_timestep']:.3e} "
                    f"obj_ts={row['objective_loss_timestep']:.3e} "
                    f"wang_pde={row['wang_scaled_loss_pde']:.3e} "
                    f"wang_ic={row['wang_scaled_loss_ic']:.3e} "
                    f"wang_bc={row['wang_scaled_loss_bc']:.3e} "
                    f"wang_ts={row['wang_scaled_loss_timestep']:.3e} "
                    f"pde_weight_anchor={row['loss_pde_for_weighting']:.3e} "
                    f"pde_ungated={row['loss_pde_ungated']:.3e} "
                    f"pde_gated={row['loss_pde_gated']:.3e} "
                    f"pde_gate_mean={row['pde_gate_mean']:.3e} "
                    f"bc_valid={row['bc_valid_fraction']:.3e} "
                    f"bc_bad_g={row['bc_invalid_g_fraction']:.3e} "
                    f"bc_bad_rec={row['bc_invalid_recruitment_fraction']:.3e} "
                    f"bc_nonfinite={row['bc_nonfinite_fraction']:.3e} "
                )

                save_history(history, run_dir)
                save_rmax_history(rmax_history, run_dir)

            if step % args.checkpoint_every == 0:
                save_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    inverse_rmax=inverse_rmax,
                )

        timing["actual_total_seconds"] = time.perf_counter() - start_time
        config["current_lr"] = current_lr(optimizer)
        config["final_lr"] = current_lr(optimizer)
        save_json(config, run_dir / "config.json")

        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )

        save_history(history, run_dir)
        save_rmax_history(rmax_history, run_dir)
        estimated_rmax_path = save_estimated_rmax(inverse_rmax, params, run_dir)
        config["estimated_rmax_csv"] = estimated_rmax_path
        save_json(config, run_dir / "config.json")

        torch.save(
            {
                "step": args.n_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                **({"inverse_parameter_state_dict": inverse_rmax.state_dict(), "inverse_parameter_config": inverse_rmax.config(), "initial_r_max": inverse_rmax.initial_r_max.detach().cpu(), "initial_log_r_max": inverse_rmax.initial_log_r_max.detach().cpu(), "current_r_max": inverse_rmax.current_r_max().detach().cpu(), "current_log_r_max": inverse_rmax.current_log_r_max().detach().cpu()} if inverse_rmax is not None else {}),
            },
            run_dir / "model_final.pt",
        )

        save_final_predictions_multispecies(
            run_dir=run_dir,
            model=model,
            params=params,
            n_times=50,
        )

        if observation_batch is not None and args.lambda_data != 0.0:
            save_data_predictions_final(
                run_dir=run_dir,
                model=model,
                params=params,
                observation_batch=observation_batch,
                eps=args.data_loss_eps,
                data_time_quadrature_points=args.data_time_quadrature_points,
            )

        save_final_residual_sample_multispecies(
            run_dir=run_dir,
            model=model,
            params=params,
            n_pp=n_pp,
            n_time=args.n_time,
            n_eval=args.n_eval,
        )

        save_training_diagnostic_plots(run_dir)
        
        save_fixed_grid_fields_and_plots_multispecies(
            model=model,
            params=params,
            n_pp=n_pp,
            n_init=n_init,
            outdir=run_dir / "fixed_grid_diagnostics",
            residual_form=args.residual_form,
            boundary_loss_form=args.boundary_loss_form,
            n_time=args.diag_final_n_time,
            n_eval=args.diag_final_n_eval,
            bc_g_min=args.bc_g_min,
            bc_use_constant_r=args.bc_use_constant_r,
            bc_constant_r=args.bc_constant_r,
            boundary_target_gradient_mode="rmax-only" if args.estimate_rmax else "detached",
        )

    except Exception:
        save_history(history, run_dir)
        save_rmax_history(rmax_history, run_dir)
        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )
        raise

    print("Finished.")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()


