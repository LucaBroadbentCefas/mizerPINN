from __future__ import annotations

import argparse
import os
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PINNmizer.pinn.models import FactorizedLinear, build_pinn_model
from PINNmizer.pinn.sampling import make_fixed_pde_batch as make_fixed_training_pde_batch, sample_pde_batch
from PINNmizer.training.checkpointing import save_checkpoint
from PINNmizer.training.config import (
    _to_float,
    causal_t_max_current,
    causal_time_fraction,
    parse_fraction_schedule,
)
from PINNmizer.training.outputs import (
    HPC_FIXED_DIAGNOSTIC_COLUMNS,
    HPC_HISTORY_COLUMNS,
    filter_hpc_fixed_diagnostic_row,
    filter_hpc_history_row,
    save_final_predictions,
    save_final_residual_sample,
    save_history,
    save_json,
    save_run_command,
)
from PINNmizer.training.loop import train_one_step, total_grad_norm_and_check, scalar_min, scalar_max, scalar_mean
from PINNmizer.pinn.r3 import make_r3_population, CausalR3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import scale_x, scale_t, active_grid_mask
from PINNmizer.diagnostics.fixed_grid import (
    make_fixed_pde_batch as make_fixed_diagnostic_pde_batch,
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
from PINNmizer.diagnostics.fields import (
    save_fixed_grid_fields_and_plots,
)
#


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_job_id is not None and slurm_array_task_id is not None:
        name = f"{stamp}_job{slurm_job_id}_task{slurm_array_task_id}"
    elif slurm_job_id is not None:
        name = f"{stamp}_job{slurm_job_id}"
    else:
        name = stamp

    run_dir = PROJECT_ROOT / "runs" / "pde_only_single_species" / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_checkpoint_weights(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str | Path,
    device,
    load_optimizer_state: bool = False,
) -> dict:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model_state_dict': {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_loaded = False
    if load_optimizer_state:
        if optimizer is None:
            raise ValueError("optimizer must be provided when load_optimizer_state=True.")
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint has no 'optimizer_state_dict': {checkpoint_path}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True

    return {
        "path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step", None),
        "optimizer_loaded": optimizer_loaded,
        "checkpoint_config": checkpoint.get("config", None),
    }





def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _fmt_metric(row: dict | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    value = row.get(key, default)
    return float(value) if value is not None else default


def should_hpc_emit(step: int, every: int = 2000) -> bool:
    return step > 0 and step % every == 0


def save_hpc_final_summary(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    config: dict,
    status: str,
    error_message: str | None,
    n_steps_completed: int,
    timing: dict,
    latest_history_row: dict | None,
    latest_fixed_diagnostic_row: dict | None,
    final_checkpoint_path: str | None,
    final_model_path: str | None,
) -> None:
    actual_total_seconds = float(timing.get("actual_total_seconds", math.nan))
    seconds_per_step = float(timing.get("seconds_per_step", math.nan))
    if not math.isfinite(seconds_per_step) and n_steps_completed > 0 and math.isfinite(actual_total_seconds):
        seconds_per_step = actual_total_seconds / n_steps_completed

    row = {
        "run_id": str(run_dir),
        "run_dir": str(run_dir),
        "seed": args.seed,
        "fourier_seed": args.fourier_seed,
        "model_arch": args.model_arch,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "fourier_num_features": args.fourier_num_features,
        "fourier_scale": args.fourier_scale,
        "fourier_include_raw_input": args.fourier_include_raw_input,
        "weight_factorization": args.weight_factorization,
        "rwf_mu": args.rwf_mu,
        "rwf_sigma": args.rwf_sigma,
        "rwf_apply_to": args.rwf_apply_to,
        "rwf_base_init": args.rwf_base_init,
        "n_steps_completed": n_steps_completed,
        "status": status,
        "error_message": error_message or "",
        "seconds_per_step": seconds_per_step,
        "actual_total_seconds": actual_total_seconds,
        "final_loss": _fmt_metric(latest_history_row, "loss"),
        "final_loss_unweighted": _fmt_metric(latest_history_row, "loss_unweighted"),
        "final_loss_pde": _fmt_metric(latest_history_row, "loss_pde"),
        "final_loss_ic": _fmt_metric(latest_history_row, "loss_ic"),
        "final_loss_bc": _fmt_metric(latest_history_row, "loss_bc"),
        "final_loss_pde_ungated": _fmt_metric(latest_history_row, "loss_pde_ungated"),
        "final_fixed_loss": _fmt_metric(latest_fixed_diagnostic_row, "fixed_loss"),
        "final_fixed_loss_unweighted": _fmt_metric(latest_fixed_diagnostic_row, "fixed_loss_unweighted"),
        "final_fixed_loss_pde": _fmt_metric(latest_fixed_diagnostic_row, "fixed_loss_pde"),
        "final_fixed_loss_ic": _fmt_metric(latest_fixed_diagnostic_row, "fixed_loss_ic"),
        "final_fixed_loss_bc": _fmt_metric(latest_fixed_diagnostic_row, "fixed_loss_bc"),
        "final_fixed_residual_log_abs_p95": _fmt_metric(latest_fixed_diagnostic_row, "fixed_residual_log_abs_p95"),
        "final_checkpoint_path": final_checkpoint_path,
        "final_model_path": final_model_path,
    }
    pd.DataFrame([row]).to_csv(run_dir / "final_summary.csv", index=False)
    save_json(row, run_dir / "final_summary.json")


def print_hpc_training_line(
    *,
    row: dict,
    fixed_row: dict | None,
    latest_checkpoint_path: str | None,
) -> None:
    checkpoint_text = latest_checkpoint_path or ""
    print(
        f"step={int(row['step'])} "
        f"elapsed={row['seconds_elapsed']:.1f}s "
        f"lr={row['lr']:.6e} "
        f"loss={row['loss']:.6e} "
        f"loss_unweighted={row['loss_unweighted']:.6e} "
        f"loss_pde={row['loss_pde']:.6e} "
        f"loss_ic={row['loss_ic']:.6e} "
        f"loss_bc={row['loss_bc']:.6e} "
        f"loss_pde_ungated={row['loss_pde_ungated']:.6e} "
        f"w_pde={row['w_pde']:.3e} "
        f"w_ic={row['w_ic']:.3e} "
        f"w_bc={row['w_bc']:.3e} "
        f"fixed_loss_pde={_fmt_metric(fixed_row, 'fixed_loss_pde'):.6e} "
        f"fixed_loss_ic={_fmt_metric(fixed_row, 'fixed_loss_ic'):.6e} "
        f"fixed_loss_bc={_fmt_metric(fixed_row, 'fixed_loss_bc'):.6e} "
        f"fixed_residual_log_abs_p95={_fmt_metric(fixed_row, 'fixed_residual_log_abs_p95'):.6e} "
        f"checkpoint={checkpoint_text}"
    )


def build_lr_scheduler(*, optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.lr_scheduler == "none":
        return None

    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.lr_cosine_t_max or args.n_steps),
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

    parser.add_argument("--input-dir", default="validation/fixtures/pde_single_species")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-time", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "step", "plateau"], default="none")
    parser.add_argument("--lr-step-size", type=int, default=500)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--lr-cosine-t-max", type=int, default=None)
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
    parser.add_argument("--timestep-loss-form", choices=["physical", "log", "relative"], default="physical")
    parser.add_argument("--detach-step-target", action="store_true", default=True)
    parser.add_argument("--no-detach-step-target", dest="detach_step_target", action="store_false")
    parser.add_argument("--timestep-dt", type=float, default=None)
    parser.add_argument("--timestep-n-pairs", type=int, default=1)

    parser.add_argument(
        "--collocation-strategy",
        choices=["uniform", "fixed-grid", "r3", "causal-r3"],
        default="uniform",
    )
    parser.add_argument("--fixed-collocation-use-mizer-x-grid", action="store_true")
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
    parser.add_argument("--hpc", action="store_true", help="Use streamlined output files for HPC batch runs.")

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

    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64


    hpc_history_every = 2000
    hpc_checkpoint_every = 4000

    if args.causal_loss == "expert" and args.time_sampling != "stratified":
        raise ValueError("--causal-loss expert requires --time-sampling stratified.")
    if args.causal_loss == "expert" and args.collocation_strategy != "uniform":
        raise ValueError("--causal-loss expert currently requires --collocation-strategy uniform.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    run_dir = make_run_dir()
    save_run_command(run_dir / "run_command.txt")

    params, n_init, n_pp = load_mizer_inputs(
        args.input_dir,
        dtype=dtype,
        device=args.device,
    )

    if args.hpc:
        diag_every = hpc_history_every
        diag_grad_every = hpc_history_every
    else:
        diag_every = args.diag_every if args.diag_every > 0 else args.print_every
        diag_grad_every = args.diag_grad_every if args.diag_grad_every > 0 else diag_every

    if args.diag_grid_csv is not None:
        fixed_diag_batch = make_fixed_pde_batch_from_csv(
            params=params,
            path=args.diag_grid_csv,
        )
    else:
        fixed_diag_batch = make_fixed_diagnostic_pde_batch(
            params=params,
            n_time=args.diag_n_time,
            n_eval=args.diag_n_eval,
            use_mizer_x_grid=args.diag_use_mizer_x_grid,
        )

    fixed_collocation_batch = None
    if args.collocation_strategy == "fixed-grid":
        fixed_collocation_batch = make_fixed_training_pde_batch(
            params=params,
            n_time=args.n_time,
            n_eval=args.n_eval,
            t_max_current=None,
            use_mizer_x_grid=args.fixed_collocation_use_mizer_x_grid,
        )

    n_species = params.interaction.shape[0]

    if n_species != 1:
        raise ValueError(f"Expected one species, got {n_species}")

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
            species_idx=0,
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
    ).to(dtype=dtype, device=params.w.device)

    if args.init_final_bias_from_ic:
        initialise_final_bias_from_ic(
            model=model,
            n_init=n_init,
            params=params,
            eps=args.loss_eps,
        )

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
        )

    loss_weights = {
        "pde": args.initial_w_pde,
        "ic": args.initial_w_ic,
        "bc": args.initial_w_bc,
        "timestep": args.initial_w_timestep,
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
        "dtype": args.dtype,
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
        "timestep_loss_form": args.timestep_loss_form,
        "detach_step_target": args.detach_step_target,
        "timestep_dt": args.timestep_dt,
        "timestep_n_pairs": args.timestep_n_pairs,
        "collocation_strategy": args.collocation_strategy,
        "fixed_collocation_use_mizer_x_grid": args.fixed_collocation_use_mizer_x_grid,
        "fixed_collocation_is_static": fixed_collocation_batch is not None,
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
        "hpc": args.hpc,
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
            "Composite PINN loss with PDE, IC, and recruitment boundary terms. "
            "IC/BC weights are adapted using Wang-style gradient statistics."
        ),
    }

    save_json(config, run_dir / "config.json")
    if args.hpc:
        pd.DataFrame(columns=HPC_FIXED_DIAGNOSTIC_COLUMNS).to_csv(
            run_dir / "fixed_diagnostic_history.csv",
            index=False,
        )

    history = []
    hpc_history = []
    latest_history_row = None
    latest_fixed_diagnostic_row = None
    latest_checkpoint_path = None
    final_model_path = None
    n_steps_completed = args.start_step
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
                fixed_collocation_batch=fixed_collocation_batch,
            )

            history.append(row)
            latest_history_row = row
            n_steps_completed = step

            if args.hpc and (should_hpc_emit(step, hpc_history_every) or step == args.n_steps):
                hpc_history.append(filter_hpc_history_row(row))
                save_history(hpc_history, run_dir, columns=HPC_HISTORY_COLUMNS)

            run_fixed_diag = (
                (not args.hpc and (step == 1 or step % diag_every == 0))
                or (args.hpc and should_hpc_emit(step, hpc_history_every))
            )
            if run_fixed_diag:
                diag_row = compute_fixed_diagnostics(
                    model=model,
                    params=params,
                    n_pp=n_pp,
                    n_init=n_init,
                    fixed_batch=fixed_diag_batch,
                    residual_form=args.residual_form,
                    boundary_loss_form=args.boundary_loss_form,
                    species_idx=0,
                    bc_g_min=args.bc_g_min,
                    compute_grad_norms=(step == 1 or step % diag_grad_every == 0),
                    bc_use_constant_r=args.bc_use_constant_r,
                    bc_constant_r=args.bc_constant_r,
                )

                diag_row = {
                    "step": step,
                    **diag_row,
                }

                if args.hpc:
                    diag_row = filter_hpc_fixed_diagnostic_row(diag_row)
                    latest_fixed_diagnostic_row = diag_row
                    append_diagnostic_row(
                        diag_row,
                        run_dir / "fixed_diagnostic_history.csv",
                    )
                else:
                    latest_fixed_diagnostic_row = diag_row
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

                if not args.hpc:
                    print(
                        "Timing:",
                        f"{seconds_per_step:.4f} sec/step;",
                        f"estimated 2000 steps = {seconds_per_step * 2000 / 60:.2f} min",
                    )

            if args.hpc and step % hpc_checkpoint_every == 0:
                latest_checkpoint_path = str(save_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    scheduler=scheduler,
                    latest_history_row=filter_hpc_history_row(row),
                    latest_fixed_diagnostic_row=latest_fixed_diagnostic_row,
                    subdir="checkpoints",
                ))

            if args.hpc:
                if should_hpc_emit(step, hpc_history_every):
                    print_hpc_training_line(
                        row=row,
                        fixed_row=latest_fixed_diagnostic_row,
                        latest_checkpoint_path=latest_checkpoint_path,
                    )
            elif step % args.print_every == 0 or step == 1:
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

            if (not args.hpc) and step % args.checkpoint_every == 0:
                latest_checkpoint_path = str(save_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                ))

        timing["actual_total_seconds"] = time.perf_counter() - start_time
        config["current_lr"] = current_lr(optimizer)
        config["final_lr"] = current_lr(optimizer)
        save_json(config, run_dir / "config.json")

        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )

        if args.hpc:
            if latest_history_row is not None and (
                not hpc_history or hpc_history[-1].get("step") != latest_history_row.get("step")
            ):
                hpc_history.append(filter_hpc_history_row(latest_history_row))
            save_history(hpc_history, run_dir, columns=HPC_HISTORY_COLUMNS)
        else:
            save_history(history, run_dir)

        final_model_path = run_dir / "model_final.pt"
        final_checkpoint = {
            "step": args.n_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        if args.hpc and scheduler is not None:
            final_checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(final_checkpoint, final_model_path)

        save_final_predictions(
            run_dir=run_dir,
            model=model,
            params=params,
            n_times=50,
        )

        if args.hpc:
            final_fixed_row = compute_fixed_diagnostics(
                model=model,
                params=params,
                n_pp=n_pp,
                n_init=n_init,
                fixed_batch=fixed_diag_batch,
                residual_form=args.residual_form,
                boundary_loss_form=args.boundary_loss_form,
                species_idx=0,
                bc_g_min=args.bc_g_min,
                compute_grad_norms=False,
                bc_use_constant_r=args.bc_use_constant_r,
                bc_constant_r=args.bc_constant_r,
            )
            latest_fixed_diagnostic_row = filter_hpc_fixed_diagnostic_row({
                "step": args.n_steps,
                **final_fixed_row,
            })
            save_fixed_grid_fields_and_plots(
                model=model,
                params=params,
                n_pp=n_pp,
                n_init=n_init,
                outdir=run_dir,
                residual_form=args.residual_form,
                boundary_loss_form=args.boundary_loss_form,
                species_idx=0,
                n_time=args.diag_final_n_time,
                n_eval=args.diag_final_n_eval,
                bc_g_min=args.bc_g_min,
                bc_use_constant_r=args.bc_use_constant_r,
                bc_constant_r=args.bc_constant_r,
                make_plots=False,
                save_boundary_diagnostics=False,
            )
            save_hpc_final_summary(
                run_dir=run_dir,
                args=args,
                config=config,
                status="completed",
                error_message=None,
                n_steps_completed=n_steps_completed,
                timing=timing,
                latest_history_row=latest_history_row,
                latest_fixed_diagnostic_row=latest_fixed_diagnostic_row,
                final_checkpoint_path=latest_checkpoint_path,
                final_model_path=str(final_model_path),
            )
        else:
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
                boundary_loss_form=args.boundary_loss_form,
                species_idx=0,
                n_time=args.diag_final_n_time,
                n_eval=args.diag_final_n_eval,
                bc_g_min=args.bc_g_min,
                bc_use_constant_r=args.bc_use_constant_r,
                bc_constant_r=args.bc_constant_r,
            )

    except Exception as exc:
        timing["actual_total_seconds"] = time.perf_counter() - start_time
        if args.hpc:
            save_history(hpc_history, run_dir, columns=HPC_HISTORY_COLUMNS)
            save_hpc_final_summary(
                run_dir=run_dir,
                args=args,
                config=config,
                status="failed",
                error_message=str(exc),
                n_steps_completed=n_steps_completed,
                timing=timing,
                latest_history_row=latest_history_row,
                latest_fixed_diagnostic_row=latest_fixed_diagnostic_row,
                final_checkpoint_path=latest_checkpoint_path,
                final_model_path=str(final_model_path) if final_model_path is not None else None,
            )
        else:
            save_history(history, run_dir)
        pd.DataFrame([timing]).to_csv(
            run_dir / "timing_summary.csv",
            index=False,
        )
        raise

    print("Finished.")
    print(f"Run directory: {run_dir}")
    if not args.hpc:
        from PINNmizer.diagnostics.output_surface import save_output_surface_diagnostics

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


