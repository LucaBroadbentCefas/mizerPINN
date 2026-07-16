from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask

from .checkpointing import load_checkpoint, save_checkpoint
from .config import (directory_identity, dtype_from_name, file_identity,
                     model_configuration_dict, validate_focused_configuration)
from .curriculum import causal_t_max_current
from .known_state import KnownStateProvider
from .losses import compute_composite_loss
from .models import build_scalar_model, final_linear_layer
from .outputs import fixed_diagnostics, save_final_outputs, save_history, save_json, save_run_command
from .pde_state import compute_pde_state
from .residual import compute_pde_residual_from_state
from .residual_scale import grid_residual_scale, set_residual_scale_from_initial_condition
from .sampling import make_fixed_pde_batch
from .training import train_one_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingContext:
    args: argparse.Namespace
    params: object
    n_init: torch.Tensor
    n_pp: torch.Tensor
    known_state: KnownStateProvider
    species_idx: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one independent scalar species PINN.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-state-csv", required=True)
    parser.add_argument("--species-idx", type=int, required=True)
    parser.add_argument("--biology-label", choices=["detailed", "trait"], required=True)
    parser.add_argument("--environment-state", choices=["dynamic-known", "frozen-initial"], default="dynamic-known")
    parser.add_argument("--known-state-interpolation", choices=["linear", "log-linear"], default="linear")
    parser.add_argument("--known-state-log-floor", type=float, default=1e-30)
    parser.add_argument("--allow-known-initial-mismatch", action="store_true")
    parser.add_argument("--n-steps", type=int, default=25000)
    parser.add_argument("--n-time", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-cosine-t-max", type=int, default=25000)
    parser.add_argument("--residual-form", choices=["reference-scaled", "log", "physical"], default="reference-scaled")
    parser.add_argument("--residual-scale-floor-fraction", type=float, default=1e-12)
    parser.add_argument("--state-parameterization", choices=["log-n", "log-u"], default="log-n")
    parser.add_argument("--boundary-loss-form", choices=["relative", "log", "physical"], default="relative")
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-timestep", type=float, default=0.0)
    parser.add_argument("--loss-eps", type=float, default=1e-30)
    parser.add_argument("--bc-g-min", type=float, default=1e-12)
    parser.add_argument("--collocation-strategy", choices=["uniform", "r3", "fixed-grid"], default="uniform")
    parser.add_argument("--time-sampling", choices=["stratified", "uniform"], default="stratified")
    parser.add_argument("--causal-loss", choices=["expert", "off"], default="expert")
    parser.add_argument("--causal-curriculum", choices=["linear", "off", "step"], default="linear")
    parser.add_argument("--causal-start-fraction", type=float, default=0.05)
    parser.add_argument("--causal-ramp-steps", type=int, default=40000)
    parser.add_argument("--causal-step-fractions", default="0.05,0.10,0.20,0.40,0.70,1.0")
    parser.add_argument("--causal-n-chunks", type=int, default=64)
    parser.add_argument("--causal-epsilon", type=float, default=1.0)
    parser.add_argument("--loss-weighting", choices=["expert-grad-norm", "none"], default="expert-grad-norm")
    parser.add_argument("--expert-weight-update-every", type=int, default=1000)
    parser.add_argument("--expert-weight-alpha", type=float, default=0.7)
    parser.add_argument("--weight-min", type=float, default=1e-3)
    parser.add_argument("--weight-max", type=float, default=1e2)
    parser.add_argument("--model-arch", choices=["fourier", "mlp"], default="fourier")
    parser.add_argument("--fourier-num-features", type=int, default=16)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--fourier-include-raw-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier-seed", type=int, default=123)
    parser.add_argument("--weight-factorization", choices=["rwf", "none"], default="rwf")
    parser.add_argument("--rwf-mu", type=float, default=1.0)
    parser.add_argument("--rwf-sigma", type=float, default=0.1)
    parser.add_argument("--rwf-apply-to", choices=["all", "hidden"], default="all")
    parser.add_argument("--rwf-base-init", choices=["xavier_uniform", "pytorch", "xavier_normal"], default="xavier_uniform")
    parser.add_argument("--hidden-width", type=int, default=384)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=4000)
    parser.add_argument("--diag-every", type=int, default=2000)
    parser.add_argument("--load-weights", default=None)
    parser.add_argument("--load-optimizer-state", action="store_true")
    return parser.parse_args(argv)


def _run_directory(args, species_name: str) -> Path:
    root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "runs" / "species_ensemble_pinn"
    label = f"{args.biology_label}_{args.environment_state}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm = os.environ.get("SLURM_JOB_ID")
    suffix = f"_job{slurm}" if slurm else ""
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in species_name)
    path = root / label / f"species_{args.species_idx:02d}_{safe_name}" / f"{stamp}{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _configuration(args, species_name: str) -> dict:
    return {
        **vars(args), "species_name": species_name,
        "state_parameterization": "log-n", "residual_form": "reference-scaled",
        "model_configuration": model_configuration_dict(),
        "runtime_defaults_source": "tranche specification plus current trainer conventions; successful run command was not attached",
    }


def _initialise_bias(model, n_init, params, species_idx: int, eps: float) -> None:
    active = active_grid_mask(params)[species_idx]
    target = torch.log(torch.clamp(n_init[species_idx, active], min=eps)).mean()
    layer = final_linear_layer(model)
    with torch.no_grad():
        layer.bias.fill_(target.to(dtype=layer.bias.dtype, device=layer.bias.device))


def _fixed_loss_bundle(model, fixed_batch, context, weights):
    state = compute_pde_state(model, fixed_batch, context.params, context.n_init, context.n_pp,
                              context.known_state, species_idx=context.species_idx)
    residual = compute_pde_residual_from_state(state, context.params, species_idx=context.species_idx)
    return compute_composite_loss(state=state, residual_out=residual, params=context.params,
        species_idx=context.species_idx, batch=fixed_batch,
        lambda_pde=context.args.lambda_pde, lambda_ic=context.args.lambda_ic,
        lambda_bc=context.args.lambda_bc, loss_weights=weights,
        causal_n_chunks=context.args.causal_n_chunks, causal_epsilon=context.args.causal_epsilon,
        eps=context.args.loss_eps, bc_g_min=context.args.bc_g_min)


def run(args: argparse.Namespace) -> Path:
    validate_focused_configuration(args)
    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    params, n_init, n_pp = load_mizer_inputs(args.input_dir, dtype=dtype, device=args.device)
    params.state_parameterization = "log-n"
    known = KnownStateProvider(args.known_state_csv, params, n_init, mode=args.environment_state,
        interpolation=args.known_state_interpolation, log_floor=args.known_state_log_floor,
        allow_initial_mismatch=args.allow_known_initial_mismatch)
    if not 0 <= args.species_idx < len(params.species):
        raise IndexError("--species-idx is outside the known species mapping.")
    species_name = params.species[args.species_idx]
    set_residual_scale_from_initial_condition(params, n_init,
        floor_fraction=args.residual_scale_floor_fraction)
    model = build_scalar_model().to(dtype=dtype, device=args.device)
    _initialise_bias(model, n_init, params, args.species_idx, args.loss_eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.lr_cosine_t_max), eta_min=args.lr_min)
    configuration = _configuration(args, species_name)
    run_dir = _run_directory(args, species_name)
    save_json(configuration, run_dir / "config.json")
    save_run_command(run_dir / "run_command.txt")
    parameter_identity = directory_identity(args.input_dir)
    known_identity = file_identity(args.known_state_csv)
    save_json(parameter_identity, run_dir / "parameter_fixture_identity.json")
    save_json(known_identity, run_dir / "known_state_file_identity.json")
    fixed_generator = torch.Generator(device="cpu")
    fixed_generator.manual_seed(args.seed + 100003)
    fixed_batch = make_fixed_pde_batch(params, args.n_time, args.n_eval,
        causal_n_chunks=args.causal_n_chunks, generator=fixed_generator)
    context = TrainingContext(args, params, n_init, n_pp, known, args.species_idx)
    weights = {"pde": 1.0, "ic": 1.0, "bc": 0.1}
    start_step = 0
    if args.load_weights:
        payload = load_checkpoint(args.load_weights, model=model, optimizer=optimizer, scheduler=scheduler,
            params=params, species_idx=args.species_idx, species_name=species_name,
            configuration=configuration, load_optimizer_state=args.load_optimizer_state)
        weights.update(payload.get("loss_weights", {}))
        start_step = int(payload["step"])
    history, diagnostics = [], []
    status = {"status": "running", "error_message": "", "species_idx": args.species_idx,
              "species_name": species_name}
    save_json(status, run_dir / "run_status.json")
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, args.n_steps + 1):
            fraction, t_current = causal_t_max_current(
                params, step, args.causal_start_fraction, args.causal_ramp_steps)
            row = train_one_step(model=model, optimizer=optimizer, scheduler=scheduler,
                context=context, step=step, t_max_current=t_current,
                loss_weights=weights, fixed_weight_batch=fixed_batch)
            row.update({"causal_fraction": fraction, "t_max_current": t_current,
                        "seconds_elapsed": time.perf_counter() - started})
            history.append(row)
            if args.print_every and step % args.print_every == 0:
                print(f"species={args.species_idx} step={step} loss={row['loss']:.6e} lr={row['lr']:.3e}")
            if args.diag_every and step % args.diag_every == 0:
                _, fixed_losses = _fixed_loss_bundle(model, fixed_batch, context, weights)
                diagnostics.append({"step": step, **fixed_diagnostics(fixed_losses)})
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(run_dir / f"checkpoint_step_{step}.pt", step=step,
                    species_idx=args.species_idx, species_name=species_name, model=model,
                    optimizer=optimizer, scheduler=scheduler, loss_weights=weights,
                    configuration=configuration, latest_history_row=row,
                    parameter_fixture_identity=parameter_identity,
                    known_state_file_identity=known_identity, params=params)
        final_step = args.n_steps
        save_checkpoint(run_dir / "checkpoint_final.pt", step=final_step,
            species_idx=args.species_idx, species_name=species_name, model=model,
            optimizer=optimizer, scheduler=scheduler, loss_weights=weights,
            configuration=configuration, latest_history_row=history[-1] if history else None,
            parameter_fixture_identity=parameter_identity,
            known_state_file_identity=known_identity, params=params)
        save_history(history, run_dir / "loss_history.csv")
        pd.DataFrame(diagnostics).to_csv(run_dir / "fixed_diagnostics.csv", index=False)
        save_final_outputs(run_dir=run_dir, model=model, params=params, n_init=n_init,
            n_pp=n_pp, known_state=known, species_idx=args.species_idx,
            species_name=species_name, batch=fixed_batch)
        log_scale, _ = grid_residual_scale(params)
        status.update({"status": "success", "steps_completed": final_step,
            "target_residual_scale_min": float(torch.exp(log_scale[args.species_idx]).min().cpu()),
            "target_residual_scale_max": float(torch.exp(log_scale[args.species_idx]).max().cpu()),
            "initial_state_max_abs_difference": known.initial_max_abs_difference,
            "initial_state_max_relative_difference": known.initial_max_relative_difference})
    except Exception as exc:
        status.update({"status": "failed", "error_message": f"{type(exc).__name__}: {exc}",
                       "steps_completed": len(history)})
        save_history(history, run_dir / "loss_history.csv")
        pd.DataFrame(diagnostics).to_csv(run_dir / "fixed_diagnostics.csv", index=False)
        save_json(status, run_dir / "run_status.json")
        raise
    save_json(status, run_dir / "run_status.json")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)))


if __name__ == "__main__":
    main()
