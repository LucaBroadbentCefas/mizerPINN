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

from PINNmizer.pinn.models import MLP
from PINNmizer.training.checkpointing import save_checkpoint
from PINNmizer.training.config import (
    _to_float,
    causal_t_max_current,
    causal_time_fraction,
    parse_fraction_schedule,
)
from PINNmizer.training.outputs import (
    save_final_predictions,
    save_final_residual_sample,
    save_history,
    save_json,
    save_run_command,
)
from PINNmizer.training.loop import train_one_step, total_grad_norm_and_check, scalar_min, scalar_max, scalar_mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from PINNmizer.io import load_mizer_inputs
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
from PINNmizer.diagnostics.fields import (
    save_fixed_grid_fields_and_plots,
)
#


def make_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / "pde_only_single_species" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir






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
        if isinstance(module, nn.Linear):
            final_linear = module
            break

    if final_linear is None:
        raise ValueError("Could not find final nn.Linear layer.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="validation/fixtures/pde_single_species")
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = make_run_dir()
    save_run_command(run_dir / "run_command.txt")

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
            params=params,
            eps=args.loss_eps,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_weights = {
        "pde": args.initial_w_pde,
        "ic": args.initial_w_ic,
        "bc": args.initial_w_bc,
        "timestep": args.initial_w_timestep,
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
                lambda_timestep=args.lambda_timestep,
                timestep_loss_form=args.timestep_loss_form,
                detach_step_target=args.detach_step_target,
                timestep_dt=args.timestep_dt,
                timestep_n_pairs=args.timestep_n_pairs,
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
                    f"loss_ts={row['loss_timestep']:.3e} "
                    f"w_ts={row['w_timestep']:.3e} "
                    f"wts={row['weighted_loss_timestep']:.3e} "
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


