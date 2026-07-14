#!/usr/bin/env python
"""
Decompose the existing multispecies PDE loss at a saved checkpoint.

This script does not define a new residual or loss. It rebuilds the model,
loads a checkpoint, calls PINNmizer.pinn.losses.compute_pde_loss(), and
retains the species and weight dimensions of the exact loss summands.

Outputs:
  pde_residual_by_weight.png
  pde_loss_contribution_by_weight.png
  pde_loss_decomposition.csv
  pde_loss_summary.csv

A checkpoint does not store the random collocation batch used at the original
training iteration. This therefore evaluates the checkpoint on a reproducible
diagnostic batch using the existing sampler and the full mizer weight grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_eval_mask, scale_x
from PINNmizer.pinn.losses import compute_pde_loss
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.state_scale import (
    DEFAULT_STATE_SCALE_EPS,
    set_state_scale_from_initial_condition,
)
from PINNmizer.training.config import causal_t_max_current
from PINNmizer.training.train_pde_multispecies import load_checkpoint_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose the current multispecies PDE loss over species and weight."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model_step_*.pt.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Mizer input directory. Defaults to checkpoint config['input_dir'].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a folder inside the run directory.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Physical time to plot. The closest diagnostic time is used.",
    )
    parser.add_argument(
        "--evaluation-t-max",
        type=float,
        default=None,
        help=(
            "Upper physical-time limit for diagnostic loss evaluation. Defaults to "
            "the causal-curriculum horizon at the checkpoint step."
        ),
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=None,
        help="Diagnostic time sample count. Defaults to checkpoint config['n_time'].",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def require_config(config: dict, key: str):
    if key not in config:
        raise KeyError(
            f"Checkpoint config is missing {key!r}; the model cannot be rebuilt safely."
        )
    return config[key]


def infer_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def infer_species_names(run_dir: Path, n_species: int) -> list[str]:
    predictions_path = run_dir / "final_predictions_grid.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(
            predictions_path,
            usecols=["species_idx", "species"],
        ).drop_duplicates("species_idx")
        names = {
            int(row.species_idx): str(row.species)
            for row in predictions.itertuples(index=False)
        }
        return [names.get(i, f"species_{i}") for i in range(n_species)]
    return [f"species_{i}" for i in range(n_species)]


def build_model_from_config(
    *, config: dict, n_species: int, device: torch.device
) -> torch.nn.Module:
    model = build_pinn_model(
        model_arch=require_config(config, "model_arch"),
        in_dim=2,
        out_dim=n_species,
        hidden_width=int(require_config(config, "hidden_width")),
        hidden_layers=int(require_config(config, "hidden_layers")),
        fourier_num_features=int(config.get("fourier_num_features", 64)),
        fourier_scale=float(config.get("fourier_scale", 1.0)),
        fourier_include_raw_input=bool(
            config.get("fourier_include_raw_input", False)
        ),
        fourier_seed=config.get("fourier_seed"),
        weight_factorization=config.get("weight_factorization", "none"),
        rwf_mu=float(config.get("rwf_mu", 1.0)),
        rwf_sigma=float(config.get("rwf_sigma", 0.1)),
        rwf_apply_to=config.get("rwf_apply_to", "all"),
        rwf_base_init=config.get("rwf_base_init", "pytorch"),
    ).to(dtype=torch.float64, device=device)
    model.state_parameterization = config.get("state_parameterization", "log-n")
    return model


def checkpoint_training_t_max(*, params, config: dict, checkpoint_step: int) -> float:
    _, t_max_current = causal_t_max_current(
        params=params,
        step=checkpoint_step,
        mode=config.get("causal_curriculum", "off"),
        start_fraction=float(config.get("causal_start_fraction", 0.05)),
        ramp_steps=int(config.get("causal_ramp_steps", 1)),
        step_fractions=config.get(
            "causal_step_fractions",
            "0.05,0.10,0.20,0.40,0.70,1.0",
        ),
    )
    return float(t_max_current)


def make_diagnostic_batch(
    *,
    params,
    n_time: int,
    t_max_current: float,
    time_sampling: str,
    causal_n_chunks: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    # Use the repository's current time sampler and expert-causal chunk indices.
    torch.manual_seed(seed)
    batch = sample_pde_batch(
        params=params,
        n_time=n_time,
        n_eval=params.w.numel(),
        t_max_current=t_max_current,
        time_sampling=time_sampling,
        causal_n_chunks=causal_n_chunks,
    )

    # Evaluate on the existing mizer grid so the weight axis is complete.
    # No residual or loss equation is changed.
    x_eval = torch.log(params.w)
    batch["x_eval"] = x_eval
    batch["x_eval_scaled"] = scale_x(x_eval, params)
    batch["w_eval"] = params.w
    return batch


def pointwise_penalty(
    residual: torch.Tensor, *, penalty: str, delta: float
) -> torch.Tensor:
    """Exact algebra used by losses._pointwise_penalty()."""
    if penalty == "squared":
        return residual.square()
    if penalty == "pseudo-huber":
        if delta <= 0.0:
            raise ValueError("Pseudo-Huber delta must be strictly positive.")
        delta_t = torch.as_tensor(
            delta,
            dtype=residual.dtype,
            device=residual.device,
        )
        return delta_t.square() * (
            torch.sqrt(1.0 + (residual / delta_t).square()) - 1.0
        )
    raise ValueError(f"Unsupported PDE penalty: {penalty!r}")


def exact_loss_contributions(
    *,
    residual: torch.Tensor,
    active_mask: torch.Tensor,
    pointwise: torch.Tensor,
    batch: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    causal_loss: str,
    causal_n_chunks: int,
) -> torch.Tensor:
    """Return exact scalar-loss summands with shape [time, species, weight]."""
    mask = active_mask[None, :, :].expand_as(residual).to(residual.dtype)

    if causal_loss == "off":
        return pointwise * mask / mask.sum()

    if causal_loss != "expert":
        raise ValueError(f"Unsupported causal loss: {causal_loss!r}")
    if "t_chunk_idx" not in batch:
        raise KeyError("Expert causal loss requires batch['t_chunk_idx'].")
    if "pde_causal_weights" not in out:
        raise KeyError("compute_pde_loss() did not return pde_causal_weights.")

    chunk_idx = batch["t_chunk_idx"]
    causal_weights = out["pde_causal_weights"]
    contributions = torch.zeros_like(pointwise)

    for chunk in range(causal_n_chunks):
        time_mask = chunk_idx == chunk
        if not bool(time_mask.any().detach().cpu()):
            raise ValueError(f"Diagnostic batch has no samples in chunk {chunk}.")
        chunk_mask = mask[time_mask]
        contributions[time_mask] = (
            causal_weights[chunk]
            * pointwise[time_mask]
            * chunk_mask
            / chunk_mask.sum()
            / causal_n_chunks
        )

    return contributions


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint does not contain a usable config dictionary.")

    checkpoint_step = checkpoint.get("step")
    if checkpoint_step is None:
        raise KeyError("Checkpoint does not contain its training step.")
    checkpoint_step = int(checkpoint_step)

    input_dir = Path(
        args.input_dir
        if args.input_dir is not None
        else require_config(config, "input_dir")
    )

    params, n_init, n_pp = load_mizer_inputs(
        input_dir,
        dtype=torch.float64,
        device=device,
    )
    params.state_parameterization = config.get("state_parameterization", "log-n")
    set_state_scale_from_initial_condition(
        params,
        n_init,
        eps=float(config.get("state_scale_eps", DEFAULT_STATE_SCALE_EPS)),
    )

    n_species = int(params.interaction.shape[0])
    model = build_model_from_config(
        config=config,
        n_species=n_species,
        device=device,
    )
    load_checkpoint_weights(
        model=model,
        optimizer=None,
        checkpoint_path=checkpoint_path,
        device=device,
        load_optimizer_state=False,
    )
    model.eval()

    training_t_max = checkpoint_training_t_max(
        params=params,
        config=config,
        checkpoint_step=checkpoint_step,
    )
    evaluation_t_max = (
        float(args.evaluation_t_max)
        if args.evaluation_t_max is not None
        else training_t_max
    )

    physical_t_min = float(torch.as_tensor(params.t_min).detach().cpu())
    physical_t_max = float(torch.as_tensor(params.t_max).detach().cpu())
    if not physical_t_min < evaluation_t_max <= physical_t_max:
        raise ValueError(
            f"--evaluation-t-max must lie in ({physical_t_min}, {physical_t_max}], "
            f"got {evaluation_t_max}."
        )

    requested_plot_time = (
        float(args.time) if args.time is not None else evaluation_t_max
    )
    if not physical_t_min <= requested_plot_time <= evaluation_t_max:
        raise ValueError(
            f"--time must lie in [{physical_t_min}, {evaluation_t_max}], "
            f"got {requested_plot_time}. Increase --evaluation-t-max explicitly "
            "to inspect a later time."
        )

    causal_loss = config.get("causal_loss", "off")
    causal_n_chunks = int(config.get("causal_n_chunks", 32))
    time_sampling = config.get("time_sampling", "uniform")
    if causal_loss == "expert":
        time_sampling = "stratified"

    n_time = int(
        args.n_time
        if args.n_time is not None
        else config.get("n_time", causal_n_chunks)
    )
    if causal_loss == "expert":
        n_time = max(n_time, causal_n_chunks)

    batch = make_diagnostic_batch(
        params=params,
        n_time=n_time,
        t_max_current=evaluation_t_max,
        time_sampling=time_sampling,
        causal_n_chunks=causal_n_chunks,
        seed=args.seed,
    )

    residual_form = config.get("residual_form", "log")
    pde_penalty = config.get("pde_penalty", "squared")
    pde_delta = float(config.get("pde_pseudo_huber_delta", 1.0))

    # Same PDE-loss function used by the current multispecies training path.
    _, out = compute_pde_loss(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        residual_form=residual_form,
        n_init=None,
        lambda_pde=1.0,
        lambda_ic=0.0,
        lambda_bc=0.0,
        causal_loss=causal_loss,
        causal_n_chunks=causal_n_chunks,
        causal_epsilon=float(config.get("causal_epsilon", 1.0)),
        pde_penalty=pde_penalty,
        pde_pseudo_huber_delta=pde_delta,
    )

    residual_key = {
        "log": "residual_log",
        "scaled": "residual_scaled",
        "physical": "residual",
    }[residual_form]
    residual = out[residual_key]
    active_mask = active_eval_mask(batch["w_eval"], params)
    penalty = pointwise_penalty(
        residual,
        penalty=pde_penalty,
        delta=pde_delta,
    )
    contributions = exact_loss_contributions(
        residual=residual,
        active_mask=active_mask,
        pointwise=penalty,
        batch=batch,
        out=out,
        causal_loss=causal_loss,
        causal_n_chunks=causal_n_chunks,
    )

    reconstructed_loss = contributions.sum()
    reported_loss = out["loss_pde"]
    absolute_error = torch.abs(reconstructed_loss - reported_loss)
    tolerance = 1e-10 + 1e-8 * abs(float(reported_loss.detach().cpu()))
    if float(absolute_error.detach().cpu()) > tolerance:
        raise RuntimeError(
            "Pointwise decomposition did not reconstruct compute_pde_loss(): "
            f"reported={float(reported_loss.detach().cpu()):.16e}, "
            f"reconstructed={float(reconstructed_loss.detach().cpu()):.16e}, "
            f"error={float(absolute_error.detach().cpu()):.16e}."
        )

    t_eval = batch["t_eval"]
    plot_time_idx = int(
        torch.argmin(
            torch.abs(
                t_eval
                - torch.as_tensor(
                    requested_plot_time,
                    dtype=t_eval.dtype,
                    device=t_eval.device,
                )
            )
        ).item()
    )
    actual_plot_time = float(t_eval[plot_time_idx].detach().cpu())

    run_dir = infer_run_dir(checkpoint_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / f"pde_loss_decomposition_step_{checkpoint_step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    species_names = infer_species_names(run_dir, n_species)
    w_cpu = batch["w_eval"].detach().cpu()
    residual_cpu = residual.detach().cpu()
    penalty_cpu = penalty.detach().cpu()
    contribution_cpu = contributions.detach().cpu()
    active_cpu = active_mask.detach().cpu()
    total_by_species_weight = contribution_cpu.sum(dim=0)

    rows = []
    for species_idx, species_name in enumerate(species_names):
        for weight_idx, weight in enumerate(w_cpu.tolist()):
            rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "training_t_max_at_checkpoint": training_t_max,
                    "evaluation_t_max": evaluation_t_max,
                    "requested_plot_time": requested_plot_time,
                    "actual_plot_time": actual_plot_time,
                    "time_index": plot_time_idx,
                    "species_idx": species_idx,
                    "species": species_name,
                    "weight_index": weight_idx,
                    "w": weight,
                    "active": bool(active_cpu[species_idx, weight_idx]),
                    "residual_form": residual_form,
                    "residual_raw": float(
                        residual_cpu[plot_time_idx, species_idx, weight_idx]
                    ),
                    "residual_used_in_loss": (
                        float(residual_cpu[plot_time_idx, species_idx, weight_idx])
                        if bool(active_cpu[species_idx, weight_idx])
                        else float("nan")
                    ),
                    "pde_penalty": pde_penalty,
                    "pointwise_penalty": float(
                        penalty_cpu[plot_time_idx, species_idx, weight_idx]
                    ),
                    "exact_loss_contribution_at_plot_time": float(
                        contribution_cpu[plot_time_idx, species_idx, weight_idx]
                    ),
                    "exact_loss_contribution_all_times": float(
                        total_by_species_weight[species_idx, weight_idx]
                    ),
                }
            )

    decomposition_path = output_dir / "pde_loss_decomposition.csv"
    pd.DataFrame(rows).to_csv(decomposition_path, index=False)

    summary_path = output_dir / "pde_loss_summary.csv"
    pd.DataFrame(
        [
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": checkpoint_step,
                "input_dir": str(input_dir),
                "residual_form": residual_form,
                "pde_penalty": pde_penalty,
                "pde_pseudo_huber_delta": pde_delta,
                "causal_loss": causal_loss,
                "causal_n_chunks": causal_n_chunks,
                "diagnostic_requested_n_time": n_time,
                "diagnostic_effective_n_time": int(t_eval.numel()),
                "diagnostic_n_weight": int(w_cpu.numel()),
                "training_t_max_at_checkpoint": training_t_max,
                "evaluation_t_max": evaluation_t_max,
                "requested_plot_time": requested_plot_time,
                "actual_plot_time": actual_plot_time,
                "reported_loss_pde": float(reported_loss.detach().cpu()),
                "reconstructed_loss_pde": float(reconstructed_loss.detach().cpu()),
                "absolute_reconstruction_error": float(
                    absolute_error.detach().cpu()
                ),
            }
        ]
    ).to_csv(summary_path, index=False)

    # Plot only residuals that actually enter compute_pde_loss().
    # Inactive weights are masked out by active_eval_mask() and therefore make
    # exactly zero contribution to the scalar PDE loss.
    fig, ax = plt.subplots(figsize=(12, 7))
    for species_idx, species_name in enumerate(species_names):
        species_active = active_cpu[species_idx, :].numpy().astype(bool)
        ax.plot(
            w_cpu.numpy()[species_active],
            residual_cpu[plot_time_idx, species_idx, :].numpy()[species_active],
            label=species_name,
        )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Weight")
    ax.set_ylabel(f"PDE residual ({residual_form})")
    ax.set_title(
        f"PDE residual by weight at t={actual_plot_time:.6g}, "
        f"checkpoint step {checkpoint_step}"
    )
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    residual_plot_path = output_dir / "pde_residual_by_weight.png"
    fig.savefig(residual_plot_path, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for species_idx, species_name in enumerate(species_names):
        ax.plot(
            w_cpu.numpy(),
            total_by_species_weight[species_idx, :].numpy(),
            label=species_name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Exact contribution to scalar PDE loss")
    ax.set_title(
        f"PDE loss contribution by weight, checkpoint step {checkpoint_step}"
    )
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    contribution_plot_path = output_dir / "pde_loss_contribution_by_weight.png"
    fig.savefig(contribution_plot_path, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Checkpoint step: {checkpoint_step}")
    print(f"Training causal horizon at checkpoint: {training_t_max:.12g}")
    print(f"Diagnostic evaluation horizon: {evaluation_t_max:.12g}")
    print(f"Requested plot time: {requested_plot_time:.12g}")
    print(f"Actual sampled plot time: {actual_plot_time:.12g}")
    print(f"compute_pde_loss() loss_pde: {float(reported_loss.detach().cpu()):.16e}")
    print(
        "Reconstructed pointwise loss: "
        f"{float(reconstructed_loss.detach().cpu()):.16e}"
    )
    print(
        "Absolute reconstruction error: "
        f"{float(absolute_error.detach().cpu()):.16e}"
    )
    print(f"Saved: {residual_plot_path}")
    print(f"Saved: {contribution_plot_path}")
    print(f"Saved: {decomposition_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
