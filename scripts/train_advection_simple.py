from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.losses import compute_expert_causal_pde_loss
from PINNmizer.training.weighting import update_expert_gradient_norm_weights_


class PeriodicAdvectionModel(nn.Module):
    """
    Model for u(t, x).

    Input columns:
        tx[:, 0] = t in [0, 1]
        tx[:, 1] = x in [0, 2*pi]

    The embedding imposes periodicity in x and gives the model a periodic
    time representation.
    """

    def __init__(
        self,
        *,
        hidden_width: int,
        hidden_layers: int,
        fourier_num_features: int,
        fourier_scale: float,
        rwf_mu: float,
        rwf_sigma: float,
        dtype: torch.dtype,
    ):
        super().__init__()

        # Trainable time period, initialized to 1 as in the paper's idea.
        self.log_time_period = nn.Parameter(torch.tensor(0.0, dtype=dtype))

        self.net = build_pinn_model(
            model_arch="fourier",
            in_dim=4,
            out_dim=1,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            fourier_num_features=fourier_num_features,
            fourier_scale=fourier_scale,
            fourier_include_raw_input=False,
            weight_factorization="rwf",
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            rwf_apply_to="all",
            rwf_base_init="xavier_uniform",
        )

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        t = tx[:, 0:1]
        x = tx[:, 1:2]

        time_period = torch.exp(self.log_time_period) + 1e-8
        omega_t = 2.0 * math.pi / time_period

        z = torch.cat(
            [
                torch.cos(omega_t * t),
                torch.sin(omega_t * t),
                torch.cos(x),
                torch.sin(x),
            ],
            dim=1,
        )

        return self.net(z)


def exact_solution(t: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    return torch.sin(x - c * t)


def sample_batch(
    *,
    n_time: int,
    n_x: int,
    n_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if n_time % n_chunks != 0:
        raise ValueError("--n-time must be divisible by --causal-n-chunks.")

    per_chunk = n_time // n_chunks
    t_list = []
    chunk_list = []

    edges = torch.linspace(0.0, 1.0, n_chunks + 1, device=device, dtype=dtype)

    for k in range(n_chunks):
        lo = edges[k]
        hi = edges[k + 1]

        t_k = lo + (hi - lo) * torch.rand(per_chunk, device=device, dtype=dtype)
        t_list.append(t_k)

        chunk_list.append(
            torch.full((per_chunk,), k, device=device, dtype=torch.long)
        )

    t = torch.cat(t_list)
    t_chunk_idx = torch.cat(chunk_list)

    x = 2.0 * math.pi * torch.rand(n_x, device=device, dtype=dtype)

    tt, xx = torch.meshgrid(t, x, indexing="ij")
    tx = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)

    return {
        "tx": tx,
        "t_chunk_idx": t_chunk_idx,
        "n_time": n_time,
        "n_x": n_x,
    }


def advection_residual(
    *,
    model: nn.Module,
    tx: torch.Tensor,
    c: float,
) -> torch.Tensor:
    tx = tx.detach().clone().requires_grad_(True)

    u = model(tx)

    grad = torch.autograd.grad(
        u,
        tx,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_t = grad[:, 0:1]
    u_x = grad[:, 1:2]

    return u_t + c * u_x


def compute_losses(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    n_ic: int,
    c: float,
    causal_n_chunks: int,
    causal_epsilon: float,
) -> dict[str, torch.Tensor]:
    residual_flat = advection_residual(
        model=model,
        tx=batch["tx"],
        c=c,
    )

    residual = residual_flat.reshape(batch["n_time"], 1, batch["n_x"])
    active_mask = torch.ones_like(residual)

    loss_pde, causal_out = compute_expert_causal_pde_loss(
        residual=residual,
        active_mask=active_mask,
        t_chunk_idx=batch["t_chunk_idx"],
        n_chunks=causal_n_chunks,
        epsilon=causal_epsilon,
    )

    x_ic = 2.0 * math.pi * torch.rand(
        n_ic,
        device=batch["tx"].device,
        dtype=batch["tx"].dtype,
    )
    t_ic = torch.zeros_like(x_ic)

    tx_ic = torch.stack([t_ic, x_ic], dim=1)
    u_pred = model(tx_ic)
    u_true = torch.sin(x_ic).reshape(-1, 1)

    loss_ic = ((u_pred - u_true) ** 2).mean()

    return {
        "loss_pde": loss_pde,
        "loss_ic": loss_ic,
        **causal_out,
    }


@torch.no_grad()
def relative_l2(
    *,
    model: nn.Module,
    c: float,
    device: torch.device,
    dtype: torch.dtype,
    n_time: int = 101,
    n_x: int = 256,
) -> float:
    t = torch.linspace(0.0, 1.0, n_time, device=device, dtype=dtype)
    x = torch.linspace(0.0, 2.0 * math.pi, n_x, device=device, dtype=dtype)

    tt, xx = torch.meshgrid(t, x, indexing="ij")
    tx = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)

    pred = model(tx).reshape(n_time, n_x)
    true = exact_solution(tt, xx, c)

    return float(torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--n-steps", type=int, default=200000)
    p.add_argument("--n-time", type=int, default=32)
    p.add_argument("--n-x", type=int, default=256)
    p.add_argument("--n-ic", type=int, default=8192)

    p.add_argument("--c", type=float, default=80.0)

    p.add_argument("--hidden-width", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--fourier-num-features", type=int, default=128)
    p.add_argument("--fourier-scale", type=float, default=1.0)

    p.add_argument("--rwf-mu", type=float, default=1.0)
    p.add_argument("--rwf-sigma", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-step-size", type=int, default=2000)
    p.add_argument("--lr-gamma", type=float, default=0.9)

    p.add_argument("--causal-n-chunks", type=int, default=32)
    p.add_argument("--causal-epsilon", type=float, default=1.0)

    p.add_argument("--loss-weighting", choices=["none", "grad-norm"], default="grad-norm")
    p.add_argument("--weight-update-every", type=int, default=1000)
    p.add_argument("--weight-alpha", type=float, default=0.9)
    p.add_argument("--weight-min", type=float, default=1e-3)
    p.add_argument("--weight-max", type=float, default=1e3)

    p.add_argument("--lambda-pde", type=float, default=1.0)
    p.add_argument("--lambda-ic", type=float, default=1.0)

    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--out-dir", default="runs/advection_simple")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    run_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    model = PeriodicAdvectionModel(
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_num_features=args.fourier_num_features,
        fourier_scale=args.fourier_scale,
        rwf_mu=args.rwf_mu,
        rwf_sigma=args.rwf_sigma,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    weights = {
        "pde": 1.0,
        "ic": 1.0,
        "bc": 1.0,
        "timestep": 1.0,
    }

    has_updated_weights = False
    history = []
    start = time.perf_counter()

    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        batch = sample_batch(
            n_time=args.n_time,
            n_x=args.n_x,
            n_chunks=args.causal_n_chunks,
            device=device,
            dtype=dtype,
        )

        out = compute_losses(
            model=model,
            batch=batch,
            n_ic=args.n_ic,
            c=args.c,
            causal_n_chunks=args.causal_n_chunks,
            causal_epsilon=args.causal_epsilon,
        )

        losses_for_weighting = {
            "pde": args.lambda_pde * out["loss_pde"],
            "ic": args.lambda_ic * out["loss_ic"],
            "bc": torch.zeros((), device=device, dtype=dtype),
            "timestep": torch.zeros((), device=device, dtype=dtype),
        }

        if (
            args.loss_weighting == "grad-norm"
            and step % args.weight_update_every == 0
        ):
            update_expert_gradient_norm_weights_(
                model=model,
                losses=losses_for_weighting,
                weights=weights,
                alpha=args.weight_alpha,
                min_weight=args.weight_min,
                max_weight=args.weight_max,
                hard_set=not has_updated_weights,
            )
            has_updated_weights = True

        if args.loss_weighting == "none":
            loss = losses_for_weighting["pde"] + losses_for_weighting["ic"]
        else:
            loss = (
                weights["pde"] * losses_for_weighting["pde"]
                + weights["ic"] * losses_for_weighting["ic"]
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.print_every == 0:
            err = relative_l2(
                model=model,
                c=args.c,
                device=device,
                dtype=dtype,
            )

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_pde": float(out["loss_pde"].detach().cpu()),
                "loss_ic": float(out["loss_ic"].detach().cpu()),
                "w_pde": float(weights["pde"]),
                "w_ic": float(weights["ic"]),
                "rel_l2": err,
                "time_period": float(torch.exp(model.log_time_period).detach().cpu()),
                "causal_weight_min": float(out["pde_causal_weight_min"].detach().cpu()),
                "seconds": time.perf_counter() - start,
            }

            history.append(row)
            print(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        run_dir / "final_model.pt",
    )


if __name__ == "__main__":
    main()
