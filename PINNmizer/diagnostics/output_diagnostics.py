from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype


def _make_model_inputs(
    x_scaled: torch.Tensor,
    t_scaled: torch.Tensor,
) -> torch.Tensor:
    n_t = t_scaled.numel()
    n_x = x_scaled.numel()

    xx = x_scaled[None, :].expand(n_t, n_x)
    tt = t_scaled[:, None].expand(n_t, n_x)

    return torch.stack(
        [xx.reshape(-1), tt.reshape(-1)],
        dim=1,
    )


def _get_fish_w_grid(params, device, dtype) -> torch.Tensor:
    if hasattr(params, "w"):
        w = params.w
    elif hasattr(params, "w_grid"):
        w = params.w_grid
    else:
        raise AttributeError(
            "Could not find params.w or params.w_grid. "
            "Pass the fish weight grid explicitly or adapt _get_fish_w_grid()."
        )

    if not torch.is_tensor(w):
        w = torch.as_tensor(w)

    return w.to(device=device, dtype=dtype)


def evaluate_logN_on_grid(
    model,
    params,
    n_t: int = 101,
    n_x: int | None = None,
    species_idx: int = 0,
) -> dict[str, torch.Tensor]:
    model.eval()

    device, dtype = _model_device_dtype(model)

    w_grid_native = _get_fish_w_grid(params, device=device, dtype=dtype)

    if n_x is None:
        w_eval = w_grid_native
    else:
        x_min = torch.log(w_grid_native.min())
        x_max = torch.log(w_grid_native.max())
        x_eval = torch.linspace(x_min, x_max, n_x, device=device, dtype=dtype)
        w_eval = torch.exp(x_eval)

    x_eval = torch.log(w_eval)
    x_scaled = (x_eval - x_eval.min()) / (x_eval.max() - x_eval.min())
    t_scaled = torch.linspace(0.0, 1.0, n_t, device=device, dtype=dtype)

    inputs = _make_model_inputs(x_scaled=x_scaled, t_scaled=t_scaled)

    with torch.no_grad():
        raw = model(inputs)

    if raw.ndim == 1:
        raw = raw[:, None]

    n_species = raw.shape[1]
    if species_idx >= n_species:
        raise ValueError(f"species_idx={species_idx} but model output has only {n_species} species.")

    log_N = raw.reshape(n_t, x_scaled.numel(), n_species).permute(0, 2, 1).contiguous()
    log_N_species = log_N[:, species_idx, :]
    N_species = torch.exp(log_N_species)

    return {
        "t_scaled": t_scaled.detach().cpu(),
        "x_eval": x_eval.detach().cpu(),
        "w_eval": w_eval.detach().cpu(),
        "x_scaled": x_scaled.detach().cpu(),
        "log_N": log_N_species.detach().cpu(),
        "N": N_species.detach().cpu(),
    }


def save_output_surface_diagnostics(
    model,
    params,
    outdir: str | Path,
    n_t: int = 101,
    n_x: int = 200,
    species_idx: int = 0,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = evaluate_logN_on_grid(
        model=model,
        params=params,
        n_t=n_t,
        n_x=n_x,
        species_idx=species_idx,
    )

    t = out["t_scaled"].numpy()
    x = out["x_eval"].numpy()
    w = out["w_eval"].numpy()
    log_N = out["log_N"].numpy()
    N = out["N"].numpy()

    np.savez(
        outdir / "nn_output_surface.npz",
        t_scaled=t,
        x_eval=x,
        w_eval=w,
        log_N=log_N,
        N=N,
    )

    log10_N = np.log10(np.maximum(N, np.finfo(float).tiny))
      
    plt.figure()
    plt.imshow(
        log10_N,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
    )
    plt.colorbar(label="log10(N)")
    plt.xlabel("log weight")
    plt.ylabel("scaled time")
    plt.title("Learned log10 N(t, w)")
    plt.tight_layout()
    plt.savefig(outdir / "surface_log10_N.png", dpi=200)
    plt.close()
    
    plt.figure()
    for idx in np.linspace(0, len(t) - 1, 6).astype(int):
        plt.plot(x, log10_N[idx, :], label=f"t={t[idx]:.2f}")
    plt.xlabel("log weight")
    plt.ylabel("log10(N)")
    plt.title("log10(N) profiles through time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "log10_N_profiles_by_time.png", dpi=200)
    plt.close()

    print(f"Saved NN output diagnostics to: {outdir}")
