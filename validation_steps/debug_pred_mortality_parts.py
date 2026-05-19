from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.continuous_biology import (
    compute_growth_direct_at_eval,
    compute_phi_and_dphi_dw,
)
from PINNmizer.params import _params_dtype_device


OUT_DIR = Path("py_pred_mort_debug")
OUT_DIR.mkdir(exist_ok=True)


def write_tensor(x: torch.Tensor, name: str) -> None:
    x = x.detach().cpu()

    if x.ndim == 1:
        pd.DataFrame({"value": x.numpy()}).to_csv(OUT_DIR / f"{name}.csv", index=False)
    elif x.ndim == 2:
        pd.DataFrame(x.numpy()).to_csv(OUT_DIR / f"{name}.csv", index=False)
    else:
        raise ValueError(f"{name} has ndim={x.ndim}; write 3D tensors by species slices.")


def report_diff(name: str, direct: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (direct - ref).detach().abs()
    rel = diff / ref.detach().abs().clamp_min(1e-12)

    flat_idx = diff.argmax().item()
    idx = torch.unravel_index(torch.tensor(flat_idx), diff.shape)

    print(
        f"{name:20s} "
        f"max_abs={diff.max().item():.6e} "
        f"mean_abs={diff.mean().item():.6e} "
        f"max_rel={rel.max().item():.6e} "
        f"mean_rel={rel.mean().item():.6e} "
        f"worst_idx={tuple(int(i) for i in idx)}"
    )


params, n, n_pp = load_mizer_inputs(
    "py_inputs",
    dtype=torch.float64,
    device="cpu",
)

dtype, device = _params_dtype_device(params)
n = n.to(dtype=dtype, device=device)
n_pp = n_pp.to(dtype=dtype, device=device)

# Same setup as compute_pred_mortality_direct_at_eval()
growth_grid = compute_growth_direct_at_eval(
    n_pp=n_pp,
    n_grid=n,
    w_eval=params.w,
    params=params,
)

feeding_grid = growth_grid["feeding_eval"]      # [species, n_w]
gamma_grid = growth_grid["gamma_eval"]          # [species, n_w]

q_grid = (1.0 - feeding_grid) * gamma_grid * n  # [species, n_w]

phi, dphi_dw = compute_phi_and_dphi_dw(
    w_pred_eval=params.w,
    w_prey_grid_or_eval=params.w,
    params=params,
)

# phi: [predator_species, predator_w, prey_w]
pred_rate = (
    phi
    * q_grid[:, :, None]
    * params.dw[None, :, None]
).sum(dim=1)

pred_mort_eval = params.interaction.T @ pred_rate

# -------------------------
# Export main 2D quantities
# -------------------------
write_tensor(params.w, "w")
write_tensor(params.dw, "dw")

write_tensor(gamma_grid, "gamma_grid")
write_tensor(params.search_vol, "search_vol_grid")
write_tensor(gamma_grid - params.search_vol, "gamma_minus_search_vol")

write_tensor(feeding_grid, "feeding_grid")
write_tensor(q_grid, "q_grid")

write_tensor(pred_rate, "pred_rate_direct")
write_tensor(pred_mort_eval, "pred_mort_direct")

# -------------------------
# Export phi per predator species
# -------------------------
for j in range(phi.shape[0]):
    write_tensor(phi[j], f"phi_pred_species_{j:02d}")
    write_tensor(dphi_dw[j], f"dphi_dw_pred_species_{j:02d}")

# -------------------------
# Useful printed differences
# -------------------------
print("\n--- grid quantity comparisons ---")
report_diff("gamma vs search_vol", gamma_grid, params.search_vol)

print("\n--- ranges ---")
for name, x in {
    "gamma_grid": gamma_grid,
    "search_vol": params.search_vol,
    "feeding_grid": feeding_grid,
    "q_grid": q_grid,
    "phi": phi,
    "pred_rate": pred_rate,
    "pred_mort": pred_mort_eval,
}.items():
    xd = x.detach()
    print(
        f"{name:20s} "
        f"min={xd.min().item():.6e} "
        f"max={xd.max().item():.6e} "
        f"mean={xd.mean().item():.6e}"
    )

print(f"\nWrote debug outputs to {OUT_DIR}/")
