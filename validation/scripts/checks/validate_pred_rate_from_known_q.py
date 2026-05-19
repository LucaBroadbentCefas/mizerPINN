from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs, load_mat
from PINNmizer.biology.kernels import compute_phi_and_dphi_dw
from PINNmizer.params import _params_dtype_device


OUT_DIR = Path("validation/fixtures/py_inputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_mat(x, name):
    pd.DataFrame(x.detach().cpu().numpy()).to_csv(
        OUT_DIR / f"{name}.csv",
        index=False,
    )


params, n, n_pp = load_mizer_inputs(
    "validation/fixtures/py_inputs",
    dtype=torch.float64,
    device="cpu",
)

dtype, device = _params_dtype_device(params)

q_known_fish = load_mat(
    Path("validation/fixtures/py_inputs"),
    "q_known_fish",
    dtype,
    device,
)

# q_known_fish already includes dw.
# shape: [species, n_w]
assert q_known_fish.shape == (params.interaction.shape[0], params.w.numel())

phi, _ = compute_phi_and_dphi_dw(
    w_pred_eval=params.w,
    w_prey_grid_or_eval=params.w,
    params=params,
)

# phi: [species, predator_w, prey_w]
# q_known_fish[:, :, None]: [species, predator_w, 1]
pred_rate_direct_known_q = (
    phi * q_known_fish[:, :, None]
).sum(dim=1)

pred_mort_direct_known_q = params.interaction.T @ pred_rate_direct_known_q

write_mat(pred_rate_direct_known_q, "pred_rate_direct_known_q")
write_mat(pred_mort_direct_known_q, "pred_mort_direct_known_q")

print("Wrote validation/outputs/py_known_q_direct/")
