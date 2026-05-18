#python -m validation_steps.check_biology
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.mizer_grid_ops import mizer_operators
from PINNmizer.continuous_biology import (
    compute_growth_direct_at_eval,
    compute_total_mortality_direct_at_eval,
)


def cmp(name, direct, ref, eps=1e-12):
    d = (direct - ref).detach().abs()
    rel = d / ref.detach().abs().clamp_min(eps)

    print(
        f"{name:20s} "
        f"max_abs={d.max().item():.6e}   "
        f"mean_abs={d.mean().item():.6e}   "
        f"max_rel={rel.max().item():.6e}   "
        f"mean_rel={rel.mean().item():.6e}"
    )


params, n, n_pp = load_mizer_inputs(
    "py_inputs",
    dtype=torch.float64,
    device="cpu",
)

ops = mizer_operators(
    n_pp=n_pp,
    n=n,
    params=params,
)

growth = compute_growth_direct_at_eval(
    n_pp=n_pp,
    n_grid=n,
    w_eval=params.w,
    params=params,
)

mort = compute_total_mortality_direct_at_eval(
    n_pp=n_pp,
    N_pred_grid=n,
    w_eval=params.w,
    params=params,
)

cmp("encounter", growth["encounter_eval"], ops["encounter"])
cmp("feeding", growth["feeding_eval"], ops["feeding_level"])
cmp("e_growth", growth["e_growth_eval"], ops["e_growth"])
cmp("pred_mort", mort["pred_mort_eval"], ops["pred_mort"])
cmp("total_mort", mort["mu_eval"], ops["mort"])
