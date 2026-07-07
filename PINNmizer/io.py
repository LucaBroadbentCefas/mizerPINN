from pathlib import Path

import pandas as pd
import torch

from .params import MizerTorchParams


def load_mat(root: Path, name: str, dtype, device) -> torch.Tensor:
    arr = pd.read_csv(root / f"{name}.csv").to_numpy().copy()
    return torch.as_tensor(arr, dtype=dtype, device=device)


def load_vec(root: Path, name: str, dtype, device) -> torch.Tensor:
    return load_mat(root, name, dtype, device).reshape(-1)

def load_complex_mat(root: Path, name: str, dtype, device) -> torch.Tensor:
    real = load_mat(root, f"{name}_real", dtype, device)
    imag = load_mat(root, f"{name}_imag", dtype, device)
    return torch.complex(real, imag)


def maybe_vec(root: Path, name: str, dtype, device):
    path = root / f"{name}.csv"
    if not path.exists():
        return None
    return load_vec(root, name, dtype, device)


def maybe_mat(root: Path, name: str, dtype, device):
    path = root / f"{name}.csv"
    if not path.exists():
        return None
    return load_mat(root, name, dtype, device)


def maybe_selectivity(root: Path, dtype, device, catchability, n_w: int):
    raw = maybe_mat(root, "selectivity", dtype, device)
    if raw is None:
        return None
    if raw.ndim == 3:
        return raw
    if catchability is None:
        return raw
    n_gear, n_species = catchability.shape
    if raw.shape == (n_gear * n_species, n_w):
        return raw.reshape(n_gear, n_species, n_w)
    if raw.shape == (n_species * n_gear, n_w):
        return raw.reshape(n_gear, n_species, n_w)
    return raw


def load_mizer_inputs(
    outdir: str | Path,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[MizerTorchParams, torch.Tensor, torch.Tensor]:
    root = Path(outdir)
    device = torch.device(device)

    n_init = load_mat(root, "n_init_full", dtype, device)
    n_pp = load_vec(root, "n_pp", dtype, device)
    dt = maybe_vec(root, "dt", dtype, device)
    t_min = maybe_vec(root, "t_min", dtype, device)
    t_max = maybe_vec(root, "t_max", dtype, device)

    mu_b_allometric_vec = maybe_vec(root, "mu_b_allometric", torch.long, device)
    catchability = maybe_mat(root, "catchability", dtype, device)
    mu_b_allometric = (
       bool(int(mu_b_allometric_vec[0].item()))
       if mu_b_allometric_vec is not None
       else False
    )

    params = MizerTorchParams(
        # grids
        w_full=load_vec(root, "w_full", dtype, device),
        w=load_vec(root, "w", dtype, device),
        dw_full=load_vec(root, "dw_full", dtype, device),
        dw=load_vec(root, "dw", dtype, device),
        w_min_idx=load_vec(root, "w_min_idx", torch.long, device),

        # FFT validation quantities
        ft_pred_kernel_e=load_complex_mat(root, "ft_pred_kernel_e", dtype, device),
        ft_pred_kernel_p=load_complex_mat(root, "ft_pred_kernel_p", dtype, device),
        ft_mask=load_mat(root, "ft_mask", dtype, device),
        search_vol=load_mat(root, "search_vol", dtype, device),
        intake_max=load_mat(root, "intake_max", dtype, device),
        metab=load_mat(root, "metab", dtype, device),
        psi=load_mat(root, "psi", dtype, device),
        mu_b=load_mat(root, "mu_b", dtype, device),

        # species parameters
        alpha=load_vec(root, "alpha", dtype, device),
        gamma=load_vec(root, "gamma", dtype, device),
        q=load_vec(root, "q", dtype, device),
        h=load_vec(root, "h", dtype, device),
        n_exp=load_vec(root, "n_exp", dtype, device),
        ks=load_vec(root, "ks", dtype, device),
        p_exp=load_vec(root, "p_exp", dtype, device),
        k_metab=load_vec(root, "k_metab", dtype, device),
        beta=load_vec(root, "beta", dtype, device),
        sigma=load_vec(root, "sigma", dtype, device),
        w_max=load_vec(root, "w_max", dtype, device),
        w_mat=load_vec(root, "w_mat", dtype, device),
        U=load_vec(root, "U", dtype, device),
        w_repro_max=load_vec(root, "w_repro_max", dtype, device),
        m_exp=load_vec(root, "m_exp", dtype, device),
        z0_pre=load_vec(root, "z0_pre", dtype, device),
        z0=load_vec(root, "z0", dtype, device),
        mu_b_allometric=mu_b_allometric,
        w_inf=load_vec(root, "w_inf", dtype, device),

        # interactions/resource/reproduction
        interaction_resource=load_vec(root, "interaction_resource", dtype, device),
        interaction=load_mat(root, "interaction", dtype, device),
        erepro=load_vec(root, "erepro", dtype, device),
        r_max=load_vec(root, "r_max", dtype, device),
        rr_pp=load_vec(root, "rr_pp", dtype, device),
        cc_pp=load_vec(root, "cc_pp", dtype, device),

        catchability=catchability,
        selectivity=maybe_selectivity(root, dtype, device, catchability, load_vec(root, "w", dtype, device).numel()),
        initial_effort=maybe_vec(root, "initial_effort", dtype, device),
        fishing_effort_time=maybe_vec(root, "fishing_effort_time", dtype, device),
        fishing_effort=maybe_mat(root, "fishing_effort", dtype, device),
        f_mort=maybe_mat(root, "f_mort", dtype, device),

        t_min=float(t_min[0]) if t_min is not None else 0.0,
        t_max=float(t_max[0]) if t_max is not None else 1.0,
        dt=dt[0] if dt is not None else None,
    )

    return params, n_init, n_pp
