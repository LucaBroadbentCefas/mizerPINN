from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class MizerTorchParams:
    # grids
    w_full: torch.Tensor                 # [k]
    w: torch.Tensor                      # [w]
    dw_full: torch.Tensor                # [k]
    dw: torch.Tensor                     # [w]
    w_min_idx: torch.Tensor              # [species], R-style 1-based indices as exported from R/TMB

    # feeding / growth / mortality parameters
    ft_pred_kernel_e: torch.Tensor       # [species, k], complex
    ft_pred_kernel_p: torch.Tensor       # [species, k], complex
    ft_mask: torch.Tensor                # [species, k]
    search_vol: torch.Tensor             # [species, w]
    intake_max: torch.Tensor             # [species, w]
    alpha: torch.Tensor                  # [species]
    metab: torch.Tensor                  # [species, w]
    psi: torch.Tensor                    # [species, w]
    mu_b: torch.Tensor                   # [species, w]

    # species/resource interaction
    interaction_resource: torch.Tensor   # [species]
    interaction: torch.Tensor            # [predator, prey]

    # reproduction/resource
    erepro: torch.Tensor                 # [species]
    r_max: torch.Tensor                  # [species]
    rr_pp: torch.Tensor                  # [k]
    cc_pp: torch.Tensor                  # [k]

    # fishing mortality for one timestep, optional
    f_mort: Optional[torch.Tensor] = None # [species, w]
    
        # species metadata
    species: Optional[list[str]] = None

    # continuous search / encounter prefactor
    gamma: Optional[torch.Tensor] = None        # [species]
    q: Optional[torch.Tensor] = None            # [species]

    # continuous maximum intake
    h: Optional[torch.Tensor] = None            # [species]
    n_exp: Optional[torch.Tensor] = None        # [species], species_params$n

    # continuous metabolism
    ks: Optional[torch.Tensor] = None           # [species]
    p_exp: Optional[torch.Tensor] = None        # [species], species_params$p
    k_metab: Optional[torch.Tensor] = None      # [species], species_params$k

    # continuous predation kernel
    beta: Optional[torch.Tensor] = None         # [species]
    sigma: Optional[torch.Tensor] = None        # [species]
    kernel_cutoff_sigma: float = None

    # continuous reproduction allocation
    w_max: Optional[torch.Tensor] = None        # [species]
    w_mat: Optional[torch.Tensor] = None        # [species]
    U: Optional[torch.Tensor] = None            # [species]
    w_repro_max: Optional[torch.Tensor] = None  # [species]
    m_exp: Optional[torch.Tensor] = None        # [species], species_params$m

    # continuous background mortality
    z0_pre: Optional[torch.Tensor] = None       # scalar or [species]
    w_inf: Optional[torch.Tensor] = None        # [species]

    # physical time domain
    t_min: float | torch.Tensor = 0.0
    t_max: float | torch.Tensor = 1.0
    
    # background mortality mode
    z0: Optional[torch.Tensor] = None
    mu_b_allometric: bool = False


def fish_start(params: MizerTorchParams) -> int:
    return params.w_full.numel() - params.w.numel()

def _params_dtype_device(params: MizerTorchParams):
    return params.w.dtype, params.w.device
  
  
def _to_param_tensor(x, params: MizerTorchParams) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def _species_vector(params: MizerTorchParams, name: str) -> torch.Tensor:
    value = getattr(params, name, None)

    if value is None:
        raise ValueError(f"Missing required continuous parameter: params.{name}")

    value = _to_param_tensor(value, params)

    n_species = params.interaction.shape[0]

    if value.ndim == 0:
        value = value.expand(n_species)

    assert value.shape == (n_species,), (
        f"params.{name} must have shape [{n_species}], got {tuple(value.shape)}"
    )

    return value


def _eval_weight_vector(w_eval: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    dtype, device = _params_dtype_device(params)
    w_eval = w_eval.to(dtype=dtype, device=device)
    assert w_eval.ndim == 1, f"w_eval must be 1D, got {tuple(w_eval.shape)}"
    assert torch.all(w_eval > 0), "All physical weights must be positive."
    return w_eval


def _x_grid(params: MizerTorchParams) -> torch.Tensor:
    return torch.log(params.w)
  
def _x_limits(params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    x_grid = _x_grid(params)
    return x_grid[0], x_grid[-1]

def _t_limits(params: MizerTorchParams) -> tuple[torch.Tensor, torch.Tensor]:
    return _to_param_tensor(params.t_min, params), _to_param_tensor(params.t_max, params)


def scale_x(x: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    x_grid = _x_grid(params)
    return (x - x_grid[0]) / (x_grid[-1] - x_grid[0])


def scale_t(t: torch.Tensor, params: MizerTorchParams) -> torch.Tensor:
    t_min, t_max = _t_limits(params)
    return (t - t_min) / (t_max - t_min)
  
def _n_species(params: MizerTorchParams) -> int:
    return params.interaction.shape[0]


def _n_w(params: MizerTorchParams) -> int:
    return params.w.numel()


def _k_full(params: MizerTorchParams) -> int:
    return params.w_full.numel()
  
###REMOVE LATER.
def validate_params_shapes(params: MizerTorchParams) -> None:
    n_species = _n_species(params)
    n_w = _n_w(params)
    k = _k_full(params)

    assert params.w_full.shape == (k,)
    assert params.w.shape == (n_w,)
    assert params.dw_full.shape == (k,)
    assert params.dw.shape == (n_w,)

    assert params.interaction.shape == (n_species, n_species)
    assert params.interaction_resource.shape == (n_species,)

    assert params.alpha.shape == (n_species,)

    for name in [
        "gamma", "q",
        "h", "n_exp",
        "ks", "p_exp", "k_metab",
        "beta", "sigma",
        "w_max", "w_mat", "U", "w_repro_max", "m_exp",
        "z0_pre", "w_inf",
    ]:
        value = getattr(params, name)
        value = _to_param_tensor(value, params)
        if value.ndim != 0:
            assert value.shape == (n_species,), f"{name} has shape {value.shape}"

    assert params.ft_pred_kernel_e.shape == (n_species, k)
    assert params.ft_pred_kernel_p.shape == (n_species, k)
    assert params.ft_mask.shape == (n_species, k)

    assert params.search_vol.shape == (n_species, n_w)
    assert params.intake_max.shape == (n_species, n_w)
    assert params.metab.shape == (n_species, n_w)
    assert params.psi.shape == (n_species, n_w)
    assert params.mu_b.shape == (n_species, n_w)
  
