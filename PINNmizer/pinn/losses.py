from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _n_w, _params_dtype_device
from PINNmizer.pinn.pde_state import compute_pde_state
from PINNmizer.pinn.residual import compute_pde_residual_from_state


def _as_species_w_matrix(x: torch.Tensor, *, n_species: int, n_w: int, name: str) -> torch.Tensor:
    if x.ndim == 1:
        if n_species != 1:
            raise ValueError(f"{name} is 1D but n_species={n_species}. Pass [n_species, n_w].")
        if x.numel() != n_w:
            raise ValueError(f"{name} has length {x.numel()}, expected {n_w}.")
        return x.reshape(1, n_w)
    if x.ndim == 2:
        if x.shape != (n_species, n_w):
            raise ValueError(f"{name} has shape {tuple(x.shape)}, expected {(n_species, n_w)}.")
        return x
    raise ValueError(f"{name} must be [n_w] or [n_species, n_w].")


def _species_grid_values_at_indices(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    n_time, n_species, _ = x.shape
    idx = idx.to(device=x.device, dtype=torch.long)
    gather_idx = idx[None, :, None].expand(n_time, n_species, 1)
    return torch.gather(x, dim=2, index=gather_idx).squeeze(-1)


def _scalar_tensor_min(x: torch.Tensor) -> torch.Tensor: return torch.min(x.detach())
def _scalar_tensor_max(x: torch.Tensor) -> torch.Tensor: return torch.max(x.detach())
def _fraction_leq(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor: return (x.detach() <= threshold).to(dtype=x.dtype).mean()
def _abs_quantile(x: torch.Tensor, q: float) -> torch.Tensor: return torch.quantile(torch.abs(x.detach()).reshape(-1), q)


def compute_initial_condition_loss_from_state(state: dict[str, object], params: MizerTorchParams, n_init: torch.Tensor, *, species_idx: int | None = None, eps: float = 1e-30) -> dict[str, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    if state["log_N_ic"] is None:
        raise ValueError("State does not contain IC output. Call compute_pde_state(..., include_ic=True).")
    n_species, n_w = _n_species(params), _n_w(params)
    n_init = _as_species_w_matrix(torch.as_tensor(n_init, dtype=dtype, device=device), n_species=n_species, n_w=n_w, name="n_init")
    log_N_pred, N_pred = state["log_N_ic"][0], state["N_ic"][0]
    log_N_target = torch.log(torch.clamp(n_init, min=eps))
    if species_idx is not None:
        log_N_pred = log_N_pred[species_idx: species_idx + 1]
        N_pred = N_pred[species_idx: species_idx + 1]
        log_N_target = log_N_target[species_idx: species_idx + 1]
        n_init = n_init[species_idx: species_idx + 1]
    loss_ic = ((log_N_pred - log_N_target) ** 2).mean()
    return {"loss_ic": loss_ic, "log_N_ic_pred": log_N_pred, "N_ic_pred": N_pred, "log_N_ic_target": log_N_target, "N_ic_target": n_init}


def compute_recruitment_boundary_loss_from_state(state: dict[str, object], params: MizerTorchParams, *, species_idx: int | None = None, loss_form: str = "log", eps: float = 1e-30) -> dict[str, torch.Tensor]:
    N_grid, growth_grid, recruitment = state["N_grid"], state["growth_grid"], state["recruitment"]
    egg_idx = params.w_min_idx.to(torch.long) - 1
    N_left = _species_grid_values_at_indices(N_grid, egg_idx)
    g_left = _species_grid_values_at_indices(growth_grid["e_growth_eval"], egg_idx)
    flux_left = g_left * N_left
    recruitment_flux = recruitment["rdd_flux"]
    if species_idx is not None:
        flux_left = flux_left[:, species_idx: species_idx + 1]
        recruitment_flux = recruitment_flux[:, species_idx: species_idx + 1]
        g_left = g_left[:, species_idx: species_idx + 1]
        N_left = N_left[:, species_idx: species_idx + 1]
    if loss_form == "physical": boundary_residual = flux_left - recruitment_flux
    elif loss_form == "log":
        eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
        boundary_residual = torch.log(torch.clamp(flux_left, min=eps_t)) - torch.log(torch.clamp(recruitment_flux, min=eps_t))
    elif loss_form == "relative": boundary_residual = (flux_left - recruitment_flux) / torch.clamp(torch.abs(recruitment_flux), min=eps)
    else: raise ValueError("loss_form must be 'physical', 'log', or 'relative'.")
    loss_bc = (boundary_residual ** 2).mean()
    eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
    return {"loss_bc": loss_bc, "boundary_residual": boundary_residual, "flux_left": flux_left, "recruitment_flux": recruitment_flux, "g_left": g_left, "N_left": N_left, "bc_eps": eps_t, "flux_left_min": _scalar_tensor_min(flux_left), "flux_left_max": _scalar_tensor_max(flux_left), "recruitment_flux_min": _scalar_tensor_min(recruitment_flux), "recruitment_flux_max": _scalar_tensor_max(recruitment_flux), "frac_flux_left_clamped": _fraction_leq(flux_left, eps_t), "frac_recruitment_flux_clamped": _fraction_leq(recruitment_flux, eps_t), "boundary_residual_abs_p95": _abs_quantile(boundary_residual, 0.95), "boundary_residual_abs_max": torch.max(torch.abs(boundary_residual.detach()))}


def compute_pde_loss(model, batch: dict[str, torch.Tensor], params: MizerTorchParams, n_pp: torch.Tensor, residual_form: str = "log", *, n_init: torch.Tensor | None = None, lambda_pde: float = 1.0, lambda_ic: float = 0.0, lambda_bc: float = 0.0, boundary_loss_form: str = "log", species_idx: int | None = None, eps: float = 1e-30, bc_eps: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    include_ic = lambda_ic != 0.0
    state = compute_pde_state(model=model, batch=batch, params=params, n_pp=n_pp, include_ic=include_ic)
    residual_out = compute_pde_residual_from_state(state)
    if residual_form == "log": residual = residual_out["residual_log"]
    elif residual_form == "physical": residual = residual_out["residual"]
    else: raise ValueError("residual_form must be either 'log' or 'physical'.")
    loss_pde = (residual ** 2).mean()
    dtype, device = _params_dtype_device(params)
    zero = torch.zeros((), dtype=dtype, device=device)
    if lambda_ic != 0.0:
        if n_init is None: raise ValueError("lambda_ic != 0 requires n_init.")
        ic_out = compute_initial_condition_loss_from_state(state=state, params=params, n_init=n_init, species_idx=species_idx, eps=eps)
        loss_ic = ic_out["loss_ic"]
    else:
        ic_out, loss_ic = {}, zero
    bc_eps_value = eps if bc_eps is None else bc_eps
    if lambda_bc != 0.0:
        bc_out = compute_recruitment_boundary_loss_from_state(state=state, params=params, species_idx=species_idx, loss_form=boundary_loss_form, eps=bc_eps_value)
        loss_bc = bc_out["loss_bc"]
    else:
        bc_out, loss_bc = {}, zero
    loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
    out = {**residual_out, **ic_out, **bc_out, "loss": loss, "loss_pde": loss_pde, "loss_ic": loss_ic, "loss_bc": loss_bc}
    return loss, out
