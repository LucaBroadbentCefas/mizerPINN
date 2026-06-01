from __future__ import annotations

import torch

from PINNmizer.params import (
    MizerTorchParams,
    _n_species,
    _n_w,
    _params_dtype_device,
    active_grid_mask,
    active_eval_mask,
)
from PINNmizer.pinn.pde_state import (
    compute_pde_state,
    compute_pde_state_paired,
    compute_pde_state_r3_slabbed,
)
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

def _masked_square_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype, device=x.device)
    mask = mask.expand_as(x)

    denom = mask.sum()
    if not bool((denom > 0).detach().cpu()):
        raise ValueError("Masked loss has zero active entries.")

    return ((x ** 2) * mask).sum() / denom

def _active_eval_mask_for_slab(
    w_slab: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    if w_slab.ndim != 2:
        raise ValueError(f"w_slab must be [K, M], got {tuple(w_slab.shape)}.")

    k, m = w_slab.shape
    flat_mask = active_eval_mask(w_slab.reshape(-1), params)
    n_species = flat_mask.shape[0]

    return flat_mask.reshape(n_species, k, m).permute(1, 0, 2).contiguous()

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
    ic_mask = active_grid_mask(params).to(dtype=log_N_pred.dtype, device=log_N_pred.device)

    if species_idx is not None:
       ic_mask = ic_mask[species_idx: species_idx + 1]

    loss_ic = _masked_square_mean(log_N_pred - log_N_target, ic_mask)
    return {"loss_ic": loss_ic, "log_N_ic_pred": log_N_pred, "N_ic_pred": N_pred, "log_N_ic_target": log_N_target, "N_ic_target": n_init}


def compute_recruitment_boundary_loss_from_state(state: dict[str, object], params: MizerTorchParams, *, species_idx: int | None = None, loss_form: str = "log", eps: float = 1e-30) -> dict[str, torch.Tensor]:
    N_grid, growth_grid, recruitment = state["N_grid"], state["growth_grid"], state["recruitment"]
    log_N_grid = state.get("log_N_grid", None)
    egg_idx = params.w_min_idx.to(torch.long) - 1
    N_left = _species_grid_values_at_indices(N_grid, egg_idx)
    g_left = _species_grid_values_at_indices(growth_grid["e_growth_eval"], egg_idx)
    
    erepog_left = _species_grid_values_at_indices(growth_grid["erepog_eval"], egg_idx)
    pos_erepog_left = _species_grid_values_at_indices(growth_grid["pos_erepog"], egg_idx)
    e_repro_left = _species_grid_values_at_indices(growth_grid["e_repro_eval"], egg_idx)
    psi_left = _species_grid_values_at_indices(growth_grid["psi_eval"], egg_idx)
    encounter_left = _species_grid_values_at_indices(growth_grid["encounter_eval"], egg_idx)
    feeding_left = _species_grid_values_at_indices(growth_grid["feeding_eval"], egg_idx)
    metab_left = _species_grid_values_at_indices(growth_grid["metab_eval"], egg_idx)
    h_left = _species_grid_values_at_indices(growth_grid["h_eval"], egg_idx)
    
    log_N_left = None
    if log_N_grid is not None:
        log_N_left = _species_grid_values_at_indices(log_N_grid, egg_idx)
    flux_left = g_left * N_left
    recruitment_flux = recruitment["rdd_flux"]
    if species_idx is not None:
        flux_left = flux_left[:, species_idx: species_idx + 1]
        recruitment_flux = recruitment_flux[:, species_idx: species_idx + 1]
        g_left = g_left[:, species_idx: species_idx + 1]
        N_left = N_left[:, species_idx: species_idx + 1]
        
        erepog_left = _species_grid_values_at_indices(growth_grid["erepog_eval"], egg_idx)
        pos_erepog_left = _species_grid_values_at_indices(growth_grid["pos_erepog"], egg_idx)
        e_repro_left = _species_grid_values_at_indices(growth_grid["e_repro_eval"], egg_idx)
        psi_left = _species_grid_values_at_indices(growth_grid["psi_eval"], egg_idx)
        encounter_left = _species_grid_values_at_indices(growth_grid["encounter_eval"], egg_idx)
        feeding_left = _species_grid_values_at_indices(growth_grid["feeding_eval"], egg_idx)
        metab_left = _species_grid_values_at_indices(growth_grid["metab_eval"], egg_idx)
        h_left = _species_grid_values_at_indices(growth_grid["h_eval"], egg_idx)
        
        if log_N_left is not None:
            log_N_left = log_N_left[:, species_idx: species_idx + 1]
    if loss_form == "physical": boundary_residual = flux_left - recruitment_flux
    elif loss_form == "log":
        eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
        boundary_residual = torch.log(torch.clamp(flux_left, min=eps_t)) - torch.log(torch.clamp(recruitment_flux, min=eps_t))
    elif loss_form == "relative": boundary_residual = (flux_left - recruitment_flux) / torch.clamp(torch.abs(recruitment_flux), min=eps)
    else: raise ValueError("loss_form must be 'physical', 'log', or 'relative'.")
    loss_bc = (boundary_residual ** 2).mean()
    eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
    return {
        "loss_bc": loss_bc,
        "boundary_residual": boundary_residual,
        "flux_left": flux_left,
        "recruitment_flux": recruitment_flux,
        "g_left": g_left,
        "N_left": N_left,
        "log_N_left": (
            log_N_left
            if log_N_left is not None
            else torch.log(torch.clamp(N_left, min=eps_t))
        ),
        "bc_eps": eps_t,
        "flux_left_min": _scalar_tensor_min(flux_left),
        "flux_left_max": _scalar_tensor_max(flux_left),
        "g_left_min": _scalar_tensor_min(g_left),
        "g_left_max": _scalar_tensor_max(g_left),
        "N_left_min": _scalar_tensor_min(N_left),
        "N_left_max": _scalar_tensor_max(N_left),
        "log_N_left_min": _scalar_tensor_min(
            log_N_left if log_N_left is not None else torch.log(torch.clamp(N_left, min=eps_t))
        ),
        "log_N_left_max": _scalar_tensor_max(
            log_N_left if log_N_left is not None else torch.log(torch.clamp(N_left, min=eps_t))
        ),
        "recruitment_flux_min": _scalar_tensor_min(recruitment_flux),
        "recruitment_flux_max": _scalar_tensor_max(recruitment_flux),
        "frac_flux_left_clamped": _fraction_leq(flux_left, eps_t),
        "frac_g_left_clamped": _fraction_leq(g_left, eps_t),
        "frac_N_left_clamped": _fraction_leq(N_left, eps_t),
        "frac_recruitment_flux_clamped": _fraction_leq(recruitment_flux, eps_t),
        "boundary_residual_abs_p95": _abs_quantile(boundary_residual, 0.95),
        "boundary_residual_abs_max": torch.max(torch.abs(boundary_residual.detach())),
        "erepog_left": erepog_left,
        "pos_erepog_left": pos_erepog_left,
        "e_repro_left": e_repro_left,
        "psi_left": psi_left,
        "encounter_left": encounter_left,
        "feeding_left": feeding_left,
        "metab_left": metab_left,
        "h_left": h_left,
        
        "erepog_left_min": _scalar_tensor_min(erepog_left),
        "erepog_left_max": _scalar_tensor_max(erepog_left),
        "pos_erepog_left_min": _scalar_tensor_min(pos_erepog_left),
        "pos_erepog_left_max": _scalar_tensor_max(pos_erepog_left),
        "e_repro_left_min": _scalar_tensor_min(e_repro_left),
        "e_repro_left_max": _scalar_tensor_max(e_repro_left),
        "psi_left_min": _scalar_tensor_min(psi_left),
        "psi_left_max": _scalar_tensor_max(psi_left),
        "encounter_left_min": _scalar_tensor_min(encounter_left),
        "encounter_left_max": _scalar_tensor_max(encounter_left),
        "feeding_left_min": _scalar_tensor_min(feeding_left),
        "feeding_left_max": _scalar_tensor_max(feeding_left),
        "metab_left_min": _scalar_tensor_min(metab_left),
        "metab_left_max": _scalar_tensor_max(metab_left),
        "h_left_min": _scalar_tensor_min(h_left),
        "h_left_max": _scalar_tensor_max(h_left),
    }


def compute_pde_loss(model, batch: dict[str, torch.Tensor], params: MizerTorchParams, n_pp: torch.Tensor, residual_form: str = "log", *, n_init: torch.Tensor | None = None, lambda_pde: float = 1.0, lambda_ic: float = 0.0, lambda_bc: float = 0.0, boundary_loss_form: str = "log", species_idx: int | None = None, eps: float = 1e-30, bc_eps: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    include_ic = lambda_ic != 0.0
    state = compute_pde_state(model=model, batch=batch, params=params, n_pp=n_pp, include_ic=include_ic)
    residual_out = compute_pde_residual_from_state(state)
    if residual_form == "log": residual = residual_out["residual_log"]
    elif residual_form == "physical": residual = residual_out["residual"]
    else: raise ValueError("residual_form must be either 'log' or 'physical'.")
    pde_mask = active_eval_mask(batch["w_eval"], params)[None, :, :]
    loss_pde = _masked_square_mean(residual, pde_mask)
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

def compute_pde_loss_paired(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    residual_form: str = "log",
    *,
    n_init: torch.Tensor | None = None,
    lambda_pde: float = 1.0,
    lambda_ic: float = 0.0,
    lambda_bc: float = 0.0,
    boundary_loss_form: str = "log",
    species_idx: int | None = None,
    eps: float = 1e-30,
    bc_eps: float | None = None,
    pde_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    include_ic = lambda_ic != 0.0

    state = compute_pde_state_paired(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=include_ic,
    )

    residual_out = compute_pde_residual_from_state(state)

    if residual_form == "log":
        residual = residual_out["residual_log"]
    elif residual_form == "physical":
        residual = residual_out["residual"]
    else:
        raise ValueError("residual_form must be 'log' or 'physical'.")

    mask = active_eval_mask(batch["w_pair"], params).to(
        dtype=residual.dtype,
        device=residual.device,
    )

    if pde_weights is None:
        weighted_mask = mask
    else:
        weighted_mask = mask * pde_weights.to(
            dtype=residual.dtype,
            device=residual.device,
        )[None, :]

    denom = mask.sum()
    if not bool((denom > 0).detach().cpu()):
        raise ValueError("Paired PDE loss has zero active entries.")

    loss_pde = ((residual ** 2) * weighted_mask).sum() / denom

    dtype, device = _params_dtype_device(params)
    zero = torch.zeros((), dtype=dtype, device=device)

    if lambda_ic != 0.0:
        if n_init is None:
            raise ValueError("lambda_ic != 0 requires n_init.")
        ic_out = compute_initial_condition_loss_from_state(
            state=state,
            params=params,
            n_init=n_init,
            species_idx=species_idx,
            eps=eps,
        )
        loss_ic = ic_out["loss_ic"]
    else:
        ic_out = {}
        loss_ic = zero

    if lambda_bc != 0.0:
        bc_out = compute_recruitment_boundary_loss_from_state(
            state=state,
            params=params,
            species_idx=species_idx,
            loss_form=boundary_loss_form,
            eps=eps if bc_eps is None else bc_eps,
        )
        loss_bc = bc_out["loss_bc"]
    else:
        bc_out = {}
        loss_bc = zero

    loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

    out = {
        **residual_out,
        **ic_out,
        **bc_out,
        "loss": loss,
        "loss_pde": loss_pde,
        "loss_ic": loss_ic,
        "loss_bc": loss_bc,
    }

    return loss, out

def compute_pde_loss_r3_slabbed(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    residual_form: str = "log",
    *,
    n_init: torch.Tensor | None = None,
    lambda_pde: float = 1.0,
    lambda_ic: float = 0.0,
    lambda_bc: float = 0.0,
    boundary_loss_form: str = "log",
    species_idx: int | None = None,
    eps: float = 1e-30,
    bc_eps: float | None = None,
    pde_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    include_ic = lambda_ic != 0.0

    state = compute_pde_state_r3_slabbed(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=include_ic,
    )

    residual_out = compute_pde_residual_from_state(state)

    if residual_form == "log":
        residual = residual_out["residual_log"]
    elif residual_form == "physical":
        residual = residual_out["residual"]
    else:
        raise ValueError("residual_form must be 'log' or 'physical'.")

    if residual.ndim != 3:
        raise ValueError(
            "Slabbed R3 residual must be [K, n_species, M], "
            f"got {tuple(residual.shape)}."
        )

    w_slab = batch["w_slab"]
    mask = _active_eval_mask_for_slab(w_slab, params).to(
        dtype=residual.dtype,
        device=residual.device,
    )

    if mask.shape != residual.shape:
        raise ValueError(
            f"Slabbed R3 mask shape {tuple(mask.shape)} does not match "
            f"residual shape {tuple(residual.shape)}."
        )

    denom = mask.sum()
    if not bool((denom > 0).detach().cpu()):
        raise ValueError("Slabbed R3 PDE loss has zero active entries.")
    
    loss_pde_ungated = ((residual ** 2) * mask).sum() / denom
    
    if pde_weights is None:
        loss_pde = loss_pde_ungated
        pde_gate_mean = torch.ones((), dtype=residual.dtype, device=residual.device)
        pde_gate_min = torch.ones((), dtype=residual.dtype, device=residual.device)
        pde_gate_max = torch.ones((), dtype=residual.dtype, device=residual.device)
    else:
        pde_weights = pde_weights.to(dtype=residual.dtype, device=residual.device)
    
        if pde_weights.ndim != 1 or pde_weights.numel() != residual.shape[0]:
            raise ValueError(
                "pde_weights must be [K] for slabbed R3. "
                f"Got {tuple(pde_weights.shape)}, expected ({residual.shape[0]},)."
            )
    
        weighted_mask = mask * pde_weights[:, None, None]
        loss_pde = ((residual ** 2) * weighted_mask).sum() / denom
    
        pde_gate_mean = pde_weights.detach().mean()
        pde_gate_min = pde_weights.detach().min()
        pde_gate_max = pde_weights.detach().max()

    dtype, device = _params_dtype_device(params)
    zero = torch.zeros((), dtype=dtype, device=device)

    if lambda_ic != 0.0:
        if n_init is None:
            raise ValueError("lambda_ic != 0 requires n_init.")
        ic_out = compute_initial_condition_loss_from_state(
            state=state,
            params=params,
            n_init=n_init,
            species_idx=species_idx,
            eps=eps,
        )
        loss_ic = ic_out["loss_ic"]
    else:
        ic_out = {}
        loss_ic = zero

    if lambda_bc != 0.0:
        bc_out = compute_recruitment_boundary_loss_from_state(
            state=state,
            params=params,
            species_idx=species_idx,
            loss_form=boundary_loss_form,
            eps=eps if bc_eps is None else bc_eps,
        )
        loss_bc = bc_out["loss_bc"]
    else:
        bc_out = {}
        loss_bc = zero

    loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

    k, m = w_slab.shape

    out = {
        **residual_out,
        **ic_out,
        **bc_out,
        "loss": loss,
        "loss_pde": loss_pde,
        "loss_ic": loss_ic,
        "loss_bc": loss_bc,
        "r3_n_time": torch.as_tensor(float(k), dtype=dtype, device=device),
        "r3_n_eval_per_time": torch.as_tensor(float(m), dtype=dtype, device=device),
        "r3_population_size": torch.as_tensor(float(k * m), dtype=dtype, device=device),
        "r3_biology_time_loops": torch.as_tensor(float(k), dtype=dtype, device=device),
        "loss_pde_ungated": loss_pde_ungated,
        "loss_pde_gated": loss_pde,
        "pde_gate_mean": pde_gate_mean,
        "pde_gate_min": pde_gate_min,
        "pde_gate_max": pde_gate_max,
    }

    return loss, out
