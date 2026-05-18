import torch

from .params import (
    MizerTorchParams,
    _params_dtype_device,
    _n_species,
    _n_w,
    _x_grid,
    _x_limits,
    _t_limits,
    scale_x,
    scale_t,
)

from .continuous_biology import (
    compute_growth_direct_at_eval,
    compute_total_mortality_direct_at_eval_from_growth_grid,
    compute_recruitment_direct_from_growth_grid,
)

##THIS WILL BE CHANGED!
def _as_species_w_matrix(
    x: torch.Tensor,
    *,
    n_species: int,
    n_w: int,
    name: str,
) -> torch.Tensor:
    if x.ndim == 1:
        if n_species != 1:
            raise ValueError(
                f"{name} is 1D but n_species={n_species}. "
                "Pass [n_species, n_w]."
            )
        if x.numel() != n_w:
            raise ValueError(f"{name} has length {x.numel()}, expected {n_w}.")
        return x.reshape(1, n_w)

    if x.ndim == 2:
        if x.shape != (n_species, n_w):
            raise ValueError(
                f"{name} has shape {tuple(x.shape)}, expected {(n_species, n_w)}."
            )
        return x

    raise ValueError(f"{name} must be [n_w] or [n_species, n_w].")

def _species_grid_values_at_indices(
    x: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    x:   [n_time, n_species, n_w]
    idx: [n_species]

    returns:
        [n_time, n_species]

    For species i, extracts x[:, i, idx[i]].
    """
    assert x.ndim == 3

    n_time, n_species, _ = x.shape

    idx = idx.to(device=x.device, dtype=torch.long)
    assert idx.shape == (n_species,)

    gather_idx = idx[None, :, None].expand(n_time, n_species, 1)

    return torch.gather(x, dim=2, index=gather_idx).squeeze(-1)

def _scalar_tensor_min(x: torch.Tensor) -> torch.Tensor:
    return torch.min(x.detach())


def _scalar_tensor_max(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x.detach())


def _fraction_leq(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return (x.detach() <= threshold).to(dtype=x.dtype).mean()


def _abs_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(torch.abs(x.detach()).reshape(-1), q)

def sample_pde_batch(
    params: MizerTorchParams,
    n_time: int,
    n_eval: int,
    *,
    t_max_current=None,
) -> dict[str, torch.Tensor]:
    dtype, device = _params_dtype_device(params)

    x_grid = _x_grid(params)
    x_min, x_max = x_grid[0], x_grid[-1]

    t_min, t_max = _t_limits(params)

    t_min_t = torch.as_tensor(t_min, dtype=dtype, device=device)
    t_max_t = torch.as_tensor(t_max, dtype=dtype, device=device)
    
    if t_max_current is None:
        t_upper = t_max_t
    else:
        t_upper = torch.as_tensor(t_max_current, dtype=dtype, device=device)
        t_upper = torch.maximum(t_upper, t_min_t)
        t_upper = torch.minimum(t_upper, t_max_t)
    
    if not bool((t_upper > t_min_t).detach().cpu()):
        raise ValueError(
            f"t_max_current must be greater than t_min. "
            f"Got t_min={float(t_min_t.detach().cpu())}, "
            f"t_max_current={float(t_upper.detach().cpu())}."
        )
    
    t_eval = t_min_t + (t_upper - t_min_t) * torch.rand(
        n_time,
        dtype=dtype,
        device=device,
    )
    x_eval = x_min + (x_max - x_min) * torch.rand(n_eval, dtype=dtype, device=device)
    w_eval = torch.exp(x_eval)

    return {
        "t_eval": t_eval,
        "t_scaled": scale_t(t_eval, params),
        "x_eval": x_eval,
        "x_eval_scaled": scale_x(x_eval, params),
        "w_eval": w_eval,
        "x_grid": x_grid,
        "x_grid_scaled": scale_x(x_grid, params),
        "w_grid": params.w,
    }
    
def _check_batch_vector(x: torch.Tensor, name: str) -> None:
    assert x.ndim == 1, f"{name} must be 1D, got shape {tuple(x.shape)}"

def _make_model_inputs(
    x_scaled: torch.Tensor,
    t_scaled: torch.Tensor,
) -> torch.Tensor:
    """
    Build model inputs with time-major ordering.

    x_scaled: [n_x]
    t_scaled: [n_time]

    returns inputs: [n_time * n_x, 2]
    columns: [x_scaled, t_scaled]
    """
    _check_batch_vector(x_scaled, "x_scaled")
    _check_batch_vector(t_scaled, "t_scaled")

    n_time = t_scaled.numel()
    n_x = x_scaled.numel()

    xx = x_scaled[None, :].expand(n_time, n_x)
    tt = t_scaled[:, None].expand(n_time, n_x)

    return torch.stack(
        [xx.reshape(-1), tt.reshape(-1)],
        dim=1,
    )


def evaluate_log_model_on_points(
    model,
    x_scaled: torch.Tensor,
    t_scaled: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """
    Model outputs log_N.

    returns:
        log_N: [n_time, n_species, n_x]
        N:     [n_time, n_species, n_x]
    """
    dtype, device = _params_dtype_device(params)

    x_scaled = x_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)

    inputs = _make_model_inputs(x_scaled, t_scaled)
    log_N_flat = model(inputs)

    n_time = t_scaled.numel()
    n_x = x_scaled.numel()
    n_species = _n_species(params)

    assert log_N_flat.shape == (n_time * n_x, n_species), (
        f"model output must be {(n_time * n_x, n_species)}, "
        f"got {tuple(log_N_flat.shape)}"
    )

    log_N = (
        log_N_flat
        .reshape(n_time, n_x, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    N = torch.exp(log_N)

    assert log_N.shape == (n_time, n_species, n_x)
    assert N.shape == log_N.shape

    return {
        "log_N": log_N,
        "N": N,
    }
    
def evaluate_log_model_with_derivatives_at_eval(
    model,
    x_eval_scaled: torch.Tensor,
    t_scaled: torch.Tensor,
    w_eval: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """
    Model outputs log_N.

    Computes:
        dlogN_dt
        dlogN_dw
        dN_dt = N * dlogN_dt
        dN_dw = N * dlogN_dw

    returns all tensors as:
        [n_time, n_species, n_eval]
    """
    dtype, device = _params_dtype_device(params)

    x_eval_scaled = x_eval_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    w_eval = w_eval.to(dtype=dtype, device=device)

    inputs = _make_model_inputs(x_eval_scaled, t_scaled)
    inputs.requires_grad_(True)

    log_N_flat = model(inputs)

    n_time = t_scaled.numel()
    n_eval = x_eval_scaled.numel()
    n_species = _n_species(params)

    assert log_N_flat.shape == (n_time * n_eval, n_species), (
        f"model output must be {(n_time * n_eval, n_species)}, "
        f"got {tuple(log_N_flat.shape)}"
    )

    dlogN_dx_scaled_rows = []
    dlogN_dt_scaled_rows = []

    for i in range(n_species):
        grad_i = torch.autograd.grad(
            log_N_flat[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        dlogN_dx_scaled_rows.append(grad_i[:, 0])
        dlogN_dt_scaled_rows.append(grad_i[:, 1])

    log_N = (
        log_N_flat
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    N = torch.exp(log_N)

    dlogN_dx_scaled = torch.stack(dlogN_dx_scaled_rows, dim=1)
    dlogN_dt_scaled = torch.stack(dlogN_dt_scaled_rows, dim=1)

    dlogN_dx_scaled = (
        dlogN_dx_scaled
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    dlogN_dt_scaled = (
        dlogN_dt_scaled
        .reshape(n_time, n_eval, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)

    dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
    dlogN_dt = dlogN_dt_scaled / (t_max - t_min)

    dlogN_dw = dlogN_dx / w_eval[None, None, :]

    dN_dt = N * dlogN_dt
    dN_dw = N * dlogN_dw

    expected = (n_time, n_species, n_eval)

    for name, value in {
        "log_N": log_N,
        "N": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,
        "dN_dt": dN_dt,
        "dN_dw": dN_dw,
    }.items():
        assert value.shape == expected, f"{name}: expected {expected}, got {value.shape}"

    return {
        "log_N_eval": log_N,
        "N_eval": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,
        "dN_dt": dN_dt,
        "dN_dw": dN_dw,
    }

def _stack_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = dicts[0].keys()
    return {
        key: torch.stack([d[key] for d in dicts], dim=0)
        for key in keys
    }
    
    
def compute_pde_residual(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = compute_pde_state(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        include_ic=False,
    )

    return compute_pde_residual_from_state(state)
  
def compute_initial_condition_loss_from_state(
    state: dict[str, object],
    params: MizerTorchParams,
    n_init: torch.Tensor,
    *,
    species_idx: int | None = None,
    eps: float = 1e-30,
) -> dict[str, torch.Tensor]:
    """
    Initial condition loss using cached t_min model output.
    """
    dtype, device = _params_dtype_device(params)

    if state["log_N_ic"] is None:
        raise ValueError(
            "State does not contain IC output. "
            "Call compute_pde_state(..., include_ic=True)."
        )

    n_species = _n_species(params)
    n_w = _n_w(params)

    n_init = torch.as_tensor(n_init, dtype=dtype, device=device)
    n_init = _as_species_w_matrix(
        n_init,
        n_species=n_species,
        n_w=n_w,
        name="n_init",
    )

    log_N_pred = state["log_N_ic"][0]
    N_pred = state["N_ic"][0]

    log_N_target = torch.log(torch.clamp(n_init, min=eps))

    if species_idx is not None:
        log_N_pred = log_N_pred[species_idx : species_idx + 1]
        N_pred = N_pred[species_idx : species_idx + 1]
        log_N_target = log_N_target[species_idx : species_idx + 1]
        n_init = n_init[species_idx : species_idx + 1]

    loss_ic = ((log_N_pred - log_N_target) ** 2).mean()

    return {
        "loss_ic": loss_ic,
        "log_N_ic_pred": log_N_pred,
        "N_ic_pred": N_pred,
        "log_N_ic_target": log_N_target,
        "N_ic_target": n_init,
    }
    
def compute_recruitment_boundary_loss_from_state(
    state: dict[str, object],
    params: MizerTorchParams,
    *,
    species_idx: int | None = None,
    loss_form: str = "log",
    eps: float = 1e-30,
) -> dict[str, torch.Tensor]:
    """
    Continuous recruitment boundary loss from cached state.

    Boundary condition:

        g_i(w_min_i, t) * N_i(w_min_i, t) = RDD_i(t)

    Uses cached:
        N_grid
        growth_grid["e_growth_eval"]
        recruitment["rdd_flux"]
    """
    N_grid = state["N_grid"]
    growth_grid = state["growth_grid"]
    recruitment = state["recruitment"]

    egg_idx = params.w_min_idx.to(torch.long) - 1

    N_left = _species_grid_values_at_indices(
        N_grid,
        egg_idx,
    )

    g_left = _species_grid_values_at_indices(
        growth_grid["e_growth_eval"],
        egg_idx,
    )

    flux_left = g_left * N_left

    recruitment_flux = recruitment["rdd_flux"]

    if species_idx is not None:
        flux_left = flux_left[:, species_idx : species_idx + 1]
        recruitment_flux = recruitment_flux[:, species_idx : species_idx + 1]
        g_left = g_left[:, species_idx : species_idx + 1]
        N_left = N_left[:, species_idx : species_idx + 1]

    if loss_form == "physical":
        boundary_residual = flux_left - recruitment_flux

    elif loss_form == "log":
        eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
    
        flux_left_clamped = torch.clamp(flux_left, min=eps_t)
        recruitment_flux_clamped = torch.clamp(recruitment_flux, min=eps_t)
    
        boundary_residual = (
            torch.log(flux_left_clamped)
            - torch.log(recruitment_flux_clamped)
        )

    elif loss_form == "relative":
        boundary_residual = (
            flux_left - recruitment_flux
        ) / torch.clamp(torch.abs(recruitment_flux), min=eps)

    else:
        raise ValueError("loss_form must be 'physical', 'log', or 'relative'.")

    loss_bc = (boundary_residual ** 2).mean()

    eps_t = torch.as_tensor(eps, dtype=flux_left.dtype, device=flux_left.device)
    
    bc_diagnostics = {
        "bc_eps": eps_t,
        "flux_left_min": _scalar_tensor_min(flux_left),
        "flux_left_max": _scalar_tensor_max(flux_left),
        "recruitment_flux_min": _scalar_tensor_min(recruitment_flux),
        "recruitment_flux_max": _scalar_tensor_max(recruitment_flux),
        "frac_flux_left_clamped": _fraction_leq(flux_left, eps_t),
        "frac_recruitment_flux_clamped": _fraction_leq(recruitment_flux, eps_t),
        "boundary_residual_abs_p95": _abs_quantile(boundary_residual, 0.95),
        "boundary_residual_abs_max": torch.max(torch.abs(boundary_residual.detach())),
    }

    return {
        "loss_bc": loss_bc,
        "boundary_residual": boundary_residual,
        "flux_left": flux_left,
        "recruitment_flux": recruitment_flux,
        "g_left": g_left,
        "N_left": N_left,
        **bc_diagnostics,
    }
  
def compute_pde_loss(
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    include_ic = lambda_ic != 0.0

    state = compute_pde_state(
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
        raise ValueError("residual_form must be either 'log' or 'physical'.")

    loss_pde = (residual ** 2).mean()

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

    bc_eps_value = eps if bc_eps is None else bc_eps

    if lambda_bc != 0.0:
        bc_out = compute_recruitment_boundary_loss_from_state(
            state=state,
            params=params,
            species_idx=species_idx,
            loss_form=boundary_loss_form,
            eps=bc_eps_value,
        )

        loss_bc = bc_out["loss_bc"]
    else:
        bc_out = {}
        loss_bc = zero

    loss = (
        lambda_pde * loss_pde
        + lambda_ic * loss_ic
        + lambda_bc * loss_bc
    )

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


def compute_pde_state(
    model,
    batch: dict[str, torch.Tensor],
    params: MizerTorchParams,
    n_pp: torch.Tensor,
    *,
    include_ic: bool = False,
) -> dict[str, object]:
    """
    Evaluate model and biology once per training step.

    This is the cache/state object used by all loss components.
    """
    dtype, device = _params_dtype_device(params)

    w_eval = batch["w_eval"]
    x_eval_scaled = batch["x_eval_scaled"]
    t_scaled = batch["t_scaled"]
    x_grid_scaled = batch["x_grid_scaled"]

    n_time = t_scaled.numel()

    eval_derivs = evaluate_log_model_with_derivatives_at_eval(
        model=model,
        x_eval_scaled=x_eval_scaled,
        t_scaled=t_scaled,
        w_eval=w_eval,
        params=params,
    )

    if include_ic:
        t_min, _ = _t_limits(params)

        if torch.is_tensor(t_min):
            t0 = t_min.reshape(1).to(dtype=dtype, device=device)
        else:
            t0 = torch.tensor([t_min], dtype=dtype, device=device)

        t0_scaled = scale_t(t0, params)

        t_grid_scaled = torch.cat([t_scaled, t0_scaled], dim=0)
    else:
        t0_scaled = None
        t_grid_scaled = t_scaled

    grid_eval_all = evaluate_log_model_on_points(
        model=model,
        x_scaled=x_grid_scaled,
        t_scaled=t_grid_scaled,
        params=params,
    )

    if include_ic:
        log_N_grid = grid_eval_all["log_N"][:n_time]
        N_grid = grid_eval_all["N"][:n_time]

        log_N_ic = grid_eval_all["log_N"][n_time:]
        N_ic = grid_eval_all["N"][n_time:]
    else:
        log_N_grid = grid_eval_all["log_N"]
        N_grid = grid_eval_all["N"]

        log_N_ic = None
        N_ic = None

    growth_eval_by_time = []
    growth_grid_by_time = []
    mortality_by_time = []
    recruitment_by_time = []

    for tt in range(n_time):
        N_t = N_grid[tt]

        growth_eval_t = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=w_eval,
            params=params,
        )

        growth_grid_t = compute_growth_direct_at_eval(
            n_pp=n_pp,
            n_grid=N_t,
            w_eval=params.w,
            params=params,
        )

        mortality_t = compute_total_mortality_direct_at_eval_from_growth_grid(
            N_pred_grid=N_t,
            w_eval=w_eval,
            params=params,
            growth_grid=growth_grid_t,
        )

        recruitment_t = compute_recruitment_direct_from_growth_grid(
            N_grid=N_t,
            params=params,
            growth_grid=growth_grid_t,
        )

        growth_eval_by_time.append(growth_eval_t)
        growth_grid_by_time.append(growth_grid_t)
        mortality_by_time.append(mortality_t)
        recruitment_by_time.append(recruitment_t)

    return {
        "batch": batch,
        "eval_derivs": eval_derivs,
        "log_N_grid": log_N_grid,
        "N_grid": N_grid,
        "log_N_ic": log_N_ic,
        "N_ic": N_ic,
        "growth_eval": _stack_dicts(growth_eval_by_time),
        "growth_grid": _stack_dicts(growth_grid_by_time),
        "mortality": _stack_dicts(mortality_by_time),
        "recruitment": _stack_dicts(recruitment_by_time),
    }


def compute_pde_residual_from_state(
    state: dict[str, object],
) -> dict[str, torch.Tensor]:
    eval_derivs = state["eval_derivs"]
    growth = state["growth_eval"]
    mortality = state["mortality"]

    log_N_eval = eval_derivs["log_N_eval"]
    N_eval = eval_derivs["N_eval"]

    dlogN_dt = eval_derivs["dlogN_dt"]
    dlogN_dw = eval_derivs["dlogN_dw"]

    dN_dt = eval_derivs["dN_dt"]
    dN_dw = eval_derivs["dN_dw"]

    g_eval = growth["e_growth_eval"]
    dg_dw = growth["dg_dw"]
    mu_eval = mortality["mu_eval"]

    residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
    residual = N_eval * residual_log

    residual_physical_check = (
        dN_dt
        + g_eval * dN_dw
        + (mu_eval + dg_dw) * N_eval
    )

    out = {
        "residual": residual,
        "residual_log": residual_log,
        "residual_physical_check": residual_physical_check,

        "log_N_eval": log_N_eval,
        "log_N_grid": state["log_N_grid"],

        "N_eval": N_eval,
        "N_grid": state["N_grid"],

        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,

        "dN_dt": dN_dt,
        "dN_dw": dN_dw,

        "g_eval": g_eval,
        "dg_dw": dg_dw,
        "mu_eval": mu_eval,
    }

    out.update({f"growth_eval_{k}": v for k, v in growth.items()})
    out.update({f"growth_grid_{k}": v for k, v in state["growth_grid"].items()})
    out.update({f"mortality_{k}": v for k, v in mortality.items()})
    out.update({f"recruitment_{k}": v for k, v in state["recruitment"].items()})

    return out






















