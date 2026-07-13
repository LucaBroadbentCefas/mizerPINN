from __future__ import annotations

import torch

from PINNmizer.params import MizerTorchParams, _n_species, _params_dtype_device, _t_limits, _x_limits
from PINNmizer.pinn.model_eval import _make_model_inputs
from PINNmizer.pinn.state_scale import reconstruct_from_model_output, state_parameterization


def evaluate_log_model_with_derivatives_at_eval(model, x_eval_scaled: torch.Tensor, t_scaled: torch.Tensor, w_eval: torch.Tensor, params: MizerTorchParams) -> dict[str, torch.Tensor]:
    """Autograd derivatives of NN output; converts scaled derivatives to physical coords.

    Returns tensors shaped [n_time, n_species, n_eval].
    """
    dtype, device = _params_dtype_device(params)
    x_eval_scaled = x_eval_scaled.to(dtype=dtype, device=device)
    t_scaled = t_scaled.to(dtype=dtype, device=device)
    w_eval = w_eval.to(dtype=dtype, device=device)

    inputs = _make_model_inputs(x_eval_scaled, t_scaled)
    inputs.requires_grad_(True)
    raw_flat = model(inputs)

    n_time = t_scaled.numel()
    n_eval = x_eval_scaled.numel()
    n_species = _n_species(params)

    draw_dx_scaled_rows, draw_dt_scaled_rows = [], []
    for i in range(n_species):
        grad_i = torch.autograd.grad(
            raw_flat[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        draw_dx_scaled_rows.append(grad_i[:, 0])
        draw_dt_scaled_rows.append(grad_i[:, 1])

    raw = raw_flat.reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()
    rec = reconstruct_from_model_output(raw, params, w=w_eval)
    log_N, N = rec["log_N"], rec["N"]

    draw_dx_scaled = torch.stack(draw_dx_scaled_rows, dim=1).reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()
    draw_dt_scaled = torch.stack(draw_dt_scaled_rows, dim=1).reshape(n_time, n_eval, n_species).permute(0, 2, 1).contiguous()

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)
    draw_dx = draw_dx_scaled / (x_max - x_min)
    draw_dt = draw_dt_scaled / (t_max - t_min)
    draw_dw = draw_dx / w_eval[None, None, :]
    dlogN_dt = draw_dt
    dlogN_dw = draw_dw + rec.get("dlogS_dw", torch.zeros_like(draw_dw))
    out = {"log_N_eval": log_N, "N_eval": N, "dlogN_dt": dlogN_dt, "dlogN_dw": dlogN_dw, "dN_dt": N * dlogN_dt, "dN_dw": N * dlogN_dw}
    if state_parameterization(params) == "log-u":
        out.update({"log_U_eval": rec["log_U"], "U_eval": rec["U"], "log_S_eval": rec["log_S"], "S_eval": rec["S"], "dlogU_dt": draw_dt, "dlogU_dw": draw_dw, "dU_dt": rec["U"] * draw_dt, "dU_dw": rec["U"] * draw_dw, "dlogS_dw": rec["dlogS_dw"]})
    return out

def evaluate_log_model_with_derivatives_at_pairs(
    model,
    x_scaled_pair: torch.Tensor,
    t_scaled_pair: torch.Tensor,
    w_pair: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """Paired off-grid derivatives. Returns [n_species, n_pair]."""
    dtype, device = _params_dtype_device(params)

    x_scaled_pair = x_scaled_pair.to(dtype=dtype, device=device)
    t_scaled_pair = t_scaled_pair.to(dtype=dtype, device=device)
    w_pair = w_pair.to(dtype=dtype, device=device)

    inputs = torch.stack([x_scaled_pair, t_scaled_pair], dim=1)
    inputs.requires_grad_(True)

    raw_pair = model(inputs)

    n_species = _n_species(params)

    dlogN_dx_scaled = []
    dlogN_dt_scaled = []

    for i in range(n_species):
        grad_i = torch.autograd.grad(
            raw_pair[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        dlogN_dx_scaled.append(grad_i[:, 0])
        dlogN_dt_scaled.append(grad_i[:, 1])

    raw = raw_pair.T.contiguous()
    rec = reconstruct_from_model_output(raw, params, w=w_pair)
    log_N, N = rec["log_N"], rec["N"]

    dlogN_dx_scaled = torch.stack(dlogN_dx_scaled, dim=0)
    dlogN_dt_scaled = torch.stack(dlogN_dt_scaled, dim=0)

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)

    dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
    dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
    dlogN_dw = dlogN_dx / w_pair[None, :]

    return {
        "log_N_eval": log_N,
        "N_eval": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw + rec.get("dlogS_dw", torch.zeros_like(dlogN_dw)),
        "dN_dt": N * dlogN_dt,
        "dN_dw": N * (dlogN_dw + rec.get("dlogS_dw", torch.zeros_like(dlogN_dw))),
        **({"log_U_eval": rec["log_U"], "U_eval": rec["U"], "log_S_eval": rec["log_S"], "S_eval": rec["S"], "dlogU_dt": dlogN_dt, "dlogU_dw": dlogN_dw, "dU_dt": rec["U"] * dlogN_dt, "dU_dw": rec["U"] * dlogN_dw, "dlogS_dw": rec["dlogS_dw"]} if state_parameterization(params) == "log-u" else {}),
    }

def evaluate_log_model_with_derivatives_at_slabs(
    model,
    x_slab_scaled: torch.Tensor,
    t_slab_scaled: torch.Tensor,
    w_slab: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """Autograd derivatives for slabbed R3 points.

    Inputs
    ------
    x_slab_scaled:
        [K, M]
    t_slab_scaled:
        [K]
    w_slab:
        [K, M]

    Returns
    -------
    All returned tensors are [K, n_species, M].
    """
    dtype, device = _params_dtype_device(params)

    x_slab_scaled = x_slab_scaled.to(dtype=dtype, device=device)
    t_slab_scaled = t_slab_scaled.to(dtype=dtype, device=device)
    w_slab = w_slab.to(dtype=dtype, device=device)

    if x_slab_scaled.ndim != 2:
        raise ValueError(f"x_slab_scaled must be [K, M], got {tuple(x_slab_scaled.shape)}.")
    if w_slab.shape != x_slab_scaled.shape:
        raise ValueError(
            f"w_slab shape {tuple(w_slab.shape)} must match "
            f"x_slab_scaled shape {tuple(x_slab_scaled.shape)}."
        )
    if t_slab_scaled.ndim != 1:
        raise ValueError(f"t_slab_scaled must be [K], got {tuple(t_slab_scaled.shape)}.")
    if t_slab_scaled.numel() != x_slab_scaled.shape[0]:
        raise ValueError(
            f"t_slab_scaled has length {t_slab_scaled.numel()}, "
            f"but x_slab_scaled has K={x_slab_scaled.shape[0]}."
        )

    k, m = x_slab_scaled.shape
    n_species = _n_species(params)

    t_expanded = t_slab_scaled[:, None].expand(k, m)

    inputs = torch.stack(
        [
            x_slab_scaled.reshape(-1),
            t_expanded.reshape(-1),
        ],
        dim=1,
    )
    inputs.requires_grad_(True)

    raw_flat = model(inputs)

    if raw_flat.shape != (k * m, n_species):
        raise ValueError(
            f"Model returned {tuple(raw_flat.shape)}, "
            f"expected {(k * m, n_species)}."
        )

    draw_dx_scaled_rows = []
    draw_dt_scaled_rows = []

    for i in range(n_species):
        grad_i = torch.autograd.grad(
            raw_flat[:, i].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        draw_dx_scaled_rows.append(grad_i[:, 0])
        draw_dt_scaled_rows.append(grad_i[:, 1])

    raw = (
        raw_flat
        .reshape(k, m, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )
    rec = reconstruct_from_model_output(raw, params, w=w_slab)
    log_N, N = rec["log_N"], rec["N"]

    dlogN_dx_scaled = (
        torch.stack(draw_dx_scaled_rows, dim=1)
        .reshape(k, m, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )
    dlogN_dt_scaled = (
        torch.stack(draw_dt_scaled_rows, dim=1)
        .reshape(k, m, n_species)
        .permute(0, 2, 1)
        .contiguous()
    )

    x_min, x_max = _x_limits(params)
    t_min, t_max = _t_limits(params)

    dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
    dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
    dlogU_dw_tmp = dlogN_dx / w_slab[:, None, :]
    dlogN_dw = dlogU_dw_tmp + rec.get("dlogS_dw", torch.zeros_like(dlogU_dw_tmp))

    dN_dt = N * dlogN_dt
    dN_dw = N * dlogN_dw

    return {
        "log_N_eval": log_N,
        "N_eval": N,
        "dlogN_dt": dlogN_dt,
        "dlogN_dw": dlogN_dw,
        "dN_dt": dN_dt,
        "dN_dw": dN_dw,
        **({"log_U_eval": rec["log_U"], "U_eval": rec["U"], "log_S_eval": rec["log_S"], "S_eval": rec["S"], "dlogU_dt": dlogN_dt, "dlogU_dw": dlogU_dw_tmp, "dU_dt": rec["U"] * dlogN_dt, "dU_dw": rec["U"] * dlogU_dw_tmp, "dlogS_dw": rec["dlogS_dw"]} if state_parameterization(params) == "log-u" else {}),
    }
