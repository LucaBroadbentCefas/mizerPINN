from __future__ import annotations

import torch
from PINNmizer.params import MizerTorchParams, _params_dtype_device, _t_limits, _x_limits
from .model_eval import make_model_inputs
from .state_scale import reconstruct_scalar_state


def evaluate_scalar_derivatives(model, x_scaled: torch.Tensor, t_scaled: torch.Tensor, w: torch.Tensor, params: MizerTorchParams, *, species_idx: int) -> dict[str, torch.Tensor]:
    dtype, device = _params_dtype_device(params)
    x_scaled=x_scaled.to(dtype=dtype, device=device); t_scaled=t_scaled.to(dtype=dtype, device=device); w=w.to(dtype=dtype, device=device)
    inputs = make_model_inputs(x_scaled, t_scaled).requires_grad_(True)
    raw = model(inputs)
    if raw.shape[1] != 1:
        raise ValueError("Scalar model derivative path expects raw log_U [P,1].")
    grad = torch.autograd.grad(raw[:,0].sum(), inputs, create_graph=True, retain_graph=True)[0]
    T, M = t_scaled.numel(), x_scaled.numel()
    log_U = raw.reshape(T, M, 1).permute(0,2,1).contiguous()
    dxs = grad[:,0].reshape(T,M,1).permute(0,2,1).contiguous()
    dts = grad[:,1].reshape(T,M,1).permute(0,2,1).contiguous()
    x_min,x_max=_x_limits(params); t_min,t_max=_t_limits(params)
    dlogU_dt = dts/(t_max-t_min)
    dlogU_dw = dxs/(x_max-x_min)/w.reshape(1,1,-1)
    rec = reconstruct_scalar_state(log_U, params, species_idx=species_idx, w=w, grid=False)
    dlogN_dt = dlogU_dt
    dlogN_dw = dlogU_dw + rec["dlogS_dw"]
    rec.update({"log_U_eval":rec["log_U"],"U_eval":rec["U"],"log_S_eval":rec["log_S"],"S_eval":rec["S"],"log_N_eval":rec["log_N"],"N_eval":rec["N"],"dlogU_dt":dlogU_dt,"dlogU_dw":dlogU_dw,"dU_dt":rec["U"]*dlogU_dt,"dU_dw":rec["U"]*dlogU_dw,"dlogN_dt":dlogN_dt,"dlogN_dw":dlogN_dw,"dN_dt":rec["N"]*dlogN_dt,"dN_dw":rec["N"]*dlogN_dw})
    return rec
