from __future__ import annotations

import torch
from PINNmizer.params import scale_x, scale_t


def make_fixed_pde_batch(params, *, n_time: int, n_eval: int) -> dict[str, torch.Tensor]:
    """Deterministic Cartesian diagnostic/calibration batch; no random sampling."""
    t = torch.linspace(float(params.t_min), float(params.t_max), n_time, dtype=params.w.dtype, device=params.w.device)
    x = torch.linspace(torch.log(params.w[0]), torch.log(params.w[-1]), n_eval, dtype=params.w.dtype, device=params.w.device)
    w = torch.exp(x)
    chunks = torch.arange(n_time, device=params.w.device) * 32 // max(n_time, 1)
    return {"t": t, "w": w, "t_scaled": scale_t(t, params), "x_scaled": scale_x(x, params), "time_chunk": chunks}
