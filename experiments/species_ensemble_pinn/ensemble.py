from __future__ import annotations

import torch
import torch.nn as nn
from PINNmizer.params import scale_x, scale_t
from .state_scale import reconstruct_scalar_state

class SpeciesPINNEnsemble(nn.Module):
    """Independent scalar models. forward returns concatenated raw log_U_all [P,S]."""
    def __init__(self, models: list[nn.Module]):
        super().__init__(); self.models = nn.ModuleList(models)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(inputs) for m in self.models], dim=1)


def evaluate_ensemble(ensemble: SpeciesPINNEnsemble, t: torch.Tensor, w: torch.Tensor, params) -> dict[str, torch.Tensor]:
    x_s=scale_x(torch.log(w), params); t_s=scale_t(t, params)
    xx=x_s[None,:].expand(t.numel(), w.numel()); tt=t_s[:,None].expand(t.numel(),w.numel())
    raw=ensemble(torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)).reshape(t.numel(), w.numel(), len(ensemble.models)).permute(0,2,1).contiguous()
    outs=[reconstruct_scalar_state(raw[:,i:i+1,:], params, species_idx=i, w=w, grid=torch.equal(w, params.w)) for i in range(len(ensemble.models))]
    return {k: torch.cat([o[k] for o in outs], dim=1) for k in ["log_U","U","log_S","S","log_N","N"]}
