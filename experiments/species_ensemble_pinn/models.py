from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, *, rwf_mu:float=1.0, rwf_sigma:float=0.1):
        super().__init__(); base=torch.empty(out_features,in_features); nn.init.xavier_uniform_(base)
        self.log_scale=nn.Parameter(torch.empty(out_features)); nn.init.normal_(self.log_scale, mean=rwf_mu, std=rwf_sigma)
        self.weight_v=nn.Parameter(base/torch.exp(self.log_scale.detach()).unsqueeze(1)); self.bias=nn.Parameter(torch.empty(out_features)); nn.init.uniform_(self.bias,-1/math.sqrt(in_features),1/math.sqrt(in_features))
    def forward(self,x): return F.linear(x, torch.exp(self.log_scale).unsqueeze(1)*self.weight_v, self.bias)

class ScalarSpeciesPINN(nn.Module):
    """Fourier/RWF scalar network. Raw forward output is log_U [P,1]."""
    def __init__(self, *, hidden_width:int=384, hidden_layers:int=5, fourier_num_features:int=16, fourier_scale:float=1.0, fourier_include_raw_input:bool=True, fourier_seed:int=123, rwf_mu:float=1.0, rwf_sigma:float=0.1):
        super().__init__(); self.state_parameterization='log-u'; self.fourier_include_raw_input=fourier_include_raw_input
        gen=torch.Generator(device='cpu'); gen.manual_seed(fourier_seed)
        B=torch.randn((2,fourier_num_features), generator=gen)*fourier_scale; self.register_buffer('B',B)
        in_dim=2*fourier_num_features+(2 if fourier_include_raw_input else 0)
        layers=[]; last=in_dim
        for _ in range(hidden_layers): layers += [FactorizedLinear(last,hidden_width,rwf_mu=rwf_mu,rwf_sigma=rwf_sigma), nn.Tanh()]; last=hidden_width
        layers.append(FactorizedLinear(last,1,rwf_mu=rwf_mu,rwf_sigma=rwf_sigma)); self.net=nn.Sequential(*layers)
    def features(self,x):
        proj=2*math.pi*x@self.B.to(dtype=x.dtype, device=x.device); feats=[torch.sin(proj), torch.cos(proj)]
        if self.fourier_include_raw_input: feats.append(x)
        return torch.cat(feats, dim=1)
    def forward(self,x): return self.net(self.features(x))

def make_scalar_species_model(**kwargs) -> nn.Module:
    cfg=dict(hidden_width=384, hidden_layers=5, fourier_num_features=16, fourier_scale=1.0, fourier_include_raw_input=True, fourier_seed=123, rwf_mu=1.0, rwf_sigma=0.1)
    cfg.update(kwargs); return ScalarSpeciesPINN(**cfg)

def initialise_log_u_bias(model: nn.Module, log_u_target: torch.Tensor) -> float:
    value=torch.as_tensor(log_u_target).mean()
    if not torch.isfinite(value): raise ValueError('Initial log_U bias target is not finite.')
    final=None
    for module in model.modules():
        if hasattr(module,'bias') and isinstance(module.bias, nn.Parameter) and module.bias is not None and module.bias.numel()==1: final=module
    if final is None: raise ValueError('Could not locate final scalar bias for log_U initialisation.')
    with torch.no_grad(): final.bias.fill_(value)
    return float(value.detach().cpu())
