from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 rwf_mu: float = 1.0, rwf_sigma: float = 0.1,
                 rwf_base_init: str = "xavier_uniform") -> None:
        super().__init__()
        if rwf_base_init != "xavier_uniform":
            raise ValueError("Tranche 1 requires rwf_base_init='xavier_uniform'.")
        self.in_features = in_features
        self.out_features = out_features
        base_weight = torch.empty(out_features, in_features)
        nn.init.xavier_uniform_(base_weight)
        self.log_scale = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.log_scale, mean=rwf_mu, std=rwf_sigma)
        self.weight_v = nn.Parameter(base_weight / torch.exp(self.log_scale.detach())[:, None])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    @property
    def effective_weight(self) -> torch.Tensor:
        return torch.exp(self.log_scale)[:, None] * self.weight_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight, self.bias)


class _TanhMLP(nn.Module):
    def __init__(self, in_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(cfg.hidden_layers):
            layers.extend([
                FactorizedLinear(last, cfg.hidden_width, rwf_mu=cfg.rwf_mu,
                                 rwf_sigma=cfg.rwf_sigma, rwf_base_init=cfg.rwf_base_init),
                nn.Tanh(),
            ])
            last = cfg.hidden_width
        layers.append(FactorizedLinear(last, 1, rwf_mu=cfg.rwf_mu,
                                       rwf_sigma=cfg.rwf_sigma,
                                       rwf_base_init=cfg.rwf_base_init))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarFourierRWFModel(nn.Module):
    state_parameterization = "log-n"
    output_dim = 1

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = cfg or ModelConfig()
        if self.config.output_dim != 1 or self.config.input_dim != 2:
            raise ValueError("The scalar model requires input_dim=2 and output_dim=1.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.fourier_seed)
        projection = torch.randn(
            self.config.input_dim,
            self.config.fourier_num_features,
            generator=generator,
        ) * self.config.fourier_scale
        self.register_buffer("B", projection)
        transformed = 2 * self.config.fourier_num_features + self.config.input_dim
        self.net = _TanhMLP(transformed, self.config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError(f"Expected inputs [P,2], got {tuple(inputs.shape)}.")
        projected = inputs @ self.B
        features = [inputs, torch.sin(2 * math.pi * projected), torch.cos(2 * math.pi * projected)]
        output = self.net(torch.cat(features, dim=-1))
        if output.shape != (inputs.shape[0], 1):
            raise RuntimeError(f"Scalar model returned {tuple(output.shape)}.")
        return output


def build_scalar_model() -> ScalarFourierRWFModel:
    return ScalarFourierRWFModel(ModelConfig())


def final_linear_layer(model: nn.Module) -> FactorizedLinear:
    for module in reversed(list(model.modules())):
        if isinstance(module, FactorizedLinear):
            return module
    raise ValueError("Could not find final FactorizedLinear layer.")
