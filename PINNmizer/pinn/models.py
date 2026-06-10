from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedLinear(nn.Module):
    """Linear layer with random weight factorisation (RWF).

    The effective weight follows PyTorch's Linear convention with shape
    [out_features, in_features] and is parameterised as
    exp(log_scale)[:, None] * weight_v.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
        rwf_base_init: str = "pytorch",
    ):
        super().__init__()

        if rwf_base_init not in {"pytorch", "xavier_uniform", "xavier_normal"}:
            raise ValueError(f"Unknown rwf_base_init: {rwf_base_init}")

        self.in_features = in_features
        self.out_features = out_features
        self.rwf_base_init = rwf_base_init

        base_weight = torch.empty(out_features, in_features)
        self._init_base_weight(base_weight, rwf_base_init)

        self.log_scale = nn.Parameter(torch.empty(out_features))
        nn.init.normal_(self.log_scale, mean=rwf_mu, std=rwf_sigma)

        scale = torch.exp(self.log_scale.detach()).unsqueeze(1)
        self.weight_v = nn.Parameter(base_weight / scale)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self._init_bias(self.bias, in_features)
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _init_base_weight(weight: torch.Tensor, rwf_base_init: str) -> None:
        if rwf_base_init == "pytorch":
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        elif rwf_base_init == "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        elif rwf_base_init == "xavier_normal":
            nn.init.xavier_normal_(weight)
        else:
            raise ValueError(f"Unknown rwf_base_init: {rwf_base_init}")

    @staticmethod
    def _init_bias(bias: torch.Tensor, in_features: int) -> None:
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    @property
    def effective_weight(self) -> torch.Tensor:
        return torch.exp(self.log_scale).unsqueeze(1) * self.weight_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rwf_base_init={self.rwf_base_init}"
        )


def make_linear_layer(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    weight_factorization: str = "none",
    rwf_mu: float = 1.0,
    rwf_sigma: float = 0.1,
    rwf_base_init: str = "pytorch",
) -> nn.Module:
    if weight_factorization == "none":
        return nn.Linear(in_features, out_features, bias=bias)
    if weight_factorization == "rwf":
        return FactorizedLinear(
            in_features,
            out_features,
            bias=bias,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            rwf_base_init=rwf_base_init,
        )
    raise ValueError(f"Unknown weight_factorization: {weight_factorization}")


class MLP(nn.Module):
    """
    Fully connected neural network used as the PINN state approximator.

    Mathematical role
    -----------------
    The network represents the unknown log abundance field

        log_N_i = f_theta(x_scaled, t_scaled)_i

    where i indexes species. The model does not receive physical weight `w`
    directly. It receives the scaled log-weight coordinate `x_scaled`, where
    x = log(w), and scaled time `t_scaled`.

    Tensor contract
    ---------------
    Input:
        x: [n_points, 2]
           column 0 = x_scaled
           column 1 = t_scaled

    Output:
        log_N: [n_points, n_species]

    Hidden layers
    -------------
    Each hidden layer applies a linear map followed by Tanh. For a width of 64,
    this means each hidden representation has 64 activations; it does not mean
    there are 64 separate layers. Tanh is smooth, which is useful because the
    PDE residual uses autograd derivatives of the network output.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_width: int = 64,
        hidden_layers: int = 3,
        weight_factorization: str = "none",
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
        rwf_apply_to: str = "all",
        rwf_base_init: str = "pytorch",
    ):
        super().__init__()

        if weight_factorization not in {"none", "rwf"}:
            raise ValueError(f"Unknown weight_factorization: {weight_factorization}")
        if rwf_apply_to not in {"hidden", "all"}:
            raise ValueError(f"Unknown rwf_apply_to: {rwf_apply_to}")

        layers = []
        last_dim = in_dim

        hidden_factorization = (
            weight_factorization if weight_factorization == "rwf" else "none"
        )
        output_factorization = (
            weight_factorization
            if (weight_factorization == "rwf" and rwf_apply_to == "all")
            else "none"
        )

        for _ in range(hidden_layers):
            layers.append(
                make_linear_layer(
                    last_dim,
                    hidden_width,
                    weight_factorization=hidden_factorization,
                    rwf_mu=rwf_mu,
                    rwf_sigma=rwf_sigma,
                    rwf_base_init=rwf_base_init,
                )
            )
            layers.append(nn.Tanh())
            last_dim = hidden_width

        layers.append(
            make_linear_layer(
                last_dim,
                out_dim,
                weight_factorization=output_factorization,
                rwf_mu=rwf_mu,
                rwf_sigma=rwf_sigma,
                rwf_base_init=rwf_base_init,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log abundance at scaled coordinate-time points.

        The caller is responsible for ensuring `x` uses the column order
        [x_scaled, t_scaled]. The output is log_N, not N. Conversion to N is
        done by the PDE/model-evaluation utilities using torch.exp(log_N).
        """
        return self.net(x)


class FourierFeatureMLP(nn.Module):
    """
    MLP with a fixed Fourier feature input transform.

    The transform preserves the existing scaled input convention
    [x_scaled, t_scaled] and maps z to [sin(2*pi*zB), cos(2*pi*zB)]. The
    projection matrix B is a buffer so device/dtype conversions follow the
    module and autograd remains connected to z.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_width: int = 64,
        hidden_layers: int = 3,
        num_features: int = 64,
        scale: float = 1.0,
        include_raw_input: bool = False,
        seed: int | None = None,
        weight_factorization: str = "none",
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
        rwf_apply_to: str = "all",
        rwf_base_init: str = "pytorch",
    ):
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")

        if seed is None:
            projection = torch.randn(in_dim, num_features) * scale
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            projection = torch.randn(in_dim, num_features, generator=generator) * scale

        self.register_buffer("B", projection)
        self.include_raw_input = include_raw_input

        transformed_dim = 2 * num_features
        if include_raw_input:
            transformed_dim += in_dim

        self.net = MLP(
            in_dim=transformed_dim,
            out_dim=out_dim,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            weight_factorization=weight_factorization,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            rwf_apply_to=rwf_apply_to,
            rwf_base_init=rwf_base_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x @ self.B
        features = [
            torch.sin(2.0 * math.pi * projected),
            torch.cos(2.0 * math.pi * projected),
        ]
        if self.include_raw_input:
            features.insert(0, x)
        return self.net(torch.cat(features, dim=-1))


def build_pinn_model(
    *,
    model_arch: str = "mlp",
    in_dim: int = 2,
    out_dim: int = 1,
    hidden_width: int = 64,
    hidden_layers: int = 3,
    fourier_num_features: int = 64,
    fourier_scale: float = 1.0,
    fourier_include_raw_input: bool = False,
    fourier_seed: int | None = None,
    weight_factorization: str = "none",
    rwf_mu: float = 1.0,
    rwf_sigma: float = 0.1,
    rwf_apply_to: str = "all",
    rwf_base_init: str = "pytorch",
) -> nn.Module:
    if model_arch == "mlp":
        return MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            weight_factorization=weight_factorization,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            rwf_apply_to=rwf_apply_to,
            rwf_base_init=rwf_base_init,
        )

    if model_arch == "fourier":
        return FourierFeatureMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            num_features=fourier_num_features,
            scale=fourier_scale,
            include_raw_input=fourier_include_raw_input,
            seed=fourier_seed,
            weight_factorization=weight_factorization,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
            rwf_apply_to=rwf_apply_to,
            rwf_base_init=rwf_base_init,
        )

    raise ValueError(f"Unknown model_arch: {model_arch}")
