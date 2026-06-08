from __future__ import annotations

import math

import torch
import torch.nn as nn


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
    ):
        super().__init__()

        layers = []
        last_dim = in_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.Tanh())
            last_dim = hidden_width

        layers.append(nn.Linear(last_dim, out_dim))

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
) -> nn.Module:
    if model_arch == "mlp":
        return MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
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
        )

    raise ValueError(f"Unknown model_arch: {model_arch}")
