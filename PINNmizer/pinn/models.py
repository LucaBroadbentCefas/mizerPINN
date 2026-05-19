from __future__ import annotations

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
