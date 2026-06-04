import torch

from PINNmizer.pinn.models import build_pinn_model


def assert_finite(name: str, x: torch.Tensor) -> None:
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def check_model(model_arch: str) -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    n_points = 7
    in_dim = 2
    out_dim = 1

    model = build_pinn_model(
        model_arch=model_arch,
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_width=8,
        hidden_layers=2,
        fourier_num_features=5,
        fourier_scale=1.25,
        fourier_include_raw_input=True,
        fourier_seed=123,
    ).to(dtype=dtype, device=device)

    z = torch.linspace(-1.0, 1.0, n_points * in_dim, dtype=dtype, device=device).reshape(n_points, in_dim)
    z.requires_grad_(True)

    y = model(z)
    assert y.shape == (n_points, out_dim), f"{model_arch} output shape was {tuple(y.shape)}"
    assert y.requires_grad, f"{model_arch} output is detached from autograd graph"
    assert_finite(f"{model_arch} output", y)

    dz = torch.autograd.grad(
        outputs=y.sum(),
        inputs=z,
        create_graph=True,
        retain_graph=True,
    )[0]

    assert dz.shape == z.shape, f"{model_arch} derivative shape was {tuple(dz.shape)}"
    assert dz.requires_grad, f"{model_arch} derivative is detached from autograd graph"
    assert_finite(f"{model_arch} derivative", dz)

    dx_scaled = dz[:, 0]
    dt_scaled = dz[:, 1]
    assert dx_scaled.shape == (n_points,)
    assert dt_scaled.shape == (n_points,)
    assert_finite(f"{model_arch} dlogN/dx_scaled", dx_scaled)
    assert_finite(f"{model_arch} dlogN/dt_scaled", dt_scaled)

    print(
        f"{model_arch}: output_shape={tuple(y.shape)} "
        f"derivative_shape={tuple(dz.shape)} finite=True"
    )


def main() -> None:
    check_model("mlp")
    check_model("fourier")
    print("Model architecture autograd validation passed.")


if __name__ == "__main__":
    main()
