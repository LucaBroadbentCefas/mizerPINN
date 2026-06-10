import torch

from PINNmizer.pinn.models import FactorizedLinear, build_pinn_model


def assert_finite(name: str, x: torch.Tensor) -> None:
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def check_factorized_layers(
    model: torch.nn.Module, weight_factorization: str, case_name: str
) -> None:
    factorized_layers = [
        module for module in model.modules() if isinstance(module, FactorizedLinear)
    ]

    if weight_factorization == "none":
        assert (
            not factorized_layers
        ), f"{case_name} unexpectedly contains FactorizedLinear layers"
        return

    assert factorized_layers, f"{case_name} did not contain any FactorizedLinear layers"
    for layer in factorized_layers:
        assert layer.log_scale.shape == (
            layer.out_features,
        ), f"{case_name} log_scale shape was {tuple(layer.log_scale.shape)}"
        assert layer.weight_v.shape == (
            layer.out_features,
            layer.in_features,
        ), f"{case_name} weight_v shape was {tuple(layer.weight_v.shape)}"
        assert layer.effective_weight.shape == (
            layer.out_features,
            layer.in_features,
        ), f"{case_name} effective_weight shape was {tuple(layer.effective_weight.shape)}"


def check_model(model_arch: str, weight_factorization: str) -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    n_points = 7
    in_dim = 2
    out_dim = 1
    case_name = f"{model_arch}+{weight_factorization}"

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
        weight_factorization=weight_factorization,
    ).to(dtype=dtype, device=device)

    check_factorized_layers(model, weight_factorization, case_name)

    z = torch.linspace(
        -1.0, 1.0, n_points * in_dim, dtype=dtype, device=device
    ).reshape(n_points, in_dim)
    z.requires_grad_(True)

    y = model(z)
    assert y.shape == (
        n_points,
        out_dim,
    ), f"{case_name} output shape was {tuple(y.shape)}"
    assert y.requires_grad, f"{case_name} output is detached from autograd graph"
    assert_finite(f"{case_name} output", y)

    dz = torch.autograd.grad(
        outputs=y.sum(),
        inputs=z,
        create_graph=True,
        retain_graph=True,
    )[0]

    assert dz.shape == z.shape, f"{case_name} derivative shape was {tuple(dz.shape)}"
    assert dz.requires_grad, f"{case_name} derivative is detached from autograd graph"
    assert_finite(f"{case_name} derivative", dz)

    dx_scaled = dz[:, 0]
    dt_scaled = dz[:, 1]
    assert dx_scaled.shape == (n_points,)
    assert dt_scaled.shape == (n_points,)
    assert_finite(f"{case_name} dlogN/dx_scaled", dx_scaled)
    assert_finite(f"{case_name} dlogN/dt_scaled", dt_scaled)

    print(
        f"{case_name}: output_shape={tuple(y.shape)} "
        f"derivative_shape={tuple(dz.shape)} finite=True "
        f"factorized_layers={sum(isinstance(m, FactorizedLinear) for m in model.modules())}"
    )


def main() -> None:
    for model_arch in ("mlp", "fourier"):
        for weight_factorization in ("none", "rwf"):
            check_model(model_arch, weight_factorization)
    print("Model architecture autograd validation passed.")


if __name__ == "__main__":
    main()
