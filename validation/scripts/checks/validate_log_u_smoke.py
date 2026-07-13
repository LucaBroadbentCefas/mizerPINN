from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask
from PINNmizer.pinn.losses import compute_pde_loss
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.sampling import sample_pde_batch
from PINNmizer.pinn.state_scale import (
    interpolate_log_state_scale,
    set_state_scale_from_initial_condition,
)
from PINNmizer.training.train_pde_only_single_species import (
    initialise_final_bias_from_ic,
)


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def assert_relative_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float = 1e-8,
) -> None:
    scale = torch.maximum(
        torch.maximum(actual.detach().abs(), expected.detach().abs()),
        torch.ones_like(actual),
    )
    error = ((actual.detach() - expected.detach()).abs() / scale).max()
    if float(error) > tolerance:
        raise AssertionError(
            f"{name} failed: maximum scaled error={float(error):.3e}, "
            f"tolerance={tolerance:.3e}."
        )


def main() -> None:
    torch.manual_seed(123)

    params, n_init, n_pp = load_mizer_inputs(
        "validation/fixtures/pde_single_species",
        dtype=torch.float64,
        device="cpu",
    )

    params.state_parameterization = "log-u"
    set_state_scale_from_initial_condition(params, n_init, eps=1e-30)

    n_species = int(params.interaction.shape[0])
    model = build_pinn_model(
        model_arch="mlp",
        in_dim=2,
        out_dim=n_species,
        hidden_width=16,
        hidden_layers=2,
    ).to(dtype=torch.float64, device=params.w.device)

    model.state_parameterization = "log-u"

    initialise_final_bias_from_ic(
        model=model,
        n_init=n_init,
        params=params,
        eps=1e-30,
        state_parameterization="log-u",
    )

    batch = sample_pde_batch(
        params=params,
        n_time=2,
        n_eval=8,
        time_sampling="uniform",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss, out = compute_pde_loss(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        residual_form="scaled",
        n_init=n_init,
        lambda_pde=1.0,
        lambda_ic=1.0,
        lambda_bc=0.0,
    )

    _, S_eval, _ = interpolate_log_state_scale(params, batch["w_eval"])
    S_eval = S_eval.unsqueeze(0).expand_as(out["N_eval"])
    U_eval = out["N_eval"] / S_eval

    assert_finite("initial loss", loss)
    assert_finite("U_eval", U_eval)
    assert_finite("N_eval", out["N_eval"])
    assert_finite("residual_log", out["residual_log"])
    assert_finite("residual_scaled", out["residual_scaled"])
    assert_finite("residual", out["residual"])

    active = active_grid_mask(params)
    u_ic_target = out["U_ic_target"]
    if not torch.allclose(
        u_ic_target[active],
        torch.ones_like(u_ic_target[active]),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError("The normalized IC target is not U(w,0)=1.")

    assert_relative_close(
        "physical reconstruction N = S * U",
        out["N_eval"],
        S_eval * U_eval,
    )
    assert_relative_close(
        "physical residual r_N = N * r_log",
        out["residual"],
        out["N_eval"] * out["residual_log"],
    )
    assert_relative_close(
        "scaled residual r_U = U * r_log",
        out["residual_scaled"],
        U_eval * out["residual_log"],
    )
    assert_relative_close(
        "residual conversion r_N = S * r_U",
        out["residual"],
        S_eval * out["residual_scaled"],
    )
    assert_relative_close(
        "physical residual assembly",
        out["residual"],
        out["residual_physical_check"],
        tolerance=1e-7,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError("No gradients reached the model.")

    for index, gradient in enumerate(gradients):
        assert_finite(f"gradient[{index}]", gradient)

    optimizer.step()

    loss_after, out_after = compute_pde_loss(
        model=model,
        batch=batch,
        params=params,
        n_pp=n_pp,
        residual_form="scaled",
        n_init=n_init,
        lambda_pde=1.0,
        lambda_ic=1.0,
        lambda_bc=0.0,
    )

    assert_finite("loss after one optimizer step", loss_after)
    assert_finite("N after one optimizer step", out_after["N_eval"])
    U_eval_after = out_after["N_eval"] / S_eval
    assert_finite("U after one optimizer step", U_eval_after)

    print("log-u smoke test passed")
    print(f"loss before step: {float(loss.detach()):.6e}")
    print(f"loss after step:  {float(loss_after.detach()):.6e}")
    print(
        f"U range: {float(U_eval_after.min()):.6e} to "
        f"{float(U_eval_after.max()):.6e}"
    )
    print(f"N range: {float(out_after['N_eval'].min()):.6e} to "
          f"{float(out_after['N_eval'].max()):.6e}")


if __name__ == "__main__":
    main()
