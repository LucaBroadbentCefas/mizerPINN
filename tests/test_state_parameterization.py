import torch
import torch.nn as nn

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import scale_x, scale_t
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.state_scale import (
    interpolate_log_state_scale,
    set_state_scale_from_initial_condition,
)


class ConstantModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, x):
        return self.values.expand(x.shape[0], -1)


def test_log_u_initial_condition_scale_is_one_on_active_grid():
    params, n_init, _ = load_mizer_inputs("validation/fixtures/pde_multispecies", dtype=torch.float64, device="cpu")
    params.state_parameterization = "log-u"
    set_state_scale_from_initial_condition(params, n_init, eps=1e-30)
    log_s = params.state_scale_log
    log_n = torch.log(torch.clamp(n_init, min=params.state_scale_eps))
    active = (params.w[None, :] <= params.w_max[:, None])
    log_u = log_n - log_s
    assert torch.allclose(torch.exp(log_u)[active], torch.ones_like(log_u[active]), atol=1e-12, rtol=1e-12)
    assert torch.allclose(log_u[active], torch.zeros_like(log_u[active]), atol=1e-12, rtol=1e-12)


def test_log_u_reconstructs_physical_state_multispecies():
    params, n_init, _ = load_mizer_inputs("validation/fixtures/pde_multispecies", dtype=torch.float64, device="cpu")
    params.state_parameterization = "log-u"
    set_state_scale_from_initial_condition(params, n_init, eps=1e-30)
    raw = torch.tensor([0.2, -0.3], dtype=torch.float64)
    model = ConstantModel(raw)
    t = torch.linspace(float(params.t_min), float(params.t_max), 2, dtype=torch.float64)
    out = evaluate_log_model_on_points(model, scale_x(torch.log(params.w), params), scale_t(t, params), params)
    assert torch.allclose(out["N"], out["S"] * out["U"])
    assert torch.allclose(out["log_N"], out["log_S"] + out["log_U"])


def test_log_state_scale_derivative_matches_finite_difference():
    params, n_init, _ = load_mizer_inputs("validation/fixtures/pde_single_species", dtype=torch.float64, device="cpu")
    params.state_parameterization = "log-u"
    set_state_scale_from_initial_condition(params, n_init, eps=1e-30)
    w0 = torch.sqrt(params.w[:-1] * params.w[1:])[:5]
    _, _, analytic = interpolate_log_state_scale(params, w0)
    h = w0 * 1e-5
    lp, _, _ = interpolate_log_state_scale(params, w0 + h)
    lm, _, _ = interpolate_log_state_scale(params, w0 - h)
    fd = (lp - lm) / (2 * h[None, :])
    assert torch.allclose(analytic, fd, atol=1e-7, rtol=1e-5)
