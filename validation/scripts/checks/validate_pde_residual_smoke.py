#python -m validation.scripts.smoketest_pde--*  - -
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import _params_dtype_device, _n_species
from PINNmizer.pinn import sample_pde_batch, compute_pde_loss

params, n_init, n_pp = load_mizer_inputs(
    "validation/fixtures/mizer_full",
    dtype=torch.float64,
    device="cpu",
)
class TinyLogPINN(torch.nn.Module):
    """
    Dummy model for testing only.
    Outputs log_N, not N.
    """
    def __init__(self, n_species: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, n_species),
        )

    def forward(self, x):
        return self.net(x)


def assert_finite(name: str, x: torch.Tensor) -> None:
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().amax().item())


dtype, device = _params_dtype_device(params)

n_pp = n_pp.to(dtype=dtype, device=device)

model = TinyLogPINN(_n_species(params)).to(dtype=dtype, device=device)

batch = sample_pde_batch(
    params=params,
    n_time=3,
    n_eval=20,
)

loss_pde, residual_out = compute_pde_loss(
    model=model,
    batch=batch,
    params=params,
    n_pp=n_pp,
    residual_form="log",
)

print("loss_pde:", float(loss_pde.detach()))

assert loss_pde.ndim == 0
assert_finite("loss_pde", loss_pde)

required_keys = [
    "residual",
    "residual_log",
    "residual_physical_check",
    "log_N_eval",
    "log_N_grid",
    "N_eval",
    "N_grid",
    "dlogN_dt",
    "dlogN_dw",
    "dN_dt",
    "dN_dw",
    "g_eval",
    "dg_dw",
    "mu_eval",
    "mu_b_eval",
    "pred_mort_eval",
]

for key in required_keys:
    assert key in residual_out, f"Missing residual_out key: {key}"
    assert_finite(key, residual_out[key])
    print(key, tuple(residual_out[key].shape))

n_time = batch["t_scaled"].numel()
n_eval = batch["w_eval"].numel()
n_species = _n_species(params)
n_w = params.w.numel()

assert residual_out["residual"].shape == (n_time, n_species, n_eval)
assert residual_out["residual_log"].shape == (n_time, n_species, n_eval)
assert residual_out["N_eval"].shape == (n_time, n_species, n_eval)
assert residual_out["N_grid"].shape == (n_time, n_species, n_w)

# Algebraic consistency checks for log(N) conversion
diff_residual = max_abs_diff(
    residual_out["residual"],
    residual_out["N_eval"] * residual_out["residual_log"],
)

diff_physical = max_abs_diff(
    residual_out["residual"],
    residual_out["residual_physical_check"],
)

print("max |residual - N * residual_log|:", diff_residual)
print("max |residual - residual_physical_check|:", diff_physical)

# These should usually be tiny. If values are enormous, inspect relative errors instead.
assert torch.allclose(
    residual_out["residual"],
    residual_out["N_eval"] * residual_out["residual_log"],
    rtol=1e-5,
    atol=1e-8,
)

assert torch.allclose(
    residual_out["residual"],
    residual_out["residual_physical_check"],
    rtol=1e-5,
    atol=1e-8,
)

loss_pde.backward()

grad_tensors = [
    p.grad
    for p in model.parameters()
    if p.grad is not None
]

assert len(grad_tensors) > 0, "No gradients reached model parameters."

for ii, grad in enumerate(grad_tensors):
    assert_finite(f"grad[{ii}]", grad)

total_grad_norm = torch.sqrt(
    sum((grad.detach() ** 2).sum() for grad in grad_tensors)
)

print("total_grad_norm:", float(total_grad_norm))
assert torch.isfinite(total_grad_norm), "Gradient norm is NaN or Inf"

print("PDE smoke test passed.")
