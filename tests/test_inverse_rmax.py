import math
import torch

from PINNmizer.inverse_parameters import BoundedLogRMax


def test_bounded_log_rmax_transform_gradients():
    initial = torch.tensor([math.exp(1.0), math.exp(2.0), math.exp(3.0)], dtype=torch.float64)
    inv = BoundedLogRMax(initial, lower=0.0, upper=50.0)
    r = inv.current_r_max(); log = inv.current_log_r_max()
    assert r.shape == (3,)
    assert torch.isfinite(r).all() and (r > 0).all()
    assert ((log >= 0.0) & (log <= 50.0)).all()
    assert torch.allclose(r, initial, rtol=1e-12, atol=1e-12)
    r.sum().backward()
    assert inv.raw_logit.grad is not None
    assert torch.isfinite(inv.raw_logit.grad).all()


def _target(loss_form):
    rdi = torch.tensor([[2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[0.5, 0.25]], dtype=torch.float64, requires_grad=True)
    rmax = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    n_left = torch.tensor([[3.0, 5.0]], dtype=torch.float64, requires_grad=True)
    rdd = rdi.detach() / (1 + rdi.detach() / rmax.reshape(1, -1))
    if loss_form == "log":
        residual = torch.log(n_left) - (torch.log(rdd) - torch.log(g.detach()))
    elif loss_form == "physical":
        residual = n_left - rdd / g.detach()
    else:
        residual = n_left * (g.detach() / rdd) - 1
    loss = (residual**2).mean()
    grads = torch.autograd.grad(loss, [rdi, g, rmax, n_left], allow_unused=True)
    return grads


def test_selective_target_gradient_log_physical_relative():
    for form in ["log", "physical", "relative"]:
        grdi, gg, grmax, gn = _target(form)
        assert grdi is None or torch.all(grdi == 0)
        assert gg is None or torch.all(gg == 0)
        assert grmax is not None and torch.isfinite(grmax).all() and torch.linalg.vector_norm(grmax) > 0
        assert gn is not None and torch.linalg.vector_norm(gn) > 0


def test_one_optimizer_step_changes_rmax():
    inv = BoundedLogRMax(torch.tensor([10.0, 20.0], dtype=torch.float64), lower=0.0, upper=50.0)
    dummy_network = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.Adam([{"params": [dummy_network], "lr": 1e-3, "name": "network"}, {"params": inv.parameters(), "lr": 1e-2, "name": "rmax"}])
    before = inv.current_r_max().detach().clone()
    rdi = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    g = torch.tensor([[0.5, 0.25]], dtype=torch.float64)
    n = torch.tensor([[3.0, 5.0]], dtype=torch.float64)
    rdd = rdi.detach() / (1 + rdi.detach() / inv.current_r_max().reshape(1, -1))
    loss = ((n * (g.detach() / rdd) - 1) ** 2).mean()
    opt.zero_grad(); loss.backward()
    assert inv.raw_logit.grad is not None and torch.isfinite(inv.raw_logit.grad).all()
    assert torch.linalg.vector_norm(inv.raw_logit.grad) > 0
    opt.step()
    after_log = inv.current_log_r_max().detach(); after = inv.current_r_max().detach()
    assert not torch.allclose(before, after)
    assert ((after_log >= 0.0) & (after_log <= 50.0)).all()
