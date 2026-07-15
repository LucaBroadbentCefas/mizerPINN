import torch

from PINNmizer.params import MizerTorchParams
from PINNmizer.pinn.residual import compute_pde_residual_from_state
from PINNmizer.pinn.residual_scale import set_residual_scale_from_initial_condition, grid_residual_scale


def _params(dtype=torch.float64):
    w = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=dtype)
    ns, nw = 2, w.numel()
    zsw = torch.zeros((ns, nw), dtype=dtype)
    zs = torch.zeros(ns, dtype=dtype)
    return MizerTorchParams(
        w_full=w, w=w, dw_full=torch.ones_like(w), dw=torch.ones_like(w), w_min_idx=torch.ones(ns, dtype=torch.long),
        ft_pred_kernel_e=torch.zeros((ns, nw), dtype=torch.complex128), ft_pred_kernel_p=torch.zeros((ns, nw), dtype=torch.complex128), ft_mask=torch.ones((ns, nw), dtype=dtype),
        search_vol=zsw, intake_max=zsw, alpha=zs, metab=zsw, psi=zsw, mu_b=zsw,
        interaction_resource=zs, interaction=torch.eye(ns, dtype=dtype), erepro=zs, r_max=zs, rr_pp=w, cc_pp=w,
        w_max=torch.tensor([4.0, 8.0], dtype=dtype),
    )


def _state(params, log_n, residual_log, w_key, w_value):
    n = torch.exp(log_n)
    zeros = torch.zeros_like(log_n)
    return {
        "batch": {w_key: w_value},
        "eval_derivs": {"log_N_eval": log_n, "N_eval": n, "dlogN_dt": residual_log, "dlogN_dw": zeros, "dN_dt": n * residual_log, "dN_dw": zeros},
        "growth_eval": {"e_growth_eval": zeros, "dg_dw": zeros},
        "mortality": {"mu_eval": zeros},
        "log_N_grid": torch.zeros((1, 2, 4), dtype=log_n.dtype),
        "N_grid": torch.ones((1, 2, 4), dtype=log_n.dtype),
        "growth_grid": {"dummy": torch.zeros((1, 2, 4), dtype=log_n.dtype)},
        "recruitment": {"dummy": torch.zeros((1, 2), dtype=log_n.dtype)},
    }


def test_residual_scale_construction_floor_and_inactive_continuation():
    params = _params()
    n0 = torch.tensor([[10.0, 1e-20, 2.0, 999.0], [3.0, 4.0, 5.0, 6.0]], dtype=torch.float64)
    set_residual_scale_from_initial_condition(params, n0, floor_fraction=1e-3)
    _, s = grid_residual_scale(params)
    assert torch.allclose(s[0, 0], n0[0, 0])
    assert torch.all(s[0, :3] >= 1e-3 * torch.max(n0[0, :3]))
    assert torch.allclose(s[0, 3], s[0, 2])
    assert not torch.allclose(s[0, 3], torch.ones((), dtype=s.dtype))


def test_reference_scaled_identity_all_layouts():
    params = _params()
    n0 = torch.tensor([[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 17.0, 19.0]], dtype=torch.float64)
    set_residual_scale_from_initial_condition(params, n0)
    for log_n, rlog, key, w in [
        (torch.log(torch.arange(1, 25, dtype=torch.float64).reshape(3, 2, 4)), torch.randn(3, 2, 4, dtype=torch.float64), "w_eval", params.w),
        (torch.log(torch.arange(1, 9, dtype=torch.float64).reshape(2, 4)), torch.randn(2, 4, dtype=torch.float64), "w_pair", params.w),
        (torch.log(torch.arange(1, 17, dtype=torch.float64).reshape(2, 2, 4)), torch.randn(2, 2, 4, dtype=torch.float64), "w_slab", params.w.repeat(2, 1)),
    ]:
        out = compute_pde_residual_from_state(_state(params, log_n, rlog, key, w), params)
        expected = torch.exp(log_n - out["log_reference_scale_eval"]) * rlog
        assert out["reference_scale_eval"].shape == log_n.shape
        assert torch.allclose(out["residual_reference_scaled"], expected, rtol=1e-12, atol=1e-12)


def test_reference_scale_independence_and_unit_scaling():
    params = _params()
    log_n = torch.log(torch.arange(1, 9, dtype=torch.float64).reshape(2, 4))
    rlog = torch.randn(2, 4, dtype=torch.float64)
    set_residual_scale_from_initial_condition(params, torch.ones(2, 4, dtype=torch.float64))
    out1 = compute_pde_residual_from_state(_state(params, log_n, rlog, "w_pair", params.w), params)
    set_residual_scale_from_initial_condition(params, 10 * torch.ones(2, 4, dtype=torch.float64))
    out2 = compute_pde_residual_from_state(_state(params, log_n, rlog, "w_pair", params.w), params)
    assert not torch.allclose(out1["residual_reference_scaled"], out2["residual_reference_scaled"])
    for name in ["log_N_eval", "N_eval", "dlogN_dt", "dlogN_dw", "g_eval", "mu_eval", "recruitment_dummy"]:
        assert torch.allclose(out1[name], out2[name])
    base = out1["residual_reference_scaled"]
    for factor in [1e-8, 1.0, 1e8]:
        set_residual_scale_from_initial_condition(params, factor * torch.ones(2, 4, dtype=torch.float64))
        out = compute_pde_residual_from_state(_state(params, log_n + torch.log(torch.tensor(factor, dtype=torch.float64)), rlog, "w_pair", params.w), params)
        assert torch.allclose(out["residual_reference_scaled"], base, rtol=1e-12, atol=1e-12)


def test_reference_scale_detached_and_backward_reaches_model_like_parameter():
    params = _params()
    set_residual_scale_from_initial_condition(params, torch.ones(2, 4, dtype=torch.float64))
    raw = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))
    out = compute_pde_residual_from_state(_state(params, raw, torch.ones(2, 4, dtype=torch.float64), "w_pair", params.w), params)
    loss = (out["residual_reference_scaled"] ** 2).mean()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert params.residual_scale_log.grad is None
    assert not out["reference_scale_eval"].requires_grad
