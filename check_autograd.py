from pathlib import Path

import pandas as pd
import torch

from mizer_torch_ops import (
    MizerTorchParams,
    as_complex,
    mizer_operators,
    step,
)


IN_DIR = Path("py_inputs")
DTYPE = torch.float64


def mat(name, dtype=DTYPE):
    arr = pd.read_csv(IN_DIR / f"{name}.csv").to_numpy().copy()
    return torch.as_tensor(arr, dtype=dtype)


def vec(name, dtype=DTYPE):
    return mat(name, dtype=dtype).reshape(-1)


def build_params():
    ft_e = as_complex(
        mat("ft_pred_kernel_e_real"),
        mat("ft_pred_kernel_e_imag"),
    )

    ft_p = as_complex(
        mat("ft_pred_kernel_p_real"),
        mat("ft_pred_kernel_p_imag"),
    )

    return MizerTorchParams(
        w_full=vec("w_full"),
        w=vec("w"),
        dw_full=vec("dw_full"),
        dw=vec("dw"),
        w_min_idx=vec("w_min_idx", dtype=torch.long),

        ft_pred_kernel_e=ft_e,
        ft_pred_kernel_p=ft_p,
        ft_mask=mat("ft_mask"),

        search_vol=mat("search_vol"),
        intake_max=mat("intake_max"),
        alpha=vec("alpha"),
        metab=mat("metab"),
        psi=mat("psi"),
        mu_b=mat("mu_b"),

        interaction_resource=vec("interaction_resource"),
        interaction=mat("interaction"),

        erepro=vec("erepro"),
        r_max=vec("r_max"),
        rr_pp=vec("rr_pp"),
        cc_pp=vec("cc_pp"),

        f_mort=mat("f_mort"),
    )


def report_grad(name, x):
    g = x.grad

    if g is None:
        print(f"{name:20s} grad=None")
        return

    print(
        f"{name:20s} "
        f"shape={tuple(g.shape)} "
        f"finite={torch.isfinite(g).all().item()} "
        f"nonzero={(g.abs() > 0).any().item()} "
        f"max_abs={g.abs().max().item():.6e}"
    )


def main():
    params = build_params()

    # These are the dynamic state variables. These are the first things to test.
    n = mat("n").requires_grad_(True)
    n_pp = vec("n_pp").requires_grad_(True)
    dt = float(vec("dt")[0])

    ops = mizer_operators(n_pp=n_pp, n=n, params=params)

    print("\n--- grad_fn checks ---")
    for key in [
        "encounter",
        "feeding_level",
        "e_repro",
        "e_growth",
        "pred_rate",
        "pred_mort",
        "mort",
        "rdi",
        "rdd",
    ]:
        y = ops[key]
        print(f"{key:20s} requires_grad={y.requires_grad} grad_fn={type(y.grad_fn).__name__ if y.grad_fn else None}")

    # Scalar objective. Use several outputs so that the test covers most operators.
    loss = (
        ops["encounter"].sum()
        + ops["feeding_level"].sum()
        + ops["e_repro"].sum()
        + ops["e_growth"].sum()
        + ops["pred_mort"].sum()
        + ops["mort"].sum()
        + ops["rdi"].sum()
        + ops["rdd"].sum()
    )

    print("\n--- backward through rates ---")
    print(f"loss requires_grad={loss.requires_grad}, grad_fn={type(loss.grad_fn).__name__}")
    loss.backward()

    report_grad("n", n)
    report_grad("n_pp", n_pp)

    # Now test the full one-step projection.
    n2 = mat("n").requires_grad_(True)
    n_pp2 = vec("n_pp").requires_grad_(True)

    n_pp_new, n_new, step_ops = step(n_pp=n_pp2, n=n2, params=params, dt=dt)

    step_loss = n_new.sum() + n_pp_new.sum()
    print("\n--- backward through full step ---")
    print(f"step_loss requires_grad={step_loss.requires_grad}, grad_fn={type(step_loss.grad_fn).__name__}")
    step_loss.backward()

    report_grad("n through step", n2)
    report_grad("n_pp through step", n_pp2)


if __name__ == "__main__":
    main()
