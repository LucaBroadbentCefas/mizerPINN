from pathlib import Path

import pandas as pd
import torch

from mizer_torch_ops import (
    MizerTorchParams,
    as_complex,
    compute_prey,
    get_encounter,
    feeding_level,
    e_repro_and_growth,
    e_repro,
    e_growth,
    compute_q_matrix,
    get_pred_rate,
    pred_mortality,
    resource_mortality,
    total_mortality,
    rdi,
    rdd,
    resource_semichemostat,
    get_a,
    get_b,
    get_s,
    step,
)

IN_DIR = Path("validation/fixtures/mizer_full")
OUT_DIR = Path("validation/outputs/python_steps")
OUT_DIR.mkdir(exist_ok=True)

DTYPE = torch.float64


def mat(name, dtype=DTYPE):
    return torch.as_tensor(
        pd.read_csv(IN_DIR / f"{name}.csv").to_numpy(),
        dtype=dtype
    )


def vec(name, dtype=DTYPE):
    return mat(name, dtype=dtype).reshape(-1)


def write_tensor(x, name):
    x = x.detach().cpu()

    if torch.is_complex(x):
        pd.DataFrame(x.real.numpy()).to_csv(OUT_DIR / f"{name}_real.csv", index=False)
        pd.DataFrame(x.imag.numpy()).to_csv(OUT_DIR / f"{name}_imag.csv", index=False)
        return

    if x.ndim == 0:
        pd.DataFrame({"value": [float(x)]}).to_csv(OUT_DIR / f"{name}.csv", index=False)
    elif x.ndim == 1:
        pd.DataFrame({"value": x.numpy()}).to_csv(OUT_DIR / f"{name}.csv", index=False)
    else:
        pd.DataFrame(x.numpy()).to_csv(OUT_DIR / f"{name}.csv", index=False)


ft_e = as_complex(
    mat("ft_pred_kernel_e_real"),
    mat("ft_pred_kernel_e_imag"),
)

ft_p = as_complex(
    mat("ft_pred_kernel_p_real"),
    mat("ft_pred_kernel_p_imag"),
)

params = MizerTorchParams(
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

n = mat("n")
n_pp = vec("n_pp")
dt = float(vec("dt")[0])

# Save inputs as Python saw them
write_tensor(n, "input_n")
write_tensor(n_pp, "input_n_pp")

# Step-by-step outputs
prey = compute_prey(n_pp, n, params)
write_tensor(prey, "01_prey")

encounter = get_encounter(n_pp, n, params)
write_tensor(encounter, "02_encounter")

feeding = feeding_level(encounter, params.intake_max)
write_tensor(feeding, "03_feeding_level")

erepog = e_repro_and_growth(feeding, encounter, params.alpha, params.metab)
write_tensor(erepog, "04_erepog")

e_repro_value = e_repro(params.psi, erepog)
write_tensor(e_repro_value, "05_e_repro")

e_growth_value = e_growth(erepog, e_repro_value)
write_tensor(e_growth_value, "06_e_growth")

q_matrix = compute_q_matrix(n, feeding, params)
write_tensor(q_matrix, "07_q_matrix")

pred_rate = get_pred_rate(n, feeding, params)
write_tensor(pred_rate, "08_pred_rate")

pred_mort = pred_mortality(pred_rate, params)
write_tensor(pred_mort, "09_pred_mort")

resource_mort = resource_mortality(pred_rate, params)
write_tensor(resource_mort, "10_resource_mort")

mort = total_mortality(pred_mort, params)
write_tensor(mort, "11_mort")

rdi_value = rdi(e_repro_value, n, params)
write_tensor(rdi_value, "12_rdi")

rdd_value = rdd(rdi_value, params.r_max)
write_tensor(rdd_value, "13_rdd")

n_pp_new_manual = resource_semichemostat(n_pp, resource_mort, params, dt)
write_tensor(n_pp_new_manual, "14_n_pp_new")

a = get_a(e_growth_value, params, dt)
write_tensor(a, "15_A")

b = get_b(e_growth_value, mort, params, dt)
write_tensor(b, "16_B")

s = get_s(n)
write_tensor(s, "17_S")

n_pp_new, n_new, ops = step(n_pp, n, params, dt)
write_tensor(n_pp_new, "18_step_n_pp_new")
write_tensor(n_new, "19_step_n_new")

print("Wrote Python step outputs to validation/outputs/python_steps/")
