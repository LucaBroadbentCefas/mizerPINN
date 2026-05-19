from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


torch.set_default_dtype(torch.float64)


def pos(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


@dataclass
class MizerTorchParams:
    # grids
    w_full: torch.Tensor                 # [k]
    w: torch.Tensor                      # [w]
    dw_full: torch.Tensor                # [k]
    dw: torch.Tensor                     # [w]
    w_min_idx: torch.Tensor              # [species], R-style 1-based indices as exported from R/TMB

    # feeding / growth / mortality parameters
    ft_pred_kernel_e: torch.Tensor       # [species, k], complex
    ft_pred_kernel_p: torch.Tensor       # [species, k], complex
    ft_mask: torch.Tensor                # [species, k]
    search_vol: torch.Tensor             # [species, w]
    intake_max: torch.Tensor             # [species, w]
    alpha: torch.Tensor                  # [species]
    metab: torch.Tensor                  # [species, w]
    psi: torch.Tensor                    # [species, w]
    mu_b: torch.Tensor                   # [species, w]

    # species/resource interaction
    interaction_resource: torch.Tensor   # [species]
    interaction: torch.Tensor            # [predator, prey]

    # reproduction/resource
    erepro: torch.Tensor                 # [species]
    r_max: torch.Tensor                  # [species]
    rr_pp: torch.Tensor                  # [k]
    cc_pp: torch.Tensor                  # [k]

    # fishing mortality for one timestep, optional
    f_mort: Optional[torch.Tensor] = None # [species, w]


def fish_start(params: MizerTorchParams) -> int:
    return params.w_full.numel() - params.w.numel()


def as_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, imag)


def fft_convolve_rows(x: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
    """
    Match TMB/Eigen convention:
        fft_forward = fft(x)
        fft_forward_scaled = fft_forward / nCols
        inverse = unscaled_ifft(fft_forward_scaled * kernel_fft)

    PyTorch default ifft is scaled. Using norm="forward" makes the inverse
    unscaled, matching Eigen::FFT with Unscaled.
    """
    n_cols = x.shape[-1]
    x_fft = torch.fft.fft(x.to(kernel_fft.real.dtype), dim=-1)
    y = torch.fft.ifft((x_fft / n_cols) * kernel_fft, dim=-1, norm="forward").real
    return y


def compute_prey(
    n_pp: torch.Tensor,
    n: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: computePrey()

    n_pp: [k]
    n:    [species, w]

    returns prey [species, k]
    """
    n_species = params.interaction_resource.numel()
    k = params.w_full.numel()
    w_len = params.w.numel()
    start = fish_start(params)

    prey = params.interaction_resource[:, None] * n_pp[None, :]
    prey = prey.clone()

    prey[:, start:start + w_len] = (
        prey[:, start:start + w_len]
        + params.interaction @ n
    )

    prey = prey * (params.w_full * params.dw_full)[None, :]
    assert prey.shape == (n_species, k)
    return prey


def get_encounter(
    n_pp: torch.Tensor,
    n: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: getEncounter()
    """
    start = fish_start(params)
    prey = compute_prey(n_pp, n, params)
    encounter_full = fft_convolve_rows(prey, params.ft_pred_kernel_e)
    encounter = encounter_full[:, start:] * params.search_vol
    return encounter


def feeding_level(
    encounter: torch.Tensor,
    intake_max: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    TMB equivalent: FeedingLevel()
    """
    return encounter / (encounter + intake_max + eps)


def e_repro_and_growth(
    feeding: torch.Tensor,
    encounter: torch.Tensor,
    alpha: torch.Tensor,
    metab: torch.Tensor,
) -> torch.Tensor:
    """
    TMB equivalent: EReproAndGrowth()

    Important: this uses (1 - feeding) * encounter * alpha - metab,
    not feeding * intake_max * alpha - metab.
    They are algebraically equivalent only because:
        feeding = encounter / (encounter + intake_max)
        (1 - feeding) * encounter = feeding * intake_max
    up to numerical precision.
    """
    return (1.0 - feeding) * encounter * alpha[:, None] - metab


def e_repro(
    psi: torch.Tensor,
    erepog: torch.Tensor,
) -> torch.Tensor:
    return pos(erepog) * psi


def e_growth(
    erepog: torch.Tensor,
    e_repro_value: torch.Tensor,
) -> torch.Tensor:
    return pos(erepog) - e_repro_value


def compute_q_matrix(
    n: torch.Tensor,
    feeding: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: computeQMatrix()
    """
    start = fish_start(params)
    q = torch.zeros(
        (n.shape[0], params.w_full.numel()),
        dtype=n.dtype,
        device=n.device,
    )

    tmp = (1.0 - feeding) * n * params.search_vol
    tmp = tmp * params.dw[None, :]

    q[:, start:] = tmp
    return q


def get_pred_rate(
    n: torch.Tensor,
    feeding: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    q = compute_q_matrix(n, feeding, params)
    pred_full = fft_convolve_rows(q, params.ft_pred_kernel_p)
    pred_full = pred_full * params.ft_mask
    pred_full = pos(pred_full)
    return pred_full


def pred_mortality(
    pred_rate_full: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: PredMort()
    """
    start = fish_start(params)
    tmp = pred_rate_full[:, start:]
    return params.interaction.T @ tmp


def resource_mortality(
    pred_rate_full: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: ResourceMort()
    """
    return pred_rate_full.T @ params.interaction_resource


def total_mortality(
    pred_mort: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: Mort()
    """
    if params.f_mort is None:
        return pred_mort + params.mu_b
    return pred_mort + params.mu_b + params.f_mort


def rdi(
    e_repro_value: torch.Tensor,
    n: torch.Tensor,
    params: MizerTorchParams,
) -> torch.Tensor:
    """
    TMB equivalent: RDI()

    params.w_min_idx must be the R/TMB exported 1-based index.
    TMB uses:
        w(w_min_idx(i) - 1)
    because C++ is 0-based.
    PyTorch does the same using index = w_min_idx - 1.
    """
    tmp1 = e_repro_value * n
    tmp2 = tmp1 @ params.dw
    idx = params.w_min_idx.to(torch.long) - 1
    egg_w = params.w[idx]
    return 0.5 * (tmp2 * params.erepro) / egg_w


def rdd(
    rdi_value: torch.Tensor,
    r_max: torch.Tensor,
) -> torch.Tensor:
    """
    TMB equivalent: RDD()
    """
    return rdi_value / (1.0 + rdi_value / r_max)


def resource_semichemostat(
    n_pp: torch.Tensor,
    resource_mort: torch.Tensor,
    params: MizerTorchParams,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """
    TMB equivalent: resource_semichemostat()
    """
    mur = params.rr_pp + resource_mort
    n_steady = params.rr_pp * params.cc_pp / mur
    return n_steady + (n_pp - n_steady) * torch.exp(-mur * dt)


def get_a(
    e_growth_value: torch.Tensor,
    params: MizerTorchParams,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """
    TMB equivalent: getA()
    """
    a = torch.zeros_like(e_growth_value)
    tmp = -e_growth_value[:, :-1] * dt / params.dw[1:][None, :]
    a[:, 1:] = tmp
    return a


def get_b(
    e_growth_value: torch.Tensor,
    mort: torch.Tensor,
    params: MizerTorchParams,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """
    TMB equivalent: getB()
    """
    return 1.0 + e_growth_value * dt / params.dw[None, :] + mort * dt


def get_s(n: torch.Tensor) -> torch.Tensor:
    """
    TMB equivalent: getS()
    """
    s = torch.zeros_like(n)
    s[:, 1:] = n[:, 1:]
    return s


def inner_project_loop(
    n_new: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    w_min_idx: torch.Tensor,
) -> torch.Tensor:
    """
    AD-safe version of the semi-implicit projection.

    No in-place writes.
    """
    rows = []
    n_sp, no_w = n_new.shape

    for i in range(n_sp):
        start_j = int(w_min_idx[i].item())

        vals = []

        for j in range(start_j):
            vals.append(n_new[i, j])

        prev = n_new[i, start_j - 1]

        for j in range(start_j, no_w):
            cur = (s[i, j] - a[i, j] * prev) / b[i, j]
            vals.append(cur)
            prev = cur

        rows.append(torch.stack(vals))

    return torch.stack(rows)


def mizer_operators(
    n_pp: torch.Tensor,
    n: torch.Tensor,
    params: MizerTorchParams,
) -> dict[str, torch.Tensor]:
    """
    Operator-only layer. No timestep update.
    """
    encounter = get_encounter(n_pp, n, params)
    feeding = feeding_level(encounter, params.intake_max)

    erepog = e_repro_and_growth(
        feeding=feeding,
        encounter=encounter,
        alpha=params.alpha,
        metab=params.metab,
    )
    e_repro_value = e_repro(params.psi, erepog)
    e_growth_value = e_growth(erepog, e_repro_value)

    pred_rate_full = get_pred_rate(n, feeding, params)
    pred_mort = pred_mortality(pred_rate_full, params)
    resource_mort = resource_mortality(pred_rate_full, params)
    mort = total_mortality(pred_mort, params)

    rdi_value = rdi(e_repro_value, n, params)
    rdd_value = rdd(rdi_value, params.r_max)

    return {
        "encounter": encounter,
        "feeding_level": feeding,
        "erepog": erepog,
        "e_repro": e_repro_value,
        "e_growth": e_growth_value,
        "pred_rate": pred_rate_full,
        "pred_mort": pred_mort,
        "resource_mort": resource_mort,
        "mort": mort,
        "rdi": rdi_value,
        "rdd": rdd_value,
    }


def step(
    n_pp: torch.Tensor,
    n: torch.Tensor,
    params: MizerTorchParams,
    dt: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    One timestep, AD-safe version.
    """
    ops = mizer_operators(n_pp, n, params)

    n_pp_new = resource_semichemostat(
        n_pp=n_pp,
        resource_mort=ops["resource_mort"],
        params=params,
        dt=dt,
    )

    a = get_a(ops["e_growth"], params, dt)
    b = get_b(ops["e_growth"], ops["mort"], params, dt)
    s = get_s(n)

    rows = []
    egg_idx = params.w_min_idx.to(torch.long) - 1
    n_sp, no_w = n.shape

    for i in range(n_sp):
        egg = int(egg_idx[i].item())
        start_j = int(params.w_min_idx[i].item())

        vals = []

        for j in range(egg):
            vals.append(n[i, j])

        egg_val = (n[i, egg] + ops["rdd"][i] * dt / params.dw[egg]) / b[i, egg]
        vals.append(egg_val)

        prev = egg_val

        for j in range(start_j, no_w):
            cur = (s[i, j] - a[i, j] * prev) / b[i, j]
            vals.append(cur)
            prev = cur

        rows.append(torch.stack(vals))

    n_new = torch.stack(rows)

    ops["A"] = a
    ops["B"] = b
    ops["S"] = s

    return n_pp_new, n_new, ops


def _read_vec(
    input_dir: str | Path,
    name: str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    path = Path(input_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    return torch.tensor(df.iloc[:, 0].to_numpy(), dtype=dtype)


def _read_mat(
    input_dir: str | Path,
    name: str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    path = Path(input_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    return torch.tensor(df.to_numpy(), dtype=dtype)


def load_params_and_state(
    input_dir: str | Path = "py_inputs",
) -> tuple[MizerTorchParams, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dir = Path(input_dir)

    ft_pred_kernel_e = as_complex(
        _read_mat(input_dir, "ft_pred_kernel_e_real"),
        _read_mat(input_dir, "ft_pred_kernel_e_imag"),
    )

    ft_pred_kernel_p = as_complex(
        _read_mat(input_dir, "ft_pred_kernel_p_real"),
        _read_mat(input_dir, "ft_pred_kernel_p_imag"),
    )

    f_mort_path = input_dir / "f_mort.csv"
    f_mort = _read_mat(input_dir, "f_mort") if f_mort_path.exists() else None

    params = MizerTorchParams(
        w_full=_read_vec(input_dir, "w_full"),
        w=_read_vec(input_dir, "w"),
        dw_full=_read_vec(input_dir, "dw_full"),
        dw=_read_vec(input_dir, "dw"),
        w_min_idx=_read_vec(input_dir, "w_min_idx"),

        ft_pred_kernel_e=ft_pred_kernel_e,
        ft_pred_kernel_p=ft_pred_kernel_p,
        ft_mask=_read_mat(input_dir, "ft_mask"),

        search_vol=_read_mat(input_dir, "search_vol"),
        intake_max=_read_mat(input_dir, "intake_max"),
        alpha=_read_vec(input_dir, "alpha"),
        metab=_read_mat(input_dir, "metab"),
        psi=_read_mat(input_dir, "psi"),
        mu_b=_read_mat(input_dir, "mu_b"),

        interaction_resource=_read_vec(input_dir, "interaction_resource"),
        interaction=_read_mat(input_dir, "interaction"),

        erepro=_read_vec(input_dir, "erepro"),
        r_max=_read_vec(input_dir, "r_max"),
        rr_pp=_read_vec(input_dir, "rr_pp"),
        cc_pp=_read_vec(input_dir, "cc_pp"),

        f_mort=f_mort,
    )

    n = _read_mat(input_dir, "n")
    n_pp = _read_vec(input_dir, "n_pp")
    dt = _read_vec(input_dir, "dt")[0]

    return params, n, n_pp, dt


def _write_mat(
    output_dir: str | Path,
    name: str,
    x: torch.Tensor,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_cpu = x.detach().cpu()

    if x_cpu.ndim == 0:
        x_cpu = x_cpu.reshape(1, 1)
    elif x_cpu.ndim == 1:
        x_cpu = x_cpu.reshape(-1, 1)

    pd.DataFrame(x_cpu.numpy()).to_csv(
        output_dir / f"{name}.csv",
        index=False,
    )


def run_one_step_from_csv(
    input_dir: str | Path = "py_inputs",
    output_dir: str | Path = "py_steps",
) -> None:
    params, n, n_pp, dt = load_params_and_state(input_dir)

    ops = mizer_operators(n_pp, n, params)
    n_pp_new, n_new, step_ops = step(n_pp, n, params, dt)

    prey = compute_prey(n_pp, n, params)
    q_matrix = compute_q_matrix(n, ops["feeding_level"], params)

    _write_mat(output_dir, "01_prey", prey)
    _write_mat(output_dir, "02_encounter", ops["encounter"])
    _write_mat(output_dir, "03_feeding_level", ops["feeding_level"])
    _write_mat(output_dir, "04_erepog", ops["erepog"])
    _write_mat(output_dir, "05_e_repro", ops["e_repro"])
    _write_mat(output_dir, "06_e_growth", ops["e_growth"])
    _write_mat(output_dir, "07_q_matrix", q_matrix)
    _write_mat(output_dir, "08_pred_rate", ops["pred_rate"])
    _write_mat(output_dir, "09_pred_mort", ops["pred_mort"])
    _write_mat(output_dir, "10_resource_mort", ops["resource_mort"])
    _write_mat(output_dir, "11_mort", ops["mort"])
    _write_mat(output_dir, "12_rdi", ops["rdi"])
    _write_mat(output_dir, "13_rdd", ops["rdd"])

    _write_mat(output_dir, "14_A", step_ops["A"])
    _write_mat(output_dir, "15_B", step_ops["B"])
    _write_mat(output_dir, "16_S", step_ops["S"])

    _write_mat(output_dir, "18_step_n_pp_new", n_pp_new)
    _write_mat(output_dir, "19_step_n_new", n_new)


if __name__ == "__main__":
    run_one_step_from_csv()
