from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import active_grid_mask, scale_t, scale_x
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.models import build_pinn_model
from PINNmizer.pinn.state_scale import set_state_scale_from_initial_condition


# ------------------------------------------------------------------
# Repository root
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# EDIT THESE 20260723_191235_445308_job3640860_task
# ------------------------------------------------------------------
CHECKPOINT = ROOT / "runs/pde_multispecies/20260820_165213_767884_job3995775_task2/model_step_10000.pt"

# Multispecies:
INPUT_DIR = ROOT / "validation/fixtures/pde_multispecies"
MIZER_CSV = ROOT / "mizer_projection_long.csv"
SPECIES_IDX = 0

# Example single-species Cod run instead:
# INPUT_DIR = ROOT / "validation/fixtures/pde_single_species/sp_11"
# MIZER_CSV = ROOT / "single_species_projection_long.csv"
# SPECIES_IDX = 0

DEVICE = "cpu"
# ------------------------------------------------------------------


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def build_model(config, n_species):
    model = build_pinn_model(
        model_arch=config["model_arch"],
        in_dim=2,
        out_dim=n_species,
        hidden_width=int(config["hidden_width"]),
        hidden_layers=int(config["hidden_layers"]),
        fourier_num_features=int(config.get("fourier_num_features", 64)),
        fourier_scale=float(config.get("fourier_scale", 1.0)),
        fourier_include_raw_input=bool(
            config.get("fourier_include_raw_input", False)
        ),
        fourier_seed=config.get("fourier_seed"),
        weight_factorization=config.get("weight_factorization", "none"),
        rwf_mu=float(config.get("rwf_mu", 1.0)),
        rwf_sigma=float(config.get("rwf_sigma", 0.1)),
        rwf_apply_to=config.get("rwf_apply_to", "all"),
        rwf_base_init=config.get("rwf_base_init", "pytorch"),
    ).to(dtype=torch.float64, device=DEVICE)

    return model


# ------------------------------------------------------------------
# Load trained PINN
# ------------------------------------------------------------------
params, n_init, _ = load_mizer_inputs(
    INPUT_DIR,
    dtype=torch.float64,
    device=DEVICE,
)

checkpoint = load_checkpoint(CHECKPOINT)
config = checkpoint["config"]

params.state_parameterization = config.get(
    "state_parameterization",
    "log-n",
)

if params.state_parameterization == "log-u":
    set_state_scale_from_initial_condition(
        params,
        n_init,
        eps=float(config.get("state_scale_eps", 1e-30)),
    )

n_species = int(params.interaction.shape[0])

model = build_model(config, n_species)
model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)
model.eval()


# ------------------------------------------------------------------
# Load mizer truth
# ------------------------------------------------------------------
truth = pd.read_csv(MIZER_CSV)

truth = truth.rename(
    columns={
        "t": "time",
        "weight": "w",
        "sp": "species",
        "abundance": "N",
    }
)


# ------------------------------------------------------------------
# Identify the corresponding mizer species
# ------------------------------------------------------------------
single_species_names = {
    "sp_3": "N.pout",
    "sp_7": "Sole",
    "sp_11": "Cod",
}

if INPUT_DIR.name in single_species_names:

    # A single-species network always has output index 0.
    # Recover the original mizer species identity from the fixture name.
    species_name = single_species_names[INPUT_DIR.name]

    if "species" not in truth.columns:
        raise ValueError(
            "Single-species truth must contain a 'species' column."
        )

    truth = truth[
        truth["species"].astype(str) == species_name
    ].copy()

else:

    # Multispecies case.
    if "species_idx" in truth.columns:

        truth["species_idx"] = pd.to_numeric(
            truth["species_idx"],
            errors="coerce",
        )

        truth = truth[
            truth["species_idx"] == SPECIES_IDX
        ].copy()

        if "species" in truth.columns and not truth.empty:
            species_name = str(truth["species"].iloc[0])
        else:
            species_name = f"species_{SPECIES_IDX}"

    elif "species" in truth.columns:

        # If the truth has names but no explicit index, preserve the
        # species ordering in the mizer export.
        species_order = (
            truth["species"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if SPECIES_IDX >= len(species_order):
            raise ValueError(
                f"SPECIES_IDX={SPECIES_IDX}, but the mizer truth "
                f"contains only {len(species_order)} species."
            )

        species_name = species_order[SPECIES_IDX]

        truth = truth[
            truth["species"].astype(str) == species_name
        ].copy()

    else:
        raise ValueError(
            "Multispecies truth needs either 'species_idx' "
            "or 'species'."
        )


if truth.empty:
    raise ValueError(
        f"No mizer truth found for {species_name} "
        f"(SPECIES_IDX={SPECIES_IDX})."
    )

print(f"Using PINN output index: {SPECIES_IDX}")
print(f"Using mizer truth species: {species_name}")


# ------------------------------------------------------------------
# Clean truth state
# ------------------------------------------------------------------
required = {"time", "w", "N"}

missing = required.difference(truth.columns)

if missing:
    raise ValueError(
        f"Mizer truth is missing columns: {sorted(missing)}"
    )

truth = truth[
    ["time", "w", "N"]
].copy()

for column in ["time", "w", "N"]:
    truth[column] = pd.to_numeric(
        truth[column],
        errors="coerce",
    )

truth = truth.dropna()

truth = (
    truth
    .groupby(["time", "w"], as_index=False)["N"]
    .mean()
)


# ------------------------------------------------------------------
# Use the mizer output times inside the PINN time domain
# ------------------------------------------------------------------
t_min = float(params.t_min)
t_max = float(params.t_max)

times = np.sort(
    truth.loc[
        truth["time"].between(t_min, t_max),
        "time",
    ].unique()
)

if len(times) == 0:
    raise ValueError(
        "No mizer truth times overlap the PINN time domain."
    )


# ------------------------------------------------------------------
# Evaluate PINN at the same times and on params.w
#
# N_pred shape:
# [n_time, n_species, n_w]
# ------------------------------------------------------------------
t = torch.tensor(
    times,
    dtype=torch.float64,
    device=DEVICE,
)

x = torch.log(params.w)

with torch.no_grad():

    state = evaluate_log_model_on_points(
        model=model,
        x_scaled=scale_x(x, params),
        t_scaled=scale_t(t, params),
        params=params,
    )

    N_pred = state["N"][:, SPECIES_IDX, :]


# ------------------------------------------------------------------
# Biomass
#
# B(t) = sum_w N(t,w) * w * dw
# ------------------------------------------------------------------
w = params.w.detach().cpu().numpy()
dw = params.dw.detach().cpu().numpy()

active = (
    active_grid_mask(params)[SPECIES_IDX]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

factor = w * dw

N_pred_np = N_pred.detach().cpu().numpy()

B_pinn = (
    N_pred_np[:, active]
    * factor[active]
).sum(axis=1)


# ------------------------------------------------------------------
# Mizer biomass on the same physical weight grid
# ------------------------------------------------------------------
B_mizer = []

for time in times:

    g = (
        truth[
            np.isclose(truth["time"], time)
        ]
        .sort_values("w")
    )

    wt = g["w"].to_numpy()
    Nt = g["N"].to_numpy()

    target_w = w[active]

    # If the grids are identical, use the mizer values directly.
    if (
        len(wt) == len(w)
        and np.allclose(
            wt,
            w,
            rtol=1e-10,
            atol=0,
        )
    ):

        N_on_grid = Nt[active]

    else:

        # Otherwise interpolate N in log-weight space.
        N_on_grid = np.exp(
            np.interp(
                np.log(target_w),
                np.log(wt),
                np.log(
                    np.clip(
                        Nt,
                        1e-300,
                        None,
                    )
                ),
                left=np.nan,
                right=np.nan,
            )
        )

        if not np.isfinite(N_on_grid).all():
            raise ValueError(
                "The mizer weight grid does not cover the "
                f"active PINN weight grid at time {time}."
            )

    biomass = np.sum(
        N_on_grid
        * factor[active]
    )

    B_mizer.append(biomass)

B_mizer = np.asarray(B_mizer)


# ------------------------------------------------------------------
# Save comparison
# ------------------------------------------------------------------
comparison = pd.DataFrame(
    {
        "time": times,
        "biomass_pinn": B_pinn,
        "biomass_mizer": B_mizer,
    }
)

out_csv = (
    CHECKPOINT.parent
    / "biomass_comparison.csv"
)

out_png = (
    CHECKPOINT.parent
    / "biomass_comparison.png"
)

comparison.to_csv(
    out_csv,
    index=False,
)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    times,
    B_mizer,
    label="mizer",
    linewidth=2,
)

plt.plot(
    times,
    B_pinn,
    label="PINN",
    linewidth=2,
)

plt.xlabel("Time")
plt.ylabel("Biomass")

plt.title(
    f"Biomass over time: {species_name}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    out_png,
    dpi=200,
)

plt.show()


# ------------------------------------------------------------------
# Print results
# ------------------------------------------------------------------
print(comparison)

print(
    f"Saved: {out_csv}"
)

print(
    f"Saved: {out_png}"
)
