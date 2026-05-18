from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.mizer_grid_ops import mizer_operators


OUT_DIR = Path("py_mizer_ops")
OUT_DIR.mkdir(exist_ok=True)


def write_tensor(x: torch.Tensor, name: str, outdir: Path = OUT_DIR) -> None:
    x = x.detach().cpu()

    if torch.is_complex(x):
        pd.DataFrame(x.real.numpy()).to_csv(outdir / f"{name}_real.csv", index=False)
        pd.DataFrame(x.imag.numpy()).to_csv(outdir / f"{name}_imag.csv", index=False)
        return

    if x.ndim == 0:
        pd.DataFrame({"value": [float(x)]}).to_csv(outdir / f"{name}.csv", index=False)
    elif x.ndim == 1:
        pd.DataFrame({"value": x.numpy()}).to_csv(outdir / f"{name}.csv", index=False)
    else:
        pd.DataFrame(x.numpy()).to_csv(outdir / f"{name}.csv", index=False)


params, n, n_pp = load_mizer_inputs(
    "py_inputs",
    dtype=torch.float64,
    device="cpu",
)

ops = mizer_operators(
    n_pp=n_pp,
    n=n,
    params=params,
)

for name, value in ops.items():
    write_tensor(value, name)

print(f"Wrote mizer_operators outputs to {OUT_DIR}/")
