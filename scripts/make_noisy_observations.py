#!/usr/bin/env python3
"""Create a fixed-CV noisy observation CSV from a noiseless observation CSV.

The generated noise is distributionally equivalent to:

    exp(rnorm(n, log(value_true), sqrt(log(cv^2 + 1))))

The input file is never modified.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Noiseless source observation CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Generated noisy observation CSV.")
    parser.add_argument("--cv", required=True, type=float, help="Global fixed CV used for noise and likelihood.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for the standard-normal draws.")
    parser.add_argument("--task-id", type=int, default=-1, help="Optional Slurm task number stored as metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input observation CSV does not exist: {args.input}")
    if not math.isfinite(args.cv) or args.cv <= 0.0:
        raise ValueError(f"--cv must be finite and strictly positive; got {args.cv}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    observations = pd.read_csv(args.input)
    if "value" not in observations.columns:
        raise ValueError("Input observation CSV must contain a 'value' column.")
    if observations.empty:
        raise ValueError("Input observation CSV contains no rows.")

    # Prefer an existing truth column, otherwise treat the source 'value' as truth.
    if "value_true" in observations.columns:
        value_true = pd.to_numeric(observations["value_true"], errors="coerce").to_numpy(dtype=float)
    else:
        value_true = pd.to_numeric(observations["value"], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(value_true).all() or not np.all(value_true > 0.0):
        bad_count = int(np.sum(~np.isfinite(value_true) | (value_true <= 0.0)))
        raise ValueError(
            f"All true observation values must be finite and strictly positive; found {bad_count} invalid rows."
        )

    sd_log = math.sqrt(math.log1p(args.cv**2))
    rng = np.random.default_rng(args.seed)
    noise_z = rng.standard_normal(value_true.size)
    noisy_value = np.exp(np.log(value_true) + sd_log * noise_z)

    if not np.isfinite(noisy_value).all() or not np.all(noisy_value > 0.0):
        raise FloatingPointError("Noise generation produced non-finite or non-positive values.")

    # The training loader may prefer an existing sd_log over cv, so overwrite both
    # with the same global fixed uncertainty used to generate these observations.
    observations["value_true"] = value_true
    observations["value"] = noisy_value
    observations["cv"] = args.cv
    observations["sd_log"] = sd_log

    # Metadata retained for later reconstruction and diagnostics.
    observations["true_cv"] = args.cv
    observations["true_sd_log"] = sd_log
    observations["noise_seed"] = args.seed
    observations["noise_z"] = noise_z
    observations["noise_task_id"] = args.task_id
    observations["noise_model"] = "lognormal_median_centered"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    observations.to_csv(temporary_output, index=False)
    temporary_output.replace(args.output)

    print(f"Created: {args.output}")
    print(f"Rows: {len(observations)}")
    print(f"CV: {args.cv:.8g}")
    print(f"sd_log: {sd_log:.8g}")
    print(f"seed: {args.seed}")
    print(f"value_true range: [{value_true.min():.8g}, {value_true.max():.8g}]")
    print(f"noisy value range: [{noisy_value.min():.8g}, {noisy_value.max():.8g}]")


if __name__ == "__main__":
    main()
