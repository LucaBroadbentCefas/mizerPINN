#!/usr/bin/env python3
"""
Local overnight smoke test for final_model_suite.slurm.

Run from the mizerPINN repository root with the mizer-torch environment active:

    python scripts/smoke_final_suite.py

Purpose:
- validate the real fixture/data paths used by the final suite;
- load every deterministic observation dataset through the real loader;
- generate and validate every CV x noise-seed dataset used by the final suite;
- exercise the four single-species state/architecture code paths;
- exercise no-data, perfect-data, discrepancy-gate, lambda_PDE=0,
  Rmax, CV, and fishing-effort inverse training paths;
- force an adaptive-weight update during the tiny smoke fits;
- keep going after failures and write logs + summary files.

This is not a convergence test. It deliberately uses only a few optimisation steps
and small collocation/diagnostic grids while preserving the model architecture,
loss forms, observation operators, inverse modules, and other important arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.io_observations import load_observation_csv
from PINNmizer.inverse_parameters import BoundedDataCV, BoundedLogRMax, LogFishingEffort


FINAL_SUITE_F_COMMIT = "836e1675e9d5598cbe05611c40d5f19ab2e81d1b"

SINGLE_SPECIES = ("sp_3", "sp_7", "sp_11")
DATA_FILES = (
    "perfect.csv",
    "every_3yr.csv",
    "gap_10_20.csv",
    "gap_30_40.csv",
    "missing_species_3.csv",
    "missing_species_7.csv",
    "missing_species_11.csv",
)
NOISE_CVS = (0.1, 0.2, 0.3, 0.4, 0.5)
NOISE_SEEDS = (41001, 41002, 41003, 41004, 41005)
RMAX_FACTORS = (0.25, 0.5, 2.0, 4.0)
CV_STARTS = (0.1, 0.3, 0.6, 1.0)
EFFORT_STARTS = (0.05, 0.5, 1.0, 3.0)


@dataclass
class Result:
    name: str
    kind: str
    status: str
    seconds: float
    returncode: int | None = None
    log: str = ""
    message: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path.cwd())
    p.add_argument(
        "--single-base-dir",
        default="validation/fixtures/pde_single_species",
    )
    p.add_argument(
        "--ms-base-dir",
        default="validation/fixtures/pde_multispecies",
    )
    p.add_argument(
        "--data-dir",
        default="final_runs/observations/final_runs",
        help="Directory containing perfect.csv and deterministic data variants.",
    )
    p.add_argument(
        "--final-input-root",
        default="final_runs/final_suite_inputs",
        help="Parent directory used by the final suite for task-specific input copies.",
    )
    p.add_argument(
        "--noise-output-root",
        default="final_runs/generated_observations/final_suite_noise",
        help="Parent directory used by the final suite for generated noisy observations.",
    )
    p.add_argument(
        "--manifest-root",
        default="final_runs/final_suite_manifest",
        help="Parent directory used by the final suite for task manifests.",
    )
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--n-time", type=int, default=4)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--full",
        action="store_true",
        help="Run extra representative fits. Static validation is always exhaustive.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed training subprocess.",
    )
    return p.parse_args()


def abs_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def run_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def check_git_version(root: Path) -> str:
    rc, head = run_capture(["git", "rev-parse", "HEAD"], root)
    if rc != 0:
        return "WARNING: could not read git HEAD."
    head = head.strip()
    rc, _ = run_capture(
        ["git", "merge-base", "--is-ancestor", FINAL_SUITE_F_COMMIT, "HEAD"],
        root,
    )
    if rc != 0:
        return (
            f"WARNING: HEAD={head} does not contain required fishing-effort "
            f"merge commit {FINAL_SUITE_F_COMMIT}. Pull/update before the HPC run."
        )
    return f"Git HEAD={head}; fishing-effort implementation is present."


def load_params(path: Path):
    return load_mizer_inputs(path, dtype=torch.float64, device="cpu")


def validate_observation_file(path: Path, params) -> dict:
    obs = load_observation_csv(path, params, default_cv=0.3, estimate_cv=False)
    return {
        "n_obs": int(obs["value"].numel()),
        "t_min": float(obs["t_start"].min().cpu()),
        "t_max": float(obs["t_start"].max().cpu()),
        "n_species": int(torch.unique(obs["species_idx"]).numel()),
        "value_min": float(obs["value"].min().cpu()),
        "value_max": float(obs["value"].max().cpu()),
    }


def write_scaled_rmax_fixture(source_dir: Path, dest_dir: Path, factor: float) -> Path:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)

    rmax = dest_dir / "r_max.csv"
    true_file = dest_dir / "r_max_true.csv"
    shutil.copy2(rmax, true_file)

    with rmax.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    updated = [rows[0]]
    for row in rows[1:]:
        if not row:
            continue
        x = float(row[0]) * factor
        if not math.isfinite(x) or x <= 0:
            raise ValueError(f"Bad scaled Rmax for factor {factor}: {x}")
        updated.append([f"{x:.17g}"])

    with rmax.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(updated)
    return true_file


def common_args(args: argparse.Namespace) -> list[str]:
    # Force the important training pathways to execute quickly.
    # expert update occurs at step 2 instead of 2000.
    return [
        "--n-steps", str(args.steps),
        "--n-time", str(args.n_time),
        "--n-eval", str(args.n_eval),
        "--lr", "1e-3",
        "--lr-scheduler", "cosine",
        "--lr-min", "1e-5",
        "--state-scale-eps", "1e-30",
        "--residual-form", "scaled",
        "--pde-penalty", "squared",
        "--pde-pseudo-huber-delta", "1.0",
        "--boundary-loss-form", "relative",
        "--bc-penalty", "squared",
        "--bc-pseudo-huber-delta", "1.0",
        "--bc-g-min", "1e-12",
        "--lambda-ic", "1.0",
        "--lambda-bc", "1.0",
        "--initial-w-pde", "1.0",
        "--initial-w-ic", "1.0",
        "--initial-w-bc", "0.1",
        "--lambda-timestep", "0.0",
        "--collocation-strategy", "uniform",
        "--time-sampling", "stratified",
        "--causal-loss", "expert",
        "--causal-curriculum", "linear",
        "--causal-start-fraction", "0.05",
        "--causal-ramp-steps", "2",
        "--causal-step-fractions", "0.05,0.10,0.20,0.40,0.70,1.0",
        "--causal-n-chunks", str(args.n_time),
        "--causal-epsilon", "1.0",
        "--loss-weighting", "expert-grad-norm",
        "--expert-weight-update-every", "2",
        "--weight-warmup-steps", "1",
        "--expert-weight-alpha", "0.9",
        "--expert-weight-batch", "fixed",
        "--weight-min", "1e-3",
        "--weight-max", "1e2",
        "--expert-weight-min", "1e-3",
        "--expert-weight-max", "1e2",
        "--no-hard-set-first-weight-update",
        "--weight-factorization", "rwf",
        "--rwf-mu", "1.0",
        "--rwf-sigma", "0.1",
        "--rwf-apply-to", "all",
        "--rwf-base-init", "xavier_uniform",
        "--hidden-width", "384",
        "--hidden-layers", "5",
        "--diag-final-n-time", "5",
        "--diag-final-n-eval", "10",
        "--seed", "123",
        "--device", "cpu",
        "--print-every", "1",
        "--checkpoint-every", "1000",
        "--hpc",
    ]


def add_architecture(cmd: list[str], arch: str) -> None:
    cmd.extend(["--model-arch", arch])
    if arch == "fourier":
        cmd.extend([
            "--fourier-num-features", "32",
            "--fourier-scale", "0.5",
            "--fourier-include-raw-input",
            "--fourier-seed", "123",
        ])


def single_command(
    args: argparse.Namespace,
    input_dir: Path,
    state: str,
    arch: str,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.train_pde_only_single_species",
        "--input-dir", str(input_dir),
        *common_args(args),
        "--lambda-pde", "1.0",
        "--state-parameterization", state,
    ]
    add_architecture(cmd, arch)
    return cmd


def multispecies_command(
    args: argparse.Namespace,
    input_dir: Path,
    *,
    data_csv: Path | None,
    lambda_pde: float = 1.0,
    lambda_data: float = 1.0,
    data_cv: float = 0.3,
    gate: bool = False,
    estimate_rmax: bool = False,
    estimate_cv: bool = False,
    cv_init: float | None = None,
    estimate_effort: bool = False,
    effort_init: float | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.train_pde_multispecies",
        "--input-dir", str(input_dir),
        "--species-mode", "all",
        *common_args(args),
        "--state-parameterization", "log-u",
        "--lambda-pde", str(lambda_pde),
        "--lambda-data", str(lambda_data),
        "--initial-w-data", "1.0",
        "--data-default-cv", str(data_cv),
        "--data-loss-eps", "1e-30",
        "--data-time-quadrature-points", "3",
    ]
    add_architecture(cmd, "fourier")

    if data_csv is not None:
        cmd.extend(["--data-csv", str(data_csv)])
    if gate:
        cmd.append("--data-discrepancy-gate")
    if estimate_rmax:
        cmd.extend([
            "--estimate-rmax",
            "--rmax-lr", "1e-3",
            "--rmax-log-lower", "0",
            "--rmax-log-upper", "50",
        ])
    if estimate_cv:
        assert cv_init is not None
        cmd.extend([
            "--estimate-data-cv",
            "--data-cv-scope", "global",
            "--data-cv-init", str(cv_init),
            "--data-cv-lower", "0.02",
            "--data-cv-upper", "1.5",
            "--data-cv-lr", "1e-3",
        ])
    if estimate_effort:
        assert effort_init is not None
        cmd.extend([
            "--estimate-effort",
            "--effort-lr", "1e-2",
            "--effort-init", str(effort_init),
        ])
    return cmd


def run_test(
    *,
    name: str,
    kind: str,
    cmd: list[str],
    root: Path,
    log_dir: Path,
    results: list[Result],
    fail_fast: bool,
) -> None:
    log_path = log_dir / f"{name}.log"
    print(f"[RUN ] {name}")
    start = time.perf_counter()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("SMOKE_THREADS", env.get("OMP_NUM_THREADS", "4"))
    env["MKL_NUM_THREADS"] = env.get("SMOKE_THREADS", env.get("MKL_NUM_THREADS", "4"))

    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(subprocess.list2cmdline(cmd))
        log.write("\n\n")
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    seconds = time.perf_counter() - start
    if proc.returncode == 0:
        print(f"[PASS] {name} ({seconds:.1f}s)")
        results.append(Result(name, kind, "PASS", seconds, 0, str(log_path)))
    else:
        print(f"[FAIL] {name} rc={proc.returncode} ({seconds:.1f}s) -> {log_path}")
        results.append(
            Result(
                name, kind, "FAIL", seconds, proc.returncode,
                str(log_path), "Training subprocess returned non-zero status.",
            )
        )
        if fail_fast:
            raise RuntimeError(f"Smoke test failed: {name}")


def main() -> int:
    args = parse_args()
    root = args.project_dir.resolve()
    single_base = abs_path(root, args.single_base_dir)
    ms_base = abs_path(root, args.ms_base_dir)
    data_dir = abs_path(root, args.data_dir)
    final_input_root = abs_path(root, args.final_input_root)
    noise_output_root = abs_path(root, args.noise_output_root)
    manifest_root = abs_path(root, args.manifest_root)

    # Exercise the same parent layout as the production final suite, but isolate
    # smoke-test products so they cannot collide with real array jobs.
    output_dir = manifest_root / "smoke_local"
    log_dir = output_dir / "logs"
    generated_noise = noise_output_root / "smoke_local"
    rmax_root = final_input_root / "smoke_local"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    generated_noise.mkdir(parents=True, exist_ok=True)
    rmax_root.mkdir(parents=True, exist_ok=True)

    os.environ["SMOKE_THREADS"] = str(args.threads)
    torch.set_num_threads(max(1, args.threads))

    results: list[Result] = []
    static_rows: list[dict] = []

    print("=" * 72)
    print("FINAL SUITE LOCAL SMOKE TEST")
    print(f"project_dir={root}")
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"steps={args.steps}, n_time={args.n_time}, n_eval={args.n_eval}")
    print(check_git_version(root))
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Core path and fixture loading
    # ------------------------------------------------------------------
    try:
        ms_params, _, _ = load_params(ms_base)
        if not math.isclose(float(ms_params.t_min), 0.0, abs_tol=1e-12):
            raise ValueError(f"Expected multispecies t_min=0; got {ms_params.t_min}")
        if not math.isclose(float(ms_params.t_max), 40.0, abs_tol=1e-12):
            raise ValueError(f"Expected multispecies t_max=40; got {ms_params.t_max}")
        print("[PASS] multispecies fixture loads; t=0..40")
        results.append(Result("load_multispecies_fixture", "static", "PASS", 0.0))
    except Exception as exc:
        print(f"[FAIL] multispecies fixture load: {exc}")
        results.append(Result("load_multispecies_fixture", "static", "FAIL", 0.0, message=str(exc)))
        pd.DataFrame([asdict(x) for x in results]).to_csv(output_dir / "summary.csv", index=False)
        return 1

    for sp in SINGLE_SPECIES:
        path = single_base / sp
        try:
            load_params(path)
            print(f"[PASS] single fixture loads: {sp}")
            results.append(Result(f"load_{sp}", "static", "PASS", 0.0))
        except Exception as exc:
            print(f"[FAIL] single fixture {sp}: {exc}")
            results.append(Result(f"load_{sp}", "static", "FAIL", 0.0, message=str(exc)))

    # ------------------------------------------------------------------
    # 2. Exhaustive deterministic data loading
    # ------------------------------------------------------------------
    for filename in DATA_FILES:
        path = data_dir / filename
        try:
            stats = validate_observation_file(path, ms_params)
            static_rows.append({"file": filename, **stats})
            print(
                f"[PASS] data {filename}: n={stats['n_obs']}, "
                f"species={stats['n_species']}, years={stats['t_min']}..{stats['t_max']}"
            )
            results.append(Result(f"load_{path.stem}", "static_data", "PASS", 0.0))
        except Exception as exc:
            print(f"[FAIL] data {filename}: {exc}")
            results.append(Result(f"load_{Path(filename).stem}", "static_data", "FAIL", 0.0, message=str(exc)))

    if static_rows:
        pd.DataFrame(static_rows).to_csv(output_dir / "deterministic_data_check.csv", index=False)

    perfect = data_dir / "perfect.csv"

    # ------------------------------------------------------------------
    # 3. Exhaustively generate + load all 25 noise files from final design
    # ------------------------------------------------------------------
    noise_paths: dict[tuple[float, int], Path] = {}
    for cv in NOISE_CVS:
        for noise_seed in NOISE_SEEDS:
            tag = f"cv_{str(cv).replace('.', 'p')}_seed_{noise_seed}"
            out = generated_noise / f"{tag}.csv"
            cmd = [
                sys.executable, "scripts/make_noisy_observations.py",
                "--input", str(perfect),
                "--output", str(out),
                "--cv", str(cv),
                "--seed", str(noise_seed),
                "--task-id", "-1",
                "--overwrite",
            ]
            try:
                rc, text = run_capture(cmd, root)
                if rc != 0:
                    raise RuntimeError(text)
                stats = validate_observation_file(out, ms_params)
                df = pd.read_csv(out)
                if not (df["cv"].astype(float) == cv).all():
                    raise ValueError("Generated CV column does not match requested CV.")
                if not (df["noise_seed"].astype(int) == noise_seed).all():
                    raise ValueError("Generated noise_seed metadata does not match requested seed.")
                noise_paths[(cv, noise_seed)] = out
                results.append(Result(f"noise_generate_{tag}", "static_noise", "PASS", 0.0))
            except Exception as exc:
                print(f"[FAIL] noise generation {tag}: {exc}")
                results.append(Result(f"noise_generate_{tag}", "static_noise", "FAIL", 0.0, message=str(exc)))

    print(f"[INFO] generated/validated {len(noise_paths)}/25 final-design noisy datasets")

    # ------------------------------------------------------------------
    # 4. Static inverse initialisation checks for every start value
    # ------------------------------------------------------------------
    for factor in RMAX_FACTORS:
        try:
            dest = rmax_root / f"factor_{str(factor).replace('.', 'p')}"
            write_scaled_rmax_fixture(ms_base, dest, factor)
            p, _, _ = load_params(dest)
            BoundedLogRMax(p.r_max, lower=0.0, upper=50.0)
            results.append(Result(f"rmax_init_{factor}", "static_inverse", "PASS", 0.0))
        except Exception as exc:
            results.append(Result(f"rmax_init_{factor}", "static_inverse", "FAIL", 0.0, message=str(exc)))

    for start in CV_STARTS:
        try:
            init = torch.tensor([start], dtype=torch.float64)
            BoundedDataCV(init, lower=0.02, upper=1.5, scope="global")
            results.append(Result(f"cv_init_{start}", "static_inverse", "PASS", 0.0))
        except Exception as exc:
            results.append(Result(f"cv_init_{start}", "static_inverse", "FAIL", 0.0, message=str(exc)))

    n_gears = int(ms_params.initial_effort.numel())
    n_effort_times = int(round(float(ms_params.t_max - ms_params.t_min))) + 1
    for start in EFFORT_STARTS:
        try:
            init = torch.full((n_effort_times, n_gears), start, dtype=torch.float64)
            LogFishingEffort(init)
            results.append(Result(f"effort_init_{start}", "static_inverse", "PASS", 0.0))
        except Exception as exc:
            results.append(Result(f"effort_init_{start}", "static_inverse", "FAIL", 0.0, message=str(exc)))

    # ------------------------------------------------------------------
    # 5. Genuine tiny training runs
    # ------------------------------------------------------------------

    # Main single-species fits: all three chosen species.
    for sp in SINGLE_SPECIES:
        run_test(
            name=f"single_{sp}_logu_fourier",
            kind="train_single",
            cmd=single_command(args, single_base / sp, "log-u", "fourier"),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # Remaining state/architecture code paths need only one species to smoke-test.
    for state, arch, tag in (
        ("log-u", "mlp", "logu_mlp"),
        ("log-n", "fourier", "logn_fourier"),
        ("log-n", "mlp", "logn_mlp"),
    ):
        run_test(
            name=f"single_sp_3_{tag}",
            kind="train_single_ablation",
            cmd=single_command(args, single_base / "sp_3", state, arch),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # Multispecies baselines.
    run_test(
        name="ms_no_data",
        kind="train_ms",
        cmd=multispecies_command(
            args, ms_base, data_csv=None, lambda_data=0.0,
        ),
        root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
    )
    run_test(
        name="ms_perfect",
        kind="train_ms",
        cmd=multispecies_command(
            args, ms_base, data_csv=perfect,
        ),
        root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
    )

    # Discrepancy gate: low/high noise ON, and matched CV=0.3 OFF.
    for cv, seed in ((0.1, NOISE_SEEDS[0]), (0.5, NOISE_SEEDS[-1])):
        path = noise_paths.get((cv, seed))
        if path is not None:
            run_test(
                name=f"noise_cv{str(cv).replace('.', 'p')}_gate_on",
                kind="train_noise_gate",
                cmd=multispecies_command(
                    args, ms_base, data_csv=path, data_cv=cv, gate=True,
                ),
                root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
            )

    path_cv03 = noise_paths.get((0.3, NOISE_SEEDS[0]))
    if path_cv03 is not None:
        run_test(
            name="noise_cv0p3_gate_off",
            kind="train_noise_control",
            cmd=multispecies_command(
                args, ms_base, data_csv=path_cv03, data_cv=0.3, gate=False,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # Representative deterministic missingness paths.
    for filename in ("every_3yr.csv", "gap_30_40.csv", "missing_species_11.csv"):
        run_test(
            name=f"data_{Path(filename).stem}",
            kind="train_data_variant",
            cmd=multispecies_command(
                args, ms_base, data_csv=data_dir / filename,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # All three lambda_PDE=0 variants from the final suite.
    run_test(
        name="lambdaPDE0_perfect",
        kind="train_lambda0",
        cmd=multispecies_command(
            args, ms_base, data_csv=perfect, lambda_pde=0.0,
        ),
        root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
    )
    if path_cv03 is not None:
        run_test(
            name="lambdaPDE0_noise_cv0p3",
            kind="train_lambda0",
            cmd=multispecies_command(
                args, ms_base, data_csv=path_cv03, lambda_pde=0.0,
                data_cv=0.3, gate=True,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )
    run_test(
        name="lambdaPDE0_gap_30_40",
        kind="train_lambda0",
        cmd=multispecies_command(
            args, ms_base, data_csv=data_dir / "gap_30_40.csv", lambda_pde=0.0,
        ),
        root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
    )

    # Rmax: test both extreme starting factors plus one noisy inverse case.
    for factor in (0.25, 4.0):
        idir = rmax_root / f"factor_{str(factor).replace('.', 'p')}"
        run_test(
            name=f"rmax_start_{str(factor).replace('.', 'p')}",
            kind="train_rmax",
            cmd=multispecies_command(
                args, idir, data_csv=perfect, estimate_rmax=True,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    if path_cv03 is not None:
        idir = rmax_root / "factor_0p5"
        run_test(
            name="rmax_noise_cv0p3_start0p5",
            kind="train_rmax_noise",
            cmd=multispecies_command(
                args, idir, data_csv=path_cv03, data_cv=0.3,
                gate=True, estimate_rmax=True,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # CV recovery: test both extreme starts; all four were statically validated above.
    for cv_init in (0.1, 1.0):
        if path_cv03 is not None:
            run_test(
                name=f"cv_recovery_start{str(cv_init).replace('.', 'p')}",
                kind="train_cv_inverse",
                cmd=multispecies_command(
                    args, ms_base, data_csv=path_cv03, data_cv=0.3,
                    estimate_cv=True, cv_init=cv_init,
                ),
                root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
            )

    # Effort/F recovery: test both extreme starts. First-step non-zero-gradient
    # assertion in the implementation is therefore exercised.
    for effort_init in (0.05, 3.0):
        run_test(
            name=f"effort_recovery_start{str(effort_init).replace('.', 'p')}",
            kind="train_effort_inverse",
            cmd=multispecies_command(
                args, ms_base, data_csv=perfect,
                estimate_effort=True, effort_init=effort_init,
            ),
            root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
        )

    # Optional fuller dynamic checks for the remaining deterministic/inverse starts.
    if args.full:
        for filename in ("gap_10_20.csv", "missing_species_3.csv", "missing_species_7.csv"):
            run_test(
                name=f"extra_data_{Path(filename).stem}",
                kind="train_extra",
                cmd=multispecies_command(args, ms_base, data_csv=data_dir / filename),
                root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
            )

        for factor in (0.5, 2.0):
            idir = rmax_root / f"factor_{str(factor).replace('.', 'p')}"
            run_test(
                name=f"extra_rmax_start_{str(factor).replace('.', 'p')}",
                kind="train_extra",
                cmd=multispecies_command(
                    args, idir, data_csv=perfect, estimate_rmax=True,
                ),
                root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
            )

        for cv_init in (0.3, 0.6):
            if path_cv03 is not None:
                run_test(
                    name=f"extra_cv_recovery_start{str(cv_init).replace('.', 'p')}",
                    kind="train_extra",
                    cmd=multispecies_command(
                        args, ms_base, data_csv=path_cv03,
                        estimate_cv=True, cv_init=cv_init,
                    ),
                    root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
                )

        for effort_init in (0.5, 1.0):
            run_test(
                name=f"extra_effort_recovery_start{str(effort_init).replace('.', 'p')}",
                kind="train_extra",
                cmd=multispecies_command(
                    args, ms_base, data_csv=perfect,
                    estimate_effort=True, effort_init=effort_init,
                ),
                root=root, log_dir=log_dir, results=results, fail_fast=args.fail_fast,
            )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    summary = pd.DataFrame([asdict(x) for x in results])
    summary.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in results], f, indent=2)

    failed = summary[summary["status"] == "FAIL"] if not summary.empty else summary
    passed = int((summary["status"] == "PASS").sum()) if not summary.empty else 0
    n_failed = int((summary["status"] == "FAIL").sum()) if not summary.empty else 0

    print("=" * 72)
    print(f"SMOKE TEST COMPLETE: {passed} passed, {n_failed} failed")
    print(f"Summary: {output_dir / 'summary.csv'}")
    print(f"Logs:    {log_dir}")
    if n_failed:
        print("\nFAILED TESTS:")
        for _, row in failed.iterrows():
            print(f"  - {row['name']}: {row['message']}  log={row['log']}")
        return 1

    print("All checked paths and representative training branches passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
