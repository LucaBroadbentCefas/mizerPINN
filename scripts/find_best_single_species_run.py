from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_history(path: Path, loss_column: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping unreadable file: {path}\n  {exc}")
        return None

    required = {"step", loss_column}
    if not required.issubset(df.columns):
        print(f"Skipping {path}: missing {required - set(df.columns)}")
        return None

    history = df[["step", loss_column]].copy()
    history["step"] = pd.to_numeric(history["step"], errors="coerce")
    history[loss_column] = pd.to_numeric(
        history[loss_column], errors="coerce"
    )

    history = (
        history.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )

    return history if not history.empty else None


def checkpoint_at_or_before(run_dir: Path, target_step: int) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    candidates: list[tuple[int, Path]] = []

    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("model_step_*.pt"):
            match = re.fullmatch(r"model_step_(\d+)\.pt", path.name)
            if match:
                step = int(match.group(1))
                if step <= target_step:
                    candidates.append((step, path))

    if not candidates:
        return ""

    return str(max(candidates, key=lambda item: item[0])[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank single-species PINN runs at a common iteration."
    )
    parser.add_argument(
        "run_root",
        nargs="?",
        default="runs/pde_only_single_species",
        help="Folder containing the single-species runs.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Exact comparison step. Runs that did not reach it are excluded. "
            "By default, the lowest final step shared by all retained runs is used."
        ),
    )
    parser.add_argument(
        "--min-final-step",
        type=int,
        default=0,
        help="Exclude failed or short runs that stopped before this step.",
    )
    parser.add_argument(
        "--training-loss",
        action="store_true",
        help="Use loss_history.csv/loss instead of fixed-grid diagnostics.",
    )
    parser.add_argument("--top", type=int, default=20)

    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()

    if args.training_loss:
        history_filename = "loss_history.csv"
        loss_column = "loss"
    else:
        history_filename = "fixed_diagnostic_history.csv"
        loss_column = "fixed_loss"

    if not root.exists():
        raise FileNotFoundError(f"Run folder does not exist: {root}")

    histories = []

    for history_path in root.rglob(history_filename):
        history = read_history(history_path, loss_column)
        if history is None:
            continue

        final_step = int(history["step"].max())

        if final_step < args.min_final_step:
            continue

        histories.append(
            {
                "run_dir": history_path.parent,
                "history": history,
                "final_step": final_step,
                "final_loss": float(history.iloc[-1][loss_column]),
            }
        )

    if not histories:
        raise RuntimeError(
            f"No valid {history_filename} files found under {root}"
        )

    if args.step is None:
        comparison_step = min(run["final_step"] for run in histories)
    else:
        comparison_step = args.step
        histories = [
            run for run in histories
            if run["final_step"] >= comparison_step
        ]

    if not histories:
        raise RuntimeError(
            f"No runs reached requested step {comparison_step:,}"
        )

    rows = []

    for run in histories:
        eligible = run["history"].loc[
            run["history"]["step"] <= comparison_step
        ]

        if eligible.empty:
            continue

        selected = eligible.iloc[-1]
        run_dir = run["run_dir"]

        rows.append(
            {
                "run": str(run_dir.relative_to(root)),
                "standardised_loss": float(selected[loss_column]),
                "comparison_step": comparison_step,
                "recorded_step_used": int(selected["step"]),
                "final_step": run["final_step"],
                "final_loss": run["final_loss"],
                "checkpoint_at_or_before": checkpoint_at_or_before(
                    run_dir, comparison_step
                ),
                "model_final": str(run_dir / "model_final.pt")
                if (run_dir / "model_final.pt").exists()
                else "",
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values("standardised_loss")
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    output_path = root / "standardised_loss_ranking.csv"
    ranking.to_csv(output_path, index=False)

    print(f"\nHistory: {history_filename}")
    print(f"Metric: {loss_column}")
    print(f"Comparison step: {comparison_step:,}")
    print(f"Runs ranked: {len(ranking)}\n")

    display_columns = [
        "rank",
        "run",
        "standardised_loss",
        "recorded_step_used",
        "final_step",
    ]

    print(ranking[display_columns].head(args.top).to_string(index=False))

    best = ranking.iloc[0]

    print("\nBEST RUN")
    print(f"Run: {best['run']}")
    print(f"Standardised loss: {best['standardised_loss']:.8g}")
    print(f"Step used: {best['recorded_step_used']:,}")
    print(f"Checkpoint: {best['checkpoint_at_or_before'] or 'not available'}")
    print(f"Final model: {best['model_final'] or 'not available'}")
    print(f"\nFull ranking saved to:\n{output_path}")


if __name__ == "__main__":
    main()
