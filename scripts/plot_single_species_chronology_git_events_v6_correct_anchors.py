from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-07-27-v6-correct-point-and-edge-anchors"

PARAMETERS = [
    "input_dir",
    "n_steps",
    "n_time",
    "n_eval",
    "lr",
    "lr_scheduler",
    "lr_min",
    "model_arch",
    "hidden_width",
    "hidden_layers",
    "fourier_num_features",
    "fourier_scale",
    "fourier_include_raw_input",
    "weight_factorization",
    "rwf_mu",
    "rwf_sigma",
    "rwf_apply_to",
    "rwf_base_init",
    "residual_form",
    "state_parameterization",
    "pde_penalty",
    "boundary_loss_form",
    "bc_penalty",
    "loss_weighting",
    "time_sampling",
    "causal_loss",
    "causal_curriculum",
    "causal_epsilon",
    "causal_n_chunks",
    "collocation_strategy",
    "lambda_pde",
    "lambda_ic",
    "lambda_bc",
    "lambda_timestep",
    "seed",
]

DISPLAY_NAMES = {
    "input_dir": "input data",
    "n_steps": "steps",
    "n_time": "n_time",
    "n_eval": "n_eval",
    "lr": "LR",
    "lr_scheduler": "LR schedule",
    "lr_min": "minimum LR",
    "model_arch": "architecture",
    "hidden_width": "width",
    "hidden_layers": "layers",
    "fourier_num_features": "Fourier features",
    "fourier_scale": "Fourier scale",
    "fourier_include_raw_input": "raw input",
    "weight_factorization": "factorisation",
    "rwf_mu": "RWF mu",
    "rwf_sigma": "RWF sigma",
    "rwf_apply_to": "RWF scope",
    "rwf_base_init": "RWF initialisation",
    "residual_form": "residual",
    "state_parameterization": "state",
    "pde_penalty": "PDE penalty",
    "boundary_loss_form": "BC loss",
    "bc_penalty": "BC penalty",
    "loss_weighting": "loss weighting",
    "time_sampling": "time sampling",
    "causal_loss": "causal loss",
    "causal_curriculum": "curriculum",
    "causal_epsilon": "causal epsilon",
    "causal_n_chunks": "causal chunks",
    "collocation_strategy": "collocation",
    "lambda_pde": "lambda PDE",
    "lambda_ic": "lambda IC",
    "lambda_bc": "lambda BC",
    "lambda_timestep": "lambda timestep",
    "seed": "seed",
}

ALIASES = {
    "learning_rate": "lr",
    "initial_lr": "lr",
}

RUNTIME_KEYS = {
    "run_dir",
    "current_lr",
    "final_lr",
    "actual_total_seconds",
    "seconds_per_step",
    "final_model_path",
    "final_checkpoint_path",
    "loaded_checkpoint_path",
    "note",
}


# Curated repository events that materially change the target problem, loss
# definition, or optimisation pathway. These are deliberately not every commit.
# They are comparability breaks that matter when interpreting the run chronology.
GIT_EVENTS = [
    {
        "date": "2026-05-28",
        "category": "Bug fix",
        "title": "PDE loss masked above species w_max",
        "description": (
            "Inactive size bins above the species maximum weight stopped "
            "contributing to the PDE loss. Losses before and after this fix "
            "are not directly comparable."
        ),
        "sha": "e1d7578",
    },
    {
        "date": "2026-06-01",
        "category": "Input and loss change",
        "title": "New fixture format and recruitment-loss repairs",
        "description": (
            "The single-species fixture/data were replaced or reformatted, "
            "the recruitment loss was repaired, and an implementation syntax/"
            "diagnostic error was corrected. This is a major comparability break."
        ),
        "sha": "d26bf18 / 62840a3 / 85f1167",
    },
    {
        "date": "2026-06-02",
        "category": "Loss-definition change",
        "title": "Relative recruitment boundary loss changed",
        "description": (
            "The relative BC objective was reformulated to score correct "
            "boundary answers more appropriately. The total loss scale and "
            "gradient balance changed."
        ),
        "sha": "8203fc5",
    },
    {
        "date": "2026-06-03",
        "category": "Input and model change",
        "title": "Steady-state fixture and constant recruitment BC introduced",
        "description": (
            "A new steady-state input was added and the boundary condition could "
            "use constant recruitment. A steady-state target is structurally "
            "easier than recovering a changing trajectory."
        ),
        "sha": "8779944",
    },
    {
        "date": "2026-06-04",
        "category": "Input correction",
        "title": "Fixture regenerated to agree with R/mizer plots",
        "description": (
            "Input arrays were updated to make the Python fixture consistent "
            "with the R/mizer comparison plots. This is the strongest Git-history "
            "candidate for the later target-data transition, but the commit text "
            "does not prove whether the replacement was non-steady-state."
        ),
        "sha": "b3de613",
    },
    {
        "date": "2026-06-11",
        "category": "Training-method change",
        "title": "Expert causal loss and gradient-norm weighting added",
        "description": (
            "Introduced the causal PDE and expert gradient-norm weighting used by "
            "the later baseline and ablation runs. This changes optimisation, not "
            "the biological input."
        ),
        "sha": "e17b603 / 800cf0c",
    },
    {
        "date": "2026-06-12",
        "category": "Output/diagnostic change",
        "title": "HPC output mode added",
        "description": (
            "HPC runs began emitting streamlined loss histories and fixed-grid "
            "diagnostics at fixed intervals. This affects which historical runs "
            "can be compared at an exact step, but does not change the PDE."
        ),
        "sha": "91dd703 / 22ae54f",
    },
    {
        "date": "2026-07-13",
        "category": "State-parameterisation change",
        "title": "Optional log-U state introduced",
        "description": (
            "The network could represent N = S U and fit log U rather than direct "
            "log N. This is a major change in state scaling and optimisation geometry."
        ),
        "sha": "6a580c0 / 8091604",
    },
    {
        "date": "2026-07-15",
        "category": "Residual-definition change",
        "title": "Reference-scaled PDE residual introduced",
        "description": (
            "A reference-scale multiplier was added to the PDE residual. Losses "
            "under this formulation have a different mathematical meaning."
        ),
        "sha": "caae922 / 282f154",
    },
    {
        "date": "2026-07-19",
        "category": "Revert",
        "title": "Reference-scaled residual reverted",
        "description": (
            "The reference-scaled residual implementation was removed, restoring "
            "the previous residual definition. Runs from the short interim should "
            "not be pooled with either side without checking their commands."
        ),
        "sha": "d176787",
    },
    {
        "date": "2026-07-21",
        "category": "Bug fix",
        "title": "Adaptive weighting calibration and BC gradients fixed",
        "description": (
            "The biological target in the relative recruitment BC was detached and "
            "expert-weight calibration was corrected. Loss weights and gradients "
            "after this point are not equivalent to earlier runs."
        ),
        "sha": "9000fc2",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a chronological overview of single-species PINN runs from "
            "multiple run roots."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        help=(
            "Run root to scan. Repeat this option for multiple roots. "
            "Defaults to the two known single-species run roots."
        ),
    )
    parser.add_argument(
        "--compare-step",
        type=int,
        default=20000,
        help="Fixed diagnostic step used for standardised comparison.",
    )
    parser.add_argument(
        "--max-step-gap",
        type=int,
        default=0,
        help=(
            "Permit the latest fixed diagnostic at or before the comparison "
            "step when it is no more than this many steps earlier. Default 0 "
            "requires an exact step."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/single_species_chronology_git_v6",
    )
    parser.add_argument(
        "--max-annotation-items",
        type=int,
        default=6,
        help="Maximum changed/varied arguments shown in each daily annotation.",
    )
    parser.add_argument(
        "--no-git-events",
        action="store_true",
        help="Do not draw curated repository comparability events.",
    )
    return parser.parse_args()


def parse_run_datetime(name: str) -> datetime | None:
    match = re.search(r"(\d{8})_(\d{6})(?:_(\d{6}))?", name)
    if not match:
        return None

    date_part, time_part, microseconds = match.groups()
    microseconds = microseconds or "000000"

    try:
        return datetime.strptime(
            f"{date_part}_{time_part}_{microseconds}",
            "%Y%m%d_%H%M%S_%f",
        )
    except ValueError:
        return None


def coerce_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip().strip('"').strip("'")
    lower = text.lower()

    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if lower in {"none", "null", "nan", ""}:
        return None

    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def parse_run_command(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    clean = text.replace("^", " ").replace("\\\n", " ").replace("\n", " ")

    try:
        tokens = shlex.split(clean, posix=False)
    except ValueError:
        tokens = clean.split()

    result: dict[str, Any] = {}
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue

        key = token[2:].replace("-", "_")

        if key.startswith("no_"):
            result[key[3:]] = False
            i += 1
            continue

        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            result[key] = coerce_scalar(tokens[i + 1])
            i += 2
        else:
            result[key] = True
            i += 1

    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in config.items():
        key = key.replace("-", "_")
        key = ALIASES.get(key, key)
        if key in RUNTIME_KEYS:
            continue
        result[key] = coerce_scalar(value)

    return result


def detect_source(run_dir: Path) -> str:
    upper_parts = {part.upper() for part in run_dir.parts}
    if "ABLATION_RUNS" in upper_parts:
        return "Ablation runs"
    return "Development runs"


def discover_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: set[Path] = set()

    marker_files = [
        "config.json",
        "run_command.txt",
        "fixed_diagnostic_history.csv",
        "loss_history.csv",
        "final_summary.json",
    ]

    for root in roots:
        if not root.exists():
            print(f"Warning: run root does not exist: {root}")
            continue

        for marker in marker_files:
            for path in root.rglob(marker):
                run_dirs.add(path.parent.resolve())

    return sorted(
        run_dirs,
        key=lambda path: parse_run_datetime(path.name) or datetime.max,
    )


def read_fixed_history(run_dir: Path) -> pd.DataFrame | None:
    candidates = sorted(run_dir.glob("fixed_diagnostic_history*.csv"))
    if not candidates:
        return None

    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not {"step", "fixed_loss"}.issubset(df.columns):
            continue

        history = df[["step", "fixed_loss"]].copy()
        history["step"] = pd.to_numeric(history["step"], errors="coerce")
        history["fixed_loss"] = pd.to_numeric(
            history["fixed_loss"], errors="coerce"
        )
        history = (
            history.replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values("step")
            .drop_duplicates("step", keep="last")
        )

        if not history.empty:
            return history

    return None


def select_standardised_loss(
    history: pd.DataFrame | None,
    compare_step: int,
    max_step_gap: int,
) -> tuple[float, float]:
    if history is None or history.empty:
        return math.nan, math.nan

    exact = history.loc[history["step"] == compare_step]
    if not exact.empty:
        row = exact.iloc[-1]
        return float(row["fixed_loss"]), float(row["step"])

    earlier = history.loc[history["step"] < compare_step]
    if earlier.empty:
        return math.nan, math.nan

    row = earlier.iloc[-1]
    gap = compare_step - int(row["step"])

    if gap <= max_step_gap:
        return float(row["fixed_loss"]), float(row["step"])

    return math.nan, math.nan


def build_run_table(
    run_dirs: list[Path],
    compare_step: int,
    max_step_gap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_time = parse_run_datetime(run_dir.name)
        if run_time is None:
            print(f"Skipping folder without timestamp prefix: {run_dir}")
            continue

        command_config = parse_run_command(run_dir / "run_command.txt")
        saved_config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "final_summary.json")

        effective = normalise_config(command_config)
        effective.update(normalise_config(saved_config))

        history = read_fixed_history(run_dir)
        standardised_loss, step_used = select_standardised_loss(
            history,
            compare_step,
            max_step_gap,
        )

        final_fixed_step = math.nan
        final_fixed_loss = math.nan
        if history is not None and not history.empty:
            final_fixed_step = float(history.iloc[-1]["step"])
            final_fixed_loss = float(history.iloc[-1]["fixed_loss"])

        n_steps_completed = summary.get("n_steps_completed")
        if n_steps_completed is None:
            n_steps_completed = final_fixed_step

        row: dict[str, Any] = {
            "run_time": run_time,
            "date": run_time.date().isoformat(),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "source": detect_source(run_dir),
            "task_id": extract_task_id(run_dir.name),
            "status": summary.get("status", "unknown"),
            "compare_step": compare_step,
            "standardised_step_used": step_used,
            "standardised_loss": standardised_loss,
            "eligible_standardised": math.isfinite(standardised_loss),
            "final_fixed_step": final_fixed_step,
            "final_fixed_loss": final_fixed_loss,
            "n_steps_completed": coerce_scalar(n_steps_completed),
        }

        for parameter in PARAMETERS:
            row[parameter] = effective.get(parameter)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("run_time").reset_index(drop=True)


def extract_task_id(name: str) -> int | None:
    match = re.search(r"task(\d+)", name)
    return int(match.group(1)) if match else None


def value_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<missing>"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, np.integer)):
        if abs(int(value)) >= 1000:
            return f"{int(value):,}"
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)


def mode_value(series: pd.Series) -> Any:
    values = [value for value in series.tolist() if value_key(value) != "<missing>"]
    if not values:
        return None

    counts: dict[str, int] = {}
    representatives: dict[str, Any] = {}
    for value in values:
        key = value_key(value)
        counts[key] = counts.get(key, 0) + 1
        representatives[key] = value

    best_key = max(counts, key=lambda key: (counts[key], key))
    return representatives[best_key]


def unique_nonmissing(series: pd.Series) -> list[Any]:
    seen: dict[str, Any] = {}
    for value in series.tolist():
        key = value_key(value)
        if key == "<missing>":
            continue
        seen[key] = value
    return list(seen.values())


def summarise_values(values: list[Any], max_values: int = 5) -> str:
    if not values:
        return ""

    numeric = all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        for value in values
    )

    if numeric and len(values) > max_values:
        numeric_values = sorted(float(value) for value in values)
        return f"{format_value(numeric_values[0])}–{format_value(numeric_values[-1])}"

    formatted = [format_value(value) for value in values]
    formatted = sorted(formatted)

    if len(formatted) <= max_values:
        return ", ".join(formatted)

    return ", ".join(formatted[:max_values]) + f" +{len(formatted) - max_values}"


def compact_items(items: list[str], maximum: int) -> str:
    if not items:
        return "none"
    if len(items) <= maximum:
        return "; ".join(items)
    return "; ".join(items[:maximum]) + f"; +{len(items) - maximum} more"


def build_daily_table(
    runs: pd.DataFrame,
    max_annotation_items: int,
) -> pd.DataFrame:
    daily_rows: list[dict[str, Any]] = []
    previous_modes: dict[str, Any] | None = None

    for date, day in runs.groupby("date", sort=True):
        day = day.sort_values("run_time")
        eligible = day.loc[day["eligible_standardised"]]

        modes = {parameter: mode_value(day[parameter]) for parameter in PARAMETERS}

        varied_items = []
        for parameter in PARAMETERS:
            values = unique_nonmissing(day[parameter])
            if len(values) > 1:
                varied_items.append(
                    f"{DISPLAY_NAMES[parameter]}: {summarise_values(values)}"
                )

        change_items = []
        if previous_modes is not None:
            for parameter in PARAMETERS:
                old = previous_modes.get(parameter)
                new = modes.get(parameter)

                if value_key(old) == "<missing>" or value_key(new) == "<missing>":
                    continue
                if value_key(old) != value_key(new):
                    change_items.append(
                        f"{DISPLAY_NAMES[parameter]} {format_value(old)}→{format_value(new)}"
                    )

        if eligible.empty:
            best_loss = math.nan
            median_loss = math.nan
            best_run = ""
        else:
            best_index = eligible["standardised_loss"].idxmin()
            best_loss = float(eligible.loc[best_index, "standardised_loss"])
            median_loss = float(eligible["standardised_loss"].median())
            best_run = str(eligible.loc[best_index, "run_name"])

        n_steps_values = unique_nonmissing(day["n_steps"])

        row: dict[str, Any] = {
            "date": date,
            "date_dt": pd.Timestamp(date),
            "n_runs": len(day),
            "n_eligible": int(day["eligible_standardised"].sum()),
            "best_standardised_loss": best_loss,
            "median_standardised_loss": median_loss,
            "best_run": best_run,
            "n_steps_tested": summarise_values(n_steps_values),
            "changes_from_previous_day": compact_items(
                change_items,
                max_annotation_items,
            ),
            "varied_within_day": compact_items(
                varied_items,
                max_annotation_items,
            ),
        }

        for parameter in PARAMETERS:
            row[f"mode_{parameter}"] = modes[parameter]

        daily_rows.append(row)
        previous_modes = modes

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily["cumulative_best_loss"] = daily["best_standardised_loss"].cummin()
    return daily


def annotate_n_steps_changes(ax: plt.Axes, daily: pd.DataFrame) -> None:
    previous = None

    for _, row in daily.iterrows():
        current = row.get("mode_n_steps")
        if value_key(current) == "<missing>":
            continue

        if previous is not None and value_key(previous) != value_key(current):
            when = pd.Timestamp(row["date_dt"])
            ax.axvline(when, linestyle="--", linewidth=1.0, alpha=0.55)
            ax.text(
                when,
                0.98,
                f"steps {format_value(previous)}→{format_value(current)}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
            )

        previous = current


def git_events_table() -> pd.DataFrame:
    events = pd.DataFrame(GIT_EVENTS)
    if events.empty:
        return events
    events["date_dt"] = pd.to_datetime(events["date"])
    events["event_id"] = [f"G{i + 1}" for i in range(len(events))]
    return events


def _build_chronology_cards(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Build one annotation card for each relevant calendar date.

    Daily run summaries and Git events occurring on the same date are combined
    into one card. This avoids multiple boxes competing for the same x-position.
    """
    daily_lookup = {
        str(row["date"]): row
        for _, row in daily.iterrows()
    }

    if events.empty:
        event_lookup: dict[str, list[pd.Series]] = {}
    else:
        in_range = events.loc[
            (events["date_dt"] >= start.normalize())
            & (events["date_dt"] <= end.normalize())
        ]
        event_lookup = {
            date: [row for _, row in group.iterrows()]
            for date, group in in_range.groupby("date", sort=True)
        }

    dates = sorted(set(daily_lookup) | set(event_lookup))
    cards: list[dict[str, Any]] = []

    for date in dates:
        day = daily_lookup.get(date)
        day_events = event_lookup.get(date, [])
        body_lines: list[str] = []

        if day is not None:
            best = day["best_standardised_loss"]
            best_text = f"{best:.2e}" if math.isfinite(best) else "not comparable"
            body_lines.append(
                f"{int(day['n_runs'])} runs; {int(day['n_eligible'])} comparable; "
                f"daily best {best_text}."
            )

            change_text = str(day["changes_from_previous_day"])
            if not cards:
                change_text = "Initial recorded configuration"
            body_lines.append(f"Changed: {change_text}.")
            body_lines.append(f"Tested: {day['varied_within_day']}.")
        else:
            body_lines.append("Repository event; no timestamped run folder on this date.")

        for event in day_events:
            body_lines.append(
                f"{event['event_id']} {event['category']}: {event['title']}. "
                f"{event['description']} [{event['sha']}]"
            )

        wrapped_lines: list[str] = []
        for line in body_lines:
            wrapped_lines.extend(textwrap.wrap(line, width=62) or [""])

        cards.append(
            {
                "date": date,
                "date_dt": pd.Timestamp(date),
                "title": pd.Timestamp(date).strftime("%d %b %Y"),
                "body": "\n".join(wrapped_lines),
                "line_count": max(1, len(wrapped_lines)),
                "has_event": bool(day_events),
                "has_daily_point": day is not None and math.isfinite(
                    float(day["best_standardised_loss"])
                ),
                "daily_best": (
                    float(day["best_standardised_loss"])
                    if day is not None and math.isfinite(float(day["best_standardised_loss"]))
                    else math.nan
                ),
            }
        )

    return cards


def _bbox_intersects_any(
    bbox,
    placed_bboxes: list,
    padding_pixels: float = 7.0,
) -> bool:
    padded = bbox.expanded(
        (bbox.width + 2.0 * padding_pixels) / max(bbox.width, 1.0),
        (bbox.height + 2.0 * padding_pixels) / max(bbox.height, 1.0),
    )
    return any(padded.overlaps(existing) for existing in placed_bboxes)


def _place_collision_aware_cards(
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    cards: list[dict[str, Any]],
    daily: pd.DataFrame,
) -> None:
    """Place cards outside the axes and connect them to truthful anchors.

    Cards with a finite standardised daily-best loss connect to that exact plotted
    point. Cards without a comparable daily point do not use an invented fallback
    y-value; instead, they connect to the top or bottom axes boundary at their date.
    """
    daily_loss = {
        str(row["date"]): float(row["best_standardised_loss"])
        for _, row in daily.iterrows()
        if math.isfinite(float(row["best_standardised_loss"]))
    }

    ordered_cards = sorted(cards, key=lambda item: item["date_dt"])

    # Measure all boxes once. Their final positions are calculated in display
    # pixels, then rendered using exact data/axes anchors.
    measurement_artists = []
    for card in ordered_cards:
        artist = ax.annotate(
            f"{card['title']}\n{card['body']}",
            xy=(0, 0),
            xycoords="axes pixels",
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.3,
            linespacing=1.22,
            annotation_clip=False,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "0.965" if card["has_event"] else "white",
                "edgecolor": "0.30" if card["has_event"] else "0.58",
                "linewidth": 1.0,
            },
            alpha=0.0,
        )
        measurement_artists.append(artist)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    measured_sizes = []
    for artist in measurement_artists:
        bbox = artist.get_window_extent(renderer=renderer)
        measured_sizes.append((bbox.width, bbox.height))
        artist.remove()

    placed_bboxes: list[Bbox] = []
    figure_bbox = fig.bbox
    axes_bbox = ax.get_window_extent(renderer=renderer)

    figure_margin = 14.0
    axes_gap = 22.0
    collision_gap = 8.0
    lane_step = 24.0
    points_to_pixels = fig.dpi / 72.0
    x_start, x_end = ax.get_xlim()

    for index, (card, (box_width, box_height)) in enumerate(
        zip(ordered_cards, measured_sizes)
    ):
        event_x = pd.Timestamp(card["date_dt"]) + pd.Timedelta(hours=12)
        x_num = mdates.date2num(event_x.to_pydatetime())

        # Only finite daily-best values are data-point anchors.
        has_point_anchor = card["date"] in daily_loss
        point_anchor_y = daily_loss.get(card["date"], math.nan)

        # x display coordinate is independent of which y anchor will be used.
        anchor_x_px = ax.transData.transform((x_num, ax.get_ylim()[0]))[0]
        x_fraction = (x_num - x_start) / max(x_end - x_start, 1e-12)

        if x_fraction < 0.72:
            dx_points = 18.0
            horizontal_alignment = "left"
        else:
            dx_points = -18.0
            horizontal_alignment = "right"

        dx_px = dx_points * points_to_pixels
        if horizontal_alignment == "left":
            x0 = anchor_x_px + dx_px
            x1 = x0 + box_width
        else:
            x1 = anchor_x_px + dx_px
            x0 = x1 - box_width

        preferred_side = "above" if index % 2 == 0 else "below"
        sides = [preferred_side, "below" if preferred_side == "above" else "above"]
        chosen = None

        for side in sides:
            # Exact connector endpoint in display pixels.
            if has_point_anchor:
                anchor_y_px = ax.transData.transform((x_num, point_anchor_y))[1]
            else:
                # No comparable point exists. Anchor to the relevant plot border
                # rather than fabricating a y-value near the geometric mean.
                anchor_y_px = axes_bbox.y1 if side == "above" else axes_bbox.y0

            for level in range(0, 100):
                lane_offset = level * lane_step

                if side == "above":
                    y0 = axes_bbox.y1 + axes_gap + lane_offset
                    y1 = y0 + box_height
                    vertical_alignment = "bottom"
                    box_edge_y_px = y0
                else:
                    y1 = axes_bbox.y0 - axes_gap - lane_offset
                    y0 = y1 - box_height
                    vertical_alignment = "top"
                    box_edge_y_px = y1

                bbox = Bbox.from_extents(x0, y0, x1, y1)
                padded = Bbox.from_extents(
                    x0 - collision_gap,
                    y0 - collision_gap,
                    x1 + collision_gap,
                    y1 + collision_gap,
                )

                inside_figure = (
                    bbox.x0 >= figure_bbox.x0 + figure_margin
                    and bbox.x1 <= figure_bbox.x1 - figure_margin
                    and bbox.y0 >= figure_bbox.y0 + figure_margin
                    and bbox.y1 <= figure_bbox.y1 - figure_margin
                )
                outside_plot = not padded.overlaps(axes_bbox)
                collision = any(
                    padded.overlaps(existing)
                    for existing in placed_bboxes
                )

                if inside_figure and outside_plot and not collision:
                    dy_px = box_edge_y_px - anchor_y_px
                    chosen = {
                        "dx_points": dx_points,
                        "dy_points": dy_px / points_to_pixels,
                        "ha": horizontal_alignment,
                        "va": vertical_alignment,
                        "placed_bbox": padded,
                        "side": side,
                        "has_point_anchor": has_point_anchor,
                        "point_anchor_y": point_anchor_y,
                    }
                    break

            if chosen is not None:
                break

        if chosen is None:
            raise RuntimeError(
                "Could not place all chronology cards outside the plotting axes. "
                "Increase figure_height or reduce annotation text."
            )

        placed_bboxes.append(chosen["placed_bbox"])

        if chosen["has_point_anchor"]:
            xy = (event_x, chosen["point_anchor_y"])
            xycoords = "data"
            connector_alpha = 0.68
            connector_style = "-"
        else:
            # A repository-only or non-comparable day is attached to the plot
            # boundary at its date. This communicates chronology without implying
            # a loss value that was never observed.
            xy = (event_x, 1.0 if chosen["side"] == "above" else 0.0)
            xycoords = ("data", "axes fraction")
            connector_alpha = 0.48
            connector_style = ":"

        ax.annotate(
            f"{card['title']}\n{card['body']}",
            xy=xy,
            xycoords=xycoords,
            xytext=(chosen["dx_points"], chosen["dy_points"]),
            textcoords="offset points",
            ha=chosen["ha"],
            va=chosen["va"],
            fontsize=8.3,
            linespacing=1.22,
            annotation_clip=False,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "0.965" if card["has_event"] else "white",
                "edgecolor": "0.30" if card["has_event"] else "0.58",
                "linewidth": 1.0,
                "alpha": 0.98,
            },
            arrowprops={
                "arrowstyle": "-",
                "linestyle": connector_style,
                "linewidth": 0.9,
                "alpha": connector_alpha,
                "shrinkA": 4,
                "shrinkB": 0,
                "connectionstyle": "angle3",
            },
            zorder=20,
        )

def plot_chronology(
    runs: pd.DataFrame,
    daily: pd.DataFrame,
    compare_step: int,
    output_path: Path,
    git_events: pd.DataFrame | None = None,
) -> None:
    if runs.empty or daily.empty:
        raise RuntimeError("No run chronology is available to plot.")

    start = pd.Timestamp(runs["run_time"].min()).normalize() - pd.Timedelta(hours=8)
    end = pd.Timestamp(runs["run_time"].max()).normalize() + pd.Timedelta(days=1, hours=8)
    events = git_events if git_events is not None else pd.DataFrame()
    cards = _build_chronology_cards(daily, events, start, end)

    # Keep a fixed-height central plotting panel and create large annotation
    # lanes above and below it. Figure height increases with the number of cards.
    figure_width = 24.0
    figure_height = max(16.0, 10.0 + 0.72 * len(cards))
    central_axes_height_inches = 6.2
    axes_height_fraction = central_axes_height_inches / figure_height
    axes_bottom = (1.0 - axes_height_fraction) / 2.0

    fig = plt.figure(figsize=(figure_width, figure_height))
    ax_loss = fig.add_axes([0.07, axes_bottom, 0.88, axes_height_fraction])

    fig.suptitle(
        "Single-species PINN development chronology",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    eligible = runs.loc[runs["eligible_standardised"]].copy()
    source_markers = {
        "Development runs": "o",
        "Ablation runs": "s",
    }

    for source, group in eligible.groupby("source"):
        ax_loss.scatter(
            group["run_time"],
            group["standardised_loss"],
            marker=source_markers.get(source, "o"),
            s=48,
            alpha=0.62,
            label=f"{source}: individual runs",
            zorder=3,
        )

    valid_daily = daily.loc[np.isfinite(daily["best_standardised_loss"])].copy()
    daily_x = valid_daily["date_dt"] + pd.to_timedelta(12, unit="h")

    ax_loss.plot(
        daily_x,
        valid_daily["best_standardised_loss"],
        marker="o",
        linewidth=2.4,
        label="Best run on each day",
        zorder=5,
    )
    ax_loss.plot(
        daily_x,
        valid_daily["cumulative_best_loss"],
        linestyle="--",
        linewidth=2.0,
        label="Best achieved so far",
        zorder=4,
    )

    annotate_n_steps_changes(ax_loss, daily)

    if not events.empty:
        visible_events = events.loc[
            (events["date_dt"] >= start.normalize())
            & (events["date_dt"] <= end.normalize())
        ].copy()
        for _, event in visible_events.iterrows():
            event_x = pd.Timestamp(event["date_dt"]) + pd.Timedelta(hours=12)
            ax_loss.axvline(
                event_x,
                linestyle=":",
                linewidth=1.3,
                alpha=0.50,
                zorder=1,
            )

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel(f"Fixed diagnostic loss at step {compare_step:,}")
    ax_loss.set_xlabel("Run initialisation date")
    ax_loss.grid(True, which="both", alpha=0.25)
    ax_loss.legend(loc="best")

    n_days = len(daily)
    locator = mdates.AutoDateLocator(
        minticks=min(4, n_days),
        maxticks=max(8, n_days + 2),
    )
    ax_loss.xaxis.set_major_locator(locator)
    ax_loss.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_loss.set_xlim(start, end)

    # Draw once so transforms and renderer dimensions are final before the
    # collision-aware placement algorithm evaluates bounding boxes.
    fig.canvas.draw()
    _place_collision_aware_cards(
        fig=fig,
        ax=ax_loss,
        cards=cards,
        daily=daily,
    )

    fig.text(
        0.5,
        0.012,
        (
            "Cards with a comparable daily loss connect to that exact plotted point. "
            "Cards without a comparable point connect to the plot boundary at their "
            "date, so no artificial y-value is implied. All cards remain in reserved "
            "white space above or below the loss axes."
        ),
        ha="center",
        fontsize=9,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_daily_argument_table(daily: pd.DataFrame, output_path: Path) -> None:
    table = daily[
        [
            "date",
            "n_runs",
            "n_eligible",
            "best_standardised_loss",
            "n_steps_tested",
            "changes_from_previous_day",
            "varied_within_day",
        ]
    ].copy()

    table["best_standardised_loss"] = table["best_standardised_loss"].map(
        lambda value: f"{value:.3e}" if math.isfinite(value) else "not comparable"
    )

    table.columns = [
        "Date",
        "Runs",
        "Comparable",
        "Best fixed loss",
        "Steps tested",
        "Shift from previous day",
        "Arguments varied within day",
    ]

    height = max(5, 0.8 * len(table) + 1.5)
    fig, ax = plt.subplots(figsize=(20, height))
    ax.axis("off")

    rendered = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.07, 0.045, 0.065, 0.085, 0.08, 0.31, 0.34],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1, 2.0)

    for column in range(len(table.columns)):
        rendered[(0, column)].set_text_props(weight="bold")
        rendered[(0, column)].set_facecolor("0.9")

    ax.set_title(
        "Daily experiment chronology and argument changes",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Running script: {Path(__file__).resolve()}")
    print(f"Script version: {SCRIPT_VERSION}")
    args = parse_args()

    roots = args.run_roots or [
        "runs/pde_only_single_species",
        "runs/ABLATION_RUNS/pde_only_single_species",
    ]
    roots = [Path(root).expanduser().resolve() for root in roots]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SCRIPT_VERSION.txt").write_text(SCRIPT_VERSION + "\n", encoding="utf-8")

    run_dirs = discover_run_dirs(roots)
    print(f"Run folders discovered: {len(run_dirs)}")

    runs = build_run_table(
        run_dirs,
        compare_step=args.compare_step,
        max_step_gap=args.max_step_gap,
    )

    if runs.empty:
        raise RuntimeError("No timestamped runs with readable metadata were found.")

    daily = build_daily_table(
        runs,
        max_annotation_items=args.max_annotation_items,
    )

    runs.to_csv(output_dir / "chronology_runs.csv", index=False)
    daily.to_csv(output_dir / "chronology_days.csv", index=False)

    events = pd.DataFrame() if args.no_git_events else git_events_table()
    if not events.empty:
        events.to_csv(output_dir / "03_curated_git_events.csv", index=False)

    plot_chronology(
        runs,
        daily,
        compare_step=args.compare_step,
        output_path=output_dir / "01_single_species_chronology.png",
        git_events=events,
    )
    plot_daily_argument_table(
        daily,
        output_path=output_dir / "02_daily_argument_summary.png",
    )

    print(f"Runs included in chronology: {len(runs)}")
    print(
        f"Runs comparable at step {args.compare_step:,}: "
        f"{int(runs['eligible_standardised'].sum())}"
    )
    print(f"Days represented: {len(daily)}")
    print("\nDaily summary:")
    print(
        daily[
            [
                "date",
                "n_runs",
                "n_eligible",
                "best_standardised_loss",
                "n_steps_tested",
                "changes_from_previous_day",
                "varied_within_day",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs saved to:\n{output_dir}")


if __name__ == "__main__":
    main()
