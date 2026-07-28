from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Ablation design from your slurm script
# ---------------------------------------------------------------------

BASELINE = {
    "scheme_name": "baseline",
    "group": "baseline",
    "time_sampling": "stratified",
    "causal_loss": "expert",
    "causal_curriculum": "linear",
    "causal_epsilon": 1.0,
    "causal_n_chunks": 32,
    "lr": 3e-4,
    "n_time": 36,
    "n_eval": 30,
}

SCHEME_OVERRIDES = {
    0:  {"scheme_name": "baseline", "group": "baseline"},
    1:  {"scheme_name": "no_causal_control", "group": "control", "time_sampling": "uniform", "causal_loss": "off", "causal_curriculum": "off"},
    2:  {"scheme_name": "linear_curriculum_only", "group": "control", "time_sampling": "uniform", "causal_loss": "off", "causal_curriculum": "linear"},
    3:  {"scheme_name": "step_curriculum_only", "group": "control", "time_sampling": "uniform", "causal_loss": "off", "causal_curriculum": "step"},
    4:  {"scheme_name": "expert_causal_only", "group": "control", "time_sampling": "stratified", "causal_loss": "expert", "causal_curriculum": "off"},
    5:  {"scheme_name": "eps_0p01", "group": "epsilon", "causal_epsilon": 0.01},
    6:  {"scheme_name": "eps_0p05", "group": "epsilon", "causal_epsilon": 0.05},
    7:  {"scheme_name": "eps_0p1", "group": "epsilon", "causal_epsilon": 0.1},
    8:  {"scheme_name": "eps_0p5", "group": "epsilon", "causal_epsilon": 0.5},
    9:  {"scheme_name": "eps_2", "group": "epsilon", "causal_epsilon": 2.0},
    10: {"scheme_name": "eps_5", "group": "epsilon", "causal_epsilon": 5.0},
    11: {"scheme_name": "eps_10", "group": "epsilon", "causal_epsilon": 10.0},
    12: {"scheme_name": "chunks_4", "group": "chunks", "causal_n_chunks": 4},
    13: {"scheme_name": "chunks_8", "group": "chunks", "causal_n_chunks": 8},
    14: {"scheme_name": "chunks_16", "group": "chunks", "causal_n_chunks": 16},
    15: {"scheme_name": "chunks_64", "group": "chunks", "causal_n_chunks": 64},
    16: {"scheme_name": "chunks_128", "group": "chunks", "causal_n_chunks": 128},
    17: {"scheme_name": "lr_3e-5", "group": "lr", "lr": 3e-5},
    18: {"scheme_name": "lr_1e-4", "group": "lr", "lr": 1e-4},
    19: {"scheme_name": "lr_1e-3", "group": "lr", "lr": 1e-3},
    20: {"scheme_name": "lr_3e-3", "group": "lr", "lr": 3e-3},
    21: {"scheme_name": "ntime_32_neval_30", "group": "time_eval", "n_time": 32, "n_eval": 30},
    22: {"scheme_name": "ntime_96_neval_30", "group": "time_eval", "n_time": 96, "n_eval": 30},
    23: {"scheme_name": "ntime_128_neval_30", "group": "time_eval", "n_time": 128, "n_eval": 30},
    24: {"scheme_name": "ntime_36_neval_15", "group": "time_eval", "n_time": 36, "n_eval": 15},
    25: {"scheme_name": "ntime_36_neval_60", "group": "time_eval", "n_time": 36, "n_eval": 60},
    26: {"scheme_name": "ntime_96_neval_60", "group": "time_eval", "n_time": 96, "n_eval": 60},
    27: {"scheme_name": "ntime_128_neval_60", "group": "time_eval", "n_time": 128, "n_eval": 60},
}

GROUP_ORDER = ["baseline", "control", "epsilon", "chunks", "lr", "time_eval"]

GROUP_COLORS = {
    "baseline": "black",
    "control": "tab:blue",
    "epsilon": "tab:orange",
    "chunks": "tab:green",
    "lr": "tab:red",
    "time_eval": "tab:purple",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lookup = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")
    return None


def build_scheme_table() -> pd.DataFrame:
    rows = []
    for task_id in sorted(SCHEME_OVERRIDES):
        cfg = BASELINE.copy()
        cfg.update(SCHEME_OVERRIDES[task_id])
        cfg["task_id"] = task_id
        rows.append(cfg)
    return pd.DataFrame(rows)


def parse_task_id_from_run_name(name: str) -> int | None:
    m = re.search(r"task(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"wave1_(\d+)_", name)
    if m:
        return int(m.group(1))
    return None


def read_history(path: Path, loss_col: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping unreadable history {path}: {exc}")
        return None

    if "step" not in df.columns or loss_col not in df.columns:
        return None

    out = df[["step", loss_col]].copy()
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out[loss_col] = pd.to_numeric(out[loss_col], errors="coerce")
    out = (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )
    if out.empty:
        return None
    return out


def scan_runs(runs_root: Path, min_final_step: int = 0) -> list[dict]:
    records = []

    scheme_table = build_scheme_table().set_index("task_id")

    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        task_id = parse_task_id_from_run_name(run_dir.name)

        history = None
        history_file = None
        loss_col = None

        fixed_path = run_dir / "fixed_diagnostic_history.csv"
        raw_path = run_dir / "loss_history.csv"

        if fixed_path.exists():
            history = read_history(fixed_path, "fixed_loss")
            history_file = fixed_path
            loss_col = "fixed_loss"
        elif raw_path.exists():
            history = read_history(raw_path, "loss")
            history_file = raw_path
            loss_col = "loss"

        if history is None:
            continue

        final_step = int(history["step"].max())
        if final_step < min_final_step:
            continue

        meta = {
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "task_id": task_id,
            "history": history,
            "history_file": history_file,
            "loss_col": loss_col,
            "final_step": final_step,
            "final_loss": float(history.iloc[-1][loss_col]),
        }

        if task_id is not None and task_id in scheme_table.index:
            row = scheme_table.loc[task_id].to_dict()
            meta.update(row)
        else:
            meta.update({
                "scheme_name": run_dir.name,
                "group": "unknown",
            })

        records.append(meta)

    return records


def rank_runs(records: list[dict], compare_step: int | None = None) -> tuple[pd.DataFrame, int]:
    if not records:
        raise RuntimeError("No valid runs found.")

    if compare_step is None:
        compare_step = min(r["final_step"] for r in records)

    rows = []
    for r in records:
        hist = r["history"]
        eligible = hist.loc[hist["step"] <= compare_step]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        rows.append({
            "run_name": r["run_name"],
            "run_dir": str(r["run_dir"]),
            "task_id": r.get("task_id"),
            "scheme_name": r.get("scheme_name"),
            "group": r.get("group"),
            "history_metric": r["loss_col"],
            "comparison_step": compare_step,
            "recorded_step_used": int(row["step"]),
            "standardised_loss": float(row[r["loss_col"]]),
            "final_step": r["final_step"],
            "final_loss": r["final_loss"],
        })

    ranking = pd.DataFrame(rows).sort_values("standardised_loss").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking, compare_step


# ---------------------------------------------------------------------
# Plot 1: ablation overview figure
# ---------------------------------------------------------------------

def display_value(row: dict, key: str):
    baseline_value = BASELINE[key]
    value = row[key]

    if row["scheme_name"] == "baseline":
        return str(value)

    if value == baseline_value:
        return "—"

    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def plot_ablation_overview(records: list[dict], out_path: Path):
    scheme_df = build_scheme_table()

    present_task_ids = sorted([r["task_id"] for r in records if r.get("task_id") is not None])
    scheme_df = scheme_df[scheme_df["task_id"].isin(present_task_ids)].copy()

    if scheme_df.empty:
        print("Skipping ablation overview: no task ids detected.")
        return

    scheme_df["group_rank"] = scheme_df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    scheme_df = scheme_df.sort_values(["group_rank", "task_id"]).reset_index(drop=True)

    table_df = pd.DataFrame({
        "Task": scheme_df["task_id"],
        "Scheme": scheme_df["scheme_name"],
        "Family": scheme_df["group"],
        "Time sampling": [display_value(r, "time_sampling") for _, r in scheme_df.iterrows()],
        "Causal loss": [display_value(r, "causal_loss") for _, r in scheme_df.iterrows()],
        "Curriculum": [display_value(r, "causal_curriculum") for _, r in scheme_df.iterrows()],
        "ε": [display_value(r, "causal_epsilon") for _, r in scheme_df.iterrows()],
        "Chunks": [display_value(r, "causal_n_chunks") for _, r in scheme_df.iterrows()],
        "LR": [display_value(r, "lr") for _, r in scheme_df.iterrows()],
        "N time": [display_value(r, "n_time") for _, r in scheme_df.iterrows()],
        "N eval": [display_value(r, "n_eval") for _, r in scheme_df.iterrows()],
    })

    group_counts = scheme_df["group"].value_counts()
    summary_lines = [
        f"Runs found: {len(scheme_df)}",
        "",
        "Families:",
        f"- baseline: {group_counts.get('baseline', 0)}",
        f"- control toggles: {group_counts.get('control', 0)}",
        f"- epsilon sweep: {group_counts.get('epsilon', 0)}",
        f"- chunk sweep: {group_counts.get('chunks', 0)}",
        f"- learning-rate sweep: {group_counts.get('lr', 0)}",
        f"- temporal grid / eval sweep: {group_counts.get('time_eval', 0)}",
    ]

    baseline_lines = [
        "Baseline configuration",
        "",
        f"time_sampling = {BASELINE['time_sampling']}",
        f"causal_loss = {BASELINE['causal_loss']}",
        f"causal_curriculum = {BASELINE['causal_curriculum']}",
        f"causal_epsilon = {BASELINE['causal_epsilon']}",
        f"causal_n_chunks = {BASELINE['causal_n_chunks']}",
        f"lr = {BASELINE['lr']}",
        f"n_time = {BASELINE['n_time']}",
        f"n_eval = {BASELINE['n_eval']}",
        "",
        "In the table, “—” means unchanged from baseline."
    ]

    fig = plt.figure(figsize=(20, max(10, 0.42 * len(table_df) + 6)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 4.5], width_ratios=[1.25, 1.0])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    fig.suptitle("Single-species ablation study design", fontsize=18, fontweight="bold", y=0.98)

    ax_left.axis("off")
    ax_right.axis("off")
    ax_table.axis("off")

    ax_left.text(
        0.01, 0.98,
        "\n".join(baseline_lines),
        va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="gray")
    )

    ax_right.text(
        0.01, 0.98,
        "\n".join(summary_lines),
        va="top", ha="left", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="gray")
    )

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.35)

    # colour rows by family
    family_col_idx = list(table_df.columns).index("Family")
    for i in range(len(table_df)):
        family = table_df.iloc[i]["Family"]
        color = GROUP_COLORS.get(family, "lightgray")
        for j in range(len(table_df.columns)):
            cell = table[(i + 1, j)]
            cell.set_alpha(0.12)
            cell.set_facecolor(color)

        # stronger cell on family column
        table[(i + 1, family_col_idx)].set_alpha(0.3)
        table[(i + 1, family_col_idx)].set_facecolor(color)

    # header formatting
    for j in range(len(table_df.columns)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#dddddd")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 2: all loss curves on same plot
# ---------------------------------------------------------------------

def plot_loss_overlay(records: list[dict], ranking: pd.DataFrame, compare_step: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 8))

    top5 = set(ranking.head(5)["run_name"])

    for r in records:
        hist = r["history"]
        x = hist["step"].values
        y = hist[r["loss_col"]].values
        group = r.get("group", "unknown")
        color = GROUP_COLORS.get(group, "gray")

        lw = 2.4 if r["run_name"] in top5 else 1.2
        alpha = 0.95 if r["run_name"] in top5 else 0.35

        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)

    # annotate top 5 at the standardisation step
    for _, row in ranking.head(5).iterrows():
        run_name = row["run_name"]
        rec = next(r for r in records if r["run_name"] == run_name)
        hist = rec["history"]
        eligible = hist.loc[hist["step"] <= compare_step]
        if eligible.empty:
            continue
        last = eligible.iloc[-1]
        ax.scatter([last["step"]], [last[rec["loss_col"]]], s=35, color="black", zorder=5)
        label = f"{int(row['rank'])}. {row['scheme_name']}"
        ax.text(last["step"], last[rec["loss_col"]], " " + label, fontsize=9, va="center")

    # legend by family only
    for family in GROUP_ORDER:
        if any(r.get("group") == family for r in records):
            ax.plot([], [], color=GROUP_COLORS[family], linewidth=3, label=family)

    ax.axvline(compare_step, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("All run losses on the same plot")
    ax.legend(title="Ablation family", loc="upper right")
    ax.grid(True, alpha=0.3)

    subtitle = f"Fixed diagnostic loss used where available; comparison step = {compare_step}"
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 3: standardised loss ranking
# ---------------------------------------------------------------------

def plot_standardised_ranking(ranking: pd.DataFrame, compare_step: int, out_path: Path, top_n: int | None = None):
    df = ranking.copy()
    if top_n is not None:
        df = df.head(top_n).copy()

    df["label"] = df.apply(lambda r: f"{int(r['task_id'])}: {r['scheme_name']}" if pd.notna(r["task_id"]) else r["scheme_name"], axis=1)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.33 * len(df))))
    colors = [GROUP_COLORS.get(g, "gray") for g in df["group"]]

    ax.barh(df["label"], df["standardised_loss"], color=colors)
    ax.set_xlabel("Standardised loss")
    ax.set_ylabel("Run")
    ax.set_title(f"Run ranking at common step = {compare_step}")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Truth / prediction loading
# ---------------------------------------------------------------------

def load_truth(path: Path, species: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = find_col(df, ["time", "t"])
    weight_col = find_col(df, ["w", "weight"])
    n_col = find_col(df, ["N", "abundance"])
    sp_col = find_col(df, ["sp", "species"], required=False)

    keep = [time_col, weight_col, n_col]
    if sp_col:
        keep.append(sp_col)

    out = df[keep].copy()
    rename_map = {time_col: "time", weight_col: "weight", n_col: "N_true"}
    if sp_col:
        rename_map[sp_col] = "species"
    out = out.rename(columns=rename_map)

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["N_true"] = pd.to_numeric(out["N_true"], errors="coerce")
    out = out.dropna()

    if "species" in out.columns and species is not None:
        out = out[out["species"] == species].copy()

    return out


def load_prediction_for_run(run_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    candidates = [
        run_dir / "final_predictions_grid.csv",
        run_dir / "data_predictions_final.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        time_col = find_col(df, ["time", "t"])
        weight_col = find_col(df, ["w", "weight"])
        pred_col = find_col(df, ["N_pred", "pred_N", "Nhat", "N_hat", "prediction", "pred", "N"])
        sp_col = find_col(df, ["sp", "species"], required=False)

        keep = [time_col, weight_col, pred_col]
        if sp_col:
            keep.append(sp_col)

        out = df[keep].copy()
        rename_map = {time_col: "time", weight_col: "weight", pred_col: "N_pred"}
        if sp_col:
            rename_map[sp_col] = "species"
        out = out.rename(columns=rename_map)

        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
        out["N_pred"] = pd.to_numeric(out["N_pred"], errors="coerce")
        out = out.dropna()

        return out, path

    return None, None


def prepare_species_subset(truth: pd.DataFrame, pred: pd.DataFrame, species: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    truth_out = truth.copy()
    pred_out = pred.copy()

    truth_has_sp = "species" in truth_out.columns
    pred_has_sp = "species" in pred_out.columns

    chosen_species = species

    if chosen_species is None:
        if pred_has_sp:
            pred_species = pred_out["species"].dropna().unique()
            if len(pred_species) == 1:
                chosen_species = pred_species[0]

    if chosen_species is None and truth_has_sp:
        truth_species = truth_out["species"].dropna().unique()
        if len(truth_species) == 1:
            chosen_species = truth_species[0]

    if truth_has_sp and chosen_species is not None:
        truth_out = truth_out[truth_out["species"] == chosen_species].copy()

    if pred_has_sp and chosen_species is not None:
        pred_out = pred_out[pred_out["species"] == chosen_species].copy()

    return truth_out, pred_out, chosen_species


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_key"] = out["time"].round(8)
    out["weight_key"] = out["weight"].round(8)
    return out


# ---------------------------------------------------------------------
# Plot 4: top 2 vs truth, final time spectrum
# ---------------------------------------------------------------------

def plot_top2_final_spectrum(truth: pd.DataFrame, pred_info: list[tuple[str, pd.DataFrame]], out_path: Path):
    truth = add_keys(truth)

    common_times = set(truth["time_key"].unique())
    common_weights = set(truth["weight_key"].unique())

    preds = []
    for label, pred in pred_info:
        pred = add_keys(pred)
        common_times &= set(pred["time_key"].unique())
        common_weights &= set(pred["weight_key"].unique())
        preds.append((label, pred))

    if not common_times or not common_weights:
        print("Skipping final spectrum plot: no common time/weight grid.")
        return

    t_final = max(common_times)
    common_weights = sorted(common_weights)

    truth_sub = truth[truth["time_key"] == t_final].copy()
    truth_sub = truth_sub[truth_sub["weight_key"].isin(common_weights)].sort_values("weight_key")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(truth_sub["weight"], truth_sub["N_true"], linewidth=3, label="True mizer")

    for label, pred in preds:
        pred_sub = pred[pred["time_key"] == t_final].copy()
        pred_sub = pred_sub[pred_sub["weight_key"].isin(common_weights)].sort_values("weight_key")
        ax.plot(pred_sub["weight"], pred_sub["N_pred"], linewidth=2, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Abundance")
    ax.set_title(f"Top 2 vs true mizer: final-time size spectrum (time = {t_final:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Plot 5: top 2 vs truth, time trajectories at 3 weights
# ---------------------------------------------------------------------

def plot_top2_time_trajectories(truth: pd.DataFrame, pred_info: list[tuple[str, pd.DataFrame]], out_path: Path):
    truth = add_keys(truth)

    common_times = set(truth["time_key"].unique())
    common_weights = set(truth["weight_key"].unique())

    preds = []
    for label, pred in pred_info:
        pred = add_keys(pred)
        common_times &= set(pred["time_key"].unique())
        common_weights &= set(pred["weight_key"].unique())
        preds.append((label, pred))

    if not common_times or len(common_weights) < 3:
        print("Skipping time trajectory plot: insufficient common grid.")
        return

    weights_sorted = sorted(common_weights)
    idxs = [max(0, int(0.2 * (len(weights_sorted) - 1))),
            max(0, int(0.5 * (len(weights_sorted) - 1))),
            max(0, int(0.8 * (len(weights_sorted) - 1)))]
    selected_weights = [weights_sorted[i] for i in idxs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, w_key in zip(axes, selected_weights):
        truth_sub = truth[truth["weight_key"] == w_key].sort_values("time_key")
        if truth_sub.empty:
            continue

        display_weight = float(truth_sub["weight"].iloc[0])

        ax.plot(truth_sub["time"], truth_sub["N_true"], linewidth=3, label="True mizer")

        for label, pred in preds:
            pred_sub = pred[pred["weight_key"] == w_key].sort_values("time_key")
            ax.plot(pred_sub["time"], pred_sub["N_pred"], linewidth=2, label=label)

        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_title(f"Weight ≈ {display_weight:g}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Abundance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Top 2 vs true mizer: time trajectories at representative weights", y=1.08)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot single-species ablation results.")
    parser.add_argument("--runs-root", default="runs/ABLATION_RUNS/pde_only_single_species")
    parser.add_argument("--truth-csv", default="mizer_long.csv")
    parser.add_argument("--species", default=None, help="Optional species name to filter truth/predictions.")
    parser.add_argument("--compare-step", type=int, default=None,
                        help="Common comparison step. Use 20000 to match your existing ranking.")
    parser.add_argument("--min-final-step", type=int, default=0,
                        help="Drop failed/short runs below this final step.")
    parser.add_argument("--top-n-ranking", type=int, default=20)
    parser.add_argument("--output-dir", default="analysis/single_species_ablation")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    truth_csv = Path(args.truth_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_runs(runs_root, min_final_step=args.min_final_step)
    if not records:
        raise RuntimeError(f"No valid runs found under {runs_root}")

    ranking, compare_step = rank_runs(records, compare_step=args.compare_step)
    ranking.to_csv(output_dir / "run_ranking_standardised.csv", index=False)

    print(f"Runs found: {len(records)}")
    print(f"Comparison step: {compare_step}")
    print()
    print(ranking.head(args.top_n_ranking).to_string(index=False))

    # plots 1-3
    plot_ablation_overview(records, output_dir / "01_ablation_overview.png")
    plot_loss_overlay(records, ranking, compare_step, output_dir / "02_all_losses_same_plot.png")
    plot_standardised_ranking(ranking, compare_step, output_dir / "03_standardised_loss_ranking.png",
                              top_n=min(args.top_n_ranking, len(ranking)))

    # top 2 vs truth
    truth = load_truth(truth_csv, species=args.species)

    top2 = ranking.head(2).copy()
    pred_info = []

    for _, row in top2.iterrows():
        run_dir = Path(row["run_dir"])
        pred, pred_path = load_prediction_for_run(run_dir)
        if pred is None:
            print(f"Warning: no prediction file found for {run_dir}")
            continue

        truth_sub, pred_sub, chosen_species = prepare_species_subset(truth, pred, species=args.species)
        label = f"{int(row['rank'])}. {row['scheme_name']}"

        pred_info.append((label, pred_sub))

        if chosen_species is not None:
            print(f"{label}: using species '{chosen_species}' from {pred_path.name}")
        else:
            print(f"{label}: using predictions from {pred_path.name}")

    if len(pred_info) >= 1:
        # for these plots we need truth subset matching the first prediction's species handling
        truth_plot = truth.copy()
        if args.species is not None and "species" in truth_plot.columns:
            truth_plot = truth_plot[truth_plot["species"] == args.species].copy()
        elif "species" in truth_plot.columns and "species" in pred_info[0][1].columns:
            pred_species = pred_info[0][1]["species"].dropna().unique()
            if len(pred_species) == 1:
                truth_plot = truth_plot[truth_plot["species"] == pred_species[0]].copy()

        plot_top2_final_spectrum(truth_plot, pred_info, output_dir / "04_top2_vs_truth_final_spectrum.png")
        if len(pred_info) >= 2:
            plot_top2_time_trajectories(truth_plot, pred_info, output_dir / "05_top2_vs_truth_time_trajectories.png")

    print(f"\nSaved outputs to:\n{output_dir}")


if __name__ == "__main__":
    main()
