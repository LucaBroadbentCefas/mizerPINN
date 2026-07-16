"""Read-only Streamlit viewer for PINNmizer HPC outputs."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_RUN_ROOT = Path("runs")
TINY = np.finfo(float).tiny
RUN_COLUMNS = [
    "run_id","run_dir","status","error_message","n_steps_completed","final_loss","final_loss_unweighted",
    "final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss","final_fixed_loss_unweighted",
    "final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc","final_fixed_residual_log_abs_p95",
    "seconds_per_step","actual_total_seconds","model_arch","hidden_width","hidden_layers","fourier_num_features",
    "fourier_scale","fourier_include_raw_input","weight_factorization","rwf_mu","rwf_sigma","rwf_apply_to",
    "rwf_base_init","residual_form","boundary_loss_form","time_sampling","causal_loss","loss_weighting",
    "collocation_strategy","r3_population_size","seed","fourier_seed","lr","n_steps","n_time","n_eval","lambda_pde","lambda_ic","lambda_bc","lambda_timestep","hpc",
]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()

def is_run_dir(path: Path) -> bool:
    if path.name == "fixed_grid_diagnostics":
        return False

    return any(
        (path / name).exists()
        for name in [
            "config.json",
            "final_summary.json",
            "final_summary.csv",
            "loss_history.csv",
            "fixed_diagnostic_history.csv",
            "fixed_grid_fields.npz",
            "fixed_grid_fields.csv",
            "fixed_grid_diagnostics/fixed_grid_fields.npz",
            "fixed_grid_diagnostics/fixed_grid_fields.csv",
        ]
    )

@st.cache_data(show_spinner=False)
def safe_read_csv(path: str | Path) -> pd.DataFrame | None:
    path = _as_path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def safe_read_json(path: str | Path) -> dict[str, Any]:
    path = _as_path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def safe_read_text(path: str | Path) -> str:
    path = _as_path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _first_row_dict(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    return df.iloc[-1].dropna().to_dict()


def _pick(*dicts: dict[str, Any], key: str, default: Any = np.nan) -> Any:
    for d in dicts:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


@st.cache_data(show_spinner=False)
def load_config(run_dir: str | Path) -> dict[str, Any]:
    return safe_read_json(_as_path(run_dir) / "config.json")


@st.cache_data(show_spinner=False)
def load_final_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = _as_path(run_dir)
    data = safe_read_json(run_dir / "final_summary.json")
    data.update(_first_row_dict(safe_read_csv(run_dir / "final_summary.csv")))
    return data


@st.cache_data(show_spinner=False)
def scan_runs(run_root: str | Path) -> pd.DataFrame:
    root = _as_path(run_root)
    rows: list[dict[str, Any]] = []
    if not root.exists() or not root.is_dir():
        return pd.DataFrame(columns=RUN_COLUMNS)

    run_dirs = [root] if is_run_dir(root) else sorted(p for p in root.rglob("*") if p.is_dir() and is_run_dir(p))

    for run_dir in run_dirs:
        summary = load_final_summary(run_dir)
        config = load_config(run_dir)
        timing = _first_row_dict(safe_read_csv(run_dir / "timing_summary.csv"))
        row = {c: np.nan for c in RUN_COLUMNS}
        run_id = run_dir.name if run_dir == root else str(run_dir.relative_to(root))
        row.update({"run_id": run_id, "run_dir": str(run_dir), "error_message": ""})
        for key in RUN_COLUMNS:
            if key in ("run_id", "run_dir"):
                continue
            default = "" if key in {"status","error_message","model_arch","weight_factorization","rwf_apply_to","rwf_base_init","residual_form","boundary_loss_form","time_sampling","causal_loss","loss_weighting","collocation_strategy"} else np.nan
            row[key] = _pick(summary, timing, config, key=key, default=default)
        if not row["status"]:
            row["status"] = "unknown" if summary or config else "no_summary"
        rows.append(row)
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


@st.cache_data(show_spinner=False)
def load_loss_history(run_dir: str | Path) -> pd.DataFrame | None:
    return safe_read_csv(_as_path(run_dir) / "loss_history.csv")


@st.cache_data(show_spinner=False)
def load_fixed_diagnostic_history(run_dir: str | Path) -> pd.DataFrame | None:
    return safe_read_csv(_as_path(run_dir) / "fixed_diagnostic_history.csv")


@st.cache_data(show_spinner=False)
def load_final_predictions(run_dir: str | Path) -> pd.DataFrame | None:
    return safe_read_csv(_as_path(run_dir) / "final_predictions_grid.csv")


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _fixed_field_candidates(run_dir: Path, filename: str) -> list[Path]:
    return [
        run_dir / filename,
        run_dir / "fixed_grid_diagnostics" / filename,
    ]


def _select_species_array(arr: np.ndarray, species_idx: int) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        idx = int(np.clip(species_idx, 0, arr.shape[1] - 1))
        return arr[:, idx, :]

    return arr


def _species_options_from_csv(path: Path | None) -> list[tuple[int, str]]:
    if path is None or not path.exists():
        return []

    df = safe_read_csv(path)
    if df is None or df.empty or "species_idx" not in df.columns:
        return []

    name_col = "species" if "species" in df.columns else None
    opts = (
        df[["species_idx"] + ([name_col] if name_col else [])]
        .drop_duplicates()
        .sort_values("species_idx")
    )

    out = []
    for row in opts.itertuples(index=False):
        idx = int(getattr(row, "species_idx"))
        name = str(getattr(row, name_col)) if name_col else f"species_{idx}"
        out.append((idx, name))

    return out


def _normalise_npz_fixed_fields(
    raw: dict[str, np.ndarray],
    *,
    species_idx: int,
    csv_path: Path | None,
    source_file: Path,
) -> dict[str, Any] | None:
    t = raw.get("t_eval", raw.get("t"))
    x = raw.get("x_eval", raw.get("x"))
    w = raw.get("w_eval", raw.get("w"))

    if t is None or x is None:
        return None

    fields: dict[str, Any] = {
        "t_eval": np.asarray(t),
        "x_eval": np.asarray(x),
        "source_file": str(source_file),
    }

    if w is not None:
        fields["w_eval"] = np.asarray(w)

    species_options = _species_options_from_csv(csv_path)

    n_species = 1
    for key in ["log10_N", "log_N_eval", "log_N", "residual_log"]:
        arr = raw.get(key)
        if arr is not None and np.asarray(arr).ndim == 3:
            n_species = int(np.asarray(arr).shape[1])
            break

    if not species_options:
        species_options = [(i, f"species_{i}") for i in range(n_species)]

    species_idx = int(np.clip(species_idx, 0, max(0, n_species - 1)))
    fields["species_options"] = species_options
    fields["selected_species_idx"] = species_idx
    fields["selected_species"] = next(
        (name for idx, name in species_options if idx == species_idx),
        f"species_{species_idx}",
    )

    if "log10_N" in raw:
        fields["log10_N"] = _select_species_array(raw["log10_N"], species_idx)
    elif "log_N_eval" in raw:
        fields["log10_N"] = _select_species_array(raw["log_N_eval"], species_idx) / np.log(10.0)
    elif "log_N" in raw:
        fields["log10_N"] = _select_species_array(raw["log_N"], species_idx) / np.log(10.0)

    direct_map = {
        "residual_log": "residual_log",
        "dlogN_dt": "dlogN_dt",
        "dlogN_dw": "dlogN_dw",
        "g_eval": "g_eval",
        "dg_dw": "dg_dw",
        "mu_eval": "mu",
        "mu": "mu",
        "advective": "advective",
    }

    for src, dst in direct_map.items():
        if src in raw and dst not in fields:
            fields[dst] = _select_species_array(raw[src], species_idx)

    if "advective" not in fields and {"g_eval", "dlogN_dw"}.issubset(fields):
        fields["advective"] = fields["g_eval"] * fields["dlogN_dw"]

    return fields


def _normalise_csv_fixed_fields(
    csv: pd.DataFrame,
    *,
    species_idx: int,
    source_file: Path,
) -> dict[str, Any] | None:
    aliases = {
        "t_eval": ["t_eval", "t"],
        "x_eval": ["x_eval", "x"],
        "w_eval": ["w_eval", "w"],
    }

    low = {c.lower(): c for c in csv.columns}

    def pick(names: list[str]) -> str | None:
        return next((low[n.lower()] for n in names if n.lower() in low), None)

    t_col = pick(aliases["t_eval"])
    x_col = pick(aliases["x_eval"])
    w_col = pick(aliases["w_eval"])

    if t_col is None or x_col is None:
        return None

    species_options = []
    if "species_idx" in csv.columns:
        name_col = "species" if "species" in csv.columns else None
        opts = (
            csv[["species_idx"] + ([name_col] if name_col else [])]
            .drop_duplicates()
            .sort_values("species_idx")
        )
        for row in opts.itertuples(index=False):
            idx = int(getattr(row, "species_idx"))
            name = str(getattr(row, name_col)) if name_col else f"species_{idx}"
            species_options.append((idx, name))

        csv = csv[csv["species_idx"] == species_idx].copy()
    else:
        species_options = [(0, "species_0")]

    if csv.empty:
        return None

    t_vals = np.sort(pd.to_numeric(csv[t_col], errors="coerce").dropna().unique())
    x_vals = np.sort(pd.to_numeric(csv[x_col], errors="coerce").dropna().unique())

    fields: dict[str, Any] = {
        "t_eval": t_vals,
        "x_eval": x_vals,
        "species_options": species_options,
        "selected_species_idx": species_idx,
        "selected_species": next(
            (name for idx, name in species_options if idx == species_idx),
            f"species_{species_idx}",
        ),
        "source_file": str(source_file),
    }

    if w_col is not None:
        fields["w_eval"] = np.array([
            csv.loc[csv[x_col].sub(x).abs().idxmin(), w_col]
            for x in x_vals
        ])

    value_cols = {
        "log10_N": ["log10_N", "log10N"],
        "log_N": ["log_N", "logN", "ln_N"],
        "residual_log": ["residual_log"],
        "dlogN_dt": ["dlogN_dt"],
        "dlogN_dw": ["dlogN_dw"],
        "advective": ["advective"],
        "mu": ["mu", "mu_eval"],
        "dg_dw": ["dg_dw"],
        "g_eval": ["g_eval"],
    }

    def pivot(col: str) -> np.ndarray:
        return (
            csv.pivot_table(index=t_col, columns=x_col, values=col, aggfunc="mean")
            .reindex(index=t_vals, columns=x_vals)
            .to_numpy()
        )

    for out_name, names in value_cols.items():
        col = pick(names)
        if col is not None:
            fields[out_name] = pivot(col)

    if "log10_N" not in fields and "log_N" in fields:
        fields["log10_N"] = fields["log_N"] / np.log(10.0)

    if "advective" not in fields and {"g_eval", "dlogN_dw"}.issubset(fields):
        fields["advective"] = fields["g_eval"] * fields["dlogN_dw"]

    return fields


@st.cache_data(show_spinner=False)
def load_fixed_fields(run_dir: str | Path, species_idx: int = 0) -> dict[str, Any] | None:
    run_dir = _as_path(run_dir)

    npz_path = _first_existing(_fixed_field_candidates(run_dir, "fixed_grid_fields.npz"))
    csv_path = _first_existing(_fixed_field_candidates(run_dir, "fixed_grid_fields.csv"))

    if npz_path is not None:
        try:
            with np.load(npz_path) as data:
                raw = {k: np.asarray(data[k]) for k in data.files}

            fields = _normalise_npz_fixed_fields(
                raw,
                species_idx=species_idx,
                csv_path=csv_path,
                source_file=npz_path,
            )
            if fields is not None:
                return fields
        except Exception:
            pass

    if csv_path is None:
        return None

    csv = safe_read_csv(csv_path)
    if csv is None or csv.empty:
        return None

    return _normalise_csv_fixed_fields(
        csv,
        species_idx=species_idx,
        source_file=csv_path,
    )


def check_required_columns(df: pd.DataFrame | None, columns: list[str]) -> list[str]:
    return columns if df is None else [c for c in columns if c not in df.columns]


def check_required_arrays(fields: dict[str, np.ndarray] | None, arrays: list[str]) -> list[str]:
    return arrays if fields is None else [a for a in arrays if a not in fields]


def nearest_value(values: np.ndarray, target: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr[np.nanargmin(np.abs(arr - target))])


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - target)))


def prepare_log_axis_data(df: pd.DataFrame, y_cols: list[str]) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    dropped = False
    for col in y_cols:
        if col in out.columns:
            mask = pd.to_numeric(out[col], errors="coerce") <= 0
            dropped = dropped or bool(mask.any())
            out.loc[mask, col] = np.nan
    return out, dropped


def add_plot_description(interpretation: str, calculation: str) -> None:
    with st.expander("Plot description", expanded=False):
        st.markdown(f"### Interpretation\n{interpretation}\n\n### How this plot is calculated\n{calculation}")


def show_missing(title: str, missing: list[str], source: str) -> None:
    st.warning(f"{title}: missing required columns/arrays in {source}: {', '.join(missing)}")


def make_multi_line_plot(df: pd.DataFrame, x: str, ys: list[str], title: str, y_label: str, log_y: bool, markers: bool) -> None:
    missing = check_required_columns(df, [x] + ys)
    if missing:
        show_missing(title, missing, "CSV") ; return
    plot_df = df[[x] + ys].copy()
    dropped = False
    if log_y:
        plot_df, dropped = prepare_log_axis_data(plot_df, ys)
        if dropped: st.warning("Non-positive values were replaced with NaN for log-scaled y-axis.")
    long = plot_df.melt(id_vars=x, value_vars=ys, var_name="metric", value_name=y_label).dropna(subset=[y_label])
    if long.empty:
        st.info(f"{title}: no plottable values."); return
    fig = px.line(long, x=x, y=y_label, color="metric", title=title, markers=markers, log_y=log_y)
    st.plotly_chart(fig, use_container_width=True)


def heatmap_limits(z: np.ndarray, mode: str, clip: bool) -> tuple[float | None, float | None]:
    finite = np.asarray(z)[np.isfinite(z)]
    if finite.size == 0 or mode == "auto": return None, None
    if mode == "symmetric around zero":
        m = float(np.nanmax(np.abs(finite))); return -m, m
    if mode == "percentile clipped" or clip:
        return tuple(np.nanpercentile(finite, [1, 99]).astype(float))
    return None, None


def make_heatmap(x, y, z, title: str, colorbar: str, mode: str, clip: bool) -> None:
    zmin, zmax = heatmap_limits(z, mode, clip)
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorbar={"title": colorbar}, zmin=zmin, zmax=zmax, colorscale="Viridis"))
    fig.update_layout(title=title, xaxis_title="log weight x = log(w)", yaxis_title="time t")
    st.plotly_chart(fig, use_container_width=True)


def line_desc(file, required, transform="none", log_y=False, dropped=False):
    return f"- File read: `{file}`.\n- Required columns/arrays: {', '.join(required)}.\n- Transformations: {transform}.\n- Nearest/interpolation/summary: none unless stated in the plot controls.\n- Y-axis log-scaled: {log_y}.\n- Non-positive values removed for log scaling: {dropped}."



def format_scalar(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.4g}"
    return f"{x:.3e}".replace("e-0", "e-").replace("e+0", "e")


def format_time_label(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def parse_comparison_times(text: str) -> list[float]:
    vals=[]
    for part in str(text).replace(";", ",").split(","):
        part=part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            st.warning(f"Ignoring non-numeric comparison time: {part}")
    return vals or [0.0]


def format_nearest_times_message(requested: list[float], used_by_source: dict[str, list[float]]) -> str:
    lines=["Requested times: " + ", ".join(format_time_label(t) for t in requested) + "."]
    for source, used in used_by_source.items():
        lines.append(f"{source} nearest times used: " + ", ".join(format_time_label(t) for t in used) + ".")
    return "\n".join(lines)


PLOT_INTERPRETATIONS = {
    "Total loss": "`loss` is the scalar optimiser objective used to update neural-network parameters: w_pde*lambda_pde*L_pde + w_ic*lambda_ic*L_ic + w_bc*lambda_bc*L_bc + w_ts*lambda_ts*L_ts. Falling values usually indicate easier optimisation; spikes can indicate unstable batches, weights, or learning-rate interactions.",
    "Total unweighted loss": "`loss_unweighted` is the combined mathematical loss before adaptive/objective weighting. It is useful for checking raw error scale separately from the weighted update objective.",
    "Unscaled loss terms": "Raw terms split the PINN constraints: `loss_pde` is mean(R_log^2) for R_log = d_t log N + g d_w log N + mu + d_w g; `loss_ic` is initial-condition mismatch; `loss_bc` is boundary-condition mismatch; `loss_timestep` is optional timestep consistency and is inactive when its weight is zero.",
    "Objective loss terms": "`objective_loss_pde`, `objective_loss_ic`, `objective_loss_bc`, and `objective_loss_timestep` are weighted optimiser contributions w_k*lambda_k*L_k. They show how strongly each term influences parameter updates, not always the raw mathematical error size.",
    "Adaptive weight trajectories": "`w_pde`, `w_ic`, `w_bc`, and `w_timestep` are adaptive multipliers in the training objective. These are training-control variables, not biological outputs; high values make a constraint dominate updates.",
    "Gradient norm": "Gradient norm is |grad_theta L_train|, the size of the optimiser update signal seen by neural-network parameters. Large spikes can indicate unstable optimisation; near-zero values can mean convergence, saturation, or a stuck optimiser, so read it alongside loss curves.",
    "Causal curriculum": "Causal curriculum diagnostics are training controls, not biological outputs. `causal_fraction` is the fraction of the time domain included in the PDE loss, and `t_max_current` is the largest physical time currently included.",
    "Causal chunk diagnostics": "Causal chunk diagnostics show curriculum weighting across time chunks: first/mean/last weights indicate how strongly early/middle/late chunks contribute; mean/max chunk losses show average and worst PDE loss over causal chunks.",
    "Fixed-grid losses": "Fixed losses are PDE/IC/BC diagnostics evaluated on a fixed validation grid, not necessarily the random collocation batch used for training. They provide comparable run-to-run checks of constraint satisfaction.",
    "Fixed residual summaries": "Fixed residual summaries measure R_log on the fixed grid: RMS is sqrt(mean(R_log^2)), mean/max are absolute residual summaries, and p95 is the 95th percentile of |R_log|. High tails identify local PDE failures.",
    "PDE term RMS balance": "PDE-term balance decomposes R_log = d_t log N + g d_w log N + mu + d_w g. `rms_dlogN_dt` is the time-derivative size, `rms_advective` is g*d_w log N, `rms_mu` is mortality, and `rms_dg_dw` is the growth-gradient term.",
}

def plot_with_desc(func, interpretation: str, calculation: str) -> None:
    func(); add_plot_description(interpretation, calculation)



def _value_group_key(value: Any) -> tuple[str, Any]:
    if pd.isna(value):
        return ("missing", "NA")
    try:
        x=float(value)
        if np.isfinite(x):
            return ("float", round(x, 12))
    except (TypeError, ValueError):
        pass
    return ("str", str(value))


def show_architecture_differences(run_df: pd.DataFrame, selected_runs: list[str]) -> None:
    st.subheader("Selected-run architecture/config differences")
    fields=["model_arch","hidden_width","hidden_layers","fourier_num_features","fourier_scale","fourier_include_raw_input","weight_factorization","rwf_mu","rwf_sigma","rwf_apply_to","rwf_base_init","residual_form","boundary_loss_form","time_sampling","causal_loss","loss_weighting","collocation_strategy","r3_population_size","seed","fourier_seed","lr","n_steps","n_time","n_eval","lambda_pde","lambda_ic","lambda_bc","lambda_timestep"]
    sel=run_df[run_df.run_id.isin(selected_runs)].set_index("run_id")
    rows=[]; common={}
    for field in fields:
        vals=[sel.at[r, field] if field in sel.columns and r in sel.index else np.nan for r in selected_runs]
        groups=[]
        for v in vals:
            placed=False
            for g in groups:
                gv=g[0]
                try:
                    if np.isclose(float(v), float(gv), rtol=1e-6, atol=1e-12, equal_nan=True):
                        g.append(v); placed=True; break
                except (TypeError, ValueError):
                    if _value_group_key(v)==_value_group_key(gv):
                        g.append(v); placed=True; break
            if not placed: groups.append([v])
        if len(groups) <= 1:
            continue
        mode=max(groups, key=len)[0]; common[field]=mode
        rows.append({"field":field, **{r:("NA" if pd.isna(v) else v) for r,v in zip(selected_runs, vals)}})
    if not rows:
        st.info("No differing architecture/config fields among the selected runs."); return
    d=pd.DataFrame(rows).set_index("field")
    def style_cell(value, field):
        mode=common.get(field, np.nan)
        try:
            same=np.isclose(float(value), float(mode), rtol=1e-6, atol=1e-12, equal_nan=True)
        except (TypeError, ValueError):
            same=_value_group_key(value)==_value_group_key(mode)
        return "" if same else "background-color: #ffe08a; font-weight: 600"
    st.dataframe(d.style.apply(lambda row:[style_cell(v, row.name) for v in row], axis=1), use_container_width=True)

def add_loss_at_step(run_df: pd.DataFrame, target_step: int, metric: str) -> pd.DataFrame:
    out = run_df.copy()
    values = []
    used_steps = []

    for run_dir in out["run_dir"]:
        df = load_loss_history(run_dir)
        if df is None or df.empty or "step" not in df.columns or metric not in df.columns:
            values.append(np.nan)
            used_steps.append(np.nan)
            continue

        d = df[["step", metric]].copy()
        d["step"] = pd.to_numeric(d["step"], errors="coerce")
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
        d = d.dropna(subset=["step", metric])

        if d.empty:
            values.append(np.nan)
            used_steps.append(np.nan)
            continue

        idx = (d["step"] - target_step).abs().idxmin()
        values.append(float(d.loc[idx, metric]))
        used_steps.append(int(d.loc[idx, "step"]))

    out[f"{metric}_at_selected_step"] = values
    out["selected_loss_step_used"] = used_steps
    return out

def run_browser_page(run_df: pd.DataFrame, selected_runs: list[str], log_y: bool, markers: bool) -> None:
    st.header("Run browser")
    filtered = run_df.copy()

    for col in ["status","model_arch","collocation_strategy","loss_weighting","causal_loss","weight_factorization","seed"]:
        vals = sorted([str(v) for v in filtered[col].dropna().unique() if str(v) != ""])
        chosen = st.multiselect(f"Filter {col}", vals, key=f"filter_{col}")
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

    loss_metric = st.selectbox("Loss-history metric for selected iteration", ["loss","loss_unweighted","loss_pde","loss_ic","loss_bc","loss_timestep"])
    loss_step = st.number_input("Iteration for direct loss comparison", min_value=0, value=1000, step=100)
    filtered = add_loss_at_step(filtered, int(loss_step), loss_metric)

    st.caption("Selected-iteration loss uses the nearest logged step in each run's loss_history.csv.")

    event = st.dataframe(
        filtered,
        use_container_width=True,
        key="run_browser_table",
        on_select="rerun",
        selection_mode="single-row",
    )

    clicked_rows = event.selection.rows
    if clicked_rows:
        clicked_run = str(filtered.iloc[clicked_rows[0]]["run_id"])
    
        # Only apply the table click when it is a NEW table selection.
        # This prevents an old selected table row from overriding the sidebar selectbox.
        if clicked_run != st.session_state.get("last_table_selected_run"):
            st.session_state["last_table_selected_run"] = clicked_run
            st.session_state["selected_run"] = clicked_run
            st.rerun()

    if len(selected_runs) >= 2:
        show_architecture_differences(filtered, selected_runs)

    selected_step_metric = f"{loss_metric}_at_selected_step"
    metrics = [selected_step_metric,"final_fixed_residual_log_abs_p95","final_loss","final_loss_unweighted","final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss","final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc","seconds_per_step"]
    metric = st.selectbox("Ranking metric", metrics)

    def rank():
        d = filtered[["run_id", metric]].dropna().sort_values(metric, ascending=True)
        if d.empty:
            st.info(f"No values available for {metric}.")
            return
        st.plotly_chart(px.bar(d, x=metric, y="run_id", orientation="h", title=f"Final metric ranking: {metric}"), use_container_width=True)

    plot_with_desc(rank, "Ranks run folders by the selected metric.", line_desc("final_summary.csv/json, timing_summary.csv, or loss_history.csv", ["run_id", metric], "sorted ascending", False, False))

    decomp_cols = ["final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc"]

    def decomp():
        d = filtered[filtered.run_id.isin(selected_runs)][["run_id"] + decomp_cols].melt("run_id", var_name="term", value_name="loss").dropna()
        if d.empty:
            st.info("No selected runs contain final loss decomposition values.")
            return
        st.plotly_chart(px.bar(d, x="run_id", y="loss", color="term", barmode="group", title="Final loss decomposition", log_y=log_y), use_container_width=True)

    plot_with_desc(decomp, "Compares final PDE/IC/BC contributions.", line_desc("final_summary.csv/json", ["run_id"]+decomp_cols, "wide-to-long reshape", log_y, False))

    group = st.selectbox("Scatter colour", ["model_arch","collocation_strategy","loss_weighting","causal_loss","weight_factorization"])

    def speed():
        st.plotly_chart(px.scatter(filtered, x="seconds_per_step", y="final_fixed_residual_log_abs_p95", color=group, hover_data=["run_id","hidden_width","hidden_layers","seed","final_loss"], title="Speed-quality scatter"), use_container_width=True)

    plot_with_desc(speed, "Shows the cost/accuracy trade-off.", line_desc("run index from summaries/config/timing", ["seconds_per_step","final_fixed_residual_log_abs_p95",group], "none", False, False))

    x = st.selectbox("Architecture scatter x", ["hidden_width","hidden_layers","fourier_scale","fourier_num_features","rwf_sigma","r3_population_size"])

    def arch():
        st.plotly_chart(px.scatter(filtered, x=x, y="final_fixed_residual_log_abs_p95", color=group, hover_data=["run_id"], title="Architecture / hyperparameter scatter"), use_container_width=True)

    plot_with_desc(arch, "Relates architecture or sampling settings to final fixed-grid residual quality.", line_desc("run index from summaries/config", [x,"final_fixed_residual_log_abs_p95",group], "none", False, False))

def history_page(run_dir: Path, fixed: bool, log_y: bool, markers: bool) -> None:
    file = "fixed_diagnostic_history.csv" if fixed else "loss_history.csv"
    df = load_fixed_diagnostic_history(run_dir) if fixed else load_loss_history(run_dir)
    st.header("Single run: fixed diagnostics" if fixed else "Single run: training")
    if df is None: st.warning(f"{file} not found for this run"); return
    if df.empty: st.info(f"{file} exists but is empty"); return
    if fixed:
        cards = ["fixed_residual_log_abs_p95","fixed_loss_pde","fixed_loss_ic","fixed_loss_bc"]
        cols = st.columns(4)
        for c, m in zip(cols, cards): c.metric(m, format_scalar(df[m].dropna().iloc[-1] if m in df and df[m].notna().any() else np.nan))
        plots = [("Fixed-grid losses", ["fixed_loss","fixed_loss_unweighted","fixed_loss_pde","fixed_loss_ic","fixed_loss_bc"]),("Fixed residual summaries", ["fixed_residual_log_rms","fixed_residual_log_abs_mean","fixed_residual_log_abs_p95","fixed_residual_log_abs_max"]),("PDE term RMS balance", ["rms_dlogN_dt","rms_advective","rms_mu","rms_dg_dw"])]
    else:
        plots = [("Total loss", ["loss"]),("Total unweighted loss", ["loss_unweighted"]),("Unscaled loss terms", ["loss_pde","loss_ic","loss_bc","loss_timestep"]),("Objective loss terms", ["objective_loss_pde","objective_loss_ic","objective_loss_bc","objective_loss_timestep"]),("Adaptive weight trajectories", ["w_pde","w_ic","w_bc","w_timestep"]),("Gradient norm", ["grad_norm"]),("Causal curriculum", ["causal_fraction","t_max_current"]),("Causal chunk diagnostics", ["pde_causal_weight_first","pde_causal_weight_mean","pde_causal_weight_last","pde_causal_chunk_loss_mean","pde_causal_chunk_loss_max"])]
    for title, ys in plots:
        use_log = log_y and ("Causal" not in title)
        plot_with_desc(lambda ys=ys,title=title,use_log=use_log: make_multi_line_plot(df, "step", ys, title, "value", use_log, markers), PLOT_INTERPRETATIONS.get(title, f"Tracks {title.lower()} from `{file}` to reveal convergence, imbalance, spikes, or curriculum behaviour."), line_desc(file, ["step"]+ys, "non-positive y values become NaN only for log plots", use_log, "reported above if any"))

# More field/compare/mizer helpers compact but explicit

def fields_page(run_dir: Path, clip: bool, mode: str, markers: bool) -> None:
    st.header("Single run: fields")
    fields0 = load_fixed_fields(run_dir, species_idx=0)
    
    if fields0 is None:
        st.warning(
            "No usable fixed-grid fields found. Checked both run-root and "
            "fixed_grid_diagnostics/ field outputs."
        )
        return
    
    species_options = fields0.get("species_options", [(0, "species_0")])
    species_labels = [f"{idx}: {name}" for idx, name in species_options]
    
    selected_label = st.selectbox(
        "PINN species",
        species_labels,
        index=0,
    )
    
    selected_species_idx = int(selected_label.split(":", 1)[0])
    fields = load_fixed_fields(run_dir, species_idx=selected_species_idx)
    
    if fields is None:
        st.warning(f"Could not load fixed-grid fields for species_idx={selected_species_idx}.")
        return
    
    st.caption(
        f"Field source: `{fields.get('source_file', 'unknown')}`; "
        f"species: `{fields.get('selected_species', selected_species_idx)}`."
    )
    
    missing = check_required_arrays(fields, ["t_eval","x_eval","log10_N","residual_log"])
    if missing: show_missing("Fields", missing, "fixed_grid_fields.npz/csv"); return
    t,x = fields["t_eval"], fields["x_eval"]
    for name,z,title in [("log10_N",fields["log10_N"],"Predicted abundance heatmap"),("residual_log",fields["residual_log"],"Signed log-residual heatmap"),("abs_residual_log",np.abs(fields["residual_log"]),"Absolute log-residual heatmap")]:
        plot_with_desc(lambda z=z,name=name,title=title: make_heatmap(x,t,z,title,name,mode if "residual" in name else "auto",clip), FIELD_INTERPRETATIONS[name], line_desc("fixed_grid_fields.npz (fallback fixed_grid_fields.csv)", ["t_eval","x_eval",name.replace("abs_","")], "absolute value for abs residual; optional percentile/symmetric colour limits", False, False))
    comp = st.selectbox("PDE component", [c for c in ["dlogN_dt","advective","mu","dg_dw","g_eval"] if c in fields]) if any(c in fields for c in ["dlogN_dt","advective","mu","dg_dw","g_eval"]) else None
    if comp: plot_with_desc(lambda: make_heatmap(x,t,fields[comp],f"PDE component heatmap: {comp}",comp,mode,clip), FIELD_INTERPRETATIONS[comp], line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval",comp], "optional colour clipping", False, False))
    default_times = [float(t[i]) for i in sorted(set(np.linspace(0, len(t)-1, min(6,len(t)), dtype=int)))]
    times = st.multiselect("Profile times", [float(v) for v in t], default=default_times)
    def profiles(resid=False):
        rows=[]; arr=np.abs(fields["residual_log"]) if resid and st.checkbox("Absolute residual profiles", True) else (fields["residual_log"] if resid else fields["log10_N"])
        for tv in times:
            i=nearest_index(t,tv); rows += [{"x":xx,"value":vv,"t":f"{t[i]:.4g}"} for xx,vv in zip(x,arr[i])]
        st.plotly_chart(px.line(pd.DataFrame(rows), x="x", y="value", color="t", markers=markers, title="Residual profiles" if resid else "Abundance profiles"), use_container_width=True)
    plot_with_desc(lambda: profiles(False), "Compares abundance shape across selected times using nearest fixed-grid time slices.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "nearest selected time", False, False))
    plot_with_desc(lambda: profiles(True), "Compares residual shape across selected times to identify weight ranges where the PDE is least satisfied.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","residual_log"], "nearest selected time; optional absolute value", False, False))
    xs = st.multiselect("Weight/log-weight slices", [float(v) for v in x], default=[float(x[i]) for i in sorted(set(np.linspace(0,len(x)-1,min(4,len(x)),dtype=int)))])
    def wseries():
        rows=[]
        for xv in xs:
            j=nearest_index(x,xv); rows += [{"t":tt,"log10_N":vv,"x":f"{x[j]:.4g}"} for tt,vv in zip(t,fields["log10_N"][:,j])]
        st.plotly_chart(px.line(pd.DataFrame(rows), x="t", y="log10_N", color="x", markers=markers, title="Weight-slice abundance time series"), use_container_width=True)
    plot_with_desc(wseries, "Tracks abundance through time at selected nearest log-weights to diagnose unstable temporal trajectories.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "nearest selected x/w", False, False))
    def hist():
        r=fields["residual_log"].ravel(); ar=np.abs(r); fig=go.Figure(); fig.add_trace(go.Histogram(x=r,name="residual_log",opacity=.55)); fig.add_trace(go.Histogram(x=ar,name="abs(residual_log)",opacity=.55));
        for q in [50,90,95,99]: fig.add_vline(x=float(np.nanpercentile(ar,q)), annotation_text=f"abs p{q}")
        fig.update_layout(title="Residual distribution histogram", barmode="overlay", xaxis_title="R_log") ; st.plotly_chart(fig,use_container_width=True)
    plot_with_desc(hist, "Summarises the distribution and tails of fixed-grid residual errors.", line_desc("fixed_grid_fields.npz/csv", ["residual_log"], "absolute residual and p50/p90/p95/p99 vertical lines", False, False))
    for axis_name, axis_vals, axis_num, title in [("time",t,1,"Residual by time summary"),("weight",x,0,"Residual by weight summary")]:
        def summ(axis_vals=axis_vals, axis_num=axis_num, title=title):
            ar=np.abs(fields["residual_log"]); d=pd.DataFrame({axis_name:axis_vals,"mean":np.nanmean(ar,axis=axis_num),"p95":np.nanpercentile(ar,95,axis=axis_num),"max":np.nanmax(ar,axis=axis_num)}).melt(axis_name,var_name="stat",value_name="abs_residual")
            st.plotly_chart(px.line(d,x=axis_name,y="abs_residual",color="stat",markers=markers,title=title),use_container_width=True)
        plot_with_desc(summ, f"Aggregates absolute residual by {axis_name} to locate broad regions of PDE mismatch.", line_desc("fixed_grid_fields.npz/csv", ["residual_log"], f"mean, p95, max of abs residual along the other axis", False, False))

# Comparison and mizer abbreviated implementation covering requested plots
FIELD_INTERPRETATIONS={"log10_N":"`log10_N` is the neural-network predicted abundance on the fixed diagnostic grid, shown as log10(N). High or low regions show the learned size spectrum over time.","residual_log":"`residual_log` is the signed log PDE residual R_log. Positive/negative regions show which side of d_t log N + g d_w log N + mu + d_w g is imbalanced.","abs_residual_log":"`abs_residual_log` is the magnitude of the PDE mismatch; high values mark where the neural-network field least satisfies the PDE.","dlogN_dt":"`dlogN_dt` is the time derivative of network-predicted log abundance on the fixed grid.","advective":"`advective` is the growth/advection contribution g*d_w log N in the log residual.","mu":"`mu` is the mortality contribution in the log residual.","dg_dw":"`dg_dw` is the body-size derivative of growth rate in the log residual.","g_eval":"`g_eval` is the growth rate evaluated on the fixed grid; it is a biological PDE coefficient used by the residual, not a network output."}

def normalise_mizer_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    aliases = {
        "t": ["time", "t", "t_eval"],
        "species": ["sp", "species", "species_id", "species_name"],
        "w": ["weight", "w", "w_eval"],
        "x": ["x", "log_weight", "log_w", "x_eval"],
        "N": ["N", "n", "abundance", "density"],
        "log_N": ["log_N", "logN", "ln_N"],
        "log10_N": ["log10_N", "log10N"],
    }

    out = pd.DataFrame(index=df.index)
    out["source_name"] = source_name
    low = {c.lower(): c for c in df.columns}

    for std, names in aliases.items():
        col = next((low[n.lower()] for n in names if n.lower() in low), None)
        out[std] = df[col] if col else np.nan

    if out["species"].isna().all():
        out["species"] = "species_0"

    for c in ["t", "w", "x", "N", "log_N", "log10_N"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Weight support: only positive physical weights can be converted to x=log(w).
    positive_w = out["w"] > 0
    missing_x_with_positive_w = out["x"].isna() & positive_w
    out.loc[missing_x_with_positive_w, "x"] = np.log(out.loc[missing_x_with_positive_w, "w"])

    # Abundance support: structural zeros in mizer output are not tiny abundances.
    # They should disappear from log-profile plots rather than become log10(TINY) ~= -308.
    positive_N = out["N"] > 0
    missing_log10_with_positive_N = out["log10_N"].isna() & positive_N
    out.loc[missing_log10_with_positive_N, "log10_N"] = np.log10(
        out.loc[missing_log10_with_positive_N, "N"]
    )

    missing_log10_with_log_N = out["log10_N"].isna() & out["log_N"].notna()
    out.loc[missing_log10_with_log_N, "log10_N"] = out.loc[missing_log10_with_log_N, "log_N"] / np.log(10)

    missing_N_with_log_N = out["N"].isna() & out["log_N"].notna()
    out.loc[missing_N_with_log_N, "N"] = np.exp(out.loc[missing_N_with_log_N, "log_N"])

    missing_N_with_log10_N = out["N"].isna() & out["log10_N"].notna()
    out.loc[missing_N_with_log10_N, "N"] = 10 ** out.loc[missing_N_with_log10_N, "log10_N"]

    required_ok = (
        out["t"].notna().any()
        and out["species"].notna().any()
        and out["w"].notna().any()
        and out["x"].notna().any()
        and out["log10_N"].notna().any()
    )
    if not required_ok:
        raise ValueError(
            "Could not use this mizer file.\n\n"
            "Expected long-format mizer data with columns equivalent to:\n"
            "time, sp/species, w/weight, and N/abundance or log_N/log10_N.\n\n"
            "The mizer object appears to be a time × species × weight array, so export it to long CSV first: "
            "one row per time/species/weight. Structural zero abundances are treated as missing for log plots."
        )

    return out[["source_name", "t", "species", "w", "x", "N", "log_N", "log10_N"]].dropna(
        subset=["t", "species", "w", "x", "log10_N"]
    )


def active_mizer_x_range(miz: pd.DataFrame, species: str) -> tuple[float, float] | None:
    sub = miz[
        (miz["species"].astype(str) == str(species))
        & np.isfinite(miz["x"])
        & np.isfinite(miz["log10_N"])
    ]

    if sub.empty:
        return None

    return float(sub["x"].min()), float(sub["x"].max())


def combined_mizer_x_range(
    mizers: dict[str, pd.DataFrame],
    species: str,
    source_names: list[str] | None = None,
) -> tuple[float, float] | None:
    names = list(mizers) if source_names is None else [name for name in source_names if name in mizers]
    ranges = [active_mizer_x_range(mizers[name], species) for name in names]
    ranges = [r for r in ranges if r is not None]

    if not ranges:
        return None

    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def mask_to_x_support(
    x_values,
    y_values,
    x_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)

    if x_range is not None:
        xmin, xmax = x_range
        mask &= (x_arr >= xmin) & (x_arr <= xmax)

    return x_arr[mask], y_arr[mask]


def mask_matrix_to_x_support(z, x_values, x_range: tuple[float, float] | None) -> np.ndarray:
    out = np.asarray(z, dtype=float).copy()
    x_arr = np.asarray(x_values, dtype=float)
    mask = np.isfinite(x_arr)

    if x_range is not None:
        xmin, xmax = x_range
        mask &= (x_arr >= xmin) & (x_arr <= xmax)

    out[:, ~mask] = np.nan
    return out


def compare_page(run_df, selected_runs, clip, mode, markers):
    st.header("Compare runs")
    if len(selected_runs) < 2:
        st.info("Select two or more runs in the sidebar.")
        return

    run_dirs = dict(zip(run_df.run_id, run_df.run_dir))

    first_fields = load_fixed_fields(run_dirs[selected_runs[0]], species_idx=0)
    species_options = first_fields.get("species_options", [(0, "species_0")]) if first_fields else [(0, "species_0")]
    species_labels = [f"{idx}: {name}" for idx, name in species_options]

    selected_species_label = st.selectbox(
        "PINN species for run comparison",
        species_labels,
        index=0,
        key="compare_pinn_species",
    )
    selected_species_idx = int(selected_species_label.split(":", 1)[0])

    for file, loader, metric_opts, title in [
        (
            "fixed_diagnostic_history.csv",
            load_fixed_diagnostic_history,
            ["fixed_residual_log_abs_p95","fixed_loss_pde","fixed_loss_ic","fixed_loss_bc","fixed_loss_unweighted"],
            "Fixed diagnostic overlay",
        ),
        (
            "loss_history.csv",
            load_loss_history,
            ["loss","loss_unweighted","loss_pde","loss_ic","loss_bc","loss_timestep"],
            "Training loss overlay",
        ),
    ]:
        metric = st.selectbox(f"{title} metric", metric_opts, key=title)

        def overlay(file=file, loader=loader, metric=metric, title=title):
            frames = []
            for rid in selected_runs:
                d = loader(run_dirs[rid])
                if d is not None and not d.empty and {"step", metric}.issubset(d.columns):
                    frames.append(d[["step", metric]].assign(run_id=rid))

            if frames:
                st.plotly_chart(
                    px.line(
                        pd.concat(frames),
                        x="step",
                        y=metric,
                        color="run_id",
                        markers=markers,
                        title=title,
                    ),
                    use_container_width=True,
                )
            else:
                st.warning(f"No selected runs contain step and {metric} in {file}.")

        plot_with_desc(
            overlay,
            f"Overlays `{metric}` across selected runs to compare convergence histories.",
            line_desc(file, ["step", metric], "concatenate selected runs; no interpolation", False, False),
        )

    st.dataframe(run_df[run_df.run_id.isin(selected_runs)], use_container_width=True)
    add_plot_description(
        "One-row-per-run table of final metrics/config for selected runs.",
        line_desc("final_summary.csv/json, config.json, timing_summary.csv", RUN_COLUMNS, "scan immediate run folders", False, False),
    )

    times = parse_comparison_times(st.text_input("Selected times for profile overlays", "0"))

    def profile(resid=False):
        rows = []
        used = {}

        for rid in selected_runs:
            f = load_fixed_fields(run_dirs[rid], species_idx=selected_species_idx)

            if f and not check_required_arrays(f, ["t_eval","x_eval","log10_N","residual_log"]):
                species_name = f.get("selected_species", selected_species_idx)

                for tv in times:
                    i = nearest_index(f["t_eval"], tv)
                    used.setdefault(rid, []).append(f["t_eval"][i])
                    arr = np.abs(f["residual_log"][i]) if resid else f["log10_N"][i]
                    rows += [
                        {
                            "x": xx,
                            "value": vv,
                            "run_id": f"{rid} | {species_name} | t={format_time_label(f['t_eval'][i])}",
                        }
                        for xx, vv in zip(f["x_eval"], arr)
                    ]

        st.caption(format_nearest_times_message(times, used))
        if rows:
            st.plotly_chart(
                px.line(
                    pd.DataFrame(rows),
                    x="x",
                    y="value",
                    color="run_id",
                    markers=markers,
                    title="Residual profile overlay" if resid else "Abundance profile overlay",
                ),
                use_container_width=True,
            )
        else:
            st.info("No selected runs contain usable fixed-field profiles for the selected species.")

    plot_with_desc(
        lambda: profile(False),
        "Overlays final abundance profiles at nearest time for the selected PINN species; different x grids are allowed because each line carries its own x coordinates.",
        line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "nearest time per run; selected species slice; no interpolation", False, False),
    )
    plot_with_desc(
        lambda: profile(True),
        "Overlays residual profiles at nearest time for the selected PINN species to compare where selected runs violate the PDE most.",
        line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","residual_log"], "nearest time; selected species slice; absolute value", False, False),
    )

    ref = st.selectbox("Reference run", selected_runs)
    cmp = st.selectbox("Comparison run", [r for r in selected_runs if r != ref])

    def diff(field="log10_N"):
        a = load_fixed_fields(run_dirs[ref], species_idx=selected_species_idx)
        b = load_fixed_fields(run_dirs[cmp], species_idx=selected_species_idx)

        if (
            not a
            or not b
            or check_required_arrays(a, ["t_eval","x_eval",field])
            or check_required_arrays(b, ["t_eval","x_eval",field])
        ):
            st.warning("Required fixed fields missing for the selected species.")
            return

        if not (np.array_equal(a["t_eval"], b["t_eval"]) and np.array_equal(a["x_eval"], b["x_eval"])):
            st.warning("The fixed grids differ; difference heatmap is not plotted.")
            return

        z = (np.abs(b[field]) - np.abs(a[field])) if field == "residual_log" else (b[field] - a[field])
        make_heatmap(
            a["x_eval"],
            a["t_eval"],
            z,
            f"Difference heatmap: {cmp} - {ref}",
            "delta",
            mode,
            clip,
        )

    plot_with_desc(
        lambda: diff("log10_N"),
        "Shows abundance difference for the selected species only when fixed t/x grids match exactly, avoiding invalid comparisons.",
        line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "comparison minus reference; selected species; exact grid equality required", False, False),
    )
    plot_with_desc(
        lambda: diff("residual_log"),
        "Shows change in absolute residual versus reference for the selected species only on identical grids.",
        line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","residual_log"], "abs(comparison)-abs(reference); selected species; exact grid equality required", False, False),
    )

def load_mizer_sources(local_paths: str, uploads) -> dict[str,pd.DataFrame]:
    out={}
    for raw in [p.strip() for p in local_paths.splitlines() if p.strip()]:
        df=safe_read_csv(Path(raw));

        if df is not None:
            try: out[Path(raw).name]=normalise_mizer_dataframe(df,Path(raw).name)
            except ValueError as e: st.error(str(e))
    for up in uploads or []:
        try: out[up.name]=normalise_mizer_dataframe(pd.read_csv(up), up.name)
        except ValueError as e: st.error(str(e))
        except Exception: st.warning(f"Could not read uploaded mizer CSV: {up.name}")
    return out


def interpolate_mizer_to_pinn(miz, species, t_grid, x_grid, *, method: str = "linear"):
    sub = miz[
        (miz.species.astype(str) == str(species))
        & np.isfinite(miz["t"])
        & np.isfinite(miz["x"])
        & np.isfinite(miz["log10_N"])
    ].copy()

    if sub.empty:
        return None

    z = []
    x_grid = np.asarray(x_grid, dtype=float)

    for tv in t_grid:
        nt = nearest_value(sub.t.to_numpy(), tv)
        s = sub[np.isclose(sub.t, nt)].sort_values("x")
        s = s[np.isfinite(s["x"]) & np.isfinite(s["log10_N"])]

        if s.empty:
            return None

        s = s.groupby("x", as_index=False)["log10_N"].mean().sort_values("x")
        sx = s["x"].to_numpy(dtype=float)
        sy = s["log10_N"].to_numpy(dtype=float)

        if method == "nearest":
            vals = np.full_like(x_grid, np.nan, dtype=float)
            for j, x in enumerate(x_grid):
                if x < sx.min() or x > sx.max():
                    continue
                vals[j] = sy[int(np.nanargmin(np.abs(sx - x)))]
        else:
            vals = np.interp(x_grid, sx, sy, left=np.nan, right=np.nan)

        z.append(vals)

    return np.asarray(z)


def mizer_page(run_df, selected_run, selected_runs, mizers, clip, mode, markers):
    st.header("Mizer comparison")
    if not mizers:
        st.info("Provide one or more mizer CSV files in the sidebar.")
        return

    run_dirs = dict(zip(run_df.run_id, run_df.run_dir))

    fields0 = load_fixed_fields(run_dirs.get(selected_run, ""), species_idx=0) if selected_run else None
    if not fields0 or check_required_arrays(fields0, ["t_eval","x_eval","log10_N"]):
        st.warning("Selected PINN run needs fixed_grid_fields.npz/csv with t_eval, x_eval, log10_N.")
        return
    allm = pd.concat(mizers.values(), ignore_index=True)
    
    # Keep mizer species in the order they appear in the file.
    mizer_species_options = pd.unique(allm["species"].dropna().astype(str)).tolist()
    
    species = st.selectbox("Species", mizer_species_options)
    
    # Same-order assumption: selected species position equals PINN species index.
    pinn_species_idx = mizer_species_options.index(species)

    fields = load_fixed_fields(run_dirs.get(selected_run, ""), species_idx=pinn_species_idx)
    if not fields or check_required_arrays(fields, ["t_eval","x_eval","log10_N"]):
        st.warning(f"Could not load PINN fields for species_idx={pinn_species_idx}.")
        return

    pinn_species_name = fields.get("selected_species", f"species_{pinn_species_idx}")


    t_options = [float(v) for v in fields["t_eval"]]
    default_times = [t_options[i] for i in sorted(set(np.linspace(0, len(t_options)-1, min(6, len(t_options)), dtype=int)))]
    times = st.multiselect("Comparison times", t_options, default=default_times)
    if not times:
        st.info("Select at least one comparison time.")
        return

    xsel = st.number_input("Comparison log-weight x", value=float(fields["x_eval"][0]))
    srcs = st.multiselect("Mizer sources", list(mizers), default=list(mizers))
    active_x_range = combined_mizer_x_range(mizers, species, srcs)

    if active_x_range is not None:
        st.caption(
            "PINN profiles and PINN-mizer differences are masked to the selected mizer species' "
            f"positive-abundance support: x in [{active_x_range[0]:.6g}, {active_x_range[1]:.6g}]."
        )
    else:
        st.warning("Could not infer positive-abundance mizer support for the selected species; PINN support masking is not applied.")

    def prof():
        rows = []
        used = {"PINN": []}

        for tv in times:
            i = nearest_index(fields["t_eval"], tv)
            pinnt = fields["t_eval"][i]
            used["PINN"].append(pinnt)
            time_label = format_time_label(tv)

            x_plot, y_plot = mask_to_x_support(fields["x_eval"], fields["log10_N"][i], active_x_range)
            if x_plot.size > 1:
                x_plot = x_plot[:-1]
                y_plot = y_plot[:-1]
            rows += [
                {
                    "x": x,
                    "log10_N": v,
                    "time": time_label,
                    "source": "PINN",
                    "series": f"PINN {pinn_species_name} | t={format_time_label(pinnt)}",
                }
                for x, v in zip(x_plot, y_plot)
            ]

        for name in srcs:
            s = mizers[name][mizers[name].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"])
            for tv in times:
                nt = nearest_value(s.t.to_numpy(), tv)
                used.setdefault(name, []).append(nt)
                time_label = format_time_label(tv)
                ss = s[np.isclose(s.t, nt)]

                rows += [
                    {
                        "x": r.x,
                        "log10_N": r.log10_N,
                        "time": time_label,
                        "source": "mizer",
                        "series": f"{name} {species} | t={format_time_label(nt)}",
                    }
                    for r in ss.itertuples()
                ]

        if not rows:
            st.info("No PINN/mizer profile rows to plot.")
            return

        st.caption(format_nearest_times_message(times, used))
        st.plotly_chart(
            px.line(
                pd.DataFrame(rows),
                x="x",
                y="log10_N",
                color="time",
                line_dash="source",
                line_dash_map={"PINN":"solid", "mizer":"dash"},
                line_group="series",
                hover_name="series",
                markers=markers,
                title="PINN and mizer abundance profile overlay",
            ),
            use_container_width=True,
        )

    plot_with_desc(
        prof,
        "Compares the selected PINN species slice to selected mizer profiles at nearest available times.",
        line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t/x/log10_N or aliases"], "same selected time has same colour; PINN solid, mizer dashed; selected species slice", False, False),
    )

    def ts():
        rows = []
        j = nearest_index(fields["x_eval"], xsel)
        selected_x = float(fields["x_eval"][j])
        if (
            active_x_range is None
            or (active_x_range[0] <= selected_x <= active_x_range[1])
        ):
            rows += [
                {"t": t, "log10_N": v, "source": "PINN"}
                for t, v in zip(fields["t_eval"], fields["log10_N"][:, j])
                if np.isfinite(v)
            ]
        else:
            st.warning("Selected PINN log-weight is outside the positive-abundance mizer support; PINN time-series is hidden.")

        for name in srcs:
            s = mizers[name][mizers[name].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"])
            for tv, g in s.groupby("t"):
                nx = nearest_value(g.x.to_numpy(), xsel)
                val = g.loc[np.isclose(g.x, nx), "log10_N"].iloc[0]
                rows.append({"t":tv,"log10_N":val,"source":name})

        st.plotly_chart(
            px.line(
                pd.DataFrame(rows),
                x="t",
                y="log10_N",
                color="source",
                markers=markers,
                title="Abundance time-series at selected weight",
            ),
            use_container_width=True,
        )

    plot_with_desc(
        ts,
        "Compares temporal abundance at the nearest log-weight in each source for the selected species.",
        line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t","x","log10_N"], "nearest x per time/source; selected species slice", False, False),
    )

    one = st.selectbox("Mizer source for error", list(mizers))
    interp = st.checkbox("Use linear-in-x interpolation", True)
    interp_method = "linear" if interp else "nearest"
    miz_grid = interpolate_mizer_to_pinn(
        mizers[one],
        species,
        fields["t_eval"],
        fields["x_eval"],
        method=interp_method,
    )

    def delta_profile():
        if miz_grid is None:
            st.warning("Interpolation impossible: mizer source needs t, x, log10_N for selected species.")
            return

        rows = []
        used = {"PINN": [], one: []}
        sub = mizers[one][mizers[one].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"])

        for tv in times:
            i = nearest_index(fields["t_eval"], tv)
            pinnt = fields["t_eval"][i]
            mt = nearest_value(sub.t.to_numpy(), pinnt)
            used["PINN"].append(pinnt)
            used[one].append(mt)
            delta = fields["log10_N"][i] - miz_grid[i]
            x_plot, delta_plot = mask_to_x_support(fields["x_eval"], delta, active_x_range)
            rows += [
                {"x": x, "delta": v, "source": f"PINN {pinn_species_name} - {one} {species} | t={format_time_label(pinnt)}"}
                for x, v in zip(x_plot, delta_plot)
            ]

        st.warning(f"Mizer comparison used nearest-time selection and {interp_method} matching in log-weight x onto the PINN grid.")
        st.caption(format_nearest_times_message(times, used))
        st.plotly_chart(
            px.line(
                pd.DataFrame(rows),
                x="x",
                y="delta",
                color="source",
                markers=markers,
                title="PINN minus mizer profile",
            ),
            use_container_width=True,
        )

    plot_with_desc(
        delta_profile,
        "Shows signed abundance error along weight at the selected time.",
        line_desc("fixed_grid_fields.npz/csv plus one mizer CSV", ["t_eval","x_eval","log10_N","mizer t/x/log10_N"], f"nearest time and {'linear interpolation in x' if interp else 'nearest x selection'}; selected species; delta=PINN-mizer", False, False),
    )

    def delta_heat():
        if miz_grid is None:
            st.warning("Interpolation impossible: mizer source needs t, x, log10_N for selected species.")
            return
        st.warning(f"Mizer values were matched using nearest time plus {interp_method} matching in log-weight x onto the PINN grid.")
        delta = mask_matrix_to_x_support(fields["log10_N"] - miz_grid, fields["x_eval"], active_x_range)
        make_heatmap(
            fields["x_eval"],
            fields["t_eval"],
            delta,
            "PINN minus mizer heatmap",
            "delta_log10_N",
            mode,
            clip,
        )

    plot_with_desc(
        delta_heat,
        "Shows PINN minus mizer abundance error across the fixed PINN grid.",
        line_desc("fixed_grid_fields.npz/csv plus one mizer CSV", ["t_eval","x_eval","log10_N","mizer t/x/log10_N"], "nearest mizer time and linear interpolation in x onto PINN grid; selected species", False, False),
    )

    def multi():
        rows = []

        for rid in selected_runs:
            f = load_fixed_fields(run_dirs[rid], species_idx=pinn_species_idx)
            if f and not check_required_arrays(f, ["t_eval","x_eval","log10_N"]):
                species_name = f.get("selected_species", pinn_species_idx)
                for tv in times:
                    i = nearest_index(f["t_eval"], tv)
                    x_plot, y_plot = mask_to_x_support(f["x_eval"], f["log10_N"][i], active_x_range)
                    rows += [
                        {
                            "x": x,
                            "log10_N": v,
                            "source": f"{rid} {species_name} | t={format_time_label(f['t_eval'][i])}",
                        }
                        for x, v in zip(x_plot, y_plot)
                    ]

        s = mizers[one][mizers[one].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"])
        for tv in times:
            nt = nearest_value(s.t.to_numpy(), tv)
            rows += [
                {"x":r.x,"log10_N":r.log10_N,"source":f"{one} {species} | t={format_time_label(nt)}"}
                for r in s[np.isclose(s.t, nt)].itertuples()
            ]

        if not rows:
            st.info("No multi-run/mizer profile rows to plot.")
            return

        st.plotly_chart(
            px.line(
                pd.DataFrame(rows),
                x="x",
                y="log10_N",
                color="source",
                markers=markers,
                title="Multiple PINN runs vs one mizer profile",
            ),
            use_container_width=True,
        )

    plot_with_desc(
        multi,
        "Compares several PINN runs for the selected PINN species with one selected mizer species profile at nearest times.",
        line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t/x/log10_N"], "nearest time per source; selected species; no cross-run grid assumption", False, False),
    )

    for by, axis, vals in [("time",1,fields["t_eval"]),("weight",0,fields["x_eval"])]:
        def err(by=by, axis=axis, vals=vals):
            if miz_grid is None:
                st.warning("Interpolation impossible.")
                return

            ad = np.abs(fields["log10_N"] - miz_grid)
            ad = mask_matrix_to_x_support(ad, fields["x_eval"], active_x_range)
            d = pd.DataFrame({
                by: vals,
                "mean_abs_delta_log10_N": np.nanmean(ad, axis=axis),
                "p95_abs_delta_log10_N": np.nanpercentile(ad, 95, axis=axis),
            }).melt(by, var_name="stat", value_name="error")

            st.plotly_chart(
                px.line(
                    d,
                    x=by,
                    y="error",
                    color="stat",
                    markers=markers,
                    title=f"Error summary by {by}",
                ),
                use_container_width=True,
            )

        plot_with_desc(
            err,
            f"Summarises absolute PINN-mizer error by {by} after placing mizer data on the PINN grid.",
            line_desc("fixed_grid_fields.npz plus mizer CSV", ["t/x/log10_N"], "nearest-time plus linear-x interpolation; selected species; mean and p95 abs delta", False, False),
        )

def file_view_page(run_dir: Path) -> None:
    st.header("File/config view")
    for name in ["config.json","final_summary.json","final_summary.csv","timing_summary.csv","run_command.txt"]:
        st.subheader(name); p=run_dir/name
        if name.endswith(".csv"):
            df=safe_read_csv(p); st.info(f"{name} not found or unreadable") if df is None else st.dataframe(df,use_container_width=True)
        elif name.endswith(".json"):
            data=safe_read_json(p); st.info(f"{name} not found or unreadable") if not data else st.json(data)
        else:
            txt=safe_read_text(p); st.info(f"{name} not found or unreadable") if not txt else st.code(txt)


def main() -> None:
    st.set_page_config(page_title="PINNmizer HPC Viewer", layout="wide")
    st.title("PINNmizer HPC run viewer")
    with st.sidebar:
        run_root=Path(st.text_input("Run root path", str(DEFAULT_RUN_ROOT))).expanduser()
        if st.button("Refresh / re-scan"): st.cache_data.clear()
        label_mode=st.selectbox("Run label mode", ["folder name","short folder name","model_arch + seed","custom label assembled from selected config fields"])
        custom_fields=st.multiselect("Custom label config fields", RUN_COLUMNS, default=["model_arch","seed"]) if label_mode.startswith("custom") else []
        log_y=st.checkbox("Log y-axis where relevant", True); markers=st.checkbox("Show points", True); clip=st.checkbox("Quantile clipping for heatmaps", True)
        heat_mode=st.selectbox("Heatmap colour range", ["auto","symmetric around zero","percentile clipped"], index=2)
        mizer_paths=st.text_area("Mizer CSV local paths (one per line)"); uploads=st.file_uploader("Upload mizer CSVs", type="csv", accept_multiple_files=True)
    run_df=scan_runs(run_root)
    if run_df.empty: st.warning(f"No run folders found under {run_root}")
    run_ids=run_df.run_id.tolist()
    
    if run_ids:
        if st.session_state.get("selected_run") not in run_ids:
            st.session_state["selected_run"] = run_ids[0]
    else:
        st.session_state["selected_run"] = None
    
    with st.sidebar:
        selected_run=st.selectbox(
            "Single selected run",
            run_ids,
            index=run_ids.index(st.session_state["selected_run"]) if run_ids and st.session_state["selected_run"] in run_ids else 0,
        ) if run_ids else None
        if selected_run:
            st.session_state["selected_run"] = selected_run
    
        selected_runs=st.multiselect("Runs for comparison", run_ids, default=run_ids[:min(3,len(run_ids))])
    
    selected_run = st.session_state.get("selected_run")
    run_dir=Path(run_df.loc[run_df.run_id==selected_run,"run_dir"].iloc[0]) if selected_run else Path("")
    mizers=load_mizer_sources(mizer_paths, uploads)
    tabs=st.tabs(["Run browser","Single run: training","Single run: fixed diagnostics","Single run: fields","Compare runs","Mizer comparison","File/config view"])
    with tabs[0]: run_browser_page(run_df, selected_runs, log_y, markers)
    with tabs[1]: history_page(run_dir, False, log_y, markers) if selected_run else st.info("Select a run.")
    with tabs[2]: history_page(run_dir, True, log_y, markers) if selected_run else st.info("Select a run.")
    with tabs[3]: fields_page(run_dir, clip, heat_mode, markers) if selected_run else st.info("Select a run.")
    with tabs[4]: compare_page(run_df, selected_runs, clip, heat_mode, markers)
    with tabs[5]: mizer_page(run_df, selected_run, selected_runs, mizers, clip, heat_mode, markers) if selected_run else st.info("Select a PINN run.")
    with tabs[6]: file_view_page(run_dir) if selected_run else st.info("Select a run.")


if __name__ == "__main__":
    main()
