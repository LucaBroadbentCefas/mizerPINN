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

DEFAULT_RUN_ROOT = Path("runs/pde_only_single_species")
TINY = np.finfo(float).tiny
RUN_COLUMNS = [
    "run_id","run_dir","status","error_message","n_steps_completed","final_loss","final_loss_unweighted",
    "final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss","final_fixed_loss_unweighted",
    "final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc","final_fixed_residual_log_abs_p95",
    "seconds_per_step","actual_total_seconds","model_arch","hidden_width","hidden_layers","fourier_num_features",
    "fourier_scale","fourier_include_raw_input","weight_factorization","rwf_mu","rwf_sigma","rwf_apply_to",
    "rwf_base_init","residual_form","boundary_loss_form","time_sampling","causal_loss","loss_weighting",
    "collocation_strategy","r3_population_size","seed","fourier_seed","hpc",
]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


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
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        summary = load_final_summary(run_dir)
        config = load_config(run_dir)
        timing = _first_row_dict(safe_read_csv(run_dir / "timing_summary.csv"))
        row = {c: np.nan for c in RUN_COLUMNS}
        row.update({"run_id": run_dir.name, "run_dir": str(run_dir), "error_message": ""})
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


@st.cache_data(show_spinner=False)
def load_fixed_fields(run_dir: str | Path) -> dict[str, np.ndarray] | None:
    run_dir = _as_path(run_dir)
    npz = run_dir / "fixed_grid_fields.npz"
    if npz.exists():
        try:
            with np.load(npz) as data:
                return {k: np.asarray(data[k]) for k in data.files}
        except Exception:
            return None
    csv = safe_read_csv(run_dir / "fixed_grid_fields.csv")
    if csv is None or csv.empty or not {"t_eval", "x_eval"}.issubset(csv.columns):
        return None
    fields: dict[str, np.ndarray] = {"t_eval": np.sort(csv["t_eval"].dropna().unique()), "x_eval": np.sort(csv["x_eval"].dropna().unique())}
    if "w_eval" in csv.columns:
        fields["w_eval"] = np.array([csv.loc[csv["x_eval"].sub(x).abs().idxmin(), "w_eval"] for x in fields["x_eval"]])
    for col in ["log10_N","residual_log","dlogN_dt","advective","mu","dg_dw","g_eval"]:
        if col in csv.columns:
            piv = csv.pivot_table(index="t_eval", columns="x_eval", values=col, aggfunc="mean").reindex(index=fields["t_eval"], columns=fields["x_eval"])
            fields[col] = piv.to_numpy()
    return fields


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


def plot_with_desc(func, interpretation: str, calculation: str) -> None:
    func(); add_plot_description(interpretation, calculation)


def run_browser_page(run_df: pd.DataFrame, selected_runs: list[str], log_y: bool, markers: bool) -> None:
    st.header("Run browser")
    filtered = run_df.copy()
    for col in ["status","model_arch","collocation_strategy","loss_weighting","causal_loss","weight_factorization","seed"]:
        vals = sorted([str(v) for v in filtered[col].dropna().unique() if str(v) != ""])
        chosen = st.multiselect(f"Filter {col}", vals, key=f"filter_{col}")
        if chosen: filtered = filtered[filtered[col].astype(str).isin(chosen)]
    st.dataframe(filtered, use_container_width=True)
    metrics = ["final_fixed_residual_log_abs_p95","final_loss","final_loss_unweighted","final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss","final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc","seconds_per_step"]
    metric = st.selectbox("Ranking metric", metrics)
    def rank():
        d = filtered[["run_id", metric]].dropna().sort_values(metric, ascending=True)
        st.plotly_chart(px.bar(d, x=metric, y="run_id", orientation="h", title=f"Final metric ranking: {metric}"), use_container_width=True)
    plot_with_desc(rank, "Ranks run folders by a final scalar metric from summaries/timing to identify best-quality or fastest runs and failed outliers.", line_desc("final_summary.csv/json and timing_summary.csv", ["run_id", metric], "sorted ascending (best to worst for losses/residuals)", False, False))
    decomp_cols = ["final_loss_pde","final_loss_ic","final_loss_bc","final_fixed_loss_pde","final_fixed_loss_ic","final_fixed_loss_bc"]
    def decomp():
        d = filtered[filtered.run_id.isin(selected_runs)][["run_id"] + decomp_cols].melt("run_id", var_name="term", value_name="loss").dropna()
        st.plotly_chart(px.bar(d, x="run_id", y="loss", color="term", barmode="group", title="Final loss decomposition", log_y=log_y), use_container_width=True)
    plot_with_desc(decomp, "Compares final PDE/IC/BC contributions so a run dominated by one constraint can be diagnosed.", line_desc("final_summary.csv/json", ["run_id"]+decomp_cols, "wide-to-long reshape", log_y, False))
    group = st.selectbox("Scatter colour", ["model_arch","collocation_strategy","loss_weighting","causal_loss","weight_factorization"])
    def speed():
        st.plotly_chart(px.scatter(filtered, x="seconds_per_step", y="final_fixed_residual_log_abs_p95", color=group, hover_data=["run_id","hidden_width","hidden_layers","seed","final_loss"], title="Speed-quality scatter"), use_container_width=True)
    plot_with_desc(speed, "Shows the cost/accuracy trade-off; slow runs with poor residuals or fast accurate runs stand out.", line_desc("run index from summaries/config/timing", ["seconds_per_step","final_fixed_residual_log_abs_p95",group], "none", False, False))
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
        for c, m in zip(cols, cards): c.metric(m, df[m].dropna().iloc[-1] if m in df and df[m].notna().any() else "NA")
        plots = [("Fixed-grid losses", ["fixed_loss","fixed_loss_unweighted","fixed_loss_pde","fixed_loss_ic","fixed_loss_bc"]),("Fixed residual summaries", ["fixed_residual_log_rms","fixed_residual_log_abs_mean","fixed_residual_log_abs_p95","fixed_residual_log_abs_max"]),("PDE term RMS balance", ["rms_dlogN_dt","rms_advective","rms_mu","rms_dg_dw"])]
    else:
        plots = [("Total loss", ["loss"]),("Total unweighted loss", ["loss_unweighted"]),("Unscaled loss terms", ["loss_pde","loss_ic","loss_bc","loss_timestep"]),("Objective loss terms", ["objective_loss_pde","objective_loss_ic","objective_loss_bc","objective_loss_timestep"]),("Adaptive weight trajectories", ["w_pde","w_ic","w_bc","w_timestep"]),("Gradient norm", ["grad_norm"]),("Causal curriculum", ["causal_fraction","t_max_current"]),("Causal chunk diagnostics", ["pde_causal_weight_first","pde_causal_weight_mean","pde_causal_weight_last","pde_causal_chunk_loss_mean","pde_causal_chunk_loss_max"])]
    for title, ys in plots:
        use_log = log_y and ("Causal" not in title)
        plot_with_desc(lambda ys=ys,title=title,use_log=use_log: make_multi_line_plot(df, "step", ys, title, "value", use_log, markers), f"Tracks {title.lower()} from `{file}` to reveal convergence, imbalance, spikes, or curriculum behaviour.", line_desc(file, ["step"]+ys, "non-positive y values become NaN only for log plots", use_log, "reported above if any"))

# More field/compare/mizer helpers compact but explicit

def fields_page(run_dir: Path, clip: bool, mode: str, markers: bool) -> None:
    st.header("Single run: fields")
    fields = load_fixed_fields(run_dir)
    if fields is None: st.warning("fixed_grid_fields.npz not found for this run, and fixed_grid_fields.csv could not be used."); return
    missing = check_required_arrays(fields, ["t_eval","x_eval","log10_N","residual_log"])
    if missing: show_missing("Fields", missing, "fixed_grid_fields.npz/csv"); return
    t,x = fields["t_eval"], fields["x_eval"]
    for name,z,title in [("log10_N",fields["log10_N"],"Predicted abundance heatmap"),("residual_log",fields["residual_log"],"Signed log-residual heatmap"),("abs_residual_log",np.abs(fields["residual_log"]),"Absolute log-residual heatmap")]:
        plot_with_desc(lambda z=z,name=name,title=title: make_heatmap(x,t,z,title,name,mode if "residual" in name else "auto",clip), f"Shows `{name}` over the fixed diagnostic grid to locate temporal or weight regions with abundance structure or residual failure.", line_desc("fixed_grid_fields.npz (fallback fixed_grid_fields.csv)", ["t_eval","x_eval",name.replace("abs_","")], "absolute value for abs residual; optional percentile/symmetric colour limits", False, False))
    comp = st.selectbox("PDE component", [c for c in ["dlogN_dt","advective","mu","dg_dw","g_eval"] if c in fields]) if any(c in fields for c in ["dlogN_dt","advective","mu","dg_dw","g_eval"]) else None
    if comp: plot_with_desc(lambda: make_heatmap(x,t,fields[comp],f"PDE component heatmap: {comp}",comp,mode,clip), "Shows one fixed-grid PDE component to diagnose imbalance in the residual equation.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval",comp], "optional colour clipping", False, False))
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
def normalise_mizer_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    aliases={"t":["time","t","t_eval"],"species":["sp","species","species_id","species_name"],"w":["weight","w","w_eval"],"x":["x","log_weight","log_w","x_eval"],"N":["N","n","abundance","density"],"log_N":["log_N","logN","ln_N"],"log10_N":["log10_N","log10N"]}
    out=pd.DataFrame(index=df.index); out["source_name"]=source_name
    low={c.lower():c for c in df.columns}
    for std,names in aliases.items():
        col=next((low[n.lower()] for n in names if n.lower() in low), None); out[std]=df[col] if col else np.nan
    if out["species"].isna().all(): out["species"]="species_0"
    for c in ["t","w","x","N","log_N","log10_N"]: out[c]=pd.to_numeric(out[c], errors="coerce")
    out.loc[out["x"].isna() & out["w"].notna(), "x"] = np.log(np.maximum(out.loc[out["x"].isna() & out["w"].notna(), "w"], TINY))
    out.loc[out["log10_N"].isna() & out["N"].notna(), "log10_N"] = np.log10(np.maximum(out.loc[out["log10_N"].isna() & out["N"].notna(), "N"], TINY))
    out.loc[out["log10_N"].isna() & out["log_N"].notna(), "log10_N"] = out["log_N"] / np.log(10)
    out.loc[out["N"].isna() & out["log_N"].notna(), "N"] = np.exp(out["log_N"])
    out.loc[out["N"].isna() & out["log10_N"].notna(), "N"] = 10 ** out["log10_N"]
    return out[["source_name","t","species","w","x","N","log_N","log10_N"]]


def compare_page(run_df, selected_runs, clip, mode, markers):
    st.header("Compare runs")
    if len(selected_runs)<2: st.info("Select two or more runs in the sidebar."); return
    run_dirs=dict(zip(run_df.run_id, run_df.run_dir))
    for file, loader, metric_opts, title in [("fixed_diagnostic_history.csv",load_fixed_diagnostic_history,["fixed_residual_log_abs_p95","fixed_loss_pde","fixed_loss_ic","fixed_loss_bc","fixed_loss_unweighted"],"Fixed diagnostic overlay"),("loss_history.csv",load_loss_history,["loss","loss_unweighted","loss_pde","loss_ic","loss_bc","loss_timestep"],"Training loss overlay")]:
        metric=st.selectbox(f"{title} metric", metric_opts, key=title)
        def overlay(file=file,loader=loader,metric=metric,title=title):
            frames=[]
            for rid in selected_runs:
                d=loader(run_dirs[rid]);
                if d is not None and not d.empty and {"step",metric}.issubset(d.columns): frames.append(d[["step",metric]].assign(run_id=rid))
            if frames: st.plotly_chart(px.line(pd.concat(frames),x="step",y=metric,color="run_id",markers=markers,title=title),use_container_width=True)
            else: st.warning(f"No selected runs contain step and {metric} in {file}.")
        plot_with_desc(overlay, f"Overlays `{metric}` across selected runs to compare convergence histories.", line_desc(file, ["step",metric], "concatenate selected runs; no interpolation", False, False))
    st.dataframe(run_df[run_df.run_id.isin(selected_runs)], use_container_width=True)
    add_plot_description("One-row-per-run table of final metrics/config for selected runs.", line_desc("final_summary.csv/json, config.json, timing_summary.csv", RUN_COLUMNS, "scan immediate run folders", False, False))
    tsel=st.number_input("Selected time for profile overlays", value=0.0)
    def profile(resid=False):
        rows=[]
        for rid in selected_runs:
            f=load_fixed_fields(run_dirs[rid]);
            if f and not check_required_arrays(f,["t_eval","x_eval","log10_N","residual_log"]):
                i=nearest_index(f["t_eval"], tsel); arr=np.abs(f["residual_log"][i]) if resid else f["log10_N"][i]
                rows += [{"x":xx,"value":vv,"run_id":rid} for xx,vv in zip(f["x_eval"],arr)]
        if rows: st.plotly_chart(px.line(pd.DataFrame(rows),x="x",y="value",color="run_id",markers=markers,title="Residual profile overlay" if resid else "Abundance profile overlay"),use_container_width=True)
    plot_with_desc(lambda: profile(False), "Overlays final abundance profiles at nearest time; different x grids are allowed because each line carries its own x coordinates.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "nearest time per run; no interpolation", False, False))
    plot_with_desc(lambda: profile(True), "Overlays residual profiles at nearest time to compare where selected runs violate the PDE most.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","residual_log"], "nearest time; absolute value", False, False))
    ref=st.selectbox("Reference run", selected_runs); cmp=st.selectbox("Comparison run", [r for r in selected_runs if r!=ref])
    def diff(field="log10_N"):
        a,b=load_fixed_fields(run_dirs[ref]),load_fixed_fields(run_dirs[cmp])
        if not a or not b or check_required_arrays(a,["t_eval","x_eval",field]) or check_required_arrays(b,["t_eval","x_eval",field]): st.warning("Required fixed fields missing."); return
        if not (np.array_equal(a["t_eval"],b["t_eval"]) and np.array_equal(a["x_eval"],b["x_eval"])): st.warning("The fixed grids differ; difference heatmap is not plotted."); return
        z=(np.abs(b[field])-np.abs(a[field])) if field=="residual_log" else (b[field]-a[field]); make_heatmap(a["x_eval"],a["t_eval"],z,f"Difference heatmap: {cmp} - {ref}","delta",mode,clip)
    plot_with_desc(lambda: diff("log10_N"), "Shows abundance difference only when fixed t/x grids match exactly, avoiding invalid comparisons.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","log10_N"], "comparison minus reference; exact grid equality required", False, False))
    plot_with_desc(lambda: diff("residual_log"), "Shows change in absolute residual versus reference only on identical grids.", line_desc("fixed_grid_fields.npz/csv", ["t_eval","x_eval","residual_log"], "abs(comparison)-abs(reference); exact grid equality required", False, False))


def load_mizer_sources(local_paths: str, uploads) -> dict[str,pd.DataFrame]:
    out={}
    for raw in [p.strip() for p in local_paths.splitlines() if p.strip()]:
        df=safe_read_csv(Path(raw));
        if df is not None: out[Path(raw).name]=normalise_mizer_dataframe(df,Path(raw).name)
    for up in uploads or []:
        try: out[up.name]=normalise_mizer_dataframe(pd.read_csv(up), up.name)
        except Exception: st.warning(f"Could not read uploaded mizer CSV: {up.name}")
    return out


def interpolate_mizer_to_pinn(miz, species, t_grid, x_grid):
    sub=miz[miz.species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"]); z=[]
    if sub.empty: return None
    for tv in t_grid:
        nt=nearest_value(sub.t.to_numpy(), tv); s=sub[np.isclose(sub.t,nt)].sort_values("x")
        if s.empty: return None
        z.append(np.interp(x_grid, s.x.to_numpy(), s.log10_N.to_numpy(), left=np.nan, right=np.nan))
    return np.asarray(z)


def mizer_page(run_df, selected_run, selected_runs, mizers, clip, mode, markers):
    st.header("Mizer comparison")
    if not mizers: st.info("Provide one or more mizer CSV files in the sidebar."); return
    run_dirs=dict(zip(run_df.run_id, run_df.run_dir)); fields=load_fixed_fields(run_dirs.get(selected_run,"")) if selected_run else None
    if not fields or check_required_arrays(fields,["t_eval","x_eval","log10_N"]): st.warning("Selected PINN run needs fixed_grid_fields.npz/csv with t_eval, x_eval, log10_N."); return
    allm=pd.concat(mizers.values(), ignore_index=True); species=st.selectbox("Species", sorted(allm.species.astype(str).dropna().unique()))
    tsel=st.number_input("Comparison time", value=float(fields["t_eval"][0])); xsel=st.number_input("Comparison log-weight x", value=float(fields["x_eval"][0])); srcs=st.multiselect("Mizer sources", list(mizers), default=list(mizers))
    def prof():
        rows=[]; i=nearest_index(fields["t_eval"],tsel); rows += [{"x":x,"log10_N":v,"source":"PINN"} for x,v in zip(fields["x_eval"],fields["log10_N"][i])]
        for name in srcs:
            s=mizers[name][mizers[name].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"]); nt=nearest_value(s.t.to_numpy(),tsel); ss=s[np.isclose(s.t,nt)]; rows += [{"x":r.x,"log10_N":r.log10_N,"source":name} for r in ss.itertuples()]
        st.plotly_chart(px.line(pd.DataFrame(rows),x="x",y="log10_N",color="source",markers=markers,title="PINN and mizer abundance profile overlay"),use_container_width=True)
    plot_with_desc(prof, "Compares PINN abundance to selected mizer profiles at nearest available times.", line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t/x/log10_N or aliases"], "nearest time in each source; aliases normalised", False, False))
    def ts():
        rows=[]; j=nearest_index(fields["x_eval"],xsel); rows += [{"t":t,"log10_N":v,"source":"PINN"} for t,v in zip(fields["t_eval"],fields["log10_N"][:,j])]
        for name in srcs:
            s=mizers[name][mizers[name].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"])
            for tv,g in s.groupby("t"):
                nx=nearest_value(g.x.to_numpy(),xsel); val=g.loc[np.isclose(g.x,nx),"log10_N"].iloc[0]; rows.append({"t":tv,"log10_N":val,"source":name})
        st.plotly_chart(px.line(pd.DataFrame(rows),x="t",y="log10_N",color="source",markers=markers,title="Abundance time-series at selected weight"),use_container_width=True)
    plot_with_desc(ts, "Compares temporal abundance at the nearest log-weight in each source.", line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t","x","log10_N"], "nearest x per time/source", False, False))
    one=st.selectbox("Mizer source for error", list(mizers)); interp=st.checkbox("Use linear-in-x interpolation", True)
    miz_grid=interpolate_mizer_to_pinn(mizers[one], species, fields["t_eval"], fields["x_eval"])
    def delta_profile():
        if miz_grid is None: st.warning("Interpolation impossible: mizer source needs t, x, log10_N for selected species."); return
        i=nearest_index(fields["t_eval"],tsel); delta=fields["log10_N"][i]-miz_grid[i]
        st.warning("Mizer values were interpolated/nearest-time selected onto the PINN grid.")
        st.plotly_chart(px.line(pd.DataFrame({"x":fields["x_eval"],"delta":delta}),x="x",y="delta",markers=markers,title="PINN minus mizer profile"),use_container_width=True)
    plot_with_desc(delta_profile, "Shows signed abundance error along weight at the selected time.", line_desc("fixed_grid_fields.npz/csv plus one mizer CSV", ["t_eval","x_eval","log10_N","mizer t/x/log10_N"], f"nearest time and {'linear interpolation in x' if interp else 'nearest x selection'}; delta=PINN-mizer", False, False))
    def delta_heat():
        if miz_grid is None: st.warning("Interpolation impossible: mizer source needs t, x, log10_N for selected species."); return
        st.warning("Mizer values were interpolated using nearest time plus linear-in-x interpolation onto the PINN grid."); make_heatmap(fields["x_eval"],fields["t_eval"],fields["log10_N"]-miz_grid,"PINN minus mizer heatmap","delta_log10_N",mode,clip)
    plot_with_desc(delta_heat, "Shows PINN minus mizer abundance error across the fixed PINN grid.", line_desc("fixed_grid_fields.npz/csv plus one mizer CSV", ["t_eval","x_eval","log10_N","mizer t/x/log10_N"], "nearest mizer time and linear interpolation in x onto PINN grid", False, False))
    def multi():
        rows=[]
        for rid in selected_runs:
            f=load_fixed_fields(run_dirs[rid])
            if f and not check_required_arrays(f,["t_eval","x_eval","log10_N"]):
                i=nearest_index(f["t_eval"],tsel); rows += [{"x":x,"log10_N":v,"source":rid} for x,v in zip(f["x_eval"],f["log10_N"][i])]
        s=mizers[one][mizers[one].species.astype(str)==str(species)].dropna(subset=["t","x","log10_N"]); nt=nearest_value(s.t.to_numpy(),tsel); rows += [{"x":r.x,"log10_N":r.log10_N,"source":one} for r in s[np.isclose(s.t,nt)].itertuples()]
        st.plotly_chart(px.line(pd.DataFrame(rows),x="x",y="log10_N",color="source",markers=markers,title="Multiple PINN runs vs one mizer profile"),use_container_width=True)
    plot_with_desc(multi, "Compares several PINN abundance profiles with one mizer source at nearest times.", line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t/x/log10_N"], "nearest time per source; no cross-run grid assumption", False, False))
    for by, axis, vals in [("time",1,fields["t_eval"]),("weight",0,fields["x_eval"] )]:
        def err(by=by,axis=axis,vals=vals):
            if miz_grid is None: st.warning("Interpolation impossible."); return
            ad=np.abs(fields["log10_N"]-miz_grid); d=pd.DataFrame({by:vals,"mean_abs_delta_log10_N":np.nanmean(ad,axis=axis),"p95_abs_delta_log10_N":np.nanpercentile(ad,95,axis=axis)}).melt(by,var_name="stat",value_name="error")
            st.plotly_chart(px.line(d,x=by,y="error",color="stat",markers=markers,title=f"Error summary by {by}"),use_container_width=True)
        plot_with_desc(err, f"Summarises absolute PINN-mizer error by {by} after placing mizer data on the PINN grid.", line_desc("fixed_grid_fields.npz/csv plus mizer CSV", ["t/x/log10_N"], "nearest-time plus linear-x interpolation; mean and p95 abs delta", False, False))


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
    with st.sidebar:
        selected_run=st.selectbox("Single selected run", run_ids, index=0 if run_ids else None) if run_ids else None
        selected_runs=st.multiselect("Runs for comparison", run_ids, default=run_ids[:min(3,len(run_ids))])
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
