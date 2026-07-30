"""Data-loss diagnostics and UI overrides for the PINNmizer HPC viewer."""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_COLUMNS = [
    "final_loss_data", "final_objective_loss_data", "final_weighted_loss_data",
    "final_n_data_obs", "final_data_log_residual_abs_mean",
    "final_data_log_residual_abs_max", "lambda_data", "data_csv",
    "data_default_cv", "estimate_data_cv", "data_cv_scope", "data_cv_init",
    "data_cv_lower", "data_cv_upper", "data_cv_lr", "initial_w_data",
]
DATA_CONFIG_FIELDS = [
    "lambda_data", "data_csv", "data_default_cv", "estimate_data_cv",
    "data_cv_scope", "data_cv_init", "data_cv_lower", "data_cv_upper",
    "data_cv_lr", "initial_w_data",
]
TRAINING_METRICS = [
    "loss", "loss_unweighted", "loss_pde", "loss_ic", "loss_bc", "loss_data",
    "objective_loss_pde", "objective_loss_ic", "objective_loss_bc",
    "objective_loss_data", "w_pde", "w_ic", "w_bc", "w_data", "n_data_obs",
    "data_log_residual_abs_mean", "data_log_residual_abs_max",
]


@st.cache_data(show_spinner=False)
def _read_csv(run_dir: str | Path, name: str) -> pd.DataFrame | None:
    path = Path(run_dir).expanduser() / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


def _last(df: pd.DataFrame, name: str) -> float:
    if name not in df:
        return float("nan")
    x = pd.to_numeric(df[name], errors="coerce").dropna()
    return float(x.iloc[-1]) if not x.empty else float("nan")


def _species_names(run_dir: Path, impl) -> dict[int, str]:
    grid = impl.load_final_predictions(run_dir)
    if grid is None or grid.empty or not {"species_idx", "species"}.issubset(grid):
        return {}
    rows = grid[["species_idx", "species"]].dropna().drop_duplicates("species_idx")
    return {int(r.species_idx): str(r.species) for r in rows.itertuples(index=False)}


def _prepare_predictions(raw: pd.DataFrame, run_dir: Path, impl) -> pd.DataFrame:
    d = raw.copy()
    numeric = [
        "species_idx", "gear_idx", "t_start", "t_end", "w_min", "w_max",
        "value", "prediction", "log_residual", "cv", "sd_log", "cv_used",
        "sd_log_used", "loss_contribution", "true_cv", "true_sd_log", "value_true",
    ]
    for name in numeric:
        if name in d:
            d[name] = pd.to_numeric(d[name], errors="coerce")
    if "sd_log_used" not in d and "sd_log" in d:
        d["sd_log_used"] = d["sd_log"]
    if "log_residual" not in d and {"value", "prediction"}.issubset(d):
        d["log_residual"] = np.log(d["value"]) - np.log(d["prediction"])
    required = ["value", "prediction", "log_residual", "sd_log_used"]
    missing = [name for name in required if name not in d]
    if missing:
        raise ValueError("data_predictions_final.csv is missing: " + ", ".join(missing))
    keep = np.ones(len(d), dtype=bool)
    for name in required:
        keep &= np.isfinite(d[name].to_numpy(dtype=float))
    keep &= (d["value"].to_numpy(dtype=float) > 0)
    keep &= (d["prediction"].to_numpy(dtype=float) > 0)
    keep &= (d["sd_log_used"].to_numpy(dtype=float) > 0)
    d = d.loc[keep].copy()
    d["standardised_log_residual"] = d["log_residual"] / d["sd_log_used"]
    d["half_z2"] = 0.5 * d["standardised_log_residual"] ** 2
    start = d["t_start"] if "t_start" in d else pd.Series(np.nan, index=d.index)
    end = d["t_end"] if "t_end" in d else start
    d["t_mid"] = 0.5 * (start + end)
    d["observation_id"] = np.arange(len(d), dtype=int)
    names = _species_names(run_dir, impl)
    if "species_idx" in d:
        d["species"] = [
            names.get(int(i), f"species_{int(i)}") if np.isfinite(i) else "unknown"
            for i in d["species_idx"]
        ]
    else:
        d["species"] = "unknown"
    for name in ["obs_type", "dataset"]:
        if name not in d:
            d[name] = "unspecified"
        d[name] = d[name].fillna("").astype(str).replace("", "unspecified")
    return d


def _normal_quantiles(n: int) -> np.ndarray:
    normal = NormalDist()
    p = (np.arange(1, n + 1, dtype=float) - 0.5) / n
    return np.asarray([normal.inv_cdf(float(v)) for v in p])


def _data_summary(run_dir: str | Path, impl) -> dict[str, float] | None:
    raw = _read_csv(run_dir, "data_predictions_final.csv")
    if raw is None or raw.empty:
        return None
    try:
        d = _prepare_predictions(raw, Path(run_dir), impl)
    except ValueError:
        return None
    if d.empty:
        return None
    z = d["standardised_log_residual"].to_numpy(dtype=float)
    nll = pd.to_numeric(d["loss_contribution"], errors="coerce").mean() if "loss_contribution" in d else np.nan
    return {
        "n_observations": float(len(d)), "mean_data_nll": float(nll),
        "mean_standardised_residual": float(np.mean(z)),
        "rmse_standardised_residual": float(np.sqrt(np.mean(z ** 2))),
        "fraction_within_1_96": float(np.mean(np.abs(z) <= 1.96)),
    }


def _data_config_differences(run_df: pd.DataFrame, selected_runs: list[str]) -> None:
    fields = [f for f in DATA_CONFIG_FIELDS if f in run_df]
    if len(selected_runs) < 2 or not fields:
        return
    selected = run_df[run_df.run_id.isin(selected_runs)].set_index("run_id")
    rows = []
    for field in fields:
        values = [selected.at[r, field] if r in selected.index else np.nan for r in selected_runs]
        keys = [("NA" if pd.isna(v) else str(v)) for v in values]
        if len(set(keys)) > 1:
            rows.append({"field": field, **dict(zip(selected_runs, keys))})
    if rows:
        st.subheader("Selected-run data-loss configuration differences")
        st.dataframe(pd.DataFrame(rows).set_index("field"), use_container_width=True)


def _data_page(run_dir: Path, markers: bool, impl) -> None:
    st.header("Single run: data diagnostics")
    raw = _read_csv(run_dir, "data_predictions_final.csv")
    if raw is None:
        config = impl.load_config(run_dir)
        suffix = f" Configured data file: `{config.get('data_csv')}`." if config.get("data_csv") else ""
        st.warning(
            "data_predictions_final.csv was not found. The QQ plot needs one prediction "
            "and one effective sd_log per observation; loss_history.csv has only summaries." + suffix
        )
        return
    if raw.empty:
        st.info("data_predictions_final.csv exists but is empty.")
        return
    try:
        d = _prepare_predictions(raw, run_dir, impl)
    except ValueError as exc:
        st.error(str(exc))
        return
    if d.empty:
        st.info("No valid positive observation/prediction pairs with positive sd_log.")
        return

    a, b, c = st.columns(3)
    species = a.multiselect("Species", sorted(d.species.unique()), default=sorted(d.species.unique()), key="data_species")
    obs_type = b.multiselect("Observation type", sorted(d.obs_type.unique()), default=sorted(d.obs_type.unique()), key="data_type")
    dataset = c.multiselect("Dataset", sorted(d.dataset.unique()), default=sorted(d.dataset.unique()), key="data_dataset")
    d = d[d.species.isin(species) & d.obs_type.isin(obs_type) & d.dataset.isin(dataset)].copy()
    if d.empty:
        st.info("The filters select no observations.")
        return

    z = d["standardised_log_residual"].to_numpy(dtype=float)
    nll = pd.to_numeric(d["loss_contribution"], errors="coerce").mean() if "loss_contribution" in d else np.nan
    cards = [
        ("Observations", len(d), False), ("Mean data NLL", nll, False),
        ("Mean standardised residual", np.mean(z), False),
        ("Standardised residual RMSE", np.sqrt(np.mean(z ** 2)), False),
        ("Within ±1.96", np.mean(np.abs(z) <= 1.96), True),
    ]
    for col, (label, value, percentage) in zip(st.columns(5), cards):
        col.metric(label, f"{value:.1%}" if percentage else impl.format_scalar(value))
    group = st.selectbox("Diagnostic colour grouping", ["species", "obs_type", "dataset"])

    def observed_predicted():
        fig = px.scatter(
            d, x="value", y="prediction", color=group, log_x=True, log_y=True,
            hover_data=["observation_id", "species", "obs_type", "dataset", "t_start", "t_end", "sd_log_used"],
            title="Observed versus predicted data",
        )
        lo = float(np.nanmin([d.value.min(), d.prediction.min()]))
        hi = float(np.nanmax([d.value.max(), d.prediction.max()]))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1"))
        st.plotly_chart(fig, use_container_width=True)

    impl.plot_with_desc(
        observed_predicted,
        "Checks calibration and scale-dependent bias. The 1:1 line is exact agreement.",
        impl.line_desc("data_predictions_final.csv", ["value", "prediction", group], "positive values; log-log axes; 1:1 line", True, False),
    )

    def qq():
        observed = np.sort(z)
        if observed.size < 2:
            st.info("At least two observations are required for a QQ plot.")
            return
        theoretical = _normal_quantiles(len(observed))
        lo = float(min(observed.min(), theoretical.min()))
        hi = float(max(observed.max(), theoretical.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=observed, mode="markers", name="Residual quantiles"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Normal reference"))
        fig.update_layout(
            title="Normal QQ plot of standardised log residuals",
            xaxis_title="Theoretical standard-normal quantile",
            yaxis_title="Observed standardised log-residual quantile",
        )
        st.plotly_chart(fig, use_container_width=True)

    impl.plot_with_desc(
        qq,
        "Assesses the lognormal observation model. Heavy tails, skew, or an S-shape indicate outliers, bias, heterogeneous uncertainty, dependence, or misspecification; this is not a formal independent-residual test.",
        impl.line_desc("data_predictions_final.csv", ["log_residual", "sd_log_used"], "z=log_residual/sd_log_used; standard-normal plotting-position quantiles", False, False),
    )

    def residual_fitted():
        fig = px.scatter(
            d, x="prediction", y="standardised_log_residual", color=group, log_x=True,
            hover_data=["observation_id", "value", "species", "obs_type", "dataset", "sd_log_used"],
            title="Standardised log residual versus fitted value",
        )
        fig.add_hline(y=0)
        fig.add_hline(y=1.96, line_dash="dash")
        fig.add_hline(y=-1.96, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    impl.plot_with_desc(
        residual_fitted,
        "Checks fitted-value bias and heteroscedasticity after dividing by observation uncertainty.",
        impl.line_desc("data_predictions_final.csv", ["prediction", "log_residual", "sd_log_used", group], "standardised residual; 0 and ±1.96 lines", False, False),
    )

    def coverage():
        if not np.isfinite(d.t_mid).any():
            st.info("Observation times are unavailable.")
            return
        fig = px.scatter(
            d, x="t_mid", y="species", color="dataset", symbol="obs_type",
            hover_data=["observation_id", "t_start", "t_end", "w_min", "w_max", "value"],
            title="Observation coverage by time, species, and type",
        )
        st.plotly_chart(fig, use_container_width=True)

    impl.plot_with_desc(
        coverage,
        "Shows where observations constrain the time/species domain; sparse regions are not directly validated by the data loss.",
        impl.line_desc("data_predictions_final.csv", ["t_start", "t_end", "species", "obs_type", "dataset"], "midpoint=(t_start+t_end)/2", False, False),
    )

    def worst():
        n = min(25, len(d))
        x = d.nlargest(n, "half_z2").copy()
        x["label"] = x.apply(lambda r: f"#{int(r.observation_id)} | {r.species} | {r.obs_type} | t={r.t_mid:.4g}", axis=1)
        fig = px.bar(
            x.sort_values("half_z2"), x="half_z2", y="label", orientation="h",
            hover_data=["value", "prediction", "standardised_log_residual", "sd_log_used", "dataset"],
            title=f"Largest {n} standardised observation misfits",
        )
        st.plotly_chart(fig, use_container_width=True)

    impl.plot_with_desc(
        worst,
        "Ranks observations by the squared standardised residual component, not full NLL, so log(sd_log) does not distort the outlier ranking.",
        impl.line_desc("data_predictions_final.csv", ["log_residual", "sd_log_used"], "0.5*(log_residual/sd_log_used)^2; top 25", False, False),
    )

    cv_history = _read_csv(run_dir, "data_cv_history.csv")
    if cv_history is not None and not cv_history.empty and {"step", "cv"}.issubset(cv_history):
        colour = "species" if "species" in cv_history else "species_idx" if "species_idx" in cv_history else None
        st.plotly_chart(
            px.line(cv_history, x="step", y="cv", color=colour, markers=markers, title="Estimated observation CV trajectory"),
            use_container_width=True,
        )
    else:
        fixed = pd.to_numeric(d["cv_used"], errors="coerce").dropna() if "cv_used" in d else pd.Series(dtype=float)
        if not fixed.empty:
            st.caption(f"No fitted-CV history. Final used CV range: {fixed.min():.4g} to {fixed.max():.4g}.")
    estimated = _read_csv(run_dir, "estimated_data_cv.csv")
    if estimated is not None and not estimated.empty:
        st.subheader("Final estimated observation CV")
        st.dataframe(estimated, use_container_width=True)
    if "true_cv" in d:
        comparison = d[[c for c in ["species", "obs_type", "true_cv", "cv_used"] if c in d]].dropna().drop_duplicates()
        if not comparison.empty:
            st.subheader("True and used/fitted CV")
            st.dataframe(comparison, use_container_width=True)
    st.subheader("Filtered observation predictions")
    st.dataframe(d, use_container_width=True)


def install(impl) -> None:
    original_history = impl.history_page
    original_compare = impl.compare_page

    impl.RUN_COLUMNS[:] = [c for c in impl.RUN_COLUMNS if c != "lambda_timestep"]
    for c in DATA_COLUMNS:
        if c not in impl.RUN_COLUMNS:
            impl.RUN_COLUMNS.append(c)
    impl.PLOT_INTERPRETATIONS.update({
        "Total loss": "The optimiser objective combines weighted PDE, IC, BC, and observation-data terms.",
        "Total unweighted loss": "The raw sum of PDE, IC, BC, and data terms before lambda and adaptive weighting.",
        "Unscaled loss terms": "Raw PDE, IC, BC, and lognormal observation-NLL terms. Their normalisations differ.",
        "Objective loss terms": "Actual optimiser contributions lambda_k*w_k*loss_k for PDE, IC, BC, and data.",
        "Adaptive weight trajectories": "Adaptive optimisation multipliers for PDE, IC, BC, and data; these are not fit metrics.",
        "Data residual summaries": "Absolute log residual summaries |log(observed)-log(predicted)| over active observations.",
        "Data value ranges": "Observed and predicted minima/maxima for scale-collapse and range-mismatch checks.",
        "Active data observations": "Number of observations admitted by the current causal time window.",
    })

    def run_browser(run_df, selected_runs, log_y, markers):
        st.header("Run browser")
        filtered = run_df.copy()
        for col in ["status", "model_arch", "collocation_strategy", "loss_weighting", "causal_loss", "weight_factorization", "seed", "estimate_data_cv", "data_cv_scope"]:
            if col not in filtered:
                continue
            values = sorted(str(v) for v in filtered[col].dropna().unique() if str(v) != "")
            chosen = st.multiselect(f"Filter {col}", values, key=f"filter_{col}")
            if chosen:
                filtered = filtered[filtered[col].astype(str).isin(chosen)]
        metric = st.selectbox("Training-history metric for selected iteration", TRAINING_METRICS)
        step = st.number_input("Iteration for direct comparison", min_value=0, value=1000, step=100)
        filtered = impl.add_loss_at_step(filtered, int(step), metric)
        st.caption("Uses the nearest logged step in each run's loss_history.csv.")
        event = st.dataframe(filtered, use_container_width=True, key="run_browser_table", on_select="rerun", selection_mode="single-row")
        if event.selection.rows:
            selected = str(filtered.iloc[event.selection.rows[0]]["run_id"])
            if selected != st.session_state.get("last_table_selected_run"):
                st.session_state["last_table_selected_run"] = selected
                st.session_state["selected_run"] = selected
                st.rerun()
        if len(selected_runs) >= 2:
            impl.show_architecture_differences(filtered, selected_runs)
            _data_config_differences(filtered, selected_runs)
        ranking = [
            f"{metric}_at_selected_step", "final_fixed_residual_log_abs_p95", "final_loss",
            "final_loss_unweighted", "final_loss_pde", "final_loss_ic", "final_loss_bc",
            "final_loss_data", "final_objective_loss_data", "final_data_log_residual_abs_mean",
            "final_data_log_residual_abs_max", "seconds_per_step",
        ]
        ranking = [x for x in ranking if x in filtered]
        rank_metric = st.selectbox("Ranking metric", ranking)
        x = filtered[["run_id", rank_metric]].copy()
        x[rank_metric] = pd.to_numeric(x[rank_metric], errors="coerce")
        x = x.dropna().sort_values(rank_metric)
        if x.empty:
            st.info(f"No values available for {rank_metric}.")
        else:
            st.plotly_chart(px.bar(x, x=rank_metric, y="run_id", orientation="h", title=f"Run ranking: {rank_metric}"), use_container_width=True)
        terms = ["final_loss_pde", "final_loss_ic", "final_loss_bc", "final_loss_data"]
        x = filtered[filtered.run_id.isin(selected_runs)][["run_id"] + terms].melt("run_id", var_name="term", value_name="loss").dropna()
        if not x.empty:
            st.plotly_chart(px.bar(x, x="run_id", y="loss", color="term", barmode="group", log_y=log_y, title="Final training-loss decomposition"), use_container_width=True)
        group = st.selectbox("Scatter colour", ["model_arch", "collocation_strategy", "loss_weighting", "causal_loss", "weight_factorization", "estimate_data_cv"])
        quality = st.selectbox("Quality metric", ["final_fixed_residual_log_abs_p95", "final_loss_data", "final_objective_loss_data", "final_data_log_residual_abs_mean", "final_data_log_residual_abs_max"])
        st.plotly_chart(px.scatter(filtered, x="seconds_per_step", y=quality, color=group, hover_data=["run_id", "final_loss", "final_loss_data"], title="Speed-quality scatter"), use_container_width=True)
        architecture = st.selectbox("Architecture/hyperparameter x", ["hidden_width", "hidden_layers", "fourier_scale", "fourier_num_features", "rwf_sigma", "r3_population_size", "lambda_data", "data_default_cv"])
        st.plotly_chart(px.scatter(filtered, x=architecture, y=quality, color=group, hover_data=["run_id"], title="Architecture / hyperparameter scatter"), use_container_width=True)

    def history(run_dir, fixed, log_y, markers):
        if fixed:
            return original_history(run_dir, fixed, log_y, markers)
        df = impl.load_loss_history(run_dir)
        st.header("Single run: training")
        if df is None:
            st.warning("loss_history.csv not found for this run")
            return
        if df.empty:
            st.info("loss_history.csv exists but is empty")
            return
        cards = [c for c in ["loss_data", "objective_loss_data", "w_data", "data_log_residual_abs_mean", "n_data_obs"] if c in df and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if cards:
            for col, name in zip(st.columns(len(cards)), cards):
                col.metric(name, impl.format_scalar(_last(df, name)))
        plots = [
            ("Total loss", ["loss"]), ("Total unweighted loss", ["loss_unweighted"]),
            ("Unscaled loss terms", ["loss_pde", "loss_ic", "loss_bc", "loss_data"]),
            ("Objective loss terms", ["objective_loss_pde", "objective_loss_ic", "objective_loss_bc", "objective_loss_data"]),
            ("Adaptive weight trajectories", ["w_pde", "w_ic", "w_bc", "w_data"]),
            ("Data residual summaries", ["data_log_residual_abs_mean", "data_log_residual_abs_max"]),
            ("Data value ranges", ["data_pred_min", "data_pred_max", "data_obs_min", "data_obs_max"]),
            ("Active data observations", ["n_data_obs"]), ("Gradient norm", ["grad_norm"]),
            ("Causal curriculum", ["causal_fraction", "t_max_current"]),
            ("Causal chunk diagnostics", ["pde_causal_weight_first", "pde_causal_weight_mean", "pde_causal_weight_last", "pde_causal_chunk_loss_mean", "pde_causal_chunk_loss_max"]),
        ]
        for title, candidates in plots:
            ys = [c for c in candidates if c in df and pd.to_numeric(df[c], errors="coerce").notna().any()]
            if not ys:
                continue
            use_log = log_y and "Causal" not in title and title != "Active data observations"
            impl.plot_with_desc(
                lambda ys=ys, title=title, use_log=use_log: impl.make_multi_line_plot(df, "step", ys, title, "value", use_log, markers),
                impl.PLOT_INTERPRETATIONS.get(title, f"Tracks {title.lower()} from loss_history.csv."),
                impl.line_desc("loss_history.csv", ["step"] + ys, "non-positive values are omitted only on log plots", use_log, False),
            )

    class _CompareStreamlitProxy:
        def __getattr__(self, name):
            return getattr(st, name)
        def selectbox(self, label, options, *args, **kwargs):
            if label == "Training loss overlay metric":
                options = TRAINING_METRICS
            return st.selectbox(label, options, *args, **kwargs)

    def compare(run_df, selected_runs, clip, mode, markers):
        if len(selected_runs) >= 2:
            rows = []
            run_dirs = dict(zip(run_df.run_id, run_df.run_dir))
            for run_id in selected_runs:
                summary = _data_summary(run_dirs[run_id], impl)
                if summary:
                    rows.append({"run_id": run_id, **summary})
            if rows:
                st.subheader("Final observation-fit comparison")
                table = pd.DataFrame(rows)
                metric = st.selectbox("Observation-fit comparison metric", ["mean_data_nll", "rmse_standardised_residual", "mean_standardised_residual", "fraction_within_1_96"])
                st.plotly_chart(px.bar(table, x="run_id", y=metric, title=metric), use_container_width=True)
                st.dataframe(table, use_container_width=True)
        old = impl.st
        impl.st = _CompareStreamlitProxy()
        try:
            original_compare(run_df, selected_runs, clip, mode, markers)
        finally:
            impl.st = old

    def file_view(run_dir):
        st.header("File/config view")
        for name in ["config.json", "final_summary.json", "final_summary.csv", "timing_summary.csv", "data_predictions_final.csv", "data_cv_history.csv", "estimated_data_cv.csv", "run_command.txt"]:
            st.subheader(name)
            path = run_dir / name
            if name.endswith(".csv"):
                d = impl.safe_read_csv(path)
                st.info(f"{name} not found or unreadable") if d is None else st.dataframe(d, use_container_width=True)
            elif name.endswith(".json"):
                d = impl.safe_read_json(path)
                st.info(f"{name} not found or unreadable") if not d else st.json(d)
            else:
                text = impl.safe_read_text(path)
                st.info(f"{name} not found or unreadable") if not text else st.code(text)

    def main():
        st.set_page_config(page_title="PINNmizer HPC Viewer", layout="wide")
        st.title("PINNmizer HPC run viewer")
        with st.sidebar:
            run_root = Path(st.text_input("Run root path", str(impl.DEFAULT_RUN_ROOT))).expanduser()
            if st.button("Refresh / re-scan"):
                st.cache_data.clear()
            label_mode = st.selectbox("Run label mode", ["folder name", "short folder name", "model_arch + seed", "custom label assembled from selected config fields"])
            if label_mode.startswith("custom"):
                st.multiselect("Custom label config fields", impl.RUN_COLUMNS, default=["model_arch", "seed"])
            log_y = st.checkbox("Log y-axis where relevant", True)
            markers = st.checkbox("Show points", True)
            clip = st.checkbox("Quantile clipping for heatmaps", True)
            heat_mode = st.selectbox("Heatmap colour range", ["auto", "symmetric around zero", "percentile clipped"], index=2)
            mizer_paths = st.text_area("Mizer CSV local paths (one per line)")
            uploads = st.file_uploader("Upload mizer CSVs", type="csv", accept_multiple_files=True)
        run_df = impl.scan_runs(run_root)
        if run_df.empty:
            st.warning(f"No run folders found under {run_root}")
        run_ids = run_df.run_id.tolist()
        if run_ids and st.session_state.get("selected_run") not in run_ids:
            st.session_state["selected_run"] = run_ids[0]
        if not run_ids:
            st.session_state["selected_run"] = None
        with st.sidebar:
            selected_run = st.selectbox("Single selected run", run_ids, index=run_ids.index(st.session_state["selected_run"]) if run_ids and st.session_state["selected_run"] in run_ids else 0) if run_ids else None
            if selected_run:
                st.session_state["selected_run"] = selected_run
            selected_runs = st.multiselect("Runs for comparison", run_ids, default=run_ids[:min(3, len(run_ids))])
        selected_run = st.session_state.get("selected_run")
        run_dir = Path(run_df.loc[run_df.run_id == selected_run, "run_dir"].iloc[0]) if selected_run else Path("")
        mizers = impl.load_mizer_sources(mizer_paths, uploads)
        tabs = st.tabs(["Run browser", "Single run: training", "Single run: data", "Single run: fixed diagnostics", "Single run: fields", "Compare runs", "Mizer comparison", "File/config view"])
        with tabs[0]: run_browser(run_df, selected_runs, log_y, markers)
        with tabs[1]: history(run_dir, False, log_y, markers) if selected_run else st.info("Select a run.")
        with tabs[2]: _data_page(run_dir, markers, impl) if selected_run else st.info("Select a run.")
        with tabs[3]: history(run_dir, True, log_y, markers) if selected_run else st.info("Select a run.")
        with tabs[4]: impl.fields_page(run_dir, clip, heat_mode, markers) if selected_run else st.info("Select a run.")
        with tabs[5]: compare(run_df, selected_runs, clip, heat_mode, markers)
        with tabs[6]: impl.mizer_page(run_df, selected_run, selected_runs, mizers, clip, heat_mode, markers) if selected_run else st.info("Select a PINN run.")
        with tabs[7]: file_view(run_dir) if selected_run else st.info("Select a run.")

    impl.run_browser_page = run_browser
    impl.history_page = history
    impl.data_page = lambda run_dir, markers: _data_page(run_dir, markers, impl)
    impl.compare_page = compare
    impl.file_view_page = file_view
    impl.main = main
