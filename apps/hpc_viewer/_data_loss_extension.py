"""Observation-data diagnostics for the PINNmizer HPC viewer.

This extension deliberately owns only the Data tab.  The run browser, training,
fixed diagnostics, fields, run comparison, mizer comparison, and file/config
pages remain the base viewer implementations.
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
    keep &= d["value"].to_numpy(dtype=float) > 0
    keep &= d["prediction"].to_numpy(dtype=float) > 0
    keep &= d["sd_log_used"].to_numpy(dtype=float) > 0
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
        ("Observations", len(d), False),
        ("Mean data NLL", nll, False),
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
    """Add only the data page while preserving every base-viewer page unchanged."""

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
            selected_run = st.selectbox(
                "Single selected run",
                run_ids,
                index=run_ids.index(st.session_state["selected_run"])
                if run_ids and st.session_state["selected_run"] in run_ids else 0,
            ) if run_ids else None
            if selected_run:
                st.session_state["selected_run"] = selected_run
            selected_runs = st.multiselect("Runs for comparison", run_ids, default=run_ids[:min(3, len(run_ids))])

        selected_run = st.session_state.get("selected_run")
        run_dir = Path(run_df.loc[run_df.run_id == selected_run, "run_dir"].iloc[0]) if selected_run else Path("")
        mizers = impl.load_mizer_sources(mizer_paths, uploads)
        tabs = st.tabs([
            "Run browser",
            "Single run: training",
            "Single run: data",
            "Single run: fixed diagnostics",
            "Single run: fields",
            "Compare runs",
            "Mizer comparison",
            "File/config view",
        ])
        with tabs[0]:
            impl.run_browser_page(run_df, selected_runs, log_y, markers)
        with tabs[1]:
            impl.history_page(run_dir, False, log_y, markers) if selected_run else st.info("Select a run.")
        with tabs[2]:
            _data_page(run_dir, markers, impl) if selected_run else st.info("Select a run.")
        with tabs[3]:
            impl.history_page(run_dir, True, log_y, markers) if selected_run else st.info("Select a run.")
        with tabs[4]:
            impl.fields_page(run_dir, clip, heat_mode, markers) if selected_run else st.info("Select a run.")
        with tabs[5]:
            impl.compare_page(run_df, selected_runs, clip, heat_mode, markers)
        with tabs[6]:
            impl.mizer_page(run_df, selected_run, selected_runs, mizers, clip, heat_mode, markers) if selected_run else st.info("Select a PINN run.")
        with tabs[7]:
            impl.file_view_page(run_dir) if selected_run else st.info("Select a run.")

    impl.data_page = lambda run_dir, markers: _data_page(run_dir, markers, impl)
    impl.main = main
