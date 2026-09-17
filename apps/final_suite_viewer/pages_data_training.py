"""Saved observation-data (D1–D8) and training-history (T1–T12) pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analytics import (available_columns, denoising_metrics,
                        interval_seconds_per_step, normal_quantiles,
                        prepare_observations, rank_misfits)
from .components import plot_explanation, plot_scale_controls
from .loaders import load_prediction_state, read_csv, read_json


def _run(rows):
    return next((row for row in rows if row["task_id"] == st.session_state.get("selected_task_id") and row["selected_instance"]), None)


def _explain(interpretation, calculation, inputs, selection="Current in-memory filters", alignment="No interpolation; saved rows only"):
    plot_explanation(st, interpretation=interpretation, calculation=calculation, inputs=inputs, selection=selection, alignment=alignment)


def _one_to_one(fig, data, columns):
    values = np.concatenate([pd.to_numeric(data[c], errors="coerce").to_numpy() for c in columns]); values = values[np.isfinite(values) & (values > 0)]
    if values.size: fig.add_trace(go.Scatter(x=[values.min(), values.max()], y=[values.min(), values.max()], mode="lines", name="1:1"))
    return fig


def _observation_equations(types):
    with st.expander("Exact saved-observation operator definitions", expanded=False):
        st.markdown("The saved `prediction` was produced during training by: **N_pred(t,w) → observation operator → y_pred → log residual → standardised residual → likelihood contribution → optional discrepancy gate**.")
        if set(types) & {"biomass", "survey_biomass"}: st.latex(r"B_{pred}=mean_t\sum_w N_i(t,w)w\,dw")
        if "survey_biomass" in types: st.markdown("Survey biomass multiplies this integral by the configured survey catchability `q`.")
        if "survey_abundance" in types: st.latex(r"A_{pred}=q\,mean_t\sum_w N_i(t,w)\,dw")
        if set(types) & {"catch_total", "catch_gear"}:
            st.latex(r"F_{g,i}(t,w)=E_g(t)q_{g,i}s_{g,i}(w)")
            st.latex(r"Y_{g,i}(t)=\sum_w F_{g,i}(t,w)N_i(t,w)w\,dw")
            st.latex(r"t_k=t_{start}+k(t_{end}-t_{start})/3,\ k=0,1,2;\quad C_{pred}=mean_kY(t_k)|t_{end}-t_{start}|")
            st.caption("The three interval points are left-closed: the right endpoint is excluded.")
        st.latex(r"r_j=\log(y_j+\epsilon)-\log(\hat y_j+\epsilon),\quad \sigma_{log,j}=\sqrt{\log(1+CV_j^2)},\quad z_j=r_j/\sigma_{log,j}")
        st.latex(r"\ell_j=0.5z_j^2+\log(\sigma_{log,j}),\quad loss_{data}=mean_j\ell_j")
        st.caption("Saved sd_log_used/cv_used are used when present; uncertainty is not re-estimated by the viewer.")


def data_page(rows):
    st.header("Selected run: Data"); row = _run(rows)
    if not row: st.info("Select a discovered run from the Suite map."); return
    run_dir = str(row["selected_instance"].run_dir); raw = read_csv(run_dir, "data_predictions_final.csv")
    if raw is None or raw.empty: st.info("This run has no saved data_predictions_final.csv; observation predictions will not be recomputed."); return
    state = load_prediction_state(run_dir); names = {} if state is None else dict(state[["species_idx","species"]].drop_duplicates().itertuples(index=False, name=None))
    try: data = prepare_observations(raw, names)
    except Exception as exc: st.error(f"Saved observation schema is unusable: {exc}"); return

    with st.sidebar:
        st.markdown("### Data filters")
        species = st.multiselect("Species", sorted(data.species.unique()), default=sorted(data.species.unique()), key="data_species_filter")
        types = st.multiselect("Observation type", sorted(data.obs_type.unique()), default=sorted(data.obs_type.unique()), key="data_type_filter")
        datasets = st.multiselect("Dataset/source", sorted(data.dataset.unique()), default=sorted(data.dataset.unique()), key="data_dataset_filter")
        gears_all = sorted(data.gear_idx.dropna().unique().tolist()) if "gear_idx" in data else []
        gears = st.multiselect("Gear", gears_all, default=gears_all, key="data_gear_filter") if gears_all else None
        time_values = np.sort(np.unique(np.r_[data.t_start, data.t_end]))
        time_range = st.select_slider("Observation time range", time_values.tolist(), value=(float(time_values[0]), float(time_values[-1])), key="data_time_filter")

    data = data[data.species.isin(species) & data.obs_type.isin(types) & data.dataset.isin(datasets) & (data.t_end >= time_range[0]) & (data.t_start <= time_range[1])]
    if gears is not None: data = data[data.gear_idx.isin(gears)]
    if data.empty: st.info("The current filters select no observations."); return
    _observation_equations(data.obs_type.unique())

    st.subheader("D1 · Observed versus predicted")
    hover = [c for c in ["species","obs_type","gear_idx","t_start","t_end","w_min","w_max","dataset","cv_used","sd_log_used"] if c in data]
    fig = _one_to_one(px.scatter(data, x="value", y="prediction", color="obs_type", log_x=True, log_y=True, hover_data=hover), data, ["value","prediction"]); st.plotly_chart(fig, use_container_width=True)
    _explain("Compares observed values with observation-operator predictions; prediction is not a direct read of N.", r"N_{pred}(t,w)\xrightarrow{observation\ operator}\hat y;\quad y=\hat y\ on\ the\ 1{:}1\ line", ["data_predictions_final.csv"])

    if "value_true" in data and np.isfinite(data.value_true).any():
        truth = data[np.isfinite(data.value_true) & (data.value_true > 0)].copy()
        st.subheader("D2 · Observation truth, noisy observation, and PINN")
        long = truth.melt(id_vars=["value_true"]+hover, value_vars=["value","prediction"], var_name="series", value_name="comparison")
        fig = _one_to_one(px.scatter(long, x="value_true", y="comparison", color="series", log_x=True, log_y=True, hover_data=hover), long.rename(columns={"value_true":"a","comparison":"b"}), ["a","b"]); st.plotly_chart(fig, use_container_width=True)
        _explain("Compares noisy observations and PINN-derived observations with observation truth; this is not latent-state truth.", r"x=y_{true};\quad y\in\{y_{obs},\hat y_{PINN}\}", ["data_predictions_final.csv: value_true, value, prediction"])
        st.subheader("D3 · Denoising")
        denoise, metrics = denoising_metrics(truth); fig = px.scatter(denoise, x="e_obs", y="e_PINN", color="obs_type", hover_data=hover); limit=max(denoise.e_obs.max(), denoise.e_PINN.max()); fig.add_trace(go.Scatter(x=[0,limit], y=[0,limit], mode="lines", name="equal error")); st.plotly_chart(fig, use_container_width=True)
        _explain("Points below the diagonal have PINN observation predictions closer to observation truth; this does not establish latent-state accuracy.", r"e_{obs}=|\log y_{obs}-\log y_{true}|;\quad e_{PINN}=|\log\hat y-\log y_{true}|", ["data_predictions_final.csv"])
        c=st.columns(3); c[0].metric("Fraction PINN closer", f"{metrics['fraction_below']:.1%}"); c[1].metric("Mean e_obs", f"{metrics['mean_e_obs']:.4g}"); c[2].metric("Mean e_PINN", f"{metrics['mean_e_PINN']:.4g}")

    st.subheader("D4 · Standardised residual versus fitted observation")
    fig=px.scatter(data,x="prediction",y="z",color="obs_type",log_x=True,hover_data=hover); fig.add_hline(y=0); fig.add_hline(y=1.96,line_dash="dash"); fig.add_hline(y=-1.96,line_dash="dash"); st.plotly_chart(fig,use_container_width=True)
    _explain("Checks fitted-value bias after scaling log residuals by saved uncertainty.", r"z=r/\sigma_{log}", ["data_predictions_final.csv: prediction, log_residual, sd_log_used"])
    st.subheader("D5 · Normal QQ of standardised residuals")
    observed=np.sort(data.z.to_numpy()); theoretical=normal_quantiles(len(observed)); fig=px.scatter(x=theoretical,y=observed,labels={"x":"theoretical Normal quantile","y":"observed z quantile"}); lo=min(theoretical.min(),observed.min()); hi=max(theoretical.max(),observed.max()); fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Normal reference")); st.plotly_chart(fig,use_container_width=True)
    _explain("Assesses compatibility with the assumed marginal Normal model for standardised log residuals; it is not an independence test.", r"z_{(i)}\ versus\ \Phi^{-1}((i-0.5)/n)", ["data_predictions_final.csv"])
    st.subheader("D6 · Observation coverage")
    st.plotly_chart(px.scatter(data,x="t_mid",y="species",color="dataset",symbol="obs_type",hover_data=hover),use_container_width=True)
    _explain("Shows where observation operators constrain species and time; interval endpoints remain in hover metadata.", r"t_{mid}=(t_{start}+t_{end})/2", ["data_predictions_final.csv"])
    st.subheader("D7 · Largest standardised misfits")
    worst=rank_misfits(data); worst["label"]=worst.apply(lambda r:f"#{int(r.observation_id)} {r.species} {r.obs_type}",axis=1); st.plotly_chart(px.bar(worst.sort_values("half_z2"),x="half_z2",y="label",orientation="h",hover_data=hover),use_container_width=True)
    _explain("Ranks residual magnitude, not full NLL: log(sigma_log) is an uncertainty-scale term rather than residual magnitude.", r"rank_j=0.5z_j^2", ["data_predictions_final.csv"], "Current filters; top 25 or fewer")
    st.subheader("D8 · Discrepancy gate")
    history=read_csv(run_dir,"loss_history.csv"); config = read_json(run_dir)
    gate_cols=["data_discrepancy_q","data_discrepancy_q95","data_loss_active","loss_data","loss_data_effective"]
    present,_=available_columns(history if history is not None else pd.DataFrame(),gate_cols)
    if not present: st.info("Saved discrepancy-gate history is unavailable.")
    else:
        enabled = bool(config.get("data_discrepancy_gate", False))
        st.caption(f"Configured discrepancy gate: {'enabled' if enabled else 'disabled'}.")
        groups = (
            ([c for c in present if c in {"data_discrepancy_q","data_discrepancy_q95"}], "Saved discrepancy statistic and threshold", r"Q=\sum_jz_j^2;\quad Q_{95}=\chi^2_{0.95}(n)"),
            ([c for c in present if c == "data_loss_active"], "Saved binary gate activity", r"active=\mathbf{1}[Q>Q_{95}]\quad (gate\ enabled)"),
            ([c for c in present if c in {"loss_data","loss_data_effective"}], "Raw and effective data loss", r"L_{effective}=L_{data}\mathbf{1}[Q>Q_{95}]\quad (gate\ enabled)"),
        )
        for index, (group, interpretation, equation) in enumerate(groups):
            if not group:
                continue
            fig = px.line(history,x="step",y=group,labels={"step":"optimization step"})
            if index in (0, 2):
                fig = plot_scale_controls(st, fig, key=f"data_gate_{index}", log_y=True)
            st.plotly_chart(fig,use_container_width=True)
            _explain(interpretation + "; explicit saved values are never inferred.", equation, ["loss_history.csv: " + ", ".join(group)], "All retained optimization steps", "Rows are aligned by saved optimization step.")


def _history_plot(data, columns, title, equation, interpretation, source="loss_history.csv", *, log_y=True, key=None):
    present,_=available_columns(data,columns)
    st.subheader(title)
    if not present: st.info("No corresponding saved columns are available."); return
    fig = px.line(data,x="step",y=present,labels={"step":"optimization step","value":title})
    if log_y:
        fig = plot_scale_controls(st, fig, key=key or title.replace(" ", "_").replace("·", "_"), log_y=True)
    st.plotly_chart(fig,use_container_width=True)
    _explain(interpretation,equation,[source+": "+", ".join(present)],"All retained optimization steps")


def training_page(rows):
    st.header("Selected run: Training"); row=_run(rows)
    if not row: st.info("Select a discovered run from the Suite map."); return
    run_dir=str(row["selected_instance"].run_dir); history=read_csv(run_dir,"loss_history.csv")
    if history is None or history.empty: st.info("No saved loss_history.csv is available."); return
    _history_plot(history,["loss"],"T1 · Total objective",r"L_{total}=L\ (saved)","Tracks the saved optimization objective.",key="T1")
    _history_plot(history,["loss_pde","loss_ic","loss_bc","loss_timestep","loss_data"],"T2 · Raw loss components",r"L_k\ before\ adaptive/objective\ weighting","Compares available explicitly saved raw constraint losses.",key="T2")
    objective=["objective_loss_pde","objective_loss_ic","objective_loss_bc","objective_loss_timestep","objective_loss_data"]
    if not available_columns(history,objective)[0]: objective=["weighted_loss_pde","weighted_loss_ic","weighted_loss_bc","weighted_loss_timestep","weighted_loss_data"]
    _history_plot(history,objective,"T3 · Weighted/objective components",r"L_{objective,k}\ (saved)","Compares exact saved weighted contributions; the viewer does not reconstruct them.",key="T3")
    _history_plot(history,["w_pde","w_ic","w_bc","w_timestep","w_data"],"T4 · Adaptive weights",r"w_k\ (saved)","Tracks optimizer loss-balancing weights.",key="T4")
    st.subheader("T5 · Per-term weighting gradients")
    expected=["grad_norm_pde_for_weighting","grad_norm_ic_for_weighting","grad_norm_bc_for_weighting","grad_norm_timestep_for_weighting","grad_norm_data_for_weighting"]
    present,missing=available_columns(history,expected)
    if present:
        fig=px.line(history,x="step",y=present,labels={"step":"optimization step"}); fig=plot_scale_controls(st,fig,key="T5",log_y=True); st.plotly_chart(fig,use_container_width=True); _explain("Shows only retained adaptive-weighting gradient norms.",r"\|\nabla_\theta L_k\|\ (saved)",["loss_history.csv: "+", ".join(present)])
    else: st.info("Per-term gradient-weighting diagnostics were not retained in this HPC output.")
    if missing: st.caption("Missing expected columns: "+", ".join(missing))
    _history_plot(history,["grad_norm"],"T6 · Overall gradient norm",r"\|\nabla_\theta L\|","Tracks the saved total network gradient norm.",key="T6")
    st.subheader("T7 · Learning rates")
    learning_rates, _ = available_columns(history, ["lr","rmax_lr","data_cv_lr","effort_lr"])
    if learning_rates:
        learning_rate = st.selectbox("Learning-rate group", learning_rates)
        fig=px.line(history, x="step", y=learning_rate, labels={"step":"optimization step"}); fig=plot_scale_controls(st,fig,key="T7",log_y=True); st.plotly_chart(fig, use_container_width=True)
        _explain("Shows one saved network or inverse-parameter learning rate without mixing incomparable scales.", r"\eta_g(step)\ (saved)", [f"loss_history.csv: {learning_rate}"])
    else: st.info("No saved learning-rate column is available.")
    st.subheader("T8 · Causal curriculum")
    causal = (
        ("causal_fraction", "Admitted domain fraction", r"f_{causal}\in[0,1]", "Shows the dimensionless admitted fraction of the time domain."),
        ("t_max_current", "Current maximum model time", r"t_{max,current}\ in\ model\ time", "Shows the physical/model-time frontier admitted by the curriculum."),
    )
    for column, title, equation, interpretation in causal:
        if column not in history or not pd.to_numeric(history[column], errors="coerce").notna().any():
            continue
        st.plotly_chart(px.line(history,x="step",y=column,title=title,labels={"step":"optimization step"}),use_container_width=True)
        _explain(interpretation, equation, [f"loss_history.csv: {column}"], "All retained optimization steps")
    _history_plot(history,["pde_causal_weight_first","pde_causal_weight_mean","pde_causal_weight_last"],"T9a · Causal chunk weights",r"w_{first},mean(w_k),w_{last}","Tracks saved causal weighting across chunks.",key="T9a")
    _history_plot(history,["pde_causal_chunk_loss_mean","pde_causal_chunk_loss_max"],"T9b · Causal chunk losses",r"mean(L_k),max(L_k)","Tracks saved causal chunk loss summaries.",key="T9b")
    fixed=read_csv(run_dir,"fixed_diagnostic_history.csv")
    if fixed is not None and not fixed.empty:
        _history_plot(fixed,["fixed_residual_log_rms","fixed_residual_log_abs_mean","fixed_residual_log_abs_p95","fixed_residual_log_abs_max"],"T10 · Fixed-grid PDE residual history",r"RMS(R_{log}),mean|R_{log}|,p_{95}|R_{log}|,max|R_{log}|","This is the deterministic fixed diagnostic grid, not stochastic training collocation residual.","fixed_diagnostic_history.csv",key="T10")
        _history_plot(fixed,["rms_dlogN_dt","rms_advective","rms_mu","rms_dg_dw"],"T11 · Fixed-grid PDE-term RMS",r"R_{log}=\partial_t\log N+g\partial_w\log N+\mu+\partial_wg","Compares RMS magnitude of each saved PDE-balance term.","fixed_diagnostic_history.csv",key="T11")
    else: st.info("T10/T11 unavailable: no fixed_diagnostic_history.csv.")
    st.subheader("T12 · Runtime")
    if {"step","seconds_elapsed"}.issubset(history):
        runtime=interval_seconds_per_step(history)
        fig=px.line(runtime,x="step",y="seconds_per_step",labels={"step":"optimization step"}); fig=plot_scale_controls(st,fig,key="T12",log_y=True); st.plotly_chart(fig,use_container_width=True)
        timing = read_csv(run_dir, "timing_summary.csv")
        total = float(runtime.seconds_elapsed.max())
        total_source = "loss_history.csv: seconds_elapsed"
        if timing is not None and not timing.empty and "actual_total_seconds" in timing:
            saved_total = pd.to_numeric(timing.actual_total_seconds, errors="coerce").dropna()
            if not saved_total.empty:
                total = float(saved_total.iloc[-1]); total_source = "timing_summary.csv: actual_total_seconds"
        _explain("Shows interval runtime per optimization step and the final saved total runtime, never file timestamps.",r"seconds/step=\Delta seconds_{elapsed}/\Delta step",["loss_history.csv: step, seconds_elapsed", total_source])
        st.metric("Total elapsed runtime",f"{total:.3f} s")
    else: st.info("Cumulative seconds_elapsed was not retained.")
