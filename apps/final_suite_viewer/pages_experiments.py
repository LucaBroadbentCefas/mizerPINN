"""Final-suite experiment-level comparisons (E1–E3, missing data, NN, A1–A2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analytics import (common_comparison_domain, error_by_species, error_by_time,
                        error_by_weight, mask_domain)
from .components import plot_explanation
from .experiment_analysis import (CVS, NN_SCENARIOS, ablation_task_matrix,
                                  common_replicate_domain, gap_mask,
                                  missing_seen_metrics, missing_species_mask,
                                  noise_design, noise_summary,
                                  paired_cv_differences,
                                  replicate_species_metrics,
                                  retained_omitted_years, year_location_mask)
from .loaders import load_fixed_fields, load_prediction_state, load_w_max, read_csv
from .metrics import fold_error, state_rmse
from .state import MULTISPECIES_TRUTH, align_states, load_truth_state, truth_source_for_task

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "HPC_clone"
METRICS = ("State RMSE_log10N", "State fold error", "Fixed-grid p95 |PDE residual|", "Observation mean |log error|")


def _row_map(rows): return {row["task_id"]: row for row in rows}


@st.cache_data(show_spinner=False)
def _aligned(run_dir: str, task_id: int):
    prediction = load_prediction_state(run_dir)
    if prediction is None: return pd.DataFrame()
    truth_path = truth_source_for_task(task_id)
    if not truth_path.is_file(): raise FileNotFoundError(f"Required truth file is missing: {truth_path}")
    return align_states(prediction, load_truth_state(truth_path), w_max=load_w_max(run_dir))[0]


@st.cache_data(show_spinner=False)
def _run_metric(run_dir: str, task_id: int, metric: str, species_idx: int | None):
    if metric.startswith("State"):
        aligned = _aligned(run_dir, task_id)
        if species_idx is not None: aligned = aligned[aligned.species_idx == species_idx]
        errors = aligned.error_log10_N
        return state_rmse(errors) if metric == METRICS[0] else fold_error(errors)
    if metric.startswith("Fixed-grid"):
        fixed = load_fixed_fields(run_dir)
        if fixed is None or "residual_log" not in fixed: return np.nan
        if species_idx is not None: fixed = fixed[fixed.species_idx == species_idx]
        values = np.abs(pd.to_numeric(fixed.residual_log, errors="coerce").dropna())
        return float(np.percentile(values, 95)) if len(values) else np.nan
    observations = read_csv(run_dir, "data_predictions_final.csv")
    if observations is None or not {"value", "prediction"}.issubset(observations): return np.nan
    if species_idx is not None and "species_idx" in observations: observations = observations[observations.species_idx == species_idx]
    valid = (observations.value > 0) & (observations.prediction > 0)
    return float(np.mean(np.abs(np.log(observations.loc[valid, "value"]) - np.log(observations.loc[valid, "prediction"])))) if valid.any() else np.nan


def _metric_controls(truth: pd.DataFrame):
    metric = st.selectbox("Run metric", METRICS, index=0)
    choices = [None] + sorted(truth.species_idx.astype(int).unique().tolist())
    species = st.selectbox("Metric species", choices, format_func=lambda x: "All species" if x is None else f"species {x}")
    return metric, species


def _metric_table(rows, task_ids, metric, species):
    by_id = _row_map(rows); records=[]
    for task_id in task_ids:
        task_id = int(task_id); row=by_id.get(task_id); run=row and row["selected_instance"]
        records.append({"task_id":task_id, "value": _run_metric(str(run.run_dir), task_id, metric, species) if run else np.nan, "available": bool(run)})
    return pd.DataFrame(records)


def _explain(interpretation, equation, inputs, selection, alignment="Task-specific truth alignment for state metrics; saved fields for other metrics."):
    plot_explanation(st, interpretation=interpretation, calculation=equation, inputs=inputs, selection=selection, alignment=alignment)


def _noise_section(rows, truth):
    st.header("Noise and CV comparisons")
    metric, species = _metric_controls(truth); design=noise_design(); values=design.merge(_metric_table(rows, design.task_id, metric, species), on="task_id")
    values=values[np.isfinite(values.value)]; gate_on=values[values.gate]
    if gate_on.empty: st.info("No discovered gate-ON noise runs have this metric."); return
    summary=noise_summary(values)
    st.subheader("E1 · Paired CV response")
    fig=px.line(gate_on,x="cv",y="value",color="noise_seed",markers=True,labels={"value":metric,"noise_seed":"matched seed"})
    fig.add_trace(go.Scatter(x=summary.cv,y=summary["mean"],error_y={"type":"data","array":summary.sd},mode="lines+markers",name="mean ± 1 SD"))
    gate_off=values[~values.gate]
    if not gate_off.empty: fig.add_trace(go.Scatter(x=gate_off.cv,y=gate_off.value,mode="lines+markers",name="gate OFF rep1",line={"dash":"dash"},marker={"symbol":"diamond"}))
    st.plotly_chart(fig,use_container_width=True)
    _explain("Shows five matched-seed trajectories and mean ± 1 sample SD; this is not a confidence interval. Gate OFF is one trajectory with no uncertainty ribbon.",r"\bar M(c)=mean_rM(c,r),\quad s(c)=\sqrt{\sum_r(M-\bar M)^2/(5-1)}",["suite catalogue","cached run metric table"],f"{metric}; {'all species' if species is None else f'species {species}'}")
    st.subheader("E2 · Replicate spread at each CV")
    fig=px.strip(gate_on,x="cv",y="value",color="noise_seed",labels={"value":metric}); fig.add_trace(go.Scatter(x=summary.cv,y=summary["mean"],error_y={"type":"data","array":summary.sd},mode="markers",name="mean ± 1 SD",marker={"size":12,"symbol":"x"}))
    if not gate_off.empty: fig.add_trace(go.Scatter(x=gate_off.cv,y=gate_off.value,mode="markers",name="gate OFF rep1",marker={"symbol":"diamond","size":10}))
    st.plotly_chart(fig,use_container_width=True)
    _explain("Keeps all five raw gate-ON values visible without estimating a density; markers show mean ± 1 SD.",r"M(c,r),\quad \bar M(c)\pm s(c)",["cached run metric table"],f"{metric}")
    st.subheader("E3 · Paired CV-to-CV change")
    pairs,wide=paired_cv_differences(values); matrix=pairs.pivot(index="cv_to",columns="cv_from",values="mean_difference")
    fig=px.imshow(matrix,text_auto=".3g",color_continuous_scale="RdBu_r",color_continuous_midpoint=0,labels={"x":"from CV c1","y":"to CV c2","color":"mean Δ"}); st.plotly_chart(fig,use_container_width=True)
    _explain("Shows mean paired change for every ordered CV pair; positive means worse at the row/second CV for error metrics.",r"mean_r[M(c_2,r)-M(c_1,r)]",["matched noise_seed catalogue fields"],metric,"Only identical noise seeds are paired; no significance test.")
    c1,c2=st.columns(2); cv1=c1.selectbox("First CV (c1)",CVS,index=0); cv2=c2.selectbox("Second CV (c2)",CVS,index=1)
    differences=(wide[cv2]-wide[cv1]).dropna(); detail=pd.DataFrame({"noise_seed":differences.index,"difference":differences.values})
    st.plotly_chart(px.bar(detail,x="noise_seed",y="difference",title=f"Five paired differences: CV {cv2} − CV {cv1}"),use_container_width=True)
    _explain("Shows all five paired replicate differences for the selected ordered CV pair.",r"\Delta M_r=M(c_2,r)-M(c_1,r)",["cached paired metric table"],f"CV {cv2} minus CV {cv1}","Matched by noise_seed.")
    st.metric("Mean paired difference",f"{differences.mean():.5g}"); st.metric("SD of paired differences",f"{differences.std(ddof=1):.5g}")
    st.dataframe(summary.rename(columns={"mean":"mean","sd":"SD","minimum":"minimum","maximum":"maximum"}),use_container_width=True)
    _explain("Positive differences mean worse performance at the second CV for the error metrics; negative means better. No significance test is performed.",r"\Delta M_r(c_2,c_1)=M(c_2,r)-M(c_1,r)", ["matched noise_seed catalogue fields"],f"CV {cv2} minus CV {cv1}; {metric}","Only identical noise seeds are paired.")
    _replicate_comparison(rows)



def _replicate_comparison(rows):
    st.header("Detailed replicate comparison")
    st.caption("Direct comparison of the five gate-ON fits at one observation CV. State errors use the same aligned mizer truth and are restricted to cells shared by every available replicate.")

    by_id = _row_map(rows)
    cv = st.selectbox("Replicate comparison CV", CVS, index=2, key="replicate_compare_cv")
    design = noise_design()
    selected = design[(design.gate) & np.isclose(design.cv.astype(float), float(cv))].sort_values("replicate")

    frames = {}
    histories = {}
    for item in selected.itertuples(index=False):
        row = by_id.get(int(item.task_id))
        run = row and row["selected_instance"]
        if not run:
            continue
        run_dir = str(run.run_dir)
        aligned = _aligned(run_dir, int(item.task_id))
        if aligned is not None and not aligned.empty:
            frames[int(item.replicate)] = aligned
        history = read_csv(run_dir, "loss_history.csv")
        if history is not None and not history.empty:
            histories[int(item.replicate)] = history

    if len(frames) < 2:
        st.info("At least two discovered gate-ON replicates with aligned state output are required.")
        return

    frames = common_replicate_domain(frames)
    if not frames or any(frame.empty for frame in frames.values()):
        st.info("The available replicates have no common state-comparison cells.")
        return

    common_species = sorted(set.intersection(*[set(frame.species_idx.astype(int).unique()) for frame in frames.values()]))
    if not common_species:
        st.info("The available replicates have no common species.")
        return
    names = {}
    for frame in frames.values():
        if "species" in frame:
            names.update(dict(frame[["species_idx", "species"]].drop_duplicates().itertuples(index=False, name=None)))
    species = st.selectbox(
        "Replicate species",
        common_species,
        format_func=lambda x: names.get(x, f"species {x}"),
        key="replicate_compare_species",
    )
    species_frames = {rep: frame[frame.species_idx.astype(int).eq(species)].copy() for rep, frame in frames.items()}
    st.caption(f"CV {cv:g}; {len(species_frames)} available replicates; comparisons use the exact common species/time/weight domain.")

    st.subheader("R1 · State fit across replicates")
    state_view = st.radio(
        "State slice",
        ["Across weight at selected time", "Through time at selected weight"],
        horizontal=True,
        key="replicate_state_slice",
    )
    state_display = st.radio(
        "State display",
        ["Prediction", "Signed error", "Absolute error"],
        horizontal=True,
        key="replicate_state_display",
    )
    state_layout = st.radio(
        "State layout",
        ["Overlay", "Small multiples"],
        horizontal=True,
        key="replicate_state_layout",
    )

    first = next(iter(species_frames.values()))
    if state_view.startswith("Across weight"):
        values = sorted(first.time.unique().tolist())
        selected_coord = st.select_slider("State time", values, value=values[len(values) // 2], key="replicate_state_time")
        x_col = "w"
        sliced = {rep: frame[np.isclose(frame.time, selected_coord)].copy() for rep, frame in species_frames.items()}
        selection_text = f"species {species}; time={selected_coord:g}"
    else:
        values = sorted(first.w.unique().tolist())
        selected_coord = st.select_slider("State body weight w", values, value=values[len(values) // 2], key="replicate_state_weight")
        x_col = "time"
        sliced = {rep: frame[np.isclose(frame.w, selected_coord)].copy() for rep, frame in species_frames.items()}
        selection_text = f"species {species}; w={selected_coord:g}"

    plot_rows = []
    for rep, frame in sliced.items():
        frame = frame.sort_values(x_col)
        if state_display == "Prediction":
            y = frame.pred_log10_N
        elif state_display == "Signed error":
            y = frame.error_log10_N
        else:
            y = frame.error_log10_N.abs()
        plot_rows.append(pd.DataFrame({x_col: frame[x_col], "value": y, "replicate": f"rep{rep}"}))
    state_plot = pd.concat(plot_rows, ignore_index=True)

    if state_layout == "Overlay":
        fig = px.line(state_plot, x=x_col, y="value", color="replicate")
        if state_display == "Prediction":
            truth_slice = next(iter(sliced.values())).sort_values(x_col)
            fig.add_trace(go.Scatter(x=truth_slice[x_col], y=truth_slice.true_log10_N, mode="lines", name="mizer truth", line={"width": 4}))
        elif state_display == "Signed error":
            fig.add_hline(y=0, line_dash="dash")
    else:
        if state_display == "Prediction":
            facet_rows = []
            truth_slice = next(iter(sliced.values())).sort_values(x_col)
            for rep, frame in sliced.items():
                frame = frame.sort_values(x_col)
                facet_rows.append(pd.DataFrame({x_col: frame[x_col], "value": frame.pred_log10_N, "series": "PINN", "replicate": f"rep{rep}"}))
                facet_rows.append(pd.DataFrame({x_col: truth_slice[x_col], "value": truth_slice.true_log10_N, "series": "mizer truth", "replicate": f"rep{rep}"}))
            facet_data = pd.concat(facet_rows, ignore_index=True)
            fig = px.line(facet_data, x=x_col, y="value", color="series", facet_col="replicate", facet_col_wrap=3)
        else:
            fig = px.line(state_plot, x=x_col, y="value", facet_col="replicate", facet_col_wrap=3)
            if state_display == "Signed error":
                fig.add_hline(y=0, line_dash="dash", row="all", col="all")
    if x_col == "w":
        fig.update_xaxes(type="log")
    fig.update_yaxes(title="log10 N" if state_display == "Prediction" else ("log10 error" if state_display == "Signed error" else "|log10 error|"))
    st.plotly_chart(fig, use_container_width=True)
    _explain(
        "Shows the fitted state functions themselves rather than reducing each replicate to one score. Prediction mode includes the common mizer truth; error modes compare the same state cells directly.",
        r"e_r(t,w)=\log_{10}\hat N_r(t,w)-\log_{10}N_{true}(t,w)",
        ["aligned replicate state predictions", "task-specific mizer truth"],
        selection_text,
        "All available replicates are restricted to identical species/time/weight cells before plotting.",
    )

    st.subheader("R2 · Error comparison")

    weights = sorted(first.w.unique().tolist())
    weight_range = st.select_slider(
        "Weight range used for RMSE through time",
        weights,
        value=(weights[0], weights[-1]),
        key="replicate_error_weight_range",
    )
    time_curves = []
    for rep, frame in species_frames.items():
        curve = error_by_time(mask_domain(frame, weight_range=weight_range)).assign(replicate=f"rep{rep}")
        time_curves.append(curve)
    time_curves = pd.concat(time_curves, ignore_index=True)
    st.plotly_chart(px.line(time_curves, x="time", y="RMSE_log10N", color="replicate"), use_container_width=True)
    _explain(
        "Shows when replicate fits separate in state accuracy. The selected weight range is applied identically to every replicate.",
        r"RMSE_r(t)=\sqrt{mean_w\{e_r(t,w)^2\}}",
        ["common-domain aligned state"],
        f"species {species}; w in [{weight_range[0]:g}, {weight_range[1]:g}]",
    )

    times = sorted(first.time.unique().tolist())
    time_range = st.select_slider(
        "Time range used for RMSE across weight",
        times,
        value=(times[0], times[-1]),
        key="replicate_error_time_range",
    )
    weight_curves = []
    for rep, frame in species_frames.items():
        curve = error_by_weight(mask_domain(frame, time_range=time_range)).assign(replicate=f"rep{rep}")
        weight_curves.append(curve)
    weight_curves = pd.concat(weight_curves, ignore_index=True)
    fig = px.line(weight_curves, x="w", y="RMSE_log10N", color="replicate")
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    _explain(
        "Shows which body-size regions differ in accuracy between replicates. The selected time range is applied identically to every replicate.",
        r"RMSE_r(w)=\sqrt{mean_t\{e_r(t,w)^2\}}",
        ["common-domain aligned state"],
        f"species {species}; t in [{time_range[0]:g}, {time_range[1]:g}]",
    )

    species_metrics = replicate_species_metrics(frames)
    metric_options = {
        "RMSE log10N": "RMSE_log10N",
        "Fold error": "fold_error",
        "Mean absolute log10 error": "MAE_log10N",
    }
    metric_label = st.selectbox("Species × replicate error metric", list(metric_options), key="replicate_species_metric")
    metric_col = metric_options[metric_label]
    species_metrics["species_label"] = species_metrics.apply(
        lambda row: str(row["species"]) if "species" in species_metrics.columns and pd.notna(row.get("species")) else f"species {int(row.species_idx)}",
        axis=1,
    )
    matrix = species_metrics.pivot(index="species_label", columns="replicate", values=metric_col)
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    fig = px.imshow(matrix, text_auto=".3g", aspect="auto", labels={"x": "replicate", "y": "species", "color": metric_label})
    st.plotly_chart(fig, use_container_width=True)
    _explain(
        "Separates a globally poor replicate from species-specific instability. Every heatmap cell is calculated on the common replicate comparison domain.",
        r"RMSE_{s,r}=\sqrt{mean_{t,w}(e_{s,r}^2)};\quad F_{s,r}=10^{mean|e_{s,r}|}",
        ["common-domain aligned state"],
        metric_label,
    )

    st.subheader("R3 · Training comparison")
    component_columns = {
        "Total": "loss",
        "PDE": "loss_pde",
        "Data": "loss_data",
        "IC": "loss_ic",
        "BC": "loss_bc",
    }
    available_components = [
        label for label, column in component_columns.items()
        if sum(column in history and pd.to_numeric(history[column], errors="coerce").notna().any() for history in histories.values()) >= 2
    ]
    if not available_components:
        st.info("At least two replicate loss histories with a common saved loss component are required.")
        return
    component = st.selectbox("Training loss component", available_components, key="replicate_training_component")
    column = component_columns[component]
    history_rows = []
    for rep, history in histories.items():
        if column not in history or "step" not in history:
            continue
        values = pd.DataFrame({
            "step": pd.to_numeric(history["step"], errors="coerce"),
            "value": pd.to_numeric(history[column], errors="coerce"),
            "replicate": f"rep{rep}",
        }).dropna()
        history_rows.append(values)
    history_plot = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    if history_plot.empty:
        st.info(f"No usable saved {component} histories are available.")
        return
    positive = bool((history_plot.value > 0).all())
    fig = px.line(history_plot, x="step", y="value", color="replicate", log_y=positive)
    st.plotly_chart(fig, use_container_width=True)
    if not positive:
        st.caption("Linear y-axis used because this saved loss component contains zero or negative values.")
    _explain(
        "Compares optimization trajectories for the same saved loss component across replicates. This helps distinguish different final fits from different training paths.",
        rf"{column}(step)\quad\text{{as saved during training}}",
        [f"loss_history.csv: step, {column}"],
        f"CV {cv:g}; {component}",
        "Optimization steps are not interpolated; each replicate uses its own retained history rows.",
    )


@st.cache_data(show_spinner=False)
def _observation_design(task_id):
    names={44:"every_3yr.csv",45:"gap_10_20.csv",46:"gap_30_40.csv",47:"missing_species_3.csv",48:"missing_species_7.csv",49:"missing_species_11.csv"}
    perfect=pd.read_csv(PROJECT_ROOT/"final_runs/observations/final_runs/perfect.csv")
    reduced=pd.read_csv(PROJECT_ROOT/"final_runs/observations/final_runs"/names[task_id])
    return perfect,reduced


def _missing_section(rows):
    st.header("Missing and sparse observations")
    by_id=_row_map(rows); options=[i for i in range(44,50) if by_id.get(i) and by_id[i]["selected_instance"]]
    if not options: st.info("No missing-data runs were discovered."); return
    task_id=st.selectbox("Missing-data run",options,format_func=lambda i:f"{i}: {by_id[i]['display_label']}"); run_dir=str(by_id[task_id]["selected_instance"].run_dir)
    aligned=_aligned(run_dir,task_id); perfect,reduced=_observation_design(task_id)
    omitted_species=sorted(set(perfect.species_idx)-set(reduced.species_idx)); retained_years,omitted_years=retained_omitted_years(perfect,reduced)
    species_options=sorted(aligned.species_idx.astype(int).unique()); default_species=omitted_species[0] if omitted_species else species_options[0]
    species=st.selectbox("Species",species_options,index=species_options.index(default_species)); selected=aligned[aligned.species_idx==species]
    times=sorted(selected.time.unique()); weights=sorted(selected.w.unique()); default_time=15 if task_id==45 else 35 if task_id==46 else times[len(times)//2]
    time=st.select_slider("Model time",times,value=min(times,key=lambda x:abs(x-default_time))); weight=st.select_slider("Physical body weight w",weights,value=weights[len(weights)//2])
    st.caption(f"Validated observation design: {len(reduced)} retained of {len(perfect)} perfect-data rows. Omitted species={omitted_species or 'none'}; retained observation years={retained_years}; omitted observation years={omitted_years}.")
    at_time=selected[np.isclose(selected.time,time)]; through=selected[np.isclose(selected.w,weight)]
    for title,data,x in (("Across body size",at_time,"w"),("Through time",through,"time")):
        long=data[[x,"true_log10_N","pred_log10_N"]].melt(x,var_name="source",value_name="log10_N"); long.source=long.source.map({"true_log10_N":"Mizer truth","pred_log10_N":"PINN"}); fig=px.line(long,x=x,y="log10_N",color="source",markers=True,title=title)
        if x=="w": fig.update_xaxes(type="log")
        if x=="time" and task_id in (45,46): fig.add_vrect(x0=10 if task_id==45 else 30,x1=20 if task_id==45 else 40,fillcolor="grey",opacity=.2,line_width=0)
        if x=="time" and task_id==44:
            for year in retained_years: fig.add_vline(x=year,line_color="green",opacity=.15)
            for year in omitted_years: fig.add_vline(x=year,line_color="orange",opacity=.15,line_dash="dot")
        st.plotly_chart(fig,use_container_width=True)
        _explain(f"Compares state truth and prediction {title.lower()}.",r"e=\log_{10}N_{pred}-\log_{10}N_{true}",["aligned state","actual reduced observation CSV"],f"species {species}; t={time}; w={weight}")
    st.subheader("Missing-region state-error heatmap")
    mode=st.radio("Missing-data error display",["Signed","Absolute"],horizontal=True); surface=selected.assign(display=selected.error_log10_N if mode=="Signed" else selected.error_log10_N.abs())
    if task_id in (45, 46):
        start, end = (10, 20) if task_id == 45 else (30, 40)
        surface = surface[gap_mask(surface, start, end)]
    grid=surface.pivot(index="time",columns="w",values="display"); limit=np.nanmax(np.abs(grid)) if mode=="Signed" else None
    fig=go.Figure(go.Heatmap(x=grid.columns,y=grid.index,z=grid,zmin=-limit if mode=="Signed" else None,zmax=limit,colorscale="RdBu_r" if mode=="Signed" else "Viridis")); fig.update_xaxes(type="log",title="physical body weight w"); fig.update_yaxes(title="model time")
    if task_id == 44:
        for year in retained_years: fig.add_hline(y=year,line_color="green",opacity=.25)
        for year in omitted_years: fig.add_hline(y=year,line_color="orange",opacity=.35,line_dash="dot")
    st.plotly_chart(fig,use_container_width=True)
    _explain("Maps latent-state error over the withheld domain without treating observation truth as true state.",r"e=\log_{10}N_{pred}-\log_{10}N_{true}",["aligned task-specific state truth","actual observation design"],f"species {species}; {mode}","Gap runs are cropped to the continuous gap; every-third-year lines mark discrete retained/omitted years.")
    if task_id in (45,46): missing=gap_mask(aligned,10 if task_id==45 else 30,20 if task_id==45 else 40)
    elif omitted_species: missing=missing_species_mask(aligned,omitted_species[0])
    else: missing=year_location_mask(aligned,omitted_years)
    metrics=missing_seen_metrics(aligned,missing)
    if omitted_species:
        observed = aligned[~missing]
        observed_rmse = error_by_species(observed)
        observed_fold = observed.groupby(["species_idx", "species"]).error_log10_N.agg(fold_error).reset_index(name="fold_error")
        metrics["RMSE_seen"] = float(observed_rmse.RMSE_log10N.mean())
        metrics["fold_seen"] = float(observed_fold.fold_error.mean())
        metrics["generalisation_penalty"] = metrics["RMSE_missing"] - metrics["RMSE_seen"]
        st.caption("Seen comparison is the mean of species-level metrics across observed species; no observed species is treated as identical to the omitted species.")
        st.dataframe(observed_rmse.merge(observed_fold, on=["species_idx", "species"]), use_container_width=True)
    st.dataframe(pd.DataFrame([metrics]),use_container_width=True)
    overview=error_by_time(aligned) if not omitted_species else error_by_species(aligned)
    if omitted_species:
        fig=px.bar(overview,x="species",y="RMSE_log10N",color=overview.species_idx.eq(omitted_species[0]).map({True:"omitted",False:"observed"}))
    else:
        fig=px.line(overview,x="time",y="RMSE_log10N",markers=True)
        if task_id in (45, 46):
            start, end = (10, 20) if task_id == 45 else (30, 40)
            fig.add_vrect(x0=start, x1=end, fillcolor="grey", opacity=.2, line_width=0)
        if task_id == 44:
            for year in retained_years: fig.add_vline(x=year,line_color="green",opacity=.2)
            for year in omitted_years: fig.add_vline(x=year,line_color="orange",opacity=.3,line_dash="dot")
    st.plotly_chart(fig,use_container_width=True)
    _explain("Separates state error on the actual withheld design from comparable seen cells; the default penalty is a difference, not a ratio.",r"penalty=RMSE_{miss}-RMSE_{seen}", ["actual perfect/reduced observation CSVs","task-specific state truth"],by_id[task_id]["display_label"])


def _nn_and_ablation(rows,truth):
    st.header("PINN versus no-PDE NN")
    by_id=_row_map(rows); scenario=st.selectbox("Matched scenario",list(NN_SCENARIOS)); pair=NN_SCENARIOS[scenario]
    if not all(by_id.get(i) and by_id[i]["selected_instance"] for i in pair): st.info("Both matched runs are required.")
    else:
        species_choices=[None]+sorted(truth.species_idx.astype(int).unique()); species=st.selectbox("NN comparison species",species_choices,format_func=lambda x:"All species" if x is None else f"species {x}")
        aligned=[]
        for task_id,label in zip(pair,("PINN","Data-only NN")):
            data=_aligned(str(by_id[task_id]["selected_instance"].run_dir),task_id); data=data if species is None else data[data.species_idx==species]; aligned.append(data)
        aligned=list(common_comparison_domain(aligned[0],aligned[1]))
        overall=pd.DataFrame({"model":["PINN","Data-only NN"],"state RMSE":[state_rmse(x.error_log10_N) for x in aligned],"fold error":[fold_error(x.error_log10_N) for x in aligned]})
        st.subheader("NN1 · Overall state recovery"); st.dataframe(overall,use_container_width=True); st.plotly_chart(px.bar(overall,x="model",y=["state RMSE","fold error"],barmode="group"),use_container_width=True)
        _explain("Compares overall latent-state recovery for matched PINN and data-only networks, not their incompatible objectives.",r"RMSE=\sqrt{mean(e^2)},\quad F=10^{mean|e|}",["matched aligned states","task-specific truth"],scenario,"Both runs are restricted to identical comparison cells.")
        st.subheader("NN2 · State error through time"); curves=pd.concat([error_by_time(x).assign(model=label) for x,label in zip(aligned,("PINN","Data-only NN"))]); fig=px.line(curves,x="time",y="RMSE_log10N",color="model")
        if scenario.startswith("Gap"): fig.add_vrect(x0=30,x1=40,fillcolor="grey",opacity=.2,line_width=0)
        st.plotly_chart(fig,use_container_width=True); _explain("Compares matched state recovery without comparing incompatible training objectives.",r"RMSE(t)=\sqrt{mean_{i,w}e^2}",["catalogue-validated matched runs","task-specific truth"],scenario)
    st.header("A1 · Single-species ablation matrix")
    task_matrix=ablation_task_matrix(); metric=st.selectbox("Ablation metric",METRICS[:3]); values=task_matrix.astype(float).copy()
    for sp in values.index:
        for variant in values.columns:
            task_id=int(task_matrix.loc[sp,variant]); row=by_id.get(task_id); values.loc[sp,variant]=_run_metric(str(row["selected_instance"].run_dir),task_id,metric,int(sp.split("_")[1])) if row and row["selected_instance"] else np.nan
    st.plotly_chart(px.imshow(values,text_auto=".4g",aspect="auto",labels={"color":metric}),use_container_width=True)
    _explain("Compares architecture and state-parameterisation ablations within each single-species fixture.",metric,["tasks 0–11", "single_species_projection_long.csv"],"rows=species; columns=parameterisation/architecture","Each cell uses the independent single-species simulation truth and its restored fixture species identity.")
    chosen=st.selectbox("Set an ablation run as global selection",task_matrix.stack().astype(int).tolist(),format_func=lambda i:f"task {i}: {by_id[i]['run_label']}")
    if st.button("Select ablation run"): st.session_state.selected_task_id=chosen; st.success(f"Selected task {chosen}.")
    st.header("A2 · Multispecies no-data versus perfect-data")
    if all(by_id.get(i) and by_id[i]["selected_instance"] for i in (12,13)):
        raw_data=[_aligned(str(by_id[i]["selected_instance"].run_dir),i) for i in (12,13)]; labels=("No data","Perfect data")
        all_data=list(common_comparison_domain(raw_data[0],raw_data[1]))
        choices=[None]+sorted(set(all_data[0].species_idx.astype(int)) & set(all_data[1].species_idx.astype(int)))
        selected_species=st.selectbox("A2 species",choices,format_func=lambda x:"All species" if x is None else f"species {x}")
        data=all_data if selected_species is None else [frame[frame.species_idx==selected_species] for frame in all_data]
        data=list(common_comparison_domain(data[0],data[1]))
        curves=pd.concat([error_by_time(x).assign(run=label) for x,label in zip(data,labels)]); st.plotly_chart(px.line(curves,x="time",y="RMSE_log10N",color="run",title="A2a · State RMSE through time"),use_container_width=True)
        _explain("Shows through-time improvement from observations beyond PDE/IC/BC information alone.",r"RMSE(t)=\sqrt{mean_{i,w}e^2}",["tasks 12 and 13","mizer_projection_long.csv"],"all species" if selected_species is None else f"species {selected_species}","Common valid cells only.")
        species=pd.concat([error_by_species(x).assign(run=label) for x,label in zip(all_data,labels)]); st.plotly_chart(px.bar(species,x="species",y="RMSE_log10N",color="run",barmode="group",title="A2b · State RMSE by species"),use_container_width=True)
        _explain("Shows which species benefit from perfect observations relative to the no-data baseline.",r"RMSE_i=\sqrt{mean_{t,w}e_i^2}",["tasks 12 and 13","mizer_projection_long.csv"],"All species","Common valid cells and species-specific active domains.")


def experiment_page(rows):
    st.title("Experiment analyses")
    if not MULTISPECIES_TRUTH.is_file():
        st.error(f"Required multispecies truth file is missing: {MULTISPECIES_TRUTH}")
        return
    try: truth=load_truth_state(MULTISPECIES_TRUTH)
    except Exception as exc: st.error(f"Truth source unavailable: {exc}"); return
    section=st.radio("Analysis",["Noise/CV E1–E3","Missing/sparse data","PINN vs NN and ablations"],horizontal=True)
    if section.startswith("Noise"): _noise_section(rows,truth)
    elif section.startswith("Missing"): _missing_section(rows)
    else: _nn_and_ablation(rows,truth)
