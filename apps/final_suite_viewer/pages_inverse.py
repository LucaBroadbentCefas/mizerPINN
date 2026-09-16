"""Saved inverse-parameter analyses R1–R5, C1–C2, and F1–F5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import plot_explanation
from .inverse_analysis import (effort_errors, effort_recovery_summary,
                               fishing_mortality, reshape_selectivity,
                               rmax_recovery, rmse_log_ratio)
from .loaders import read_csv, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RMAX_TASKS = tuple(range(53, 62)); CV_TASKS = tuple(range(62, 66)); EFFORT_TASKS = tuple(range(66, 70))


def _rows(rows): return {row["task_id"]: row for row in rows}


def _available(rows, task_ids):
    by_id=_rows(rows); return [i for i in task_ids if by_id.get(i) and by_id[i]["selected_instance"]]


@st.cache_data(show_spinner=False)
def _read_numeric(path: str) -> np.ndarray:
    return pd.read_csv(path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _input_dir(run_dir: str) -> Path | None:
    configured=read_json(run_dir).get("input_dir"); candidates=[]
    if configured:
        raw=Path(str(configured)); candidates += [raw, PROJECT_ROOT/raw, Path(run_dir)/raw, Path(run_dir).parent/raw]
    candidates.append(PROJECT_ROOT/"validation/fixtures/pde_multispecies")
    return next((path for path in candidates if path.is_dir()),None)


def _rmax_truth(run_dir: str) -> tuple[np.ndarray | None,str]:
    candidates=[Path(run_dir)/"r_max_true.csv"]
    root=_input_dir(run_dir)
    if root: candidates += [root/"r_max_true.csv",root/"r_max.csv"]
    for path in candidates:
        if path.is_file(): return _read_numeric(str(path)).reshape(-1),str(path)
    return None,"r_max_true.csv (run directory or configured input directory)"


def _explain(interpretation,equation,inputs,selection,alignment="Saved rows only; no checkpoint or optimization reconstruction."):
    plot_explanation(st,interpretation=interpretation,calculation=equation,inputs=inputs,selection=selection,alignment=alignment)


def _one_to_one(fig,x,y):
    values=np.r_[x,y]; values=values[np.isfinite(values)&(values>0)]
    if len(values): fig.add_trace(go.Scatter(x=[values.min(),values.max()],y=[values.min(),values.max()],mode="lines",name="1:1"))
    return fig


def _rmax(rows):
    st.header("Rmax recovery"); by_id=_rows(rows); available=_available(rows,RMAX_TASKS)
    if not available: st.info("Rmax diagnostics unavailable: no tasks 53–61 were discovered."); return
    task=st.selectbox("Rmax recovery run",available,format_func=lambda i:f"{i}: {by_id[i]['display_label']}"); run=str(by_id[task]["selected_instance"].run_dir)
    estimated=read_csv(run,"estimated_rmax.csv"); truth,truth_source=_rmax_truth(run)
    ready = estimated is not None and not estimated.empty and truth is not None
    selected = 0
    if not ready:
        missing = "estimated_rmax.csv" if estimated is None or estimated.empty else truth_source
        st.warning(f"R1–R3 unavailable for the selected run: {missing} is required. Starting values are not used as truth. R4/R5 remain available where their run files exist.")
    else:
        recovered=rmax_recovery(estimated,truth); species=list(recovered.species_idx.astype(int).unique()); selected=st.selectbox("Rmax species",species,format_func=lambda i:str(recovered.loc[recovered.species_idx==i,"species"].iloc[0] or f"species {i}"))
        st.subheader("R1 · True versus final estimated Rmax")
        fig=_one_to_one(px.scatter(recovered,x="true_r_max",y="estimated_r_max",hover_data=["species","true_r_max","estimated_r_max","ratio_to_truth"],log_x=True,log_y=True),recovered.true_r_max,recovered.estimated_r_max); st.plotly_chart(fig,use_container_width=True)
        _explain("Checks final per-species recovery against the actual reference Rmax.",r"\rho_i=\widehat R_{max,i}/R_{max,i}^{true}",["estimated_rmax.csv",truth_source],f"task {task}")
        st.subheader("R2 · Rmax optimization trajectory"); history=read_csv(run,"rmax_history.csv")
        if history is None or history.empty or not {"step","species_idx","r_max"}.issubset(history): st.info("R2 unavailable: rmax_history.csv with step, species_idx, and r_max is required.")
        else:
            show_all=st.checkbox("Show all species trajectories",False); plotted=history if show_all else history[history.species_idx==selected]
            fig=px.line(plotted,x="step",y="r_max",color="species" if "species" in plotted else "species_idx",labels={"step":"optimization step"}); fig.add_hline(y=float(truth[selected]),line_dash="dash",annotation_text="true Rmax"); st.plotly_chart(fig,use_container_width=True)
            _explain("Shows saved parameter optimization history, not model time.",r"\widehat R_{max,i}(optimization\ step)",["rmax_history.csv",truth_source],"all species" if show_all else f"species {selected}")
        st.subheader("R3 · Final Rmax ratio"); fig=px.bar(recovered,x="species",y="ratio_to_truth"); fig.add_hline(y=1,line_dash="dash"); st.plotly_chart(fig,use_container_width=True)
        _explain("A ratio of one is exact parameter recovery.",r"\rho_i=\widehat R_{max,i}/R_{max,i}^{true}",["estimated_rmax.csv",truth_source],f"task {task}")
    st.subheader("R4 · Initialization sensitivity"); species_mode=st.checkbox("R4 selected-species error",False,key="rmax_species_mode"); records=[]
    for task_id,factor in zip(range(53,57),(0.25,0.5,2.0,4.0)):
        row=by_id.get(task_id); rd=str(row["selected_instance"].run_dir) if row and row["selected_instance"] else ""; est=read_csv(rd,"estimated_rmax.csv") if rd else None; true,_=_rmax_truth(rd) if rd else (None,"")
        value=np.nan
        if est is not None and not est.empty and true is not None: value=abs(np.log(float(est.loc[est.species_idx==selected,"estimated_r_max"].iloc[0])/true[selected])) if st.session_state.get("rmax_species_mode",False) else rmse_log_ratio(est.estimated_r_max,true)
        records.append({"start_factor":factor,"parameter_error":value})
    st.plotly_chart(px.line(pd.DataFrame(records),x="start_factor",y="parameter_error",markers=True),use_container_width=True)
    _explain("Tests final Rmax sensitivity to four starting factors without treating starts as truth.",r"RMSE_{logR}=\sqrt{mean_i[\log(\widehat R_i/R_i^{true})]^2}",["tasks 53–56 estimated_rmax.csv","r_max_true.csv"],f"{'species '+str(selected) if species_mode else 'all species'}")
    st.subheader("R5 · Noisy Rmax recovery"); noisy=[]
    for task_id in range(57,62):
        row=by_id.get(task_id); rd=str(row["selected_instance"].run_dir) if row and row["selected_instance"] else ""; est=read_csv(rd,"estimated_rmax.csv") if rd else None; true,_=_rmax_truth(rd) if rd else (None,"")
        if est is not None and not est.empty and true is not None: noisy.append({"replicate":task_id-56,"noise_seed":41000+task_id-56,"error":abs(np.log(float(est.loc[est.species_idx==selected,"estimated_r_max"].iloc[0])/true[selected])) if species_mode else rmse_log_ratio(est.estimated_r_max,true)})
    noisy=pd.DataFrame(noisy)
    if noisy.empty: st.info("R5 unavailable: tasks 57–61 need estimated_rmax.csv and r_max_true.csv.")
    else:
        fig=px.scatter(noisy,x="replicate",y="error",hover_data=["noise_seed"]); fig.add_trace(go.Scatter(x=[3],y=[noisy.error.mean()],error_y={"type":"data","array":[noisy.error.std(ddof=1)]},mode="markers",name="mean ± 1 SD",marker={"symbol":"x","size":12})); st.plotly_chart(fig,use_container_width=True)
        _explain("Summarises five noisy recovery replicates at true observation CV 0.3 and initial Rmax factor 0.5; error bars are ±1 sample SD, not a confidence interval.",r"RMSE_{logR}\ or\ |\log(\widehat R_i/R_i^{true})|",["tasks 57–61 inverse outputs"],f"{'species '+str(selected) if species_mode else 'all species'}")


def _cv(rows):
    st.header("Fitted observation CV"); by_id=_rows(rows); starts={62:.1,63:.3,64:.6,65:1.0}; histories=[]; finals=[]
    mode=st.radio("CV trajectory scale",["CV","sigma_log"],horizontal=True)
    for task,start in starts.items():
        row=by_id.get(task); rd=str(row["selected_instance"].run_dir) if row and row["selected_instance"] else ""
        history=read_csv(rd,"data_cv_history.csv") if rd else None; final=read_csv(rd,"estimated_data_cv.csv") if rd else None
        column="cv" if mode=="CV" else "sd_log"
        if history is not None and not history.empty and column in history: histories.append(history.assign(start_cv=start))
        if final is not None and not final.empty and "estimated_cv" in final: finals.append({"initial_cv":start,"final_cv":float(final.estimated_cv.mean()),"absolute_error":abs(float(final.estimated_cv.mean())-.3)})
    st.subheader("C1 · CV optimization trajectories")
    if histories:
        data=pd.concat(histories); column="cv" if mode=="CV" else "sd_log"; fig=px.line(data,x="step",y=column,color="start_cv",labels={"step":"optimization step"}); reference=.3 if mode=="CV" else np.sqrt(np.log1p(.3**2)); fig.add_hline(y=reference,line_dash="dash",annotation_text="truth"); st.plotly_chart(fig,use_container_width=True)
        _explain("Tests convergence from four CV initializations; default display is CV.",r"\sigma_{log}=\sqrt{\log(1+CV^2)}",["data_cv_history.csv"],mode)
    else: st.info(f"C1 unavailable: data_cv_history.csv with {'cv' if mode=='CV' else 'sd_log'} is required.")
    st.subheader("C2 · Final fitted CV versus initialization")
    if finals:
        final=pd.DataFrame(finals); fig=px.line(final,x="initial_cv",y="final_cv",markers=True); fig.add_hline(y=.3,line_dash="dash",annotation_text="true CV=0.3"); st.plotly_chart(fig,use_container_width=True); st.dataframe(final,use_container_width=True)
        _explain("Shows initialization sensitivity and convergence to known observation CV 0.3.",r"error=|\widehat{CV}-0.3|",["estimated_data_cv.csv"],"tasks 62–65")
    else: st.info("C2 unavailable: estimated_data_cv.csv is required for tasks 62–65.")


@st.cache_data(show_spinner=False)
def _fishing_inputs(run_dir: str):
    root=_input_dir(run_dir)
    required=("catchability.csv","selectivity.csv","w.csv","initial_effort.csv")
    if root is None or any(not (root/name).is_file() for name in required): return None,required
    catch=_read_numeric(str(root/"catchability.csv")); w=_read_numeric(str(root/"w.csv")).reshape(-1); raw=_read_numeric(str(root/"selectivity.csv")); effort=_read_numeric(str(root/"initial_effort.csv")).reshape(-1)
    try: selectivity=reshape_selectivity(raw,*catch.shape,len(w))
    except ValueError: return None,required
    return {"root":str(root),"catchability":catch,"selectivity":selectivity,"w":w,"effort":effort},required


def _effort(rows):
    st.header("Fishing-effort recovery"); by_id=_rows(rows); available=_available(rows,EFFORT_TASKS)
    if not available: st.info("Effort diagnostics unavailable: no tasks 66–69 were discovered."); return
    task=st.selectbox("Effort-start run",available,format_func=lambda i:f"{i}: {by_id[i]['display_label']}"); run=str(by_id[task]["selected_instance"].run_dir); effort=read_csv(run,"estimated_effort.csv")
    if effort is None or effort.empty: st.warning("F1–F5 unavailable: estimated_effort.csv is required; no effort history is reconstructed from checkpoints."); return
    effort=effort_errors(effort); gears=sorted(effort.gear_idx.astype(int).unique()); gear=st.selectbox("Gear",gears); starts={66:.05,67:.5,68:1.,69:3.}
    st.caption("No effort optimization trajectory is shown because training does not save an explicit effort-history CSV. These are final recovered physical-time series.")
    st.subheader("F1 · Estimated / true effort")
    grid=effort.pivot(index="gear_idx",columns="physical_time",values="effort_ratio"); fig=px.imshow(grid,aspect="auto",color_continuous_scale="RdBu_r",color_continuous_midpoint=1,labels={"x":"physical model time","y":"gear","color":"estimated / true"}); st.plotly_chart(fig,use_container_width=True)
    zero=effort[effort.true_reference_effort==0]
    if not zero.empty: st.warning(f"{len(zero)} true-zero cells have undefined ratios and are omitted; their absolute errors are reported separately.")
    _explain("Ratio one is exact effort recovery; true-zero effort is never divided by epsilon.",r"\rho_E(t,g)=\widehat E(t,g)/E^{true}(t,g),\ E^{true}>0",["estimated_effort.csv"],f"task {task}")
    st.subheader("F2 · Final effort through physical model time"); overlay=st.checkbox("Overlay all four effort-start runs",False); frames=[]
    for task_id in (available if overlay else [task]):
        rd=str(by_id[task_id]["selected_instance"].run_dir); data=read_csv(rd,"estimated_effort.csv");
        if data is not None and not data.empty:
            data=data[data.gear_idx==gear]; frames += [pd.DataFrame({"physical_time":data.physical_time,"effort":data.estimated_effort,"series":f"estimated start {starts[task_id]}"}),pd.DataFrame({"physical_time":data.physical_time,"effort":data.true_reference_effort,"series":"true/reference"})]
    st.plotly_chart(px.line(pd.concat(frames),x="physical_time",y="effort",color="series"),use_container_width=True)
    _explain("Compares final recovered effort time series with reference effort; this is not an optimization trajectory.",r"\widehat E_g(t)\ versus\ E_g^{true}(t)",["estimated_effort.csv"],f"gear {gear}")
    st.subheader("F3 · Effort error through time"); selected=effort[effort.gear_idx==gear]; positive=selected[selected.true_reference_effort>0]; zero_selected=selected[selected.true_reference_effort==0]
    if not positive.empty: st.plotly_chart(px.line(positive,x="physical_time",y="positive_truth_log_error"),use_container_width=True); _explain("Shows multiplicative error only where reference effort is positive.",r"e_E=|\log(\widehat E/E^{true})|",["estimated_effort.csv"],f"gear {gear}; positive truth")
    if not zero_selected.empty: st.plotly_chart(px.line(zero_selected,x="physical_time",y="zero_truth_absolute_error"),use_container_width=True); _explain("Separately shows absolute error for true-zero effort cells.",r"e_{E,zero}=|\widehat E-E^{true}|",["estimated_effort.csv"],f"gear {gear}; zero truth")
    aggregate_positive=effort[effort.true_reference_effort>0].groupby("physical_time",as_index=False).positive_truth_log_error.mean()
    if not aggregate_positive.empty:
        st.plotly_chart(px.line(aggregate_positive,x="physical_time",y="positive_truth_log_error",title="Mean positive-truth log error across gears"),use_container_width=True)
        _explain("Aggregates multiplicative error across positive-reference gears only.",r"mean_{g:E_g^{true}>0}|\log(\widehat E_g/E_g^{true})|",["estimated_effort.csv"],"all positive-reference gears")
    st.dataframe(pd.DataFrame([effort_recovery_summary(effort,gear)]),use_container_width=True)
    st.subheader("F4 · Effort-start sensitivity"); records=[]
    for task_id,start in starts.items():
        row=by_id.get(task_id); data=read_csv(str(row["selected_instance"].run_dir),"estimated_effort.csv") if row and row["selected_instance"] else None
        if data is not None and not data.empty: records.append({"start":start,**effort_recovery_summary(data,gear)})
    sensitivity=pd.DataFrame(records)
    if sensitivity.empty: st.info("F4 unavailable: estimated_effort.csv is missing for tasks 66–69.")
    else:
        st.plotly_chart(px.line(sensitivity,x="start",y="RMSE_logE_positive_truth",markers=True),use_container_width=True); st.dataframe(sensitivity,use_container_width=True)
        _explain("Assesses final recovery sensitivity to scalar initialization; zero-reference error remains a separate table column.",r"RMSE_{logE}=\sqrt{mean[\log(\widehat E/E^{true})]^2}",["tasks 66–69 estimated_effort.csv"],f"gear {gear}")
    st.subheader("F5 · Recovered fishing mortality"); inputs,required=_fishing_inputs(run)
    if inputs is None: st.warning("F5 unavailable. Required locally resolvable inputs: "+", ".join(required)); return
    species=st.selectbox("Fishing-mortality species",list(range(inputs["catchability"].shape[1]))); gear_choice=st.selectbox("Fishing-mortality gear",[None]+gears,format_func=lambda x:"Total across gears" if x is None else f"gear {x}"); times=sorted(effort.physical_time.unique()); time=st.select_slider("Fishing-mortality physical model time",times,value=times[len(times)//2]); at=effort[np.isclose(effort.physical_time,time)].sort_values("gear_idx")
    estimated_f=fishing_mortality(at.estimated_effort.to_numpy(),inputs["catchability"],inputs["selectivity"],species,gear_choice); true_f=fishing_mortality(at.true_reference_effort.to_numpy(),inputs["catchability"],inputs["selectivity"],species,gear_choice)
    profile=pd.DataFrame({"body_weight":inputs["w"],"recovered estimated F":estimated_f,"true/reference F":true_f}).melt("body_weight",var_name="series",value_name="F"); fig=px.line(profile,x="body_weight",y="F",color="series"); fig.update_xaxes(type="log"); st.plotly_chart(fig,use_container_width=True)
    _explain("Reconstructs fishing mortality without refitting, using the identical gear/species/weight indexing as the model operator.",r"F_{g,i}(t,w)=E_g(t)q_{g,i}s_{g,i}(w),\quad F_i=\sum_gF_{g,i}",["estimated_effort.csv",inputs["root"]+"/catchability.csv",inputs["root"]+"/selectivity.csv",inputs["root"]+"/w.csv"],f"species {species}; {'total' if gear_choice is None else 'gear '+str(gear_choice)}; physical model time {time}","Exact saved weight grid and [species, gear] CSV reshape to internal [gear, species, weight]; no approximation or fit.")


def inverse_page(rows):
    st.title("Inverse parameter analyses")
    section=st.radio("Inverse analysis",["Rmax recovery","CV recovery","Effort recovery"],horizontal=True)
    if section.startswith("Rmax"): _rmax(rows)
    elif section.startswith("CV"): _cv(rows)
    else: _effort(rows)
