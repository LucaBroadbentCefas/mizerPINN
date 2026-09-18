"""Selected-run State (N3–N9) and saved PDE diagnostic (P1–P7) pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analytics import (common_comparison_domain, error_by_species,
                        error_by_species_time, error_by_time, error_by_weight,
                        fold_by_species, mask_domain, pde_balance,
                        residual_aggregate, residual_summary)
from .components import plot_explanation, plot_scale_controls
from .loaders import load_fixed_fields, load_prediction_state, load_w_max
from .state import align_states, load_truth_state, truth_source_for_task


@st.cache_data(show_spinner=False)
def _aligned(run_dir: str, task_id: int):
    prediction = load_prediction_state(run_dir)
    if prediction is None:
        return pd.DataFrame(), {}
    truth_path = truth_source_for_task(task_id)
    if not truth_path.is_file():
        raise FileNotFoundError(f"Required truth file is missing: {truth_path}")
    return align_states(prediction, load_truth_state(truth_path), w_max=load_w_max(run_dir))


def _instance_options(rows, current):
    found = [row for row in rows if row["selected_instance"]]
    ids = [row["task_id"] for row in found]
    if not ids: return None, None, found
    if current not in ids: current = ids[0]
    labels = {row["task_id"]: f"{row['task_id']:02d} · {row['run_label']}" for row in found}
    with st.sidebar:
        st.markdown("### Analysis selection")
        selected = st.selectbox("Selected run", ids, index=ids.index(current), format_func=labels.get, key="analysis_selected_task")
        compare = st.selectbox("Optional comparison run", [None] + [i for i in ids if i != selected], format_func=lambda x: "None" if x is None else labels[x], key="analysis_comparison_task")
    st.session_state.selected_task_id = selected
    return next(row for row in found if row["task_id"] == selected), next((row for row in found if row["task_id"] == compare), None), found


def _shared_controls(row, comparison, source: pd.DataFrame):
    if row is None or source is None or source.empty: return row, comparison, None
    species_rows = source[["species_idx", "species"]].drop_duplicates().sort_values("species_idx")
    options = species_rows.species_idx.astype(int).tolist(); names = dict(zip(options, species_rows.species.astype(str)))
    if not options:
        return row, comparison, None
    with st.sidebar:
        st.markdown("### State/PDE coordinates")
        species_idx = st.selectbox("Species", options, format_func=lambda i: names[i], key=f"species_{row['task_id']}")
        active = source[source.species_idx == species_idx]
        times = np.sort(active.time.unique()); weights = np.sort(active.w.unique())
        if len(times) == 0 or len(weights) == 0:
            return row, comparison, None

        # Batch coordinate changes into a single rerun. While the user moves
        # these sliders, Streamlit does not start another expensive alignment.
        with st.form(key=f"state_pde_coordinates_{row['task_id']}_{species_idx}", clear_on_submit=False):
            time = st.select_slider("Model time", options=times.tolist(), value=float(times[len(times)//2]), key=f"time_{row['task_id']}_{species_idx}")
            weight = st.select_slider("Physical body weight w", options=weights.tolist(), value=float(weights[len(weights)//2]), key=f"weight_{row['task_id']}_{species_idx}")
            time_range = st.select_slider("Model-time range", options=times.tolist(), value=(float(times[0]), float(times[-1])), key=f"time_range_{row['task_id']}_{species_idx}")
            weight_range = st.select_slider("Body-weight range", options=weights.tolist(), value=(float(weights[0]), float(weights[-1])), key=f"weight_range_{row['task_id']}_{species_idx}")
            st.form_submit_button("Apply coordinates", type="primary", use_container_width=True)
        st.caption("Adjust time/weight controls, then apply once. Slider movement alone does not start a new calculation.")
    return row, comparison, {"species_idx": species_idx, "species": names[species_idx], "time": float(time), "weight": float(weight), "time_range": time_range, "weight_range": weight_range}

def _heatmap(data, value, title, symmetric=False):
    if data is None or data.empty or value not in data:
        st.info(f"{title} unavailable: no valid cells are present in the selected domain.")
        return False
    grid = data.pivot(index="time", columns="w", values=value).sort_index().sort_index(axis=1)
    values = grid.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        st.info(f"{title} unavailable: the selected cells contain no finite values.")
        return False
    limit = None
    if symmetric:
        limit = float(np.max(np.abs(finite)))
        if limit == 0:
            limit = np.finfo(float).eps
    fig = go.Figure(go.Heatmap(x=grid.columns, y=grid.index, z=grid, colorscale="RdBu_r" if symmetric else "Viridis", zmin=-limit if symmetric else None, zmax=limit, colorbar_title=value))
    fig.update_layout(title=title, xaxis_title="physical body weight w", yaxis_title="model time"); fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    return True


def _multi_species_state_overview(aligned: pd.DataFrame, row, truth_path):
    species_rows = aligned[["species_idx", "species"]].drop_duplicates().sort_values("species_idx")
    if len(species_rows) < 2:
        return
    species_order = species_rows.species.astype(str).tolist()
    species_ids = species_rows.species_idx.astype(int).tolist()
    names = dict(zip(species_ids, species_order))

    counts = aligned.groupby("time").species_idx.nunique()
    common_times = counts[counts == len(species_ids)].index.to_numpy(dtype=float)
    times = common_times if len(common_times) else np.sort(aligned.time.unique())
    if len(times) == 0:
        return

    st.subheader("Multi-species state overview")
    overview_time = st.select_slider(
        "Multi-species profile time",
        options=times.tolist(),
        value=float(times[len(times) // 2]),
        key=f"multi_species_time_{row['task_id']}",
    )
    at_time = aligned[np.isclose(aligned.time, overview_time)].copy()
    if at_time.empty:
        st.info("No aligned state cells are available at the selected overview time.")
        return

    st.markdown("**All species as small multiples**")
    y_scale = st.radio(
        "Small-multiple y scale",
        ["Shared", "Independent"],
        horizontal=True,
        key=f"multi_species_y_scale_{row['task_id']}",
    )
    facets = at_time[["species_idx", "species", "w", "true_log10_N", "pred_log10_N"]].melt(
        id_vars=["species_idx", "species", "w"],
        value_vars=["true_log10_N", "pred_log10_N"],
        var_name="source",
        value_name="log10_N",
    )
    facets["source"] = facets.source.map({
        "true_log10_N": "Mizer truth",
        "pred_log10_N": "PINN",
    })
    fig = px.line(
        facets,
        x="w",
        y="log10_N",
        color="source",
        line_dash="source",
        facet_col="species",
        facet_col_wrap=4,
        category_orders={"species": species_order},
    )
    fig.update_xaxes(type="log", matches=None, title="physical body weight w")
    if y_scale == "Independent":
        fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=max(700, 230 * int(np.ceil(len(species_order) / 4))))
    st.plotly_chart(fig, use_container_width=True)
    plot_explanation(
        st,
        interpretation="Shows every species' fitted size spectrum against its own mizer truth at the same model time. Independent y scaling emphasises shape agreement; shared y scaling preserves between-species abundance differences.",
        calculation=r"N_i(t,w)\rightarrow\log_{10}N_i(t,w)",
        inputs=["aligned predicted state", str(truth_path)],
        selection=f"all species; time={overview_time:g}",
        alignment="Each species retains its own active body-weight domain; no cross-species weight extrapolation is introduced.",
    )

    st.markdown("**All species on one graph**")
    displayed = st.multiselect(
        "Displayed species",
        species_ids,
        default=species_ids,
        format_func=lambda i: names[i],
        key=f"multi_species_overlay_species_{row['task_id']}",
    )
    overlay = at_time[at_time.species_idx.astype(int).isin(displayed)].copy()
    if overlay.empty:
        st.info("Select at least one species to display.")
    else:
        palette = px.colors.qualitative.Dark24
        fig = go.Figure()
        for index, species_idx in enumerate(species_ids):
            if species_idx not in displayed:
                continue
            frame = overlay[overlay.species_idx.astype(int).eq(species_idx)].sort_values("w")
            if frame.empty:
                continue
            colour = palette[index % len(palette)]
            species_name = names[species_idx]
            fig.add_trace(go.Scatter(
                x=frame.w,
                y=frame.true_log10_N,
                mode="lines",
                name=f"{species_name} truth",
                legendgroup=species_name,
                showlegend=False,
                line={"color": colour, "dash": "dash"},
                hovertemplate=f"{species_name}<br>truth<br>w=%{{x:.4g}}<br>log10N=%{{y:.4g}}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=frame.w,
                y=frame.pred_log10_N,
                mode="lines",
                name=species_name,
                legendgroup=species_name,
                showlegend=True,
                line={"color": colour},
                hovertemplate=f"{species_name}<br>PINN<br>w=%{{x:.4g}}<br>log10N=%{{y:.4g}}<extra></extra>",
            ))
        fig.update_layout(
            xaxis_title="physical body weight w",
            yaxis_title="log10 N",
            legend={"groupclick": "togglegroup", "title": {"text": "Species"}},
        )
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Solid = PINN; dashed = mizer truth. Clicking a species in the legend toggles both traces for that species. The multiselect can also remove species explicitly.")
        plot_explanation(
            st,
            interpretation="Places all selected species on the same axes for direct between-species comparison while keeping truth and PINN paired by species.",
            calculation=r"\{\log_{10}N_i^{true}(t,w),\ \log_{10}\hat N_i(t,w)\}_{i=1}^{S}",
            inputs=["aligned predicted state", str(truth_path)],
            selection=f"time={overview_time:g}; displayed species={len(displayed)}",
            alignment="Species keep their own active body-weight domains. Legend groups toggle the truth and PINN traces together.",
        )

    st.markdown("**Species × time state error**")
    by_species_time = error_by_species_time(aligned)
    heatmap = by_species_time.pivot(index="species", columns="time", values="RMSE_log10N").reindex(species_order)
    fig = go.Figure(go.Heatmap(
        x=heatmap.columns,
        y=heatmap.index,
        z=heatmap.to_numpy(dtype=float),
        colorscale="Viridis",
        colorbar_title="RMSE log10N",
        hovertemplate="species=%{y}<br>time=%{x:.4g}<br>RMSE=%{z:.4g}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="model time", yaxis_title="species")
    st.plotly_chart(fig, use_container_width=True)
    plot_explanation(
        st,
        interpretation="Shows which species are inaccurate and when. Each cell aggregates over that species' valid active body-weight bins only.",
        calculation=r"RMSE_i(t)=\sqrt{\operatorname{mean}_{w\in active_i}e_i(t,w)^2}",
        inputs=["aligned predicted and task-specific mizer truth states", "species active-size masks"],
        selection="all species; all aligned times",
        alignment="No averaging across species; each row uses that species' own valid body-weight domain.",
    )


def state_page(rows):
    st.header("Selected run: State")
    row, comparison, _ = _instance_options(rows, st.session_state.get("selected_task_id"))
    if row is None:
        st.info("Select a discovered run from the Suite map.")
        return
    truth_path = truth_source_for_task(row["task_id"])
    if not truth_path.is_file():
        st.error(f"Required truth file is missing: {truth_path}")
        return
    run_dir = str(row["selected_instance"].run_dir)
    try:
        aligned, meta = _aligned(run_dir, row["task_id"])
    except Exception as exc:
        st.error(f"Truth/state alignment failed: {exc}")
        return
    if aligned.empty:
        st.warning("No overlapping valid prediction/truth state cells are available for this run.")
        print(f"[state.page] empty aligned state task={row['task_id']} run_dir={run_dir}", flush=True)
        return
    row, comparison, ctl = _shared_controls(row, comparison, aligned)
    if not ctl:
        st.warning("No valid saved prediction-state coordinates are available.")
        return
    chosen = mask_domain(aligned[aligned.species_idx == ctl["species_idx"]], ctl["time_range"], ctl["weight_range"])
    if chosen.empty:
        st.warning("The selected species/time/weight domain contains no aligned prediction/truth cells.")
        print(f"[state.page] empty selected domain task={row['task_id']} species={ctl['species_idx']} time_range={ctl['time_range']} weight_range={ctl['weight_range']} aligned_rows={len(aligned)}", flush=True)
        return
    compare_aligned = None
    if comparison:
        compare_aligned, _ = _aligned(str(comparison["selected_instance"].run_dir), comparison["task_id"])
        compare_aligned = mask_domain(compare_aligned[compare_aligned.species_idx == ctl["species_idx"]], ctl["time_range"], ctl["weight_range"])
        if compare_aligned.empty:
            compare_aligned = None
            st.info("The comparison run has no cells in the selected species/time/weight domain; comparison overlays are omitted.")

    st.caption(f"Truth source for task {row['task_id']}: {truth_path.name}")
    _multi_species_state_overview(aligned, row, truth_path)
    st.subheader("N3/N4 · State-error heatmap")
    mode = st.radio("Error display", ["Signed", "Absolute"], horizontal=True)
    display = chosen.assign(display=chosen.error_log10_N.abs() if mode == "Absolute" else chosen.error_log10_N)
    _heatmap(display, "display", f"{mode} state error", mode == "Signed")
    plot_explanation(st, interpretation="Locates signed over/under-prediction or its magnitude. e=0 is exact; e≈+0.301 is about twice truth; e≈−0.301 is about half truth; e=+1 is ten times truth.", calculation=r"e(t,w)=\log_{10}N_{pred}-\log_{10}N_{true}", inputs=["fixed_grid_fields.csv or final_predictions_grid.csv", str(truth_path)], selection=f"{ctl['species']}; t={ctl['time_range']}; w={ctl['weight_range']}", alignment=f"{meta.get('time_method')}; {meta.get('weight_method')}; no extrapolation; active species bins only")

    st.subheader("N5 · Error through time")
    primary = error_by_time(chosen).assign(run=row["run_label"]); frames = [primary]
    if compare_aligned is not None:
        a, b = common_comparison_domain(chosen, compare_aligned); frames = [error_by_time(a).assign(run=row["run_label"]), error_by_time(b).assign(run=comparison["run_label"])]
    fig = px.line(pd.concat(frames), x="time", y="RMSE_log10N", color="run"); fig = plot_scale_controls(st, fig, key="N5", log_y=True); st.plotly_chart(fig, use_container_width=True)
    plot_explanation(st, interpretation="Tracks state error through model time over the selected body-size interval.", calculation=r"RMSE(t)=\sqrt{\operatorname{mean}_w e(t,w)^2}", inputs=["aligned predicted and task-specific mizer truth states"], selection=f"{ctl['species']}; w={ctl['weight_range']}", alignment="Comparison runs are each aligned to their own authoritative simulation truth, then restricted to identical species/time/weight coordinates before aggregation.")

    st.subheader("N6 · Error through body size")
    frames = [error_by_weight(chosen).assign(run=row["run_label"])]
    if compare_aligned is not None:
        a, b = common_comparison_domain(chosen, compare_aligned); frames = [error_by_weight(a).assign(run=row["run_label"]), error_by_weight(b).assign(run=comparison["run_label"])]
    fig = px.line(pd.concat(frames), x="w", y="RMSE_log10N", color="run"); fig = plot_scale_controls(st, fig, key="N6", log_y=True, log_x=True, default_log_x=True); fig.update_xaxes(title="physical body weight w"); st.plotly_chart(fig, use_container_width=True)
    plot_explanation(st, interpretation="Shows which physical body sizes have greatest state error over the selected times.", calculation=r"RMSE(w)=\sqrt{\operatorname{mean}_t e(t,w)^2}", inputs=["aligned predicted and task-specific mizer truth states"], selection=f"{ctl['species']}; t={ctl['time_range']}", alignment="Linear alignment is in log-weight; display axis can be switched between physical linear and logarithmic w.")

    all_primary = mask_domain(aligned, ctl["time_range"], ctl["weight_range"])
    st.subheader("N7 · Error by species")
    bars = [error_by_species(all_primary).assign(run=row["run_label"])]
    if comparison:
        other, _ = _aligned(str(comparison["selected_instance"].run_dir), comparison["task_id"]); other = mask_domain(other, ctl["time_range"], ctl["weight_range"]); a, b = common_comparison_domain(all_primary, other); bars = [error_by_species(a).assign(run=row["run_label"]), error_by_species(b).assign(run=comparison["run_label"])]
    fig = px.bar(pd.concat(bars), x="species", y="RMSE_log10N", color="run", barmode="group"); fig = plot_scale_controls(st, fig, key="N7", log_y=True); st.plotly_chart(fig, use_container_width=True)
    plot_explanation(st, interpretation="Compares state accuracy among species, with each species restricted to its own active size domain.", calculation=r"RMSE_i=\sqrt{\operatorname{mean}_{t,w}e_i^2}", inputs=["aligned states", "species w_max"], selection=f"t={ctl['time_range']}; requested w={ctl['weight_range']}", alignment="Species active-weight masks are applied before aggregation.")

    st.subheader("N8 · Absolute fold error")
    fold_mode = st.radio("Fold-error aggregation", ["By species", "All valid cells"], horizontal=True)
    folds = fold_by_species(all_primary) if fold_mode == "By species" else pd.DataFrame({"group": ["All valid cells"], "fold_error": [10 ** np.mean(np.abs(all_primary.error_log10_N))]})
    x = "species" if fold_mode == "By species" else "group"; fig = px.bar(folds, x=x, y="fold_error"); fig = plot_scale_controls(st, fig, key="N8", log_y=True); st.plotly_chart(fig, use_container_width=True)
    plot_explanation(st, interpretation="Summarises typical unsigned multiplicative disagreement: 1 is exact and 2 is a typical factor-of-two difference.", calculation=r"F=10^{\operatorname{mean}|e|}", inputs=["aligned predicted and task-specific mizer truth states"], selection=f"t={ctl['time_range']}; w={ctl['weight_range']}", alignment="All aggregation uses valid active cells only.")

    st.subheader("N9 · State comparison profile")
    profile_mode = st.radio("Profile mode", ["Across body size", "Through time"], horizontal=True); scale = st.radio("Display", ["log10(N)", "physical N"], horizontal=True)
    nearest_time = float(chosen.time.iloc[np.argmin(np.abs(chosen.time.to_numpy()-ctl["time"]))]); nearest_w = float(chosen.w.iloc[np.argmin(np.abs(chosen.w.to_numpy()-ctl["weight"]))])
    if profile_mode == "Across body size": profile = chosen[np.isclose(chosen.time, nearest_time)]; x, xlabel = "w", "physical body weight w"
    else: profile = chosen[np.isclose(chosen.w, nearest_w)]; x, xlabel = "time", "model time"
    value_cols = ["true_log10_N", "pred_log10_N"]; labels = {"true_log10_N": "Task-specific mizer truth", "pred_log10_N": row["run_label"]}
    long = profile[[x]+value_cols].melt(x, var_name="source", value_name="log10_N"); long.source = long.source.map(labels)
    if compare_aligned is not None:
        axis_values = compare_aligned.time if profile_mode == "Across body size" else compare_aligned.w; target = ctl["time"] if profile_mode == "Across body size" else ctl["weight"]
        comparison_coordinate = float(axis_values.iloc[np.argmin(np.abs(axis_values.to_numpy() - target))])
        cp = compare_aligned[np.isclose(axis_values, comparison_coordinate)]
        extra = cp[[x, "pred_log10_N"]].rename(columns={"pred_log10_N":"log10_N"}); extra["source"] = comparison["run_label"]; long = pd.concat([long, extra])
    y = "log10_N" if scale == "log10(N)" else "N"; long["N"] = 10**long.log10_N
    fig = px.line(long, x=x, y=y, color="source"); fig = plot_scale_controls(st, fig, key="N9", log_y=(scale == "physical N"), log_x=(x == "w"), default_log_x=(x == "w")); fig.update_xaxes(title=xlabel); st.plotly_chart(fig, use_container_width=True)
    plot_explanation(st, interpretation="Always compares saved PINN state with the authoritative mizer truth for that task at one requested coordinate.", calculation=r"N=10^{\log_{10}N}", inputs=["aligned predicted state", str(truth_path)], selection=f"requested t={ctl['time']}, mapped to t={nearest_time}; requested w={ctl['weight']}, mapped to w={nearest_w}", alignment=f"{meta.get('weight_method')}; {meta.get('time_method')}; plotted coordinate uses the nearest available aligned cell.")


def pde_page(rows):
    st.header("Selected run: PDE")
    row, comparison, _ = _instance_options(rows, st.session_state.get("selected_task_id"))
    if row is None: st.info("Select a discovered run from the Suite map."); return
    selected_dir = str(row["selected_instance"].run_dir); fixed = load_fixed_fields(selected_dir)
    if fixed is None or fixed.empty: st.warning("Saved fixed-grid diagnostics are unavailable; the viewer will not recompute the PDE."); return
    if not {"time", "w", "species_idx"}.issubset(fixed.columns):
        missing = sorted({"time", "w", "species_idx"}.difference(fixed.columns)); st.error(f"Saved fixed-grid diagnostics are missing required columns: {missing}"); print(f"[pde.page] missing required columns task={row['task_id']} columns={list(fixed.columns)}", flush=True); return
    limits = load_w_max(selected_dir); fixed = fixed[fixed.apply(lambda item: item.w <= limits.get(int(item.species_idx), np.inf), axis=1)]
    row, comparison, ctl = _shared_controls(row, comparison, fixed)
    if not ctl: st.warning("No valid saved fixed-grid coordinates are available."); return
    selected = mask_domain(fixed[fixed.species_idx == ctl["species_idx"]], ctl["time_range"], ctl["weight_range"])
    if selected.empty:
        st.warning("The selected species/time/weight domain contains no saved fixed-grid diagnostics."); print(f"[pde.page] empty selected domain task={row['task_id']} species={ctl['species_idx']} time_range={ctl['time_range']} weight_range={ctl['weight_range']} fixed_rows={len(fixed)}", flush=True); return
    st.caption("All fields below are loaded from saved fixed-grid diagnostics. No checkpoint, model evaluation, PDE call, derivative, or autograd calculation occurs in this viewer.")

    st.subheader("P1/P2 · Residual heatmap")
    mode = st.radio("Residual display", ["Signed R_log", "Absolute |R_log|"], horizontal=True); selected = selected.assign(display=selected.residual_log if mode.startswith("Signed") else selected.residual_log.abs())
    _heatmap(selected, "display", mode, mode.startswith("Signed"))
    plot_explanation(st, interpretation="Locates signed PDE imbalance or its magnitude on the saved fixed grid.", calculation=r"R_{log}=\partial_t\log N+g\partial_w\log N+\mu+\partial_wg", inputs=["fixed_grid_fields.csv: residual_log"], selection=f"{ctl['species']}; t={ctl['time_range']}; w={ctl['weight_range']}", alignment="No interpolation; selected saved grid cells only. State derivatives were generated by network autograd during diagnostics; biological terms were generated by model operators; this viewer only reads them.")

    st.subheader("P3 · Residual distribution")
    st.plotly_chart(px.histogram(selected, x="residual_log", nbins=60), use_container_width=True); summary = residual_summary(selected.residual_log)
    plot_explanation(st, interpretation="Describes residual centre and tails; fixed-grid cells are not statistically independent samples.", calculation=r"RMS=\sqrt{mean(R_{log}^2)};\quad p_q=quantile_q(|R_{log}|)", inputs=["fixed_grid_fields.csv: residual_log"], selection=f"{ctl['species']}; selected time/body-weight domain", alignment="Finite saved cells only; no statistical independence is assumed.")
    for col, (name, value) in zip(st.columns(7), summary.items()): col.metric(name.replace("_", " ").upper(), f"{value:.4g}")

    for index, (title, by, x, logx) in enumerate((("P4 · Residual through time", "time", "time", False), ("P5 · Residual through body size", "w", "w", True)), start=4):
        st.subheader(title); agg = residual_aggregate(selected, by).melt(by, var_name="summary", value_name="absolute residual")
        fig = px.line(agg, x=x, y="absolute residual", color="summary"); fig = plot_scale_controls(st, fig, key=f"P{index}", log_y=True, log_x=logx, default_log_x=logx); fig.update_xaxes(title="physical body weight w" if logx else "model time"); st.plotly_chart(fig, use_container_width=True)
        plot_explanation(st, interpretation=f"Shows residual magnitude variation through {'body size' if by == 'w' else 'model time'}.", calculation=r"mean(|R|),\quad p_{95}(|R|),\quad max(|R|)", inputs=["fixed_grid_fields.csv: residual_log"], selection=f"{ctl['species']}; selected domain", alignment=f"Aggregated over selected {'times' if by == 'w' else 'active body weights'}.")

    st.subheader("P6 · PDE component heatmap")
    names = {"dlogN_dt":"d(log N)/dt", "advective":"g*d(log N)/dw", "mu":"mu", "dg_dw":"dg/dw", "g":"g"}; available = [name for name in names if name in selected and selected[name].notna().any()]
    if available:
        component = st.selectbox("Saved PDE component", available, format_func=names.get); _heatmap(selected, component, names[component], False)
        plot_explanation(st, interpretation="Displays one saved PDE component without fabricating unavailable fields.", calculation=names[component], inputs=[f"fixed_grid_fields.csv: {component}"], selection=f"{ctl['species']}; selected domain", alignment="No viewer-side derivative calculation or interpolation.")
    else: st.info("No supported saved PDE components are available.")

    st.subheader("P7 · PDE balance")
    try: balance, maximum, inconsistent = pde_balance(selected)
    except ValueError as exc: st.info(str(exc)); return
    balance_mode = st.radio("Balance mode", ["Across body weight", "Through time"], horizontal=True)
    if balance_mode == "Across body weight": slice_ = balance[np.isclose(balance.time, balance.time.iloc[np.argmin(np.abs(balance.time.to_numpy()-ctl["time"]))])]; x="w"
    else: slice_ = balance[np.isclose(balance.w, balance.w.iloc[np.argmin(np.abs(balance.w.to_numpy()-ctl["weight"]))])]; x="time"
    terms = ["dlogN_dt", "advective", "mu", "dg_dw", "term_sum"] + (["residual_log"] if "residual_log" in slice_ else [])
    fig = px.line(slice_.melt(x, terms, "term", "value"), x=x, y="value", color="term")
    if x == "w": fig = plot_scale_controls(st, fig, key="P7", log_x=True, default_log_x=True)
    st.plotly_chart(fig, use_container_width=True)
    if inconsistent: st.warning(f"Saved four-term sum and stored residual disagree beyond numerical tolerance (maximum absolute difference {maximum:.4g}); this indicates a diagnostic/output inconsistency.")
    plot_explanation(st, interpretation="Checks signed balance among saved derivative and biological/operator terms and compares their sum with the stored residual.", calculation=r"A=\partial_t\log N,\ B=g\partial_w\log N,\ C=\mu,\ D=\partial_wg,\ R_{log}=A+B+C+D", inputs=["fixed_grid_fields.csv saved components and residual"], selection=f"{ctl['species']}; requested t={ctl['time']}; requested w={ctl['weight']}", alignment="Nearest saved coordinate for the profile; viewer sums saved columns only.")
