"""Across-weight abundance surfaces for PINN versus mizer comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _surface_figure(x, t, z, title: str, zmin: float, zmax: float):
    fig = go.Figure(
        go.Heatmap(
            x=np.asarray(x, dtype=float),
            y=np.asarray(t, dtype=float),
            z=np.asarray(z, dtype=float),
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "log10(N)"},
            hovertemplate="t=%{y:.5g}<br>x=%{x:.5g}<br>log10(N)=%{z:.5g}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="log-weight x = log(w)", yaxis_title="time")
    return fig


def _abundance_surfaces(run_df, selected_run, mizers, impl) -> None:
    st.subheader("Abundance across time and weight")
    st.caption(
        "PINN and mizer are shown on the same PINN time/log-weight grid and the same log10(N) colour scale. "
        "This is the full-field counterpart to the selected-weight abundance time-series."
    )
    if not mizers or not selected_run:
        st.info("Select a PINN run and provide a mizer CSV to show abundance surfaces.")
        return

    run_dirs = dict(zip(run_df.run_id, run_df.run_dir))
    allm = pd.concat(mizers.values(), ignore_index=True)
    species_options = pd.unique(allm["species"].dropna().astype(str)).tolist()
    if not species_options:
        st.info("No mizer species are available for the abundance surface.")
        return

    c1, c2, c3 = st.columns(3)
    species = c1.selectbox("Surface species", species_options, key="mizer_surface_species")
    source = c2.selectbox("Surface mizer source", list(mizers), key="mizer_surface_source")
    method = c3.selectbox("Surface x matching", ["linear", "nearest"], key="mizer_surface_method")

    pinn_species_idx = species_options.index(species)
    fields = impl.load_fixed_fields(run_dirs.get(selected_run, ""), species_idx=pinn_species_idx)
    if not fields or impl.check_required_arrays(fields, ["t_eval", "x_eval", "log10_N"]):
        st.warning(f"Could not load PINN fixed fields for species_idx={pinn_species_idx}.")
        return

    miz_grid = impl.interpolate_mizer_to_pinn(
        mizers[source],
        species,
        fields["t_eval"],
        fields["x_eval"],
        method=method,
    )
    if miz_grid is None:
        st.warning("Could not place the selected mizer abundance onto the PINN grid.")
        return

    active_x_range = impl.combined_mizer_x_range(mizers, species, [source])
    pinn_grid = impl.mask_matrix_to_x_support(fields["log10_N"], fields["x_eval"], active_x_range)
    miz_grid = impl.mask_matrix_to_x_support(miz_grid, fields["x_eval"], active_x_range)

    finite_pinn = pinn_grid[np.isfinite(pinn_grid)]
    finite_mizer = miz_grid[np.isfinite(miz_grid)]
    if finite_pinn.size == 0 or finite_mizer.size == 0:
        st.warning("No overlapping finite PINN/mizer abundance values remain after support masking.")
        return
    shared = np.concatenate([finite_pinn, finite_mizer])
    zmin = float(np.nanmin(shared))
    zmax = float(np.nanmax(shared))
    if zmin == zmax:
        zmax = zmin + 1e-12

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _surface_figure(
                fields["x_eval"], fields["t_eval"], pinn_grid,
                f"PINN abundance: {fields.get('selected_species', species)}", zmin, zmax,
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            _surface_figure(
                fields["x_eval"], fields["t_eval"], miz_grid,
                f"Mizer abundance: {species}", zmin, zmax,
            ),
            use_container_width=True,
        )
    st.caption(
        f"Mizer matching uses nearest available time and {method} matching in log-weight x. "
        "Both panels use the identical colour limits, so visual differences are directly comparable."
    )


def install(impl) -> None:
    original_mizer_page = impl.mizer_page

    def mizer_page(run_df, selected_run, selected_runs, mizers, clip, mode, markers):
        original_mizer_page(run_df, selected_run, selected_runs, mizers, clip, mode, markers)
        _abundance_surfaces(run_df, selected_run, mizers, impl)

    impl.mizer_page = mizer_page
