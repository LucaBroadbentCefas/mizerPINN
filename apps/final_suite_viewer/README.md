# PINNmizer final-suite viewer

The independent acceptance review, evidence, limitations, and per-feature
PASS/FIXED/BLOCKED matrix are recorded in [FINAL_AUDIT.md](FINAL_AUDIT.md).

This is a purpose-built, read-only Streamlit application for the fixed 70-task
final HPC experimental design. It is separate from `apps/hpc_viewer`.

```bash
streamlit run apps/final_suite_viewer/streamlit_app.py
```

The app scans `runs/pde_only_single_species` and `runs/pde_multispecies` by
default. It identifies runs by `final_suite_label.txt`, enriches them from
key/value, CSV, or JSON final-suite manifests, and only then falls back to
configuration metadata. A stale absolute `run_dir` in a copied HPC manifest is
not trusted: local labels and folder basenames are used for matching. Duplicate
identities remain visible and the newest local instance is merely the labelled
default.

No repository-tracked full time × species × weight canonical truth export was
found during implementation. In particular, fixture initial-state arrays and
`final_runs/observations/final_runs/perfect.csv` are not full truth state. Set
**Technical settings → Canonical truth CSV** to an explicit long-form export.
The normalised schema is `time, species_idx, species, w, x, N, log_N, log10_N`.

The foundation never loads checkpoints, evaluates a neural network, or calls a
PINNmizer PDE. Disk discovery and truth reads are cached. Later page controls
will filter the already-loaded tables.

## Implemented selected-run diagnostics

The **State** page reads the best saved state grid (`fixed_grid_fields.csv`
where available, otherwise `final_predictions_grid.csv`) and compares it only
with the configured canonical mizer truth. It provides signed/absolute error,
RMSE through time and body size, species and fold-error summaries, and state
profiles. A multi-species overview shows all species simultaneously as small
multiples, as a single toggleable overlay, and as a species-by-time RMSE
heatmap. The overlay groups each species' PINN and mizer-truth traces so one
legend click hides or restores both, and also provides an explicit species
multiselect. Optional run comparisons are restricted to common aligned cells.

The **PDE** page reads saved fixed-grid residuals and component fields. It never
recomputes derivatives or calls the PDE. The **Data** page reads
`data_predictions_final.csv` and explains the repository's observation
operators and likelihood chain. The **Training** page reads `loss_history.csv`,
`fixed_diagnostic_history.csv`, and retained timing columns. All file reads and
deterministic alignment are cached; controls only filter in-memory tables.

The **Experiment analyses** page uses the catalogue fields rather than parsing
run-folder names. Its noise section opens with a dedicated replicate-comparison
view for one CV: all available gate-ON fits can be compared directly against the
same mizer truth across weight or time, with signed/absolute state error, RMSE
through time, RMSE across weight, a species-by-replicate error matrix, and
overlaid saved training-loss histories. Replicate state diagnostics are first
restricted to the exact common species/time/weight cells. The page also retains
paired noise/CV summaries, actual observation-file validation for sparse and
missing-data designs, matched PINN versus no-PDE comparisons, the single-species
ablation matrix, and no-data versus perfect-data multispecies baselines.
Run/truth metric tables and state alignments are cached.

## Dependencies and saved outputs

Install Streamlit, pandas, NumPy, and Plotly (the existing viewer requirements
provide these). PyTorch is not required for ordinary browsing. Run directories
may independently omit diagnostics: the affected plot explains the required
filename/columns while the rest of the application remains usable.

State comparisons use the canonical mizer truth CSV only. Truth is interpolated
onto saved prediction cells linearly in `x = log(w)` and model time, without
extrapolation and with species `w_max` masks. State metrics are log10 RMSE and
`10 ** mean(abs(log10(N_pred)-log10(N_true)))`. The saved PDE equation is
`R_log = dlogN_dt + g*dlogN_dw + mu + dg_dw`; the viewer does not recompute it.

Observation plots use the repository operators: abundance or biomass size
integrals, survey catchability, and fishing catch based on
`F[g,i](t,w) = E[g](t) * q[g,i] * s[g,i](w)`. Interval catch uses three
left-closed time points. The lognormal loss uses saved uncertainty,
`z = (log(y)-log(yhat))/sd_log`, and the discrepancy gate compares `sum(z^2)`
with the saved chi-square threshold.

The noise sweep contains five matched seeds at each CV and only one gate-OFF
replicate. Error bars are sample SD, never confidence intervals. Missing-data
pages distinguish continuous temporal gaps, omitted species, and discrete
omitted observation years. Matched PINN/no-PDE pairs are tasks 13/50, 24/51,
and 46/52, sourced from the catalogue.

## Inverse parameters

Rmax pages use `estimated_rmax.csv`, `rmax_history.csv`, and actual
`r_max_true.csv`/reference inputs. CV pages use `data_cv_history.csv` and
`estimated_data_cv.csv`, with true CV 0.3. Effort pages treat
`estimated_effort.csv` as the authoritative final recovered physical-time
series. Positive-reference effort uses log-ratio error; true-zero effort uses a
separate absolute error. Fishing mortality is reconstructed only when local
`catchability.csv`, `selectivity.csv`, `w.csv`, and `initial_effort.csv` inputs
can be resolved, using the repository's exact gear/species/weight indexing.
There is no effort optimization-history plot because no explicit effort-history
CSV is written.

## Performance and limitations

Suite discovery, file reads, truth loading, aligned grids, observation designs,
and per-run metric summaries are cached. Slider changes filter cached frames and
never load checkpoints, run a network forward pass, invoke autograd, or call the
PDE. Known limitations are that canonical truth must be supplied externally,
copied runs must retain sufficient config/input metadata to resolve biological
inputs, and unavailable historical diagnostics cannot be reconstructed from a
final checkpoint by this read-only app.
