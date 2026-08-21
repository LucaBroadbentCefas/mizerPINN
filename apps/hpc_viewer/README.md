# PINNmizer HPC Viewer

A read-only Streamlit app for browsing, plotting, and comparing PINNmizer HPC run outputs.

The viewer is intentionally file-based and conservative: it reads existing CSV/JSON/NPZ output files from run folders and optional mizer CSV files. It does **not** load PyTorch checkpoints, recompute model predictions, call PINNmizer PDE/model functions, modify run outputs, regenerate PNGs, or start background jobs.

## Run

From the repository root:

```bash
streamlit run apps/hpc_viewer/streamlit_app.py
```

The default run root is:

```text
runs/pde_only_single_species
```

You can override this path in the sidebar. The app scans only immediate subdirectories of the selected run root.

## Tabs and data diagnostics

The standard run browser, training, fixed-diagnostic, field, run-comparison, mizer-comparison, and file/config pages retain the base viewer behaviour. Observation-data diagnostics are isolated to the **Single run: data** tab rather than being injected into the general training and run-comparison pages.

The Data tab uses `data_predictions_final.csv` when present and provides observed-versus-predicted, normal QQ, standardised residual, observation coverage, largest-misfit, and fitted-CV diagnostics.

## Expected run folder structure

A typical HPC run lives under:

```text
runs/pde_only_single_species/<timestamp>/
```

The viewer uses files when they are present and skips plots gracefully when files or columns are missing:

```text
config.json
final_summary.csv
final_summary.json
timing_summary.csv
run_command.txt
loss_history.csv
fixed_diagnostic_history.csv
fixed_grid_fields.npz
fixed_grid_fields.csv
final_predictions_grid.csv
```

Important field files:

- `fixed_grid_fields.npz` is preferred for field heatmaps and profiles.
- `fixed_grid_fields.csv` is used as a fallback when the NPZ is missing.
- `final_predictions_grid.csv` may be used for long-format PINN abundance data where convenient.

Expected fixed-grid arrays/columns include:

```text
t_eval, x_eval, w_eval, log10_N, residual_log,
dlogN_dt, advective, mu, dg_dw, g_eval
```

## Recover HPC outputs from an older non-HPC run

Do this outside the Streamlit app. The recovery utility loads the completed run's final checkpoint (or the numerically highest `model_step_<N>.pt` if `model_final.pt` is absent), replays the saved `run_command.txt`, forces zero training steps, and enables `--hpc`.

For one run:

```bash
python -m scripts.recover_hpc_outputs runs/<run_type>/<legacy_run>
```

For several runs:

```bash
python -m scripts.recover_hpc_outputs \
  runs/<run_type>/<legacy_run_1> \
  runs/<run_type>/<legacy_run_2>
```

Inspect the exact replay command without executing it:

```bash
python -m scripts.recover_hpc_outputs runs/<run_type>/<legacy_run> --dry-run
```

Override the original device if necessary:

```bash
python -m scripts.recover_hpc_outputs runs/<run_type>/<legacy_run> --device cpu
```

The legacy run is not modified. A new run directory is created by the relevant trainer. Because `--n-steps 0` is forced, the loaded neural-network weights are evaluated but not optimised. `run_command.txt` is required: rebuilding an old checkpoint using current CLI defaults is unsafe because architecture and state-parameterisation settings must match the checkpoint.

## Mizer CSV support

Mizer CSVs can be provided by local path or upload in the sidebar. Multiple files are supported.

The app normalises flexible column names into this standard long format:

```text
source_name, t, species, w, x, N, log_N, log10_N
```

Accepted aliases:

| Standard column | Accepted aliases |
| --- | --- |
| `t` | `time`, `t`, `t_eval` |
| `species` | `sp`, `species`, `species_id`, `species_name` |
| `w` | `weight`, `w`, `w_eval` |
| `x` | `x`, `log_weight`, `log_w`, `x_eval` |
| `N` | `N`, `n`, `abundance`, `density` |
| `log_N` | `log_N`, `logN`, `ln_N` |
| `log10_N` | `log10_N`, `log10N` |

Normalisation rules:

- If `x` is missing but `w` is present, `x = log(w)`.
- If `log10_N` is missing but `N` is present, `log10_N = log10(max(N, tiny))`.
- If `log10_N` is missing but `log_N` is present, `log10_N = log_N / log(10)`.
- If `N` is missing but `log_N` is present, `N = exp(log_N)`.
- If `N` is missing but `log10_N` is present, `N = 10 ** log10_N`.
- If species is absent, a single species named `species_0` is assumed.

## PINN-mizer comparison across weight

The mizer page retains the selected-weight abundance time-series for detailed traces. It also includes an **Abundance across time and weight** section that shows paired PINN and mizer `time × log(weight)` heatmaps. Mizer abundance is placed on the PINN grid using nearest available time and selectable linear/nearest matching in log-weight. Both panels use one shared `log10(N)` colour range, so differences in abundance can be judged without selecting individual weights.

The existing signed PINN-minus-mizer heatmap and error summaries by time and weight remain available for quantitative error inspection.

## Interpolation notes

The app never assumes mizer and PINN grids match. For profile overlays it uses nearest available times or weights and states this in the plot description. For mizer-minus-PINN heatmaps, abundance surfaces, and summaries, mizer data is placed onto the PINN fixed grid using nearest-time selection plus linear or nearest interpolation/matching in log-weight `x` as stated on the page.

Run-to-run difference heatmaps are stricter: they are plotted only when `t_eval` and `x_eval` match exactly between the reference and comparison run. If grids differ, the app shows a message and does not plot the difference.

## Troubleshooting

- **No runs found**: check the run root path in the sidebar. The app scans immediate subdirectories only.
- **Missing plot**: the corresponding file may be absent, empty, or missing required columns/arrays. The plot area reports the missing inputs instead of raising a traceback.
- **Empty CSV**: empty CSV files are tolerated and reported as empty.
- **Log-axis warning**: non-positive values are replaced with `NaN` before log-scaled plots so Plotly does not display invalid values.
- **Mizer interpolation impossible**: ensure the mizer CSV has time, log-weight or weight, and abundance/log-abundance columns after alias normalisation.
- **Legacy run cannot be recovered**: ensure it contains `run_command.txt` and either `model_final.pt` or at least one `model_step_<N>.pt` checkpoint.

### Required long CSV for mizer array exports

For a mizer abundance array with dimensions `time × species × weight`, export a long CSV before loading it in the viewer. The CSV should contain one row per time/species/weight combination:

```text
time, sp, w, N
0, Sprat, 0.001, 123.4
0, Sprat, 0.00111, 120.7
1, Sprat, 0.001, 118.9
```

The viewer normalises this internally to `t, species, w, x, N, log_N, log10_N`, with `x = log(w)` and `log10_N = log10(max(N, tiny))`. `.rds` and `.RData` files are intentionally not read; export the R object to CSV first to keep the app portable and read-only.
