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

## Interpolation notes

The app never assumes mizer and PINN grids match. For profile overlays it uses nearest available times or weights and states this in the plot description. For mizer-minus-PINN heatmaps and summaries, mizer data is placed onto the PINN fixed grid using nearest-time selection plus linear interpolation in log-weight `x`. The page shows a visible warning whenever this interpolation is used.

Run-to-run difference heatmaps are stricter: they are plotted only when `t_eval` and `x_eval` match exactly between the reference and comparison run. If grids differ, the app shows a message and does not plot the difference.

## Troubleshooting

- **No runs found**: check the run root path in the sidebar. The app scans immediate subdirectories only.
- **Missing plot**: the corresponding file may be absent, empty, or missing required columns/arrays. The plot area reports the missing inputs instead of raising a traceback.
- **Empty CSV**: empty CSV files are tolerated and reported as empty.
- **Log-axis warning**: non-positive values are replaced with `NaN` before log-scaled plots so Plotly does not display invalid values.
- **Mizer interpolation impossible**: ensure the mizer CSV has time, log-weight or weight, and abundance/log-abundance columns after alias normalisation.
