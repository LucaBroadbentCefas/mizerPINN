# Data and Shape Conventions

## Core dimensions

| Symbol | Meaning |
|---|---|
| `n_time` | Number of sampled/diagnostic time values in Cartesian batches; also current slab count `K` in slabbed R3. |
| `n_eval` | Number of arbitrary off-grid PDE evaluation weights in Cartesian batches; also default per-slab count `M` when `--r3-population-size` is omitted. |
| `K` | Number of R3 time slabs. Current implementation uses `K = n_time`. |
| `M` | Number of R3 x/log-weight points per time slab. |
| `n_pair` | Historical flat-paired R3 point count; current effective slabbed population is `K * M`. |
| `n_species` | Number of fish species. Current training script expects one species. |
| `n_w` | Number of fish-grid size classes. |
| `k_full` | Number of full grid points, including resource and fish grid points. |
| `n_x` | Generic number of x/log-weight points used for model evaluation. |

## Core grids

| Object | Shape | Meaning |
|---|---:|---|
| `params.w_full` | `[k_full]` | Full resource + fish physical weight grid. |
| `params.w` | `[n_w]` | Fish physical weight grid. |
| `params.dw_full` | `[k_full]` | Full-grid integration widths. |
| `params.dw` | `[n_w]` | Fish-grid integration widths. |
| `params.w_min_idx` | `[n_species]` | R/TMB exported 1-based species egg/min index. |
| `params.w_max` | `[n_species]` or scalar | Species active maximum body mass for masking/sampling. |
| `_x_grid(params)` | `[n_w]` | `log(params.w)`. |

## Model input convention

The model takes scaled coordinates:

```text
input columns = [x_scaled, t_scaled]
```

For Cartesian vectors:

```text
x_scaled: [n_x]
t_scaled: [n_time]
```

`_make_model_inputs(x_scaled, t_scaled)` returns:

```text
inputs: [n_time * n_x, 2]
```

The ordering is time-major Cartesian product:

```text
for each time:
    for each x:
        [x_scaled, t_scaled]
```

This ordering matters because outputs are reshaped as:

```text
raw output: [n_time * n_x, n_species]
reshape:    [n_time, n_x, n_species]
permute:    [n_time, n_species, n_x]
```

For current slabbed R3 inputs:

```text
t_slab:        [K]
x_slab:        [K, M]
w_slab:        [K, M]
t_slab_scaled: [K]
x_slab_scaled: [K, M]
```

Each model input row corresponds to a specific slab cell `(k,m)`:

```text
row(k,m) = [x_slab_scaled[k,m], t_slab_scaled[k]]
```

## Model output convention

The active model output is interpreted as `log_N`.

For generic Cartesian grid evaluation:

| Object | Shape |
|---|---:|
| `log_N_flat` | `[n_time * n_x, n_species]` |
| `log_N` | `[n_time, n_species, n_x]` |
| `N = exp(log_N)` | `[n_time, n_species, n_x]` |

For Cartesian PDE collocation evaluation:

| Object | Shape |
|---|---:|
| `log_N_eval` | `[n_time, n_species, n_eval]` |
| `N_eval` | `[n_time, n_species, n_eval]` |
| `dlogN_dt` | `[n_time, n_species, n_eval]` |
| `dlogN_dw` | `[n_time, n_species, n_eval]` |
| `dN_dt` | `[n_time, n_species, n_eval]` |
| `dN_dw` | `[n_time, n_species, n_eval]` |

For historical flat-paired R3 PDE collocation evaluation:

| Object | Shape |
|---|---:|
| `log_N_eval` | `[n_species, n_pair]` |
| `N_eval` | `[n_species, n_pair]` |
| `dlogN_dt` | `[n_species, n_pair]` |
| `dlogN_dw` | `[n_species, n_pair]` |
| `dN_dt` | `[n_species, n_pair]` |
| `dN_dw` | `[n_species, n_pair]` |

For current slabbed R3 PDE collocation evaluation:

| Object | Shape |
|---|---:|
| `log_N_eval` | `[K, n_species, M]` |
| `N_eval` | `[K, n_species, M]` |
| `dlogN_dt` | `[K, n_species, M]` |
| `dlogN_dw` | `[K, n_species, M]` |
| `dN_dt` | `[K, n_species, M]` |
| `dN_dw` | `[K, n_species, M]` |

## Coordinate and derivative scaling

Definitions:

```text
x = log(w)
w = exp(x)
x_scaled = (x - x_min) / (x_max - x_min)
t_scaled = (t - t_min) / (t_max - t_min)
```

Autograd gives:

```text
dlogN_dx_scaled
dlogN_dt_scaled
```

The PDE code converts to physical derivatives:

```text
dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
dlogN_dw = dlogN_dx / w

dN_dt = N * dlogN_dt
dN_dw = N * dlogN_dw
```

The derivative conversion is identical for Cartesian, flat-paired, and slabbed R3 paths. Only tensor geometry differs.

## Batch dictionary from `sample_pde_batch()`

This is the current uniform Cartesian PDE batch.

| Key | Shape | Meaning |
|---|---:|---|
| `t_eval` | `[n_time]` | Physical sampled times. |
| `t_scaled` | `[n_time]` | Scaled sampled times. |
| `x_eval` | `[n_eval]` | Physical log-weight collocation values. |
| `x_eval_scaled` | `[n_eval]` | Scaled log-weight collocation values. |
| `w_eval` | `[n_eval]` | Physical collocation weights, `exp(x_eval)`. |
| `x_grid` | `[n_w]` | Fixed fish-grid log weights. |
| `x_grid_scaled` | `[n_w]` | Scaled fixed fish-grid log weights. |
| `w_grid` | `[n_w]` | Fixed fish physical weights. |

The current Cartesian PDE state evaluates the model both at:

1. arbitrary `w_eval` collocation points for residuals;
2. fixed `params.w` grid points for nonlocal biological computations.

The Cartesian residual shape is:

```text
[n_time, n_species, n_eval]
```

## Current slabbed R3 batch dictionary

Current R3/Causal R3 uses time slabs and per-slab x points.

`R3Population`:

| Object | Shape | Meaning |
|---|---:|---|
| `t_points` | `[K]` | Physical time slabs. Resampled each step under `t_max_current`. |
| `x_points` | `[K, M]` | Physical log body mass values. Retained/resampled by R3. |
| `population_size` | scalar | `K * M`. |

Slabbed batch from `R3Population.as_batch()`:

| Key | Shape | Meaning |
|---|---:|---|
| `t_slab` | `[K]` | Physical slab times. |
| `t_slab_scaled` | `[K]` | Scaled slab times. |
| `x_slab` | `[K, M]` | Physical log-weight collocation values. |
| `x_slab_scaled` | `[K, M]` | Scaled log-weight collocation values. |
| `w_slab` | `[K, M]` | Physical weights, `exp(x_slab)`. |
| `x_grid` | `[n_w]` | Fixed fish-grid log weights. |
| `x_grid_scaled` | `[n_w]` | Scaled fixed fish-grid log weights. |
| `w_grid` | `[n_w]` | Fixed fish physical weights. |

Slabbed residual shape:

```text
[K, n_species, M]
```

Slabbed fixed-grid model state for nonlocal biology:

```text
N_grid:     [K, n_species, n_w]
log_N_grid: [K, n_species, n_w]
```

Slabbed continuous biological outputs at residual points:

```text
growth_eval terms: [K, n_species, M]
mortality terms:   [K, n_species, M]
```

Slabbed fixed-grid biological outputs for recruitment/nonlocal terms:

```text
growth_grid terms: [K, n_species, n_w]
recruitment terms: [K, n_species]
```

Historical flat-paired R3 used:

```text
x_pair, t_pair, w_pair: [n_pair]
residual: [n_species, n_pair]
N_grid/log_N_grid: [n_pair, n_species, n_w]
```

Keep that only as historical context unless current source reintroduces a flat-paired path.

## Active-size masks

Grid mask:

```text
active_grid_mask(params): [n_species, n_w]
```

Evaluation masks:

```text
active_eval_mask(w_eval, params): [n_species, n_eval]
active_eval_mask(w_pair, params): [n_species, n_pair]
active_eval_mask(w_slab.reshape(-1), params): [n_species, K*M]
```

For slabbed R3, the flattened active mask is reshaped/permuted to:

```text
[K, n_species, M]
```

The active mask is based on:

```text
w <= w_max_i
```

Sampling inside `w_max` reduces wasted points, but loss masking remains the required safety condition. Do not rely only on sampling bounds.

## `MizerTorchParams` shapes

### FFT/reference-grid quantities

| Object | Shape |
|---|---:|
| `ft_pred_kernel_e` | `[n_species, k_full]`, complex |
| `ft_pred_kernel_p` | `[n_species, k_full]`, complex |
| `ft_mask` | `[n_species, k_full]` |
| `search_vol` | `[n_species, n_w]` |
| `intake_max` | `[n_species, n_w]` |
| `metab` | `[n_species, n_w]` |
| `psi` | `[n_species, n_w]` |
| `mu_b` | `[n_species, n_w]` |

### Species vectors

| Object | Shape |
|---|---:|
| `alpha` | `[n_species]` |
| `gamma` | `[n_species]` |
| `q` | `[n_species]` |
| `h` | `[n_species]` |
| `n_exp` | `[n_species]` |
| `ks` | `[n_species]` |
| `p_exp` | `[n_species]` |
| `k_metab` | `[n_species]` |
| `beta` | `[n_species]` |
| `sigma` | `[n_species]` |
| `w_max` | `[n_species]` |
| `w_mat` | `[n_species]` |
| `U` | `[n_species]` |
| `w_repro_max` | `[n_species]` |
| `m_exp` | `[n_species]` |
| `z0_pre` | scalar or `[n_species]` |
| `z0` | scalar or `[n_species]` |
| `w_inf` | `[n_species]` |

### Interaction/resource/reproduction quantities

| Object | Shape |
|---|---:|
| `interaction_resource` | `[n_species]` |
| `interaction` | `[n_species, n_species]` |
| `erepro` | `[n_species]` |
| `r_max` | `[n_species]` |
| `rr_pp` | `[k_full]` |
| `cc_pp` | `[k_full]` |
| `f_mort` | optional `[n_species, n_w]` |

## Continuous biological outputs

For any `w_eval: [n_eval]`, continuous biological functions return species-by-evaluation arrays.

| Quantity | Shape |
|---|---:|
| `gamma_eval` | `[n_species, n_eval]` |
| `dgamma_dw` | `[n_species, n_eval]` |
| `encounter_eval` | `[n_species, n_eval]` |
| `dencounter_dw` | `[n_species, n_eval]` |
| `h_eval` | `[n_species, n_eval]` |
| `dh_dw` | `[n_species, n_eval]` |
| `feeding_eval` | `[n_species, n_eval]` |
| `dfeeding_dw` | `[n_species, n_eval]` |
| `metab_eval` | `[n_species, n_eval]` |
| `dmetab_dw` | `[n_species, n_eval]` |
| `erepog_eval` | `[n_species, n_eval]` |
| `derepog_dw` | `[n_species, n_eval]` |
| `psi_eval` | `[n_species, n_eval]` |
| `dpsi_dw` | `[n_species, n_eval]` |
| `e_repro_eval` | `[n_species, n_eval]` |
| `e_growth_eval` | `[n_species, n_eval]` |
| `dg_dw` | `[n_species, n_eval]` |
| `mu_b_eval` | `[n_species, n_eval]` |
| `pred_mort_eval` | `[n_species, n_eval]` |
| `mu_eval` | `[n_species, n_eval]` |

When stacked over Cartesian time by `_stack_dicts()`, these become:

```text
[n_time, n_species, n_eval]
```

For slabbed R3, current downstream outputs become:

```text
[K, n_species, M]
```

## PDE residual outputs

Cartesian outputs:

| Quantity | Shape |
|---|---:|
| `residual_log` | `[n_time, n_species, n_eval]` |
| `residual` | `[n_time, n_species, n_eval]` |
| `residual_physical_check` | `[n_time, n_species, n_eval]` |

Historical flat-paired outputs:

| Quantity | Shape |
|---|---:|
| `residual_log` | `[n_species, n_pair]` |
| `residual` | `[n_species, n_pair]` |
| `residual_physical_check` | `[n_species, n_pair]` |

Current slabbed R3 outputs:

| Quantity | Shape |
|---|---:|
| `residual_log` | `[K, n_species, M]` |
| `residual` | `[K, n_species, M]` |
| `residual_physical_check` | `[K, n_species, M]` |

Current definitions:

```text
residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
residual = N_eval * residual_log
residual_physical_check = dN_dt + g_eval * dN_dw + (mu_eval + dg_dw) * N_eval
```

`residual` and `residual_physical_check` should be numerically consistent up to floating-point differences.

## Boundary-loss shapes

The recruitment boundary extracts species-specific left/egg positions from fixed-grid arrays.

Cartesian path:

| Quantity | Shape |
|---|---:|
| `log_N_left` | `[n_time, n_species]` |
| `N_left` | `[n_time, n_species]` |
| `g_left` | `[n_time, n_species]` |
| `flux_left = g_left * N_left` | `[n_time, n_species]` |
| `recruitment_flux` | `[n_time, n_species]` |
| `bc_valid_mask` | `[n_time, n_species]` or species-sliced equivalent |
| `bc_target_log_N` | `[n_time, n_species]` or species-sliced equivalent |
| `bc_target_N` | `[n_time, n_species]` or species-sliced equivalent |
| `boundary_residual` | `[n_time, n_species]` or species-sliced equivalent |

Historical flat-paired path:

| Quantity | Shape |
|---|---:|
| `N_left` | `[n_pair, n_species]` |
| `g_left` | `[n_pair, n_species]` |
| `flux_left = g_left * N_left` | `[n_pair, n_species]` |
| `recruitment_flux` | `[n_pair, n_species]` |
| `boundary_residual` | `[n_pair, n_species]` or species-sliced equivalent |

Current slabbed R3 path uses one boundary/recruitment state per slab time:

| Quantity | Shape |
|---|---:|
| `log_N_left` | `[K, n_species]` |
| `N_left` | `[K, n_species]` |
| `g_left` | `[K, n_species]` |
| `flux_left = g_left * N_left` | `[K, n_species]` |
| `recruitment_flux` | `[K, n_species]` |
| `bc_valid_mask` | `[K, n_species]` or species-sliced equivalent |
| `bc_target_log_N` | `[K, n_species]` or species-sliced equivalent |
| `bc_target_N` | `[K, n_species]` or species-sliced equivalent |
| `boundary_residual` | `[K, n_species]` or species-sliced equivalent |

Current BC validity rule:

```text
valid = finite(log_N_left, N_left, g_left, recruitment_flux)
        and g_left > bc_g_min
        and recruitment_flux > 0
```

## R3 score shapes

For current slabbed residuals:

```text
residual: [K, n_species, M]
mask:     [K, n_species, M]
score:    [K, M]
```

Default absolute score:

```text
score_km = sum_i mask_kim * abs(residual_kim) / max(sum_i mask_kim, 1)
```

Squared score:

```text
score_km = sum_i mask_kim * residual_kim^2 / max(sum_i mask_kim, 1)
```

Retain rule:

```text
retain_km = score_km > mean(score)
```

Only `x_points[retain]` are kept across the R3 update. Time slabs are resampled every step before the batch is formed.

## Current single-species training assumption

`PINNmizer/training/train_pde_only_single_species.py` raises an error unless:

```text
n_species == 1
```

Multi-species support should not be assumed for the current training script even though many lower-level functions are shape-general.
