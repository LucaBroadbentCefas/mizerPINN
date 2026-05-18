# Data and Shape Conventions

## Core dimensions

| Symbol | Meaning |
|---|---|
| `n_time` | Number of sampled or diagnostic time values. |
| `n_eval` | Number of arbitrary off-grid PDE evaluation weights. |
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
| `_x_grid(params)` | `[n_w]` | `log(params.w)`. |

## Model input convention

The model takes scaled coordinates:

```text
input columns = [x_scaled, t_scaled]
```

For vectors:

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

## Model output convention

The active model output is interpreted as `log_N`.

For generic grid evaluation:

| Object | Shape |
|---|---:|
| `log_N_flat` | `[n_time * n_x, n_species]` |
| `log_N` | `[n_time, n_species, n_x]` |
| `N = exp(log_N)` | `[n_time, n_species, n_x]` |

For PDE collocation evaluation:

| Object | Shape |
|---|---:|
| `log_N_eval` | `[n_time, n_species, n_eval]` |
| `N_eval` | `[n_time, n_species, n_eval]` |
| `dlogN_dt` | `[n_time, n_species, n_eval]` |
| `dlogN_dw` | `[n_time, n_species, n_eval]` |
| `dN_dt` | `[n_time, n_species, n_eval]` |
| `dN_dw` | `[n_time, n_species, n_eval]` |

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

## Batch dictionary from `sample_pde_batch()`

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

The current PDE state evaluates the model both at:

1. arbitrary `w_eval` collocation points for residuals;
2. fixed `params.w` grid points for nonlocal biological computations.

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

When stacked over time by `_stack_dicts()`, these become:

```text
[n_time, n_species, n_eval]
```

## PDE residual outputs

| Quantity | Shape |
|---|---:|
| `residual_log` | `[n_time, n_species, n_eval]` |
| `residual` | `[n_time, n_species, n_eval]` |
| `residual_physical_check` | `[n_time, n_species, n_eval]` |

Current definitions:

```text
residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
residual = N_eval * residual_log
residual_physical_check = dN_dt + g_eval * dN_dw + (mu_eval + dg_dw) * N_eval
```

`residual` and `residual_physical_check` should be numerically consistent up to floating-point differences.

## Boundary-loss shapes

The recruitment boundary extracts species-specific left/egg positions from fixed-grid arrays.

| Quantity | Shape |
|---|---:|
| `N_left` | `[n_time, n_species]` |
| `g_left` | `[n_time, n_species]` |
| `flux_left = g_left * N_left` | `[n_time, n_species]` |
| `recruitment_flux` | `[n_time, n_species]` |
| `boundary_residual` | `[n_time, n_species]` or species-sliced equivalent |

## Current single-species training assumption

`validation_steps/train_pde_only_single_species.py` raises an error unless:

```text
n_species == 1
```

Multi-species support should not be assumed for the current training script even though many lower-level functions are shape-general.
