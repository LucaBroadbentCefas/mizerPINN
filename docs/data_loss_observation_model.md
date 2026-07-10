# Data-loss observation model

This tranche supports lognormal observation likelihoods for long-format CSV data in physical time and physical weight units.

## Supported observation types

- `biomass`: species-specific biomass over a size range.
- `survey_biomass`: biomass index with fixed CSV catchability `q` (default 1) and flat selectivity on `[w_min, w_max]`.
- `survey_abundance`: abundance index with fixed CSV catchability `q` (default 1) and flat selectivity on `[w_min, w_max]`.
- `catch_total`: total fisheries catch/yield summed over gears.
- `catch_gear`: gear-specific catch/yield using exported gear-level effort, catchability, and selectivity.

## CSV schema

Required columns are `obs_type`, `species_idx`, `t_start`, and `value`. Optional columns are `dataset`, `gear_idx`, `t_end`, `w_min`, `w_max`, `cv`, `sd_log`, `unit`, `include`, and `q`.

`species_idx` is zero-based. `gear_idx` is required for `catch_gear`. `t_end` defaults to `t_start`; `w_min` and `w_max` default to the model weight-grid bounds; `include` defaults to true; `q` defaults to 1.

## Observation equations

Biomass for species `i` is

```text
B_i(t; w_a, w_b) = integral_{w_a}^{w_b} N_i(w,t) w dw.
```

Survey biomass and abundance are

```text
I_{d,i}(t)   = q_d integral S_d(w) N_i(w,t) w dw
I^N_{d,i}(t) = q_d integral S_d(w) N_i(w,t) dw
```

with flat selectivity inside the CSV weight range in this tranche.

Gear catch rate and total catch rate are

```text
Y_{g,i}(t) = integral F_{g,i}(w,t) N_i(w,t) w dw
Y_i(t)     = sum_g Y_{g,i}(t)
F_{g,i}(w,t) = E_g(t) q_{g,i} S_{g,i}(w).
```

For intervals, catch is approximated by simple quadrature over the available observation time grid and multiplied by interval length.

## Lognormal likelihood

The retained per-observation negative log likelihood omits only the additive Normal constant:

```text
ell_j = 0.5 * ((log(y_j + eps) - log(yhat_j + eps)) / sd_log_j)^2 + log(sd_log_j).
```

If `sd_log` is missing and `cv` is present, `sd_log = sqrt(log(1 + cv^2))`. Otherwise the CLI default CV is used.

## Current limitations

- No learned catchability.
- No learned selectivity.
- No gamma likelihood.
- No composition likelihood.
- Gear-specific catch requires gear-level fishing inputs (`catchability`, `selectivity`, and effort).
