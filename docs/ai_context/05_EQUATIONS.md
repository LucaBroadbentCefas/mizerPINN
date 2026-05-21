# Equations and Conventions

## Target PDE

For species `i`:

```text
dN_i/dt + g_i(w,t) dN_i/dw + [mu_i(w,t) + dg_i/dw] N_i(w,t) = 0
```

Current implemented log-form:

```text
dlogN_i/dt + g_i(w,t) dlogN_i/dw + mu_i(w,t) + dg_i/dw = 0
```

The implementation computes both:

```text
residual_log = dlogN_dt + g_eval * dlogN_dw + mu_eval + dg_dw
residual = N_eval * residual_log
```

and a physical-form consistency check:

```text
residual_physical_check = dN_dt + g_eval * dN_dw + (mu_eval + dg_dw) * N_eval
```

## Coordinates

```text
x = log(w)
w = exp(x)
```

Scaling:

```text
x_scaled = (x - x_min) / (x_max - x_min)
t_scaled = (t - t_min) / (t_max - t_min)
```

Derivative conversion:

```text
dlogN_dx = dlogN_dx_scaled / (x_max - x_min)
dlogN_dt = dlogN_dt_scaled / (t_max - t_min)
dlogN_dw = dlogN_dx / w

dN_dt = N * dlogN_dt
dN_dw = N * dlogN_dw
```

## Model output

The active model outputs `log_N`.

```text
N = exp(log_N)
```

This is not just a numerical detail. Loss scaling, IC loss, timestep consistency, and collapse diagnostics depend on this convention.

## Prey construction

The fixed-grid and continuous encounter paths both rely on prey over the full grid. Conceptually:

```text
prey_full_i(w_p) = resource contribution + fish-prey contribution
```

The fish contribution uses the interaction matrix:

```text
fish_contribution_i(w_p) = sum_j interaction[i,j] * N_j(w_p)
```

The implementation multiplies by integration weights:

```text
prey = prey * (w_full * dw_full)
```

This convention matters for matching mizer/TMB exports.

## Continuous search / encounter prefactor

Current continuous convention:

```text
gamma_i(w) = gamma_i * w^{q_i}
```

Derivative:

```text
dgamma_i/dw = gamma_i * q_i * w^{q_i - 1}
```

## Continuous intake maximum

```text
h_i(w) = h_i * w^{n_i}
```

Derivative:

```text
dh_i/dw = h_i * n_i * w^{n_i - 1}
```

## Continuous metabolism

```text
metab_i(w) = ks_i * w^{p_i} + k_i * w
```

Derivative:

```text
dmetab_i/dw = ks_i * p_i * w^{p_i - 1} + k_i
```

## Predation kernel

Current continuous kernel implementation uses:

```text
ppmr = w_pred / w_prey
log_term = log(ppmr) - log(beta_i)
phi_raw = exp(-(log_term^2) / (2 sigma_i^2))
```

The active mask is currently:

```text
active = ppmr > 1
```

Then:

```text
phi = phi_raw if active else 0
```

Derivative with respect to predator weight:

```text
dphi/dw_pred = -phi * log_term / (sigma_i^2 * w_pred)
```

with zero derivative where inactive.

Important unresolved issue: earlier discussions included kernel truncation conventions. The current repository implementation only masks `ppmr > 1` in `compute_phi_and_dphi_dw()`. If further kernel truncation is required to match mizer exactly, that should be handled through a new decision record and validation run.

## Encounter

For predator species `i` at predator weight `w`:

```text
E_i(w,t) = gamma_i(w) * sum_p prey_full_i(w_p,t) * phi_i(w,w_p)
```

Current direct implementation uses a discrete sum over `params.w_full` after prey has already been multiplied by `w_full * dw_full`.

Derivative:

```text
dE_i/dw = dgamma_i/dw * conv_i(w,t) + gamma_i(w) * dconv_i/dw
```

where:

```text
conv_i(w,t) = sum_p prey_full_i(w_p,t) * phi_i(w,w_p)
dconv_i/dw = sum_p prey_full_i(w_p,t) * dphi_i/dw
```

## Feeding level

```text
f_i(w,t) = E_i(w,t) / [E_i(w,t) + h_i(w) + eps]
```

Derivative used in source:

```text
df_i/dw = [dE_i/dw * (h_i + eps) - E_i * dh_i/dw] / [E_i + h_i + eps]^2
```

If `eps = 0`, this reduces to the usual expression.

## Energy available for reproduction and growth

Current convention:

```text
erepog_i(w,t) = alpha_i * [1 - f_i(w,t)] * E_i(w,t) - metab_i(w)
```

Derivative:

```text
derepog_i/dw = alpha_i * ([1 - f_i] * dE_i/dw - E_i * df_i/dw) - dmetab_i/dw
```

## Positive part

```text
pos(x) = max(x, 0)
```

Derivative convention for the manual path:

```text
dpos(x)/dw = dx/dw if x > 0 else 0
```

The derivative is undefined at exactly zero. The current implementation uses the branch above.

## Reproduction allocation

Current source computes:

```text
psi_i(w) = maturity_i(w) * repro_prop_i(w)
```

with:

```text
A = (w / w_mat_i)^(-U_i)
maturity_raw = 1 / (1 + A)
dmaturity_dw_raw = U_i * maturity_raw * (1 - maturity_raw) / w
```

and:

```text
exponent = m_i - n_i
repro_prop = (w / w_repro_max_i)^exponent
drepro_prop_dw = exponent * repro_prop / w
```

Then:

```text
psi_raw = maturity_raw * repro_prop
dpsi_dw_raw = dmaturity_dw_raw * repro_prop + maturity_raw * drepro_prop_dw
```

Piecewise conventions:

```text
if maturity_raw < maturity_floor:
    psi = 0
    dpsi_dw = 0

if w >= w_repro_max:
    psi = 1
    dpsi_dw = 0
```

## Reproduction and growth split

```text
pos_erepog = pos(erepog)
e_repro = pos_erepog * psi
e_growth = pos_erepog - e_repro
```

Derivative:

```text
de_repro/dw = dpos_erepog/dw * psi + pos_erepog * dpsi/dw
```

Growth derivative:

```text
dg/dw = dpos_erepog/dw - de_repro/dw
```

Equivalently:

```text
dg/dw = (1 - psi) * dpos_erepog/dw - pos_erepog * dpsi/dw
```

## Background mortality

Current source supports two modes.

If `params.mu_b_allometric` is false:

```text
mu_b_i(w) = z0_i
```

If `params.mu_b_allometric` is true:

```text
z0_i = z0_pre_i * w_inf_i^(1 - n_i)
mu_b_i(w) = z0_i * w^(n_i - 1)
```

## Direct predation mortality

For prey evaluation weight `w_prey`, current direct implementation computes a predator-rate-like quantity:

```text
q_j(w_pred,t) = [1 - f_j(w_pred,t)] * gamma_j(w_pred) * N_j(w_pred,t)
```

Then integrates over the fixed predator grid:

```text
pred_rate_j(w_prey,t) = sum_over_w_pred phi_j(w_pred,w_prey) * q_j(w_pred,t) * dw_pred
```

Species interaction then maps predator rates to prey mortality:

```text
pred_mort_eval = interaction^T @ pred_rate
```

Total mortality:

```text
mu_eval = mu_b_eval + pred_mort_eval
```

The current continuous total mortality function used by PDE residual ignores fishing mortality.

## Recruitment flux used in boundary loss

Current direct recruitment from growth grid:

```text
repro_integrand = e_repro_grid * N_grid
repro_integral = trapz(repro_integrand, x=params.w, dim=1)
egg_w = params.w[params.w_min_idx - 1]
rdi_flux = 0.5 * repro_integral * erepro / egg_w
rdd_flux = rdi_flux / (1 + rdi_flux / r_max)
```

Boundary condition:

```text
g_i(w_min_i,t) * N_i(w_min_i,t) = RDD_i(t)
```

The implemented left flux is:

```text
flux_left = g_left * N_left
```

Boundary residual forms:

```text
physical: flux_left - recruitment_flux
log:      log(clamp(flux_left, eps)) - log(clamp(recruitment_flux, eps))
relative: (flux_left - recruitment_flux) / clamp(abs(recruitment_flux), eps)
```

## Timestep-consistency loss

This is not the continuous PDE residual. It is a fixed-grid temporal consistency loss.

For selected physical times `t0` and timestep `dt`:

```text
N0_pred = N_theta(w_grid, t0)
N1_pred = N_theta(w_grid, t0 + dt)
N1_step = step(n_pp, N0_pred, params, dt)
```

The three main model/step tensors use this shape convention:

```text
N0_pred: [n_pairs, n_species_or_1, n_w]
N1_pred: [n_pairs, n_species_or_1, n_w]
N1_step: [n_pairs, n_species_or_1, n_w]
```

Residual forms:

```text
physical:
    r_ts = N1_pred - N1_step

log:
    r_ts = log(clamp(N1_pred, eps)) - log(clamp(N1_step, eps))

relative:
    r_ts = (N1_pred - N1_step) / clamp(abs(N1_step), relative_eps)
```

Loss:

```text
loss_timestep = mean(r_ts^2)
```

Detach convention:

```text
if detach_step_target = True:
    step_target_input = detach(N0_pred)
else:
    step_target_input = N0_pred
```

If `detach_step_target=True`, `N0_pred` is detached before calling `step(...)`. Then the timestep loss mainly trains `N_theta(w,t+dt)` toward a fixed one-step target. If false, gradients can flow through the step target as well. This changes the optimisation problem and should be compared explicitly before interpreting training behaviour.

Timestep source convention:

```text
if --timestep-dt is supplied:
    dt = --timestep-dt
else:
    dt = params.dt
```

`params.dt` should represent the mizer timestep used by the fixture/export. `--timestep-dt` is an explicit experimental override.

## Wang-style gradient-statistic weighting

Current training uses PDE as the anchor:

```text
w_pde = 1
```

For every non-PDE loss component currently present in `raw_losses`, including IC, BC, and timestep:

```text
target_weight_component = max_abs_grad(loss_pde) / mean_abs_grad(loss_component)
```

Then clipped to `[weight_min, weight_max]` and updated either by hard set or exponential smoothing:

```text
w_component = (1 - alpha) * w_component + alpha * target_weight_component
```

This is an optimisation heuristic, not a biological equation. Because timestep is now a possible non-PDE component, adding a future loss to `raw_losses` also adds it to the adaptive weighting system unless deliberately excluded.

## Causal time curriculum

Current training supports:

```text
t_upper = t_min + fraction(step) * (t_max - t_min)
```

The PDE sampler draws times from:

```text
Uniform(t_min, t_upper)
```

Modes:

- `off`: always full time domain;
- `linear`: fraction ramps linearly from start fraction to 1;
- `step`: fraction follows supplied schedule.
