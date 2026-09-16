# Final-suite viewer: independent final audit

Audit date: 2026-09-16. Scope: the read-only viewer, the 70-task catalogue,
`final_model_suite.slurm`, and the current PINNmizer diagnostic, observation,
loss, and inverse-output implementations.

## Overall assessment

The viewer is technically fit for navigating the fixed suite and for inspecting
saved diagnostics. It is scientifically fit **only after an explicit canonical
full mizer state export is supplied**. No such export and no completed labelled
final-suite run are repository-tracked, so state-recovery results and end-to-end
real-output behaviour are blocked in this checkout. The viewer correctly refuses
to substitute task 13, observation `value_true`, or an initial state as latent
truth. Diagnostics missing from saved output are reported as unavailable and are
not reconstructed from checkpoints.

## Evidence and source-of-truth checks

* Tasks 0–69 were compared with `scripts/final_model_suite.slurm`: the 3×4
  single-species matrix, baselines, 25 paired gate-ON runs, five unreplicated
  gate-OFF controls, six missing-data designs (with no 20–30 task), three NN
  controls, and all inverse starts match. Noise seeds 41001–41005 are reused by
  replicate across CV.
* `PINNmizer/pinn/derivatives.py` converts derivatives from scaled network
  coordinates to physical model time and physical weight before returning them.
  `PINNmizer/pinn/residual.py` then forms
  `R_log = dlogN_dt + g*dlogN_dw + mu + dg_dw`. Fixed-field writers save these
  returned physical terms. The viewer only reads and sums those columns.
* `PINNmizer/pinn/observation_operators.py` implements abundance/biomass sums,
  survey catchability, and catch with `F=E*q*s`. Final-suite runs request three
  equally spaced left-closed interval points. The catch is their mean rate times
  interval duration.
* `PINNmizer/pinn/data_losses.py` implements the epsilon-protected log residual,
  `0.5*z^2 + log(sd_log)`, and `Q=sum(z^2)`, with an active gate when disabled or
  when `Q > chi2_0.95(n)`. The viewer consumes saved uncertainty and saved gate
  history.
* `estimated_rmax.csv`, `rmax_history.csv`, `data_cv_history.csv`,
  `estimated_data_cv.csv`, and `estimated_effort.csv` are genuine writers in the
  training code. There is no effort-history CSV. Rmax truth must be
  `r_max_true.csv`; the deliberately perturbed `r_max.csv` is never accepted as
  truth. Effort fishing mortality uses the same `[gear,species,weight]` reshape
  and multiplication as the model.

## Issues found

| Severity | Issue | Consequence | Fix/status |
|---|---|---|---|
| High | Rmax truth lookup could fall back to perturbed `r_max.csv`. | Recovery could be scored against its start rather than truth. | **FIXED**: require `r_max_true.csv`. |
| High | No canonical full time×species×weight truth is present. | N3–N9 and truth-dependent experiment metrics cannot be scientifically evaluated here. | **BLOCKED**: provide the canonical long-form export; no surrogate is used. |
| Medium | Species names such as `sp_7` were appearance-renumbered when `species_idx` was absent. | Single-species truth could fail to join task predictions or join the wrong species. | **FIXED**: preserve numeric species identity and independently test interpolation/non-extrapolation. |
| Medium | A duplicate-folder choice made in Technical details was lost after navigating away. | A user could inspect a different run instance than selected. | **FIXED**: retain and reapply the session choice. |
| Medium | No labelled completed final-suite outputs are present locally. | Family-by-family real-run smoke and numerical spot checks cannot be claimed. | **BLOCKED** in this checkout; controlled tests cover schemas/calculations. |
| Low | Per-term weighting-gradient columns may not exist in old HPC histories. | T5 cannot be shown for those runs. | **PASS** graceful unavailable message; future runs must save the columns if required. |
| Low | Effort optimisation history is not written. | A trajectory plot cannot be produced. | **NOT APPLICABLE** by design; final matrices support F1–F5. |

## Requirement matrix

“PASS” means the implementation and its source contract were inspected and a
controlled test or static source comparison was available. “BLOCKED” means the
viewer behaves safely but scientific validation needs absent external output.

| ID | Status | Implementation | Source data | Validation / caveat |
|---|---|---|---|---|
| S1 | PASS | `streamlit_app.py` | catalogue, manifests, label/config metadata | 70 unique tasks; missing/error/duplicate states visible; scientific map. |
| N3/N4 | BLOCKED | `pages_state_pde.py`, `state.py` | prediction grid + canonical truth | Sign, absolute mode, symmetric scale and masks inspected; real truth absent. |
| N5 | BLOCKED | `analytics.error_by_time` | aligned state | Independent expected-value test; real truth absent. |
| N6 | BLOCKED | `analytics.error_by_weight` | aligned state | Independent expected-value test; real truth absent. |
| N7 | BLOCKED | `error_by_species` | aligned active cells | Active-mask controlled test; real truth absent. |
| N8 | BLOCKED | `fold_by_species`, `fold_error` | aligned active cells | Exact factor-ten controlled test; real truth absent. |
| N9 | BLOCKED | State page profile modes | aligned state(s) | Both orientations and common-domain comparison inspected; real truth absent. |
| P1/P2 | PASS | PDE page | `fixed_grid_fields.csv: residual_log` | Signed/absolute, selected physical saved cells only. |
| P3 | PASS | `residual_summary` | saved residual | All requested summaries and finite filtering tested. |
| P4/P5 | PASS | `residual_aggregate` | saved residual | Mean/p95/max grouping tested. |
| P6 | PASS | PDE component selector | saved component columns | Only present columns offered; no reconstruction. |
| P7 | PASS | `pde_balance` | four saved terms + residual | Exact and inconsistent controlled cases tested; real suite spot check blocked. |
| D1 | PASS | Data page | `data_predictions_final.csv` | Observation prediction explicitly distinguished from state. |
| D2 | PASS | Data page | saved `value_true/value/prediction` | Observation truth explicitly not state truth. |
| D3 | PASS | `denoising_metrics` | saved observation values | Independent log-error test. |
| D4 | PASS | `prepare_observations` | saved log residual / SD | Standardisation checked numerically. |
| D5 | PASS | `normal_quantiles` | standardised residuals | Ordered marginal Normal QQ; limitation stated. |
| D6 | PASS | Data page | saved intervals/species/type/dataset | Midpoint display retains endpoints in hover. |
| D7 | PASS | `rank_misfits` | `0.5*z^2` | Ranking test proves scale term excluded. |
| D8 | PASS | Data/training page | saved Q, q95, active/effective loss | Gate equations compared with training source; controlled test. |
| T1 | PASS | Training page | `loss_history.csv: loss` | Saved objective only. |
| T2 | PASS | Training page | saved raw loss columns | Availability-gated. |
| T3 | PASS | Training page | saved objective/weighted columns | No reconstruction. |
| T4 | PASS | Training page | saved adaptive weights | Availability-gated. |
| T5 | PASS | Training page | saved per-term gradient norms | Clear unavailable state when not retained. |
| T6 | PASS | Training page | `grad_norm` | Optimisation-step axis. |
| T7 | PASS | Training page | saved LR columns | One group at a time; optimisation-step axis. |
| T8 | PASS | Training page | causal fraction/frontier | Fraction and model time are distinguished. |
| T9 | PASS | Training page | saved causal summaries | No fabricated chunk detail. |
| T10 | PASS | Training page | `fixed_diagnostic_history.csv` | Fixed grid distinguished from collocation. |
| T11 | PASS | Training page | saved PDE-term RMS | Physical saved terms. |
| T12 | PASS | `interval_seconds_per_step` | elapsed seconds / timing summary | Difference calculation independently tested. |
| E1 | BLOCKED | Experiment page | tasks 14–43 + run metrics | Pairing/SD logic tested; real run metrics absent. |
| E2 | BLOCKED | Experiment page | five gate-ON values/CV | Raw points + sample SD, no density; outputs absent. |
| E3 | BLOCKED | `paired_cv_differences` | matched seeds | Sign and matched-key arithmetic tested; outputs absent. |
| Missing interactive | BLOCKED | Experiment page | tasks 44–49 + aligned state | Gap/species/every-third-year modes present; real truth absent. |
| Missing seen/withheld | BLOCKED | `missing_seen_metrics` | aligned errors + actual observation coverage | Masks/math tested; real outputs absent. |
| NN1/NN2 | BLOCKED | Experiment page | 13/50, 24/51, 46/52 | Pairings verified; truth/run outputs absent. |
| A1 | BLOCKED | Experiment page | tasks 0–11 | Matrix mapping verified; run metrics absent. |
| A2 | BLOCKED | Experiment page | tasks 12/13 | Pair verified; state truth/run outputs absent. |
| R1 | BLOCKED | Inverse page | estimated Rmax + `r_max_true.csv` | Truth contract fixed; outputs absent. |
| R2 | BLOCKED | Inverse page | `rmax_history.csv` | Optimisation step explicit; outputs absent. |
| R3 | BLOCKED | Inverse page | estimated/true Rmax | Ratio math tested; outputs absent. |
| R4 | BLOCKED | `rmse_log_ratio` | tasks 53–56 | Formula independently tested; outputs absent. |
| R5 | BLOCKED | Inverse page | tasks 57–61 | Five points + sample SD; outputs absent. |
| C1 | BLOCKED | Inverse page | `data_cv_history.csv` | CV and sigma_log separated; outputs absent. |
| C2 | BLOCKED | Inverse page | `estimated_data_cv.csv` | Known truth 0.3; outputs absent. |
| F1 | BLOCKED | Inverse page | `estimated_effort.csv` | Positive ratios only; outputs absent. |
| F2 | BLOCKED | Inverse page | final effort matrix | Physical model-time axis; outputs absent. |
| F3 | BLOCKED | `effort_errors` | final effort matrix | Log error/zero absolute error independently tested. |
| F4 | BLOCKED | `effort_recovery_summary` | tasks 66–69 | RMSE logE and zero group independently tested. |
| F5 | BLOCKED | `fishing_mortality` | effort + q/selectivity/w inputs | Exact indexing arithmetic tested against source; outputs absent. |

## Runtime and UI validation

The app imports and starts without loading PyTorch. Discovery, CSV/JSON reads,
truth loading, alignment, metric tables, and fishing inputs are cached with
`st.cache_data`. Interactive controls operate on cached dataframes; no viewer
path loads checkpoints, invokes a PINN forward pass, autograd, or a PDE call.
A headless Streamlit health check is the available UI smoke test in this
non-interactive environment. Plot interaction with genuine final-suite folders
remains blocked because none are present.

## Blocked items and future-output requirements

1. **Canonical latent-state truth:** export long-form `time, species_idx/species,
   w/x, N/log_N/log10_N` from the mizer simulations. This enables all state and
   state-dependent cross-run analyses; it does not require retraining.
2. **Real final-suite run directories:** copy labelled run outputs/manifests to a
   configured run root to complete family-by-family smoke checks.
3. **T5 on old runs:** if per-term weighting-gradient columns were not retained,
   only a future training run with those diagnostics enabled can provide them.
   The viewer intentionally does not reconstruct them.
4. **Effort history:** no history file is currently written. F1–F5 use final
   matrices and remain valid; an optimisation trajectory would require a future
   output-format change and is not a viewer fix.

## Parking lot

* Perform a separate, data-bearing acceptance run after the canonical truth and
  HPC output bundle are transferred. Record one independent real-cell check for
  state error, PDE balance, observation residual/Q, each paired experiment, and
  each inverse family.
* Consider a model-side output schema/version manifest for future suites. This
  is optional reproducibility work, not a defect in the read-only viewer.
