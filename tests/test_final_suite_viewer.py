from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from apps.final_suite_viewer.catalogue import BY_TASK_ID, build_catalogue
from apps.final_suite_viewer.analytics import (
    available_columns, common_comparison_domain, cv_to_sigma,
    denoising_metrics, discrepancy_action, discrepancy_q, error_by_species,
    error_by_time, error_by_weight, interval_seconds_per_step, mask_domain,
    pde_balance, prepare_observations, rank_misfits, residual_aggregate,
    state_error_table,
)
from apps.final_suite_viewer.discovery import RunInstance, discover_local_runs, match_catalogue
from apps.final_suite_viewer.metrics import aggregate_state_metrics, fold_error, state_rmse
from apps.final_suite_viewer.state import align_states, normalise_state
from apps.final_suite_viewer.experiment_analysis import (
    NN_SCENARIOS, ablation_task_matrix, gap_mask, missing_seen_metrics,
    missing_species_mask, noise_design, noise_summary, paired_cv_differences,
    retained_omitted_years, validate_baseline_pair, validate_nn_pairings,
    year_location_mask,
)
from apps.final_suite_viewer.loaders import load_fixed_fields, load_prediction_state
from apps.final_suite_viewer.inverse_analysis import (
    effort_errors, effort_recovery_summary, fishing_mortality,
    reshape_selectivity, rmax_recovery, rmse_log_ratio,
)
from apps.final_suite_viewer.pages_inverse import _rmax_truth


def _state(rows):
    return pd.DataFrame(rows, columns=["time", "species_idx", "species", "w", "N"])


def test_catalogue_has_exactly_tasks_zero_to_69():
    catalogue = build_catalogue()
    assert len(catalogue) == 70
    assert [task.task_id for task in catalogue] == list(range(70))
    assert len({task.run_label for task in catalogue}) == 70


def test_noise_task_cv_rep_seed_and_gate_mapping():
    for task_id in range(14, 39):
        task = BY_TASK_ID[task_id]
        offset = task_id - 14
        assert task.cv == pytest.approx(0.1 + 0.1 * (offset // 5))
        assert task.replicate == offset % 5 + 1
        assert task.noise_seed == 41000 + task.replicate
        assert task.gate is True
    for task_id in range(39, 44):
        task = BY_TASK_ID[task_id]
        assert task.cv == pytest.approx(0.1 + 0.1 * (task_id - 39))
        assert (task.replicate, task.noise_seed, task.gate) == (1, 41001, False)


def test_single_species_ablation_mapping():
    expected = {
        3: ("sp_3", "log-u", "mlp"), 4: ("sp_3", "log-n", "fourier"), 5: ("sp_3", "log-n", "mlp"),
        6: ("sp_7", "log-u", "mlp"), 7: ("sp_7", "log-n", "fourier"), 8: ("sp_7", "log-n", "mlp"),
        9: ("sp_11", "log-u", "mlp"), 10: ("sp_11", "log-n", "fourier"), 11: ("sp_11", "log-n", "mlp"),
    }
    assert {task_id: (BY_TASK_ID[task_id].species, BY_TASK_ID[task_id].state, BY_TASK_ID[task_id].architecture) for task_id in expected} == expected


def test_duplicate_matching_keeps_instances_and_defaults_to_newest(tmp_path: Path):
    old = RunInstance(tmp_path / "old", "ms_perfect", modified=10, complete=True)
    new = RunInstance(tmp_path / "new", "ms_perfect", modified=20, complete=True)
    row = match_catalogue([old, new])[13]
    assert row["status"] == "duplicate"
    assert row["instances"] == [new, old]
    assert row["selected_instance"] is new


def test_discovery_prefers_label_and_uses_manifest_metadata(tmp_path: Path):
    run = tmp_path / "runs/pde_multispecies/copied_timestamp"
    run.mkdir(parents=True)
    (run / "final_suite_label.txt").write_text("ms_perfect\n")
    (run / "config.json").write_text("{}")
    (run / "final_summary.json").write_text("{}")
    (run / "final_predictions_grid.csv").write_text("t,w,N\n0,1,2\n")
    manifest = tmp_path / "final_runs/final_suite_manifest/job_1/task_13_ms_perfect.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("task_id=13\nrun_label=ms_perfect\nrun_dir=/obsolete/HPC/path\ngit_commit=abc\n")
    result = discover_local_runs(tmp_path)
    assert len(result) == 1
    assert result[0].complete
    assert result[0].metadata["git_commit"] == "abc"


def test_state_rmse_and_fold_error():
    errors = [-1.0, 1.0]
    assert state_rmse(errors) == 1.0
    assert fold_error(errors) == 10.0
    assert np.isnan(state_rmse([np.nan]))


def test_state_alignment_interpolates_in_log_weight_and_time():
    # log10 truth is x + time; the midpoint therefore has an exact answer.
    truth_rows = []
    for time in (0.0, 2.0):
        for w in (1.0, np.e**2):
            log10_n = np.log(w) + time
            truth_rows.append((time, 0, "sp", w, 10**log10_n))
    pred = _state([(1.0, 0, "sp", np.e, 10**2.0)])
    aligned, metadata = align_states(pred, _state(truth_rows))
    assert aligned.true_log10_N.iloc[0] == pytest.approx(2.0)
    assert aligned.error_log10_N.iloc[0] == pytest.approx(0.0)
    assert metadata["interpolated_time_cells"] == 1
    assert metadata["extrapolation"] == "none"


def test_state_alignment_never_extrapolates_and_preserves_named_species_index():
    truth = pd.DataFrame({
        "time": [0.0, 0.0, 2.0, 2.0],
        "species": ["sp_7"] * 4,
        "w": [1.0, np.e**2, 1.0, np.e**2],
        "N": [1.0, 100.0, 100.0, 10000.0],
    })
    normalised = normalise_state(truth)
    assert normalised.species_idx.unique().tolist() == [7]
    prediction = pd.DataFrame({
        "time": [1.0, 1.0, 3.0], "species": ["sp_7"] * 3,
        "w": [np.e, np.e**3, np.e], "N": [100.0, 1.0, 1.0],
    })
    aligned, _ = align_states(prediction, truth)
    assert len(aligned) == 1
    assert aligned.species_idx.iloc[0] == 7
    assert aligned.true_log10_N.iloc[0] == pytest.approx(2.0)


def test_species_active_weight_mask_and_zero_bins():
    truth = _state([
        (0, 0, "a", 1, 10), (0, 0, "a", 10, 20),
        (0, 1, "b", 1, 10), (0, 1, "b", 10, 0),  # zero is an inactive mask
    ])
    pred = _state([
        (0, 0, "a", 1, 10), (0, 0, "a", 10, 20),
        (0, 1, "b", 1, 10), (0, 1, "b", 10, 99),
    ])
    aligned, _ = align_states(pred, truth, w_max={0: 5, 1: 10})
    assert set(zip(aligned.species_idx, aligned.w)) == {(0, 1.0), (1, 1.0)}
    metrics, reason = aggregate_state_metrics(aligned, group_by="species_idx")
    assert reason is None
    assert metrics.n_cells.tolist() == [1.0, 1.0]


def test_metric_selection_reports_empty_domain():
    aligned = pd.DataFrame({"time": [0], "w": [1], "species_idx": [0], "error_log10_N": [0.2]})
    result, reason = aggregate_state_metrics(aligned, time_range=(2, 3))
    assert result.empty
    assert reason == "No valid comparison cells after selection."


# Imported at the bottom so a missing assertion dependency is obvious rather
# than being confused with one of the app's optional UI dependencies.
import pytest


def _aligned_errors():
    return pd.DataFrame({
        "species_idx": [0, 0, 0, 0, 1], "species": ["a"] * 4 + ["b"],
        "time": [0, 0, 1, 1, 0], "w": [1, 10, 1, 10, 1],
        "pred_log10_N": [2, 1, 3, 1, 0], "true_log10_N": [1, 2, 1, 1, 1],
    }).pipe(state_error_table)


def test_error_sign_and_n5_n6_aggregations():
    data = _aligned_errors()
    assert data.error_log10_N.tolist() == [1, -1, 2, 0, -1]
    n5 = error_by_time(data[data.species_idx == 0])
    assert n5.shape == (2, 2)
    assert n5.RMSE_log10N.tolist() == pytest.approx([1, np.sqrt(2)])
    n6 = error_by_weight(data[data.species_idx == 0])
    assert n6.shape == (2, 2)
    assert n6.RMSE_log10N.tolist() == pytest.approx([np.sqrt(2.5), np.sqrt(0.5)])


def test_species_aggregation_fold_and_range_masking():
    data = _aligned_errors()
    species = error_by_species(data)
    assert species.RMSE_log10N.tolist() == pytest.approx([np.sqrt(1.5), 1])
    selected = mask_domain(data, (1, 1), (1, 1))
    assert len(selected) == 1 and selected.error_log10_N.iloc[0] == 2


def test_comparison_domain_is_exactly_common():
    first = _aligned_errors().iloc[:4]
    second = first.iloc[[0, 3]].copy()
    a, b = common_comparison_domain(first, second)
    assert len(a) == len(b) == 2
    assert set(zip(a.time, a.w)) == {(0, 1), (1, 10)}


def test_residual_aggregation_and_pde_balance():
    data = pd.DataFrame({"time": [0, 0, 1, 1], "w": [1, 2, 1, 2], "residual_log": [-1, 3, 2, -2]})
    aggregate = residual_aggregate(data, "time")
    assert aggregate.mean_abs.tolist() == [2, 2]
    assert aggregate.max_abs.tolist() == [3, 2]
    assert aggregate.p95_abs.iloc[0] == pytest.approx(2.9)
    fields = data.assign(dlogN_dt=1, advective=2, mu=3, dg_dw=4, residual_log=10)
    balanced, maximum, warning = pde_balance(fields)
    assert np.allclose(balanced.term_sum, balanced.residual_log)
    assert maximum == 0 and not warning
    _, _, warning = pde_balance(fields.assign(residual_log=9))
    assert warning


def test_observation_math_denoising_ranking_and_gate():
    assert cv_to_sigma([0.3])[0] == pytest.approx(np.sqrt(np.log(1.09)))
    raw = pd.DataFrame({
        "value": [2.0, 4.0], "prediction": [1.0, 8.0], "value_true": [1.0, 4.0],
        "sd_log_used": [0.5, 0.25], "log_residual": [np.log(2), -np.log(2)],
        "species_idx": [0, 0], "t_start": [0, 1], "t_end": [0, 1],
    })
    prepared = prepare_observations(raw)
    assert prepared.z.tolist() == pytest.approx([2*np.log(2), -4*np.log(2)])
    denoised, summary = denoising_metrics(prepared)
    assert summary["fraction_below"] == 0.5
    assert rank_misfits(prepared).observation_id.tolist()[0] == 1
    assert discrepancy_q(prepared.z) == pytest.approx(np.sum(prepared.z**2))
    assert discrepancy_action(4, 5, 2, enabled=True) == (0.0, False)
    assert discrepancy_action(6, 5, 2, enabled=True) == (2.0, True)
    assert discrepancy_action(0, 5, 2, enabled=False) == (2.0, True)


def test_runtime_and_training_column_availability():
    history = pd.DataFrame({"step": [1, 3, 7], "seconds_elapsed": [2, 6, 10], "loss": [3, 2, 1], "empty": [np.nan] * 3})
    runtime = interval_seconds_per_step(history)
    assert runtime.seconds_per_step.iloc[1:].tolist() == [2, 1]
    present, missing = available_columns(history, ["loss", "empty", "absent"])
    assert present == ["loss"] and missing == ["empty", "absent"]


def test_observation_schema_requires_saved_uncertainty():
    raw = pd.DataFrame({
        "value": [1.0],
        "prediction": [1.0],
        "species_idx": [0],
        "t_start": [0.0],
        "t_end": [0.0],
    })
    with pytest.raises(ValueError, match="sd_log_used/sd_log"):
        prepare_observations(raw)


def test_noise_design_preserves_matched_seeds_and_e1_summary():
    design = noise_design()
    gate_on = design[design.gate]
    assert gate_on.groupby("cv").size().tolist() == [5] * 5
    for seed, group in gate_on.groupby("noise_seed"):
        assert group.cv.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        assert group.replicate.nunique() == 1
    values = gate_on.assign(value=gate_on.cv * 10 + gate_on.replicate)
    summary = noise_summary(values)
    assert summary.n.tolist() == [5] * 5
    assert summary["mean"].tolist() == pytest.approx([4, 5, 6, 7, 8])
    assert summary.sd.tolist() == pytest.approx([np.sqrt(2.5)] * 5)


def test_paired_cv_differences_use_same_noise_seed():
    values = noise_design()
    values = values[values.gate].assign(value=lambda x: x.cv * x.noise_seed)
    pairs, wide = paired_cv_differences(values)
    selected = pairs[(pairs.cv_from == 0.1) & (pairs.cv_to == 0.5)].iloc[0]
    expected = 0.4 * np.mean(wide.index)
    assert selected.mean_difference == pytest.approx(expected)
    assert selected.n == 5


def test_missing_design_masks_and_metrics():
    data = pd.DataFrame({
        "time": [0, 10, 15, 20, 30], "species_idx": [3, 3, 7, 7, 11],
        "error_log10_N": [0, 1, 2, 1, 0],
    })
    assert gap_mask(data, 10, 20).tolist() == [False, True, True, True, False]
    assert missing_species_mask(data, 7).tolist() == [False, False, True, True, False]
    summary = missing_seen_metrics(data, gap_mask(data, 10, 20))
    assert summary["RMSE_missing"] == pytest.approx(np.sqrt(2))
    assert summary["RMSE_seen"] == 0
    assert summary["generalisation_penalty"] == pytest.approx(np.sqrt(2))


def test_every_third_year_design_uses_actual_observation_years():
    perfect = pd.DataFrame({"t_start": range(7)})
    reduced = pd.DataFrame({"t_start": [0, 3, 6]})
    retained, omitted = retained_omitted_years(perfect, reduced)
    assert retained == [0, 3, 6]
    assert omitted == [1, 2, 4, 5]
    state = pd.DataFrame({"time": [0, 1, 1.5, 2, 3]})
    assert year_location_mask(state, omitted).tolist() == [False, True, False, True, False]


def test_nn_ablation_and_baseline_catalogue_mappings():
    assert validate_nn_pairings() == NN_SCENARIOS
    assert NN_SCENARIOS == {
        "Perfect data": (13, 50),
        "CV 0.3, rep1": (24, 51),
        "Gap years 30–40": (46, 52),
    }
    matrix = ablation_task_matrix()
    assert matrix.loc["sp_3"].tolist() == [0, 3, 4, 5]
    assert matrix.loc["sp_7"].tolist() == [1, 6, 7, 8]
    assert matrix.loc["sp_11"].tolist() == [2, 9, 10, 11]
    assert validate_baseline_pair() == (12, 13)


def test_single_species_loader_restores_fixture_species_identity(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    (run / "config.json").write_text('{"input_dir": "validation/fixtures/pde_single_species/sp_7"}')
    pd.DataFrame({"t": [0], "w": [1], "N": [2]}).to_csv(run / "final_predictions_grid.csv", index=False)
    state = load_prediction_state.clear() or load_prediction_state(str(run))
    assert state.species_idx.tolist() == [7]
    assert state.species.tolist() == ["sp_7"]
    diagnostics = run / "fixed_grid_diagnostics"; diagnostics.mkdir()
    pd.DataFrame({"t_eval": [0], "w_eval": [1], "x_eval": [0], "residual_log": [0]}).to_csv(diagnostics / "fixed_grid_fields.csv", index=False)
    load_fixed_fields.clear()
    fixed = load_fixed_fields(str(run))
    assert fixed.species_idx.tolist() == [7]


def test_rmax_recovery_uses_truth_and_log_ratio_metric():
    estimated = pd.DataFrame({
        "species_idx": [0, 1], "species": ["a", "b"],
        "estimated_r_max": [2.0, 2.0],
    })
    recovered = rmax_recovery(estimated, np.array([1.0, 4.0]))
    assert recovered.ratio_to_truth.tolist() == [2.0, 0.5]
    assert recovered.abs_log_error.tolist() == pytest.approx([np.log(2), np.log(2)])
    assert rmse_log_ratio(estimated.estimated_r_max, [1, 4]) == pytest.approx(np.log(2))


def test_rmax_truth_never_uses_perturbed_start_as_truth(tmp_path):
    run = tmp_path / "run"
    inputs = tmp_path / "inputs"
    run.mkdir(); inputs.mkdir()
    (run / "config.json").write_text(f'{{"input_dir": "{inputs}"}}')
    pd.DataFrame({"r_max": [2.0]}).to_csv(inputs / "r_max.csv", index=False)
    truth, source = _rmax_truth(str(run))
    assert truth is None
    assert "r_max_true.csv" in source
    pd.DataFrame({"r_max": [1.0]}).to_csv(inputs / "r_max_true.csv", index=False)
    truth, source = _rmax_truth(str(run))
    assert truth.tolist() == [1.0]
    assert source.endswith("r_max_true.csv")


def test_effort_errors_keep_zero_truth_separate():
    effort = pd.DataFrame({
        "physical_time": [0, 0, 1], "gear_idx": [0, 1, 1],
        "estimated_effort": [2.0, 0.2, 0.3],
        "true_reference_effort": [1.0, 0.0, 0.0],
    })
    errors = effort_errors(effort)
    assert errors.effort_ratio.iloc[0] == 2
    assert np.isnan(errors.effort_ratio.iloc[1])
    assert errors.zero_truth_absolute_error.iloc[1:].tolist() == pytest.approx([0.2, 0.3])
    summary = effort_recovery_summary(effort)
    assert summary["RMSE_logE_positive_truth"] == pytest.approx(np.log(2))
    assert summary["mean_absolute_error_zero_truth"] == pytest.approx(0.25)
    assert (summary["n_positive_truth"], summary["n_zero_truth"]) == (1, 2)


def test_selectivity_reshape_and_exact_fishing_operator():
    # Raw rows are [species, gear], as exported by the repository R fixture.
    raw = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    selectivity = reshape_selectivity(raw, n_gear=2, n_species=2, n_weight=2)
    assert selectivity[:, 0, :].tolist() == [[1, 2], [3, 4]]
    catchability = np.array([[2, 10], [3, 20]])
    effort = np.array([0.5, 2.0])
    gear_zero = fishing_mortality(effort, catchability, selectivity, species_idx=0, gear_idx=0)
    total = fishing_mortality(effort, catchability, selectivity, species_idx=0)
    assert gear_zero.tolist() == pytest.approx([1, 2])
    assert total.tolist() == pytest.approx([19, 26])


def test_inverse_task_catalogue_ranges():
    assert [BY_TASK_ID[i].start_value for i in range(53, 57)] == [0.25, 0.5, 2.0, 4.0]
    assert [(BY_TASK_ID[i].cv, BY_TASK_ID[i].start_value, BY_TASK_ID[i].noise_seed) for i in range(57, 62)] == [
        (0.3, 0.5, 41001), (0.3, 0.5, 41002), (0.3, 0.5, 41003), (0.3, 0.5, 41004), (0.3, 0.5, 41005),
    ]
    assert [BY_TASK_ID[i].start_value for i in range(62, 66)] == [0.1, 0.3, 0.6, 1.0]
    assert [BY_TASK_ID[i].start_value for i in range(66, 70)] == [0.05, 0.5, 1.0, 3.0]
