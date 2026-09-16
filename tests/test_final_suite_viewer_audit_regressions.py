from pathlib import Path

import pandas as pd

from apps.final_suite_viewer.experiment_analysis import gap_mask
from apps.final_suite_viewer.loaders import load_w_max


def test_missing_gap_is_half_open():
    data = pd.DataFrame({"time": [0, 10, 15, 20, 30]})
    assert gap_mask(data, 10, 20).tolist() == [False, True, True, False, False]


def test_w_max_is_never_inferred_from_prediction_support(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text("{}")
    pd.DataFrame({"t": [0, 0], "species_idx": [0, 0], "w": [1, 100], "N": [2, 3]}).to_csv(
        run / "final_predictions_grid.csv", index=False
    )
    load_w_max.clear()
    assert load_w_max(str(run)) == {}

    pd.DataFrame({"value": [10]}).to_csv(run / "w_max.csv", index=False)
    load_w_max.clear()
    assert load_w_max(str(run)) == {0: 10.0}
