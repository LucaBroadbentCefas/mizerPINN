from pathlib import Path

from PINNmizer.diagnostics.checkpoint_snapshots import (
    discover_checkpoints,
    discover_snapshots,
    parse_emitted_run_dir,
    snapshot_dir,
)
from scripts.materialize_checkpoint_outputs import override_input_dir_args


def test_discover_checkpoints_supports_both_layouts(tmp_path: Path):
    run = tmp_path / "runs" / "example"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_step_4000.pt").touch()
    (run / "checkpoints" / "model_step_8000.pt").touch()
    (run / "checkpoints" / "not_a_checkpoint.pt").touch()

    found = discover_checkpoints(run)
    assert list(found) == [4000, 8000]
    assert found[4000] == run / "model_step_4000.pt"
    assert found[8000] == run / "checkpoints" / "model_step_8000.pt"


def test_snapshot_layout_and_metadata_filter(tmp_path: Path):
    project = tmp_path / "project"
    source = project / "runs" / "pde_multispecies" / "run_a"
    source.mkdir(parents=True)
    root = project / "checkpoint_views"
    snap = snapshot_dir(source, root, 4000, project_root=project)
    snap.mkdir(parents=True)
    (snap / "checkpoint_snapshot.json").write_text(
        '{"source_run_dir": "%s", "checkpoint_step": 4000}' % source.resolve(),
        encoding="utf-8",
    )

    assert snap == root / "runs" / "pde_multispecies" / "run_a" / "step_00004000"
    assert discover_snapshots(source, root, project_root=project) == {4000: snap}


def test_parse_emitted_run_dir_uses_last_match(tmp_path: Path):
    text = "Run directory: runs/old\nnoise\nRun directory: runs/new\n"
    assert parse_emitted_run_dir(text, repo_root=tmp_path) == (tmp_path / "runs" / "new").resolve()


def test_override_input_dir_remaps_child_data_csv(tmp_path: Path):
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "n_init_full.csv").write_text("x\n1\n", encoding="utf-8")
    original = [
        "--input-dir", "validation/fixtures/old_bundle",
        "--data-csv", "validation/fixtures/old_bundle/observations.csv",
        "--diag-grid-csv", "external/grid.csv",
        "--n-steps", "10000",
    ]

    changed = override_input_dir_args(original, replacement)

    assert changed[changed.index("--input-dir") + 1] == str(replacement.resolve())
    assert changed[changed.index("--data-csv") + 1] == str(replacement.resolve() / "observations.csv")
    assert changed[changed.index("--diag-grid-csv") + 1] == "external/grid.csv"
