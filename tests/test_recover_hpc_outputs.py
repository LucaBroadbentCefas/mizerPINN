from pathlib import Path

from scripts.recover_hpc_outputs import (
    build_recovery_command,
    find_latest_checkpoint,
    recovery_args,
)


def test_checkpoint_prefers_model_final(tmp_path: Path):
    (tmp_path / "model_step_100.pt").touch()
    final = tmp_path / "model_final.pt"
    final.touch()
    assert find_latest_checkpoint(tmp_path) == final


def test_checkpoint_uses_highest_numeric_step(tmp_path: Path):
    for step in [20, 1000, 300]:
        (tmp_path / f"model_step_{step}.pt").touch()
    assert find_latest_checkpoint(tmp_path).name == "model_step_1000.pt"


def test_recovery_args_force_zero_steps_and_hpc(tmp_path: Path):
    checkpoint = tmp_path / "model_final.pt"
    original = [
        "--input-dir", "validation/fixtures/pde_multispecies",
        "--n-steps", "30000",
        "--start-step", "25000",
        "--load-weights", "old.pt",
        "--load-optimizer-state",
        "--device", "cuda",
        "--hpc",
        "--model-arch", "fourier",
    ]
    args = recovery_args(original, checkpoint, device="cpu")
    assert args.count("--n-steps") == 1
    assert args[args.index("--n-steps") + 1] == "0"
    assert args.count("--start-step") == 1
    assert args[args.index("--start-step") + 1] == "0"
    assert args.count("--load-weights") == 1
    assert args[args.index("--load-weights") + 1] == str(checkpoint.resolve())
    assert "--load-optimizer-state" not in args
    assert args.count("--hpc") == 1
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("--model-arch") + 1] == "fourier"


def test_build_command_replays_saved_module_and_settings(tmp_path: Path):
    (tmp_path / "model_step_4000.pt").touch()
    (tmp_path / "run_command.txt").write_text(
        "python -m scripts.train_pde_only_single_species ^\n"
        "  --input-dir validation/fixtures/pde_single_species ^\n"
        "  --n-steps 4000 ^\n"
        "  --state-parameterization log-u\n",
        encoding="utf-8",
    )
    command = build_recovery_command(tmp_path)
    assert command[1:3] == ["-m", "scripts.train_pde_only_single_species"]
    assert command[command.index("--n-steps") + 1] == "0"
    assert command[command.index("--load-weights") + 1].endswith("model_step_4000.pt")
    assert command[command.index("--state-parameterization") + 1] == "log-u"
    assert command[-1] == "--hpc"
