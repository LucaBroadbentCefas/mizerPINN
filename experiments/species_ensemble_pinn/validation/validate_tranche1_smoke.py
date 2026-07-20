from __future__ import annotations

import argparse
import tempfile

from experiments.species_ensemble_pinn.train_species import parse_args, run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-state-csv", required=True)
    parser.add_argument("--species-idx", type=int, default=0)
    parser.add_argument("--biology-label", required=True, choices=["detailed", "trait"])
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        run_args = parse_args([
            "--input-dir", args.input_dir, "--known-state-csv", args.known_state_csv,
            "--species-idx", str(args.species_idx), "--biology-label", args.biology_label,
            "--n-steps", "3", "--n-time", "64", "--n-eval", "8",
            "--causal-n-chunks", "32", "--diag-every", "1",
            "--checkpoint-every", "2", "--print-every", "0", "--output-root", tmp,
        ])
        run_dir = run(run_args)
        required = [
            "config.json", "run_command.txt", "status.json", "final_summary.csv",
            "final_summary.json", "loss_history.csv", "fixed_diagnostic_history.csv",
            "checkpoint_latest.pt", "model_final.pt", "predictions_final.csv",
            "residuals_final.csv", "biology_sample_final.csv",
        ]
        missing = [name for name in required if not (run_dir / name).exists()]
        assert not missing, missing
    print("PASS validate_tranche1_smoke")


if __name__ == "__main__":
    main()
