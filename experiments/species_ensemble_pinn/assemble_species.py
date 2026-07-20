from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from PINNmizer.io import load_mizer_inputs
from PINNmizer.params import scale_t, scale_x

from .config import directory_identity, file_identity, model_configuration_dict, tensor_checksum
from .ensemble import SpeciesPINNEnsemble
from .known_state import KnownStateProvider
from .model_eval import make_model_inputs
from .models import build_scalar_model
from .residual_scale import set_residual_scale_from_initial_condition


def _successful_run(root: Path, species_idx: int) -> Path:
    candidates = []
    for status_file in root.glob(f"species_{species_idx:02d}_*/**/status.json"):
        status = json.loads(status_file.read_text())
        if status.get("status") == "success" and (status_file.parent / "checkpoint_latest.pt").exists():
            candidates.append(status_file.parent)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one successful run for species {species_idx}; found {len(candidates)}.")
    return candidates[0]


def assemble(*, runs_root: str | Path, input_dir: str | Path, known_state_csv: str | Path,
             output_dir: str | Path, dtype: torch.dtype = torch.float64,
             device: str = "cpu") -> Path:
    params, n_init, _ = load_mizer_inputs(input_dir, dtype=dtype, device=device)
    known = KnownStateProvider(known_state_csv, params, n_init, mode="dynamic-known")
    set_residual_scale_from_initial_condition(params, n_init)
    expected_parameter_identity = directory_identity(input_dir)
    expected_known_identity = file_identity(known_state_csv)
    models, run_paths = [], []
    for species_idx, species_name in enumerate(params.species):
        run = _successful_run(Path(runs_root), species_idx)
        checkpoint = torch.load(run / "checkpoint_latest.pt", map_location=device)
        if checkpoint["species_idx"] != species_idx or checkpoint["species_name"] != species_name:
            raise ValueError("Checkpoint species metadata does not match canonical ordering.")
        configuration = checkpoint.get("configuration", {})
        if configuration.get("state_parameterization") != "log-n":
            raise ValueError("Checkpoint is not direct log-N.")
        if configuration.get("residual_form") != "reference-scaled":
            raise ValueError("Checkpoint does not use the required PDE residual form.")
        if configuration.get("model_configuration") != model_configuration_dict():
            raise ValueError("Checkpoint scalar architecture is incompatible.")
        if checkpoint.get("parameter_fixture_identity") != expected_parameter_identity:
            raise ValueError("Checkpoint parameter fixture identity is incompatible.")
        if checkpoint.get("known_state_file_identity") != expected_known_identity:
            raise ValueError("Checkpoint known-state identity is incompatible.")
        current_scale = params.residual_scale_log[species_idx].detach().cpu()
        if checkpoint.get("target_residual_scale_checksum") != tensor_checksum(current_scale):
            raise ValueError("Checkpoint target residual scale is incompatible.")
        model = build_scalar_model().to(dtype=dtype, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model)
        run_paths.append(str(run))
    ensemble = SpeciesPINNEnsemble(models)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": ensemble.state_dict(), "species": params.species,
                "source_runs": run_paths, "parameter_fixture_identity": expected_parameter_identity,
                "known_state_file_identity": expected_known_identity}, output / "ensemble_checkpoint.pt")
    t = known.known_times
    x = torch.log(params.w)
    inputs = make_model_inputs(scale_x(x, params), scale_t(t, params))
    log_n_flat = ensemble(inputs)
    expected = (t.numel() * x.numel(), len(params.species))
    if log_n_flat.shape != expected:
        raise ValueError(f"Ensemble returned {tuple(log_n_flat.shape)}, expected {expected}.")
    log_n = log_n_flat.reshape(t.numel(), x.numel(), len(params.species)).permute(0, 2, 1).contiguous()
    prediction = torch.exp(log_n)
    known_n = known.at(t)
    tt = t[:, None, None].expand(-1, len(params.species), params.w.numel()).detach().cpu().reshape(-1).numpy()
    ss = torch.arange(len(params.species))[:, None].expand(-1, params.w.numel()).reshape(-1)
    ss = ss[None, :].expand(t.numel(), -1).reshape(-1).numpy()
    ww = params.w[None, None, :].expand(t.numel(), len(params.species), -1).detach().cpu().reshape(-1).numpy()
    pd.DataFrame({"time": tt, "species_idx": ss, "weight": ww,
        "log_N_pred": log_n.detach().cpu().reshape(-1).numpy(),
        "N_pred": prediction.detach().cpu().reshape(-1).numpy(),
        "N_known": known_n.detach().cpu().reshape(-1).numpy()}).to_csv(
            output / "ensemble_predictions.csv", index=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--known-state-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(assemble(**vars(args)))


if __name__ == "__main__":
    main()
