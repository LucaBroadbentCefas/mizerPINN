from __future__ import annotations

from pathlib import Path
import torch

from .config import model_configuration_dict, tensor_checksum
from .residual_scale import grid_residual_scale


def save_checkpoint(path: str | Path, *, step: int, species_idx: int, species_name: str,
                    model, optimizer, scheduler, loss_weights: dict[str, float],
                    configuration: dict, latest_history_row: dict | None,
                    parameter_fixture_identity: dict, known_state_file_identity: dict,
                    params) -> Path:
    path = Path(path)
    log_scale, _ = grid_residual_scale(params)
    target = log_scale[species_idx].detach().cpu()
    payload = {
        "step": step, "species_idx": species_idx, "species_name": species_name,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "loss_weights": dict(loss_weights),
        "configuration": configuration, "latest_history_row": latest_history_row,
        "parameter_fixture_identity": parameter_fixture_identity,
        "known_state_file_identity": known_state_file_identity,
        "residual_scale_floor_fraction": params.residual_scale_floor_fraction,
        "residual_scale_source": params.residual_scale_source,
        "residual_scale_interpolation": params.residual_scale_interpolation,
        "residual_scale_extrapolation": params.residual_scale_extrapolation,
        "target_residual_scale_log": target,
        "target_residual_scale_checksum": tensor_checksum(target),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, model, optimizer, scheduler, params,
                    species_idx: int, species_name: str, configuration: dict,
                    load_optimizer_state: bool) -> dict:
    payload = torch.load(Path(path), map_location=params.w.device)
    expected = {
        "state_parameterization": "log-n", "residual_form": "reference-scaled",
        "model_configuration": model_configuration_dict(), "species_idx": species_idx,
        "species_name": species_name,
    }
    stored = payload.get("configuration", {})
    for key, value in expected.items():
        actual = payload.get(key) if key in {"species_idx", "species_name"} else stored.get(key)
        if actual != value:
            raise ValueError(f"Checkpoint incompatibility for {key}: {actual!r} != {value!r}.")
    current = params.residual_scale_log[species_idx].detach().cpu()
    if tensor_checksum(current) != payload.get("target_residual_scale_checksum"):
        raise ValueError("Checkpoint residual reference scale does not match current n_init.")
    if not torch.equal(current, payload["target_residual_scale_log"]):
        raise ValueError("Checkpoint target residual scale tensor differs from reconstruction.")
    model.load_state_dict(payload["model_state_dict"])
    if load_optimizer_state:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
