from __future__ import annotations

import hashlib, torch


def tensor_checksum(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()


def checkpoint_payload(*, step:int, species_index:int, species_name:str, model, optimizer=None, scheduler=None, loss_weights=None, configuration:dict, params, latest_history_row=None, parameter_fixture_identity=None, known_state_file_identity=None) -> dict:
    target = params.state_scale_log[species_index].detach().clone()
    return {"step":step,"species_index":species_index,"species_name":species_name,"model_state_dict":model.state_dict(),"optimizer_state_dict":None if optimizer is None else optimizer.state_dict(),"scheduler_state_dict":None if scheduler is None else scheduler.state_dict(),"loss_weights":loss_weights or {"pde":1.0,"ic":1.0,"bc":1.0},"configuration":configuration,"latest_history_row":latest_history_row,"parameter_fixture_identity":parameter_fixture_identity,"known_state_file_identity":known_state_file_identity,"state_scale_eps":params.state_scale_eps,"state_scale_source":params.state_scale_source,"state_scale_interpolation":params.state_scale_interpolation,"state_scale_extrapolation":getattr(params,"state_scale_extrapolation","constant_nearest_active"),"target_state_scale_log":target,"target_state_scale_checksum":tensor_checksum(target)}


def validate_checkpoint(ckpt: dict, *, params, species_index:int, species_name:str|None=None, configuration:dict|None=None) -> None:
    cfg = ckpt.get("configuration", {})
    if cfg.get("state_parameterization") != "log-u" or cfg.get("residual_form") != "scaled":
        raise ValueError("Incompatible checkpoint: direct log_N/reference-scaled checkpoints cannot be loaded as log_U scaled runs.")
    if int(ckpt.get("species_index", -1)) != int(species_index):
        raise ValueError("Checkpoint species_index mismatch.")
    if species_name is not None and ckpt.get("species_name") != species_name:
        raise ValueError("Checkpoint species_name mismatch.")
    target = params.state_scale_log[species_index].detach().cpu()
    saved = ckpt.get("target_state_scale_log")
    if saved is None or not torch.equal(saved.detach().cpu(), target) or ckpt.get("target_state_scale_checksum") != tensor_checksum(target):
        raise ValueError("Checkpoint state scale is incompatible with current initial condition.")
