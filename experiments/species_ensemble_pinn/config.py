from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


SUPPORTED = {
    "residual_form": "reference-scaled",
    "state_parameterization": "log-n",
    "boundary_loss_form": "relative",
    "collocation_strategy": "uniform",
    "time_sampling": "stratified",
    "causal_loss": "expert",
    "causal_curriculum": "linear",
    "loss_weighting": "expert-grad-norm",
    "model_arch": "fourier",
    "weight_factorization": "rwf",
    "rwf_apply_to": "all",
    "rwf_base_init": "xavier_uniform",
    "lr_scheduler": "cosine",
}


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 2
    output_dim: int = 1
    fourier_num_features: int = 16
    fourier_scale: float = 1.0
    fourier_include_raw_input: bool = True
    fourier_seed: int = 123
    hidden_width: int = 384
    hidden_layers: int = 5
    weight_factorization: str = "rwf"
    rwf_mu: float = 1.0
    rwf_sigma: float = 0.1
    rwf_apply_to: str = "all"
    rwf_base_init: str = "xavier_uniform"


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float32' or 'float64'.")


def tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def file_identity(path: str | Path) -> dict[str, object]:
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(p), "size": stat.st_size, "sha256": h.hexdigest()}


def directory_identity(path: str | Path) -> dict[str, object]:
    root = Path(path).expanduser().resolve()
    entries = []
    for item in sorted(root.glob("*.csv")):
        entries.append((item.name, item.stat().st_size, file_identity(item)["sha256"]))
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return {"path": str(root), "sha256": hashlib.sha256(payload).hexdigest(), "files": len(entries)}


def validate_focused_configuration(args) -> None:
    for attr, expected in SUPPORTED.items():
        actual = getattr(args, attr)
        if actual != expected:
            flag = "--" + attr.replace("_", "-")
            raise ValueError(f"{flag} supports only {expected!r} in Tranche 1; got {actual!r}.")
    if float(args.lambda_timestep) != 0.0:
        raise ValueError("--lambda-timestep must be 0 in Tranche 1.")
    if args.n_time <= 0 or args.n_eval <= 0 or args.n_steps < 0:
        raise ValueError("n_time and n_eval must be positive; n_steps must be non-negative.")
    if args.n_time % args.causal_n_chunks != 0:
        raise ValueError("--n-time must be exactly divisible by --causal-n-chunks.")
    expected = ModelConfig()
    checks = {
        "fourier_num_features": expected.fourier_num_features,
        "fourier_scale": expected.fourier_scale,
        "fourier_include_raw_input": expected.fourier_include_raw_input,
        "fourier_seed": expected.fourier_seed,
        "hidden_width": expected.hidden_width,
        "hidden_layers": expected.hidden_layers,
        "rwf_mu": expected.rwf_mu,
        "rwf_sigma": expected.rwf_sigma,
    }
    for attr, value in checks.items():
        if getattr(args, attr) != value:
            raise ValueError(f"--{attr.replace('_','-')} is fixed to {value!r} in Tranche 1.")
    if args.residual_scale_floor_fraction <= 0:
        raise ValueError("--residual-scale-floor-fraction must be strictly positive.")
    if args.known_state_log_floor <= 0:
        raise ValueError("--known-state-log-floor must be strictly positive.")
    if args.lr_min > args.lr:
        raise ValueError("--lr-min must be no greater than --lr.")
    if args.environment_state not in {"dynamic-known", "frozen-initial"}:
        raise ValueError("Unsupported environment-state.")
    if args.known_state_interpolation not in {"linear", "log-linear"}:
        raise ValueError("Unsupported known-state interpolation.")
    if args.biology_label not in {"detailed", "trait"}:
        raise ValueError("--biology-label must be detailed or trait.")


def model_configuration_dict() -> dict[str, object]:
    return asdict(ModelConfig())
