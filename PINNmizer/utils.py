import torch

def pos(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)
