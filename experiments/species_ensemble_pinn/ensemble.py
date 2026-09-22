from __future__ import annotations

import torch
import torch.nn as nn


class SpeciesPINNEnsemble(nn.Module):
    """Independent scalar log-N networks assembled in species-index order."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one scalar model is required.")
        self.models = nn.ModuleList(models)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [model(inputs) for model in self.models]
        for output in outputs:
            if output.ndim != 2 or output.shape[1] != 1:
                raise ValueError("Each species model must return [P,1] direct log_N.")
        return torch.cat(outputs, dim=1)
