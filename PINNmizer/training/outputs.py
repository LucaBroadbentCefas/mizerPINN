from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import sys

from PINNmizer.params import scale_t, scale_x
from PINNmizer.pinn.residual import compute_pde_residual
from PINNmizer.pinn.model_eval import evaluate_log_model_on_points
from PINNmizer.pinn.sampling import sample_pde_batch

