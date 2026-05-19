"""
Neural-network evaluation utilities for scaled PINN coordinates.

The current implementation is re-exported from `PINNmizer.pde_residual` to keep
this refactor mechanical. This file documents the intended ownership boundary:
model evaluation belongs here; residual assembly and biological operators do not.

Coordinate convention
---------------------
Physical mass is represented through log-weight

    x = log(w)

and the network input is scaled to [0, 1]

    x_scaled = (x - x_min) / (x_max - x_min)
    t_scaled = (t - t_min) / (t_max - t_min)

Output convention
-----------------
The neural network returns log_N. The evaluation utility converts this to N with
exp(log_N) and reshapes flat network output into time-major tensors:

    log_N: [n_time, n_species, n_x]
    N:     [n_time, n_species, n_x]
"""

from PINNmizer.pde_residual import evaluate_log_model_on_points

__all__ = ["evaluate_log_model_on_points"]
