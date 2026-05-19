"""
Plotting helpers for PINN diagnostics.

The current plotting functions live in `validation_steps.pinn_diagnostics`. This
module provides a clearer future import path for diagnostic plots while keeping
the existing implementation unchanged.
"""

from validation_steps.pinn_diagnostics import (
    save_training_diagnostic_plots,
    save_fixed_grid_fields_and_plots,
)

__all__ = [
    "save_training_diagnostic_plots",
    "save_fixed_grid_fields_and_plots",
]
