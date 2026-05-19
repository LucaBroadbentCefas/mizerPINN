"""
Fixed-grid diagnostics for PDE residual checks.

The existing implementation lives in `validation_steps.pinn_diagnostics`. This
module documents the intended destination for deterministic grid checks that
compare the PINN residual on a fixed Cartesian grid.

Fixed-grid diagnostics are separate from training because they should be stable
validation baselines rather than part of the stochastic collocation sampler.
"""

from validation_steps.pinn_diagnostics import (
    make_fixed_pde_batch,
    make_fixed_pde_batch_from_csv,
    compute_fixed_diagnostics,
)

__all__ = [
    "make_fixed_pde_batch",
    "make_fixed_pde_batch_from_csv",
    "compute_fixed_diagnostics",
]
