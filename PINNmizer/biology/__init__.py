"""
Biological operators for the mizer-style size-spectrum PINN.

The modules in this package hold the hand-coded biological terms used by the
PDE residual. These functions should not evaluate the neural network or assemble
PINN losses. They operate on tensors that already represent spectra, grids, and
species parameters.

Public compatibility imports are exposed through `PINNmizer.biology.continuous`
and the legacy `PINNmizer.continuous_biology` module.
"""

from .continuous import *
