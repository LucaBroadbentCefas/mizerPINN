"""
Continuous/off-grid biological operators used by the PINN residual.

This module currently re-exports the legacy implementation in
`PINNmizer.continuous_biology`. It gives the codebase a clearer import target
while preserving numerical behaviour.

Conceptual boundary
-------------------
Functions here compute biological quantities from a predicted spectrum and mizer
parameters. They should not evaluate the neural network, sample collocation
points, or assemble losses.

Main tensor conventions
-----------------------
- spectra on the fish grid: [n_species, n_w]
- evaluation weights:       [n_eval]
- biological outputs:       [n_species, n_eval]
- kernels:                  [n_species, n_pred, n_prey]

Derivative convention
---------------------
Biological derivatives such as `dg_dw` are computed manually from analytical
formulae. They should not be obtained by autograd through the biological path.
"""

from PINNmizer.continuous_biology import *
