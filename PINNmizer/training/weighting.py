"""
Adaptive loss-weighting utilities for PINN training.

Loss weighting is a training concern, not part of the biological model or PDE
residual. The current Wang-style weighting implementation remains in the legacy
single-species training script until it can be moved with test coverage.

Future target
-------------
Move `update_wang_gradient_weights_` and its gradient-statistic helpers here.
When that move is made, preserve the existing behaviour exactly and add a small
smoke test that confirms one backward pass still works.
"""
