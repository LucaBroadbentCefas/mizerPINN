"""
Encounter-rate functions for continuous/off-grid mizer biology.

The encounter path maps resource and fish prey spectra to the available food
encountered by each predator species at evaluation weights.

Key distinction
---------------
The neural network supplies a predicted fish spectrum on the fixed mizer grid.
Encounter is then computed biologically from that spectrum. It is not a neural
network layer and its biological derivative is not obtained by autograd.
"""

from PINNmizer.continuous_biology import (
    compute_encounter_direct_at_eval,
    evaluate_gamma_continuous,
)

__all__ = ["compute_encounter_direct_at_eval", "evaluate_gamma_continuous"]
