"""
Checkpointing helpers for PINN training runs.

Checkpointing is infrastructure: it serialises model state, optimizer state, and
configuration. It should not compute losses, diagnostics, or biological terms.

Future target
-------------
Move the existing `save_checkpoint` helper here once the training script is
split mechanically. The saved dictionary schema should remain stable:

    step
    model_state_dict
    optimizer_state_dict
    config
"""
