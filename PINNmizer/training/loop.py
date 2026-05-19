"""
Training-loop helpers for PINN experiments.

The training loop coordinates sampling, loss evaluation, backward passes,
optimiser steps, logging, checkpointing, and diagnostics. It should call the PDE
and biology modules rather than implementing their details inline.

Future target
-------------
Move `train_one_step` and repeated loop bookkeeping here after the public module
split has been reviewed. Preserve the existing validation path while doing so.
"""
