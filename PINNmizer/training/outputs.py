"""
Output-writing helpers for PINN training runs.

This module is for CSVs, final prediction grids, residual samples, and run
artefacts. Output writing should stay separate from the mathematical loss and
from biological operator code.

Future target
-------------
Move final prediction and final residual sample writers here. Preserve output
column names so existing diagnostic scripts keep working.
"""
