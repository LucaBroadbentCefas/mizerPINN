"""Smoke-check import paths used by training entry points.

This catches stale module paths after repository restructuring.
"""

from __future__ import annotations


def main() -> None:
    from PINNmizer.training.train_pde_only_single_species import main as train_main
    from PINNmizer.training.loop import train_one_step
    from PINNmizer.diagnostics.fields import save_fixed_grid_fields_and_plots
    from PINNmizer.diagnostics.output_surface import save_output_surface_diagnostics

    _ = (train_main, train_one_step, save_fixed_grid_fields_and_plots, save_output_surface_diagnostics)
    print("OK: training and diagnostics import smoke check passed")


if __name__ == "__main__":
    main()
