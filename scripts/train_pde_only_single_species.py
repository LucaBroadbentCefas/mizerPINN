"""
Entry point for the single-species PDE-only PINN run.

This wrapper delegates to the existing implementation during the structure
refactor.
"""

from validation_steps.train_pde_only_single_species import main


if __name__ == "__main__":
    main()
