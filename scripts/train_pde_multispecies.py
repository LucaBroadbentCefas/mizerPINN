import sys

from PINNmizer.training.train_pde_multispecies import main


def _strip_hpc_compat_flag() -> None:
    """Accept --hpc for CLI parity with the single-species script.

    The multispecies training module does not currently implement the
    single-species HPC output mode, so this wrapper treats --hpc as a
    compatibility flag and leaves training behaviour unchanged.
    """
    sys.argv = [arg for arg in sys.argv if arg != "--hpc"]


if __name__ == "__main__":
    _strip_hpc_compat_flag()
    main()
