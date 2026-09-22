# Required fixture contract

Each biological parameter directory must satisfy `PINNmizer.io.load_mizer_inputs()`. The known ecosystem state is a long CSV with exactly:

`time,species_idx,species,weight,N`

`species_idx` is zero-based, complete and stable. Every stored time must contain all species and all `params.w` values exactly once. Times must span `params.t_min` through `params.t_max`; interpolation never extrapolates. The first stored state must match `n_init_full.csv` unless `--allow-known-initial-mismatch` is explicitly used.

The repository currently does not populate `params.species` in `load_mizer_inputs()`. This experiment therefore takes canonical species names from the validated known-state mapping and assigns them locally.
