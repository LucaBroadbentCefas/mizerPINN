import pandas as pd
import pytest
import torch

from PINNmizer.io import maybe_selectivity


def test_selectivity_species_major_rows_are_converted_to_gear_major_tensor(tmp_path):
    n_species = 3
    n_gear = 2
    n_w = 4
    raw = torch.arange(
        n_species * n_gear * n_w,
        dtype=torch.float64,
    ).reshape(n_species * n_gear, n_w)
    pd.DataFrame(raw.numpy()).to_csv(tmp_path / "selectivity.csv", index=False)

    catchability = torch.ones((n_gear, n_species), dtype=torch.float64)
    selectivity = maybe_selectivity(
        tmp_path,
        torch.float64,
        "cpu",
        catchability,
        n_w,
    )

    assert selectivity.shape == (n_gear, n_species, n_w)
    for species_idx in range(n_species):
        for gear_idx in range(n_gear):
            source_row = species_idx * n_gear + gear_idx
            assert torch.equal(
                selectivity[gear_idx, species_idx],
                raw[source_row],
            )


def test_selectivity_loader_rejects_unexpected_shape(tmp_path):
    pd.DataFrame(torch.zeros((5, 4), dtype=torch.float64).numpy()).to_csv(
        tmp_path / "selectivity.csv",
        index=False,
    )
    catchability = torch.ones((2, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="Unexpected selectivity shape"):
        maybe_selectivity(
            tmp_path,
            torch.float64,
            "cpu",
            catchability,
            4,
        )
