# PINNs

PyTorch PINN code for a mizer-style marine size-spectrum PDE model.

This repository now separates the codebase by scientific and implementation role. The split is deliberately conservative: new thematic modules have been added first, while legacy modules remain available so existing validation commands do not break during the refactor.

## Main layout

```text
PINNmizer/
  biology/       # biological operators: kernels, encounter, growth, mortality, recruitment
  pinn/          # neural-network evaluation, autograd derivatives, PDE state, residuals, losses
  training/      # configuration, training-loop utilities, checkpointing, output helpers
  diagnostics/   # fixed-grid diagnostics and plotting helpers
  io.py          # CSV input loading from exported mizer/R inputs
  params.py      # parameter dataclass, grids, scaling helpers

scripts/         # runnable experiment entry points
validation/scripts # validation-only scripts
validation/fixtures # validation fixtures/data
```

## Conceptual boundaries

### Biology

Biology modules compute mizer-style biological quantities from tensors that already represent spectra and parameters. They should not evaluate the neural network or assemble losses.

Important examples:

- `PINNmizer.biology.kernels`: predation kernel and kernel derivative.
- `PINNmizer.biology.encounter`: encounter-rate calculation.
- `PINNmizer.biology.growth`: feeding, metabolism, reproduction allocation, growth, and `dg_dw`.
- `PINNmizer.biology.mortality`: background and predation mortality.
- `PINNmizer.biology.recruitment`: recruitment flux used by the boundary loss.

### PINN/PDE

PINN modules handle the neural approximation and PDE residual. The network input columns remain:

```text
[x_scaled, t_scaled]
```

The network output remains `log_N`, not `N`.

Physical coordinates and scaled coordinates must stay distinct:

```text
x = log(w)
x_scaled = (x - x_min) / (x_max - x_min)
t_scaled = (t - t_min) / (t_max - t_min)
```

Autograd derivatives are taken with respect to scaled network inputs and then converted to physical derivatives. Biological derivatives such as `dg_dw` remain manually calculated in the biology path.

### Training

Training modules are for infrastructure: configuration, optimisation loops, checkpointing, adaptive weighting, and outputs. Training code should call the PDE and biology modules rather than implementing mathematical terms inline.

### Diagnostics

Diagnostics are read-only checks. Fixed-grid mizer-style checks should remain stable validation baselines.

## Current migration state

This PR introduces the clearer structure without deleting existing entry points. Some new modules currently re-export functions from older files. That is intentional: it gives the project a navigable architecture before the higher-risk step of physically moving every implementation.

Existing commands should continue to work. A script wrapper has also been added:

```bash
python -m scripts.train_pde_only_single_species
```

The existing command remains valid during migration:

```bash
```

## Refactor rule for future changes

When moving code, preserve behaviour first. Do not change mathematical equations, biological operators, loss definitions, tensor shapes, or training defaults in the same commit as a file move.
