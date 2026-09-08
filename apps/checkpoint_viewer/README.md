# PINNmizer checkpoint viewer

This app answers a specific stopping-iteration question: **what would the fitted fields and observation diagnostics have looked like if training had stopped at an earlier saved checkpoint?**

It reuses the plotting functions from `apps/hpc_viewer`; it does not implement a second set of PINN plots.

## Why checkpoint materialisation is separate

A `model_step_<N>.pt` checkpoint contains neural-network weights and optional inverse-parameter state, but it does not contain the fixed-grid abundance/residual arrays or observation predictions used by the viewer. The normal HPC viewer is deliberately read-only and does not load PyTorch models.

Therefore checkpoint weights are evaluated once, outside Streamlit, with the original saved `run_command.txt`, zero optimisation steps, and `--hpc`. This produces the same CSV/JSON/NPZ diagnostic format as a final run without changing the checkpoint weights.

## 1. Materialise checkpoints

All saved checkpoints for one run:

```bash
python -m scripts.materialize_checkpoint_outputs runs/pde_multispecies/<RUN> --device cpu
```

Selected steps only:

```bash
python -m scripts.materialize_checkpoint_outputs \
  runs/pde_multispecies/<RUN> \
  --steps 4000 8000 12000 \
  --device cpu
```

The utility finds both checkpoint layouts currently used in the repository:

```text
<run>/model_step_<N>.pt
<run>/checkpoints/model_step_<N>.pt
```

Snapshots are written under `checkpoint_views/` by default, preserving the source run path. The generated duplicate `model_final.pt` is removed because the original checkpoint is the source of truth.

For each snapshot, source `*history.csv` files are truncated to `step <= checkpoint_step`. Thus the training page represents the trajectory that would have been visible if the run had stopped at that checkpoint rather than the zero-step replay used to generate diagnostic fields.

Use `--dry-run` to inspect replay commands and `--overwrite` to regenerate existing snapshots.

## 2. Run the viewer

From repository root:

```bash
streamlit run apps/checkpoint_viewer/streamlit_app.py
```

The sidebar provides:

- source-run selection;
- a checkpoint/stopping-step slider;
- checkpoint multiselect for comparison;
- the same heatmap/marker/mizer controls as the HPC viewer.

The pages reuse the existing field, training, data, run-comparison, mizer-comparison, and file/config views.

## Interpretation

A checkpoint snapshot is an **evaluation of the exact saved model state at step N**. It does not rerun the first N optimisation steps and does not continue training. For model outputs this is exactly the appropriate counterfactual for “what would I have obtained if I stopped at N?”, provided that the checkpoint was saved at N.

The method cannot reconstruct unsaved steps. If checkpoints exist only at 4,000-step intervals, for example, it cannot show the exact 6,000-step model without a 6,000-step checkpoint.
