# PINNmizer checkpoint viewer

This app answers a specific stopping-iteration question: **what would the fitted fields and observation diagnostics have looked like if training had stopped at an earlier saved checkpoint?**

It reuses the plotting functions from `apps/hpc_viewer`; it does not implement a second set of PINN plots.

## Run the app

From the repository root:

```bash
streamlit run apps/checkpoint_viewer/streamlit_app.py
```

No run paths need to be typed on the command line.

The app workflow is:

1. choose a run folder, for example `runs/pde_multispecies`, `runs/pde_only_single_species`, or `All runs`;
2. choose the run you want to inspect from the discovered run list;
3. under **Prepare checkpoint views**, select one or more discovered runs;
4. for a single selected run, either generate all saved checkpoints or choose particular checkpoint steps;
5. choose CPU, original run device, or CUDA;
6. click **Generate checkpoint views**;
7. use the checkpoint/stopping-step slider and checkpoint comparison selector to inspect the generated states.

The app then provides:

- checkpoint field/residual plots;
- training history truncated to the chosen stopping step;
- data diagnostics at that checkpoint;
- within-run checkpoint comparison;
- mizer comparison;
- file/config inspection.

## What “Generate checkpoint views” does

A `model_step_<N>.pt` checkpoint contains neural-network weights and optional inverse-parameter state, but it does not contain the fixed-grid abundance/residual arrays or observation predictions used by the viewer.

When you press **Generate checkpoint views**, the app runs the checkpoint materialiser locally. For each selected checkpoint it:

- replays the source run's saved `run_command.txt`;
- loads the exact saved checkpoint;
- forces zero optimisation steps;
- enables HPC diagnostic output;
- moves the generated CSV/JSON/NPZ files under `checkpoint_views/`;
- truncates copied training histories to `step <= checkpoint_step`;
- removes the replay-generated duplicate `model_final.pt` because the original checkpoint remains the source of truth.

This is model evaluation, not retraining.

The app discovers both checkpoint layouts currently used by the repository:

```text
<run>/model_step_<N>.pt
<run>/checkpoints/model_step_<N>.pt
```

Generated checkpoint outputs are stored under the ignored `checkpoint_views/` directory rather than `runs/`, so they are not confused with independently trained runs.

## Optional command-line use

The materialiser remains available as a standalone utility for batch/HPC use:

```bash
python -m scripts.materialize_checkpoint_outputs runs/pde_multispecies/<RUN> --device cpu
```

or for selected steps:

```bash
python -m scripts.materialize_checkpoint_outputs \
  runs/pde_multispecies/<RUN> \
  --steps 4000 8000 12000 \
  --device cpu
```

This command-line route is optional; the normal interactive workflow is entirely through the Streamlit app.

## Interpretation

A checkpoint view is an **evaluation of the exact saved model state at step N**. It does not rerun the first N optimisation steps and does not continue training. For model outputs this is the appropriate counterfactual for “what would I have obtained if I stopped at N?”, provided that the checkpoint was saved at N.

The method cannot reconstruct unsaved steps. If checkpoints exist only at 4,000-step intervals, for example, it cannot show the exact 6,000-step model without a 6,000-step checkpoint.
