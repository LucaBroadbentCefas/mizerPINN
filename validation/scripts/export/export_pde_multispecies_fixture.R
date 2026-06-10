#!/usr/bin/env Rscript

# Export a two-species PDE fixture for the Python multi-species PINN smoke tests.
# The shared export helper is currently defined in the legacy mizer-step export
# script; parse only that function to avoid running legacy side effects.

library(mizer)

legacy_lines <- readLines("validation/scripts/legacy/check_mizer_step_export.R")
end <- grep("^n <- params@initial_n", legacy_lines)[1] - 1
eval(parse(text = legacy_lines[seq_len(end)]))

params <- NS_params
species_params <- params@species_params[seq_len(2), , drop = FALSE]
params_2 <- newMultispeciesParams(species_params = species_params)
params_2 <- steady(params_2)

sim <- project(params_2, t_max = 1, dt = params_2@time_step)
n <- sim@n[1, , , drop = FALSE]
n <- n[1, , ]
n_pp <- sim@n_pp[1, ]
dt <- params_2@time_step

outdir <- "validation/fixtures/pde_multispecies"
export_mizer_inputs_for_python(params_2, n, n_pp, dt, outdir = outdir)
write.csv(data.frame(value = 0), file.path(outdir, "t_min.csv"), row.names = FALSE)
write.csv(data.frame(value = dt), file.path(outdir, "t_max.csv"), row.names = FALSE)
