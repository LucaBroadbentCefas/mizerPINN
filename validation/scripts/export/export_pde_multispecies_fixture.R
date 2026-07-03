#!/usr/bin/env Rscript

# Export a two-species PDE fixture for the Python multi-species PINN smoke tests.
# The shared export helper is currently defined in the legacy mizer-step export
# script; parse only that function to avoid running legacy side effects.

library(mizer)

legacy_lines <- readLines("validation/scripts/legacy/check_mizer_step_export.R")
end <- grep("^n <- params@initial_n", legacy_lines)[1] - 1
eval(parse(text = legacy_lines[seq_len(end)]))

params <- NS_params
species_params <- params@species_params[c(1,11), , drop = FALSE]
params_2 <- newMultispeciesParams(species_params = species_params)
params_2 <- steady(params_2)

sim <- project(params_2, t_max = 40, dt = 0.1)
n <- sim@n[1, , , drop = FALSE]
n <- n[1, , ]
n_pp <- sim@n_pp[1, ]
dt <- 0.1

outdir <- "validation/fixtures/pde_multispecies"
export_mizer_inputs_for_python(params_2, n, n_pp, dt, outdir = outdir)
write.csv(data.frame(value = 0), file.path(outdir, "t_min.csv"), row.names = FALSE)
write.csv(data.frame(value = 40), file.path(outdir, "t_max.csv"), row.names = FALSE)

pro <- project(params_2, effort = 0, t_max = 40, dt = 0.1)

tt <- seq(0, 40, length.out = dim(pro@n)[1])
ii <- unique(round(seq(1, length(tt), length.out = 6)))

w <- params_2@w
x <- log(w)

species_names <- rownames(params_2@species_params)
if (is.null(species_names)) {
  species_names <- paste0("species_", seq_len(dim(pro@n)[2]) - 1)
}

df_r <- bind_rows(lapply(seq_along(ii), function(j) {
  bind_rows(lapply(seq_along(species_names), function(s) {
    data.frame(
      time_num = tt[ii[j]],
      time_label = sprintf("%.2f", tt[ii[j]]),
      species_idx = s - 1,
      species = species_names[s],
      x = x,
      log10_N = log10(pmax(pro@n[ii[j], s, ], .Machine$double.xmin)),
      source = "R/mizer"
    )
  }))
}))

p_true <- ggplot(df_r, aes(x = x, y = log10_N, colour = time_label, group = time_label)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ species, scales = "free_y") +
  labs(
    x = "log weight",
    y = "log10(N)",
    colour = "time",
    title = "R/mizer multispecies log10(N) profiles through time"
  ) +
  theme_bw()

print(p_true)

outdir <- "validation/fixtures/pde_multispecies"

