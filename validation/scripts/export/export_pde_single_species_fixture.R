outdir <- "validation/fixtures/pde_single_species"

export_mizer_inputs_for_python(
  params = params_1,
  n = params_1@initial_n,
  n_pp = params_1@initial_n_pp,
  dt = 0.1,
  outdir = outdir
)

write_vec <- function(x, name) {
  write.csv(
    data.frame(value = as.numeric(x)),
    file.path(outdir, paste0(name, ".csv")),
    row.names = FALSE
  )
}

write_vec(0, "t_min")
write_vec(40, "t_max")


params1 <- newMultispeciesParams(NS_params@species_params[1,])

projection@n
