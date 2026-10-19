# species runs already defined in the original single_species.R
run_spec <- data.frame(
  species_idx = c(3L, 7L, 11L),
  species     = c("N.pout", "Sole", "Cod"),
  kappa       = c(1.5e11, 0.5e11, 1e11),
  fixture     = c("sp_3", "sp_7", "sp_11")
)

projection_long <- vector("list", nrow(run_spec))

for (i in seq_len(nrow(run_spec))) {

  sp_idx <- run_spec$species_idx[i]

  new_params <- newMultispeciesParams(
    NS_params@species_params[sp_idx, ],
    kappa = run_spec$kappa[i]
  )

  fixture <- paste0("validation/fixtures/pde_single_species/",run_spec[i,4])

  n_pp_csv <- read.csv(
    file.path(fixture, "n_pp.csv")
  )$value

  initialNResource(new_params) <- n_pp_csv
  resource_dynamics(new_params) <- "resource_constant"

  sim <- project(new_params, t_max = 40, t_save = 0.1)

  sim_long <- as.data.frame.table(
    sim@n,
    responseName = "N",
    stringsAsFactors = FALSE
  )

  names(sim_long)[1:3] <- c("time", "species", "weight")

  # export_mizer_inputs_for_python(new_params,
  #                                outdir = paste0("validation/fixtures/pde_single_species/",run_spec[i,4])
  #                                )

  sim_long$time   <- as.numeric(sim_long$time)
  sim_long$weight <- as.numeric(sim_long$weight)
  sim_long$species <- as.character(sim_long$species)
  sim_long$N      <- as.numeric(sim_long$N)

  projection_long[[i]] <- sim_long[, c("weight", "time", "species", "N")]
}

single_species_projection_long <- do.call(rbind, projection_long)
row.names(single_species_projection_long) <- NULL

write.csv(
  single_species_projection_long,
  "single_species_projection_long.csv",
  row.names = FALSE
)
