# producing the single-species runs

library(mizer)

<<<<<<< HEAD
new_params_1 <- newMultispeciesParams(NS_params@species_params[1,], kappa = 1.5e11)

plot(project(new_params_1, t_max = 40))

export_mizer_inputs_for_python(params = new_params_1,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_1")

new_params_2 <- newMultispeciesParams(NS_params@species_params[3,], kappa = 1.5e11)

plot(project(new_params_2, t_max = 40))

export_mizer_inputs_for_python(params = new_params_2,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_3")
=======
# Keep the same species-specific resource carrying capacities used previously.
single_species_kappa <- c(
  rep(1.5e11, 5),
  rep(0.5e11, 3),
  rep(1e11, 4)
)

projection_long <- vector("list", nrow(NS_params@species_params))
>>>>>>> 1a2546a321bd7b7154135088d2daf354d91e9557

for (i in seq_len(nrow(NS_params@species_params))) {
  new_params <- newMultispeciesParams(
    NS_params@species_params[i, ],
    kappa = single_species_kappa[i]
  )

<<<<<<< HEAD
new_params_3 <- newMultispeciesParams(NS_params@species_params[5,], kappa = 1.5e11)

plot(project(new_params_3, t_max = 40))

export_mizer_inputs_for_python(params = new_params_3,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_5")
=======
  sim <- project(new_params, t_max = 30)
  plot(sim)

  # sim@n has dimensions time x species x weight.
  # Export the latent projection in the same long format used by
  # mizer_projection_long.csv so it can be used as truth in the viewer.
  sim_long <- as.data.frame.table(
    sim@n,
    responseName = "N",
    stringsAsFactors = FALSE
  )
  names(sim_long)[1:3] <- c("time", "species", "weight")
>>>>>>> 1a2546a321bd7b7154135088d2daf354d91e9557

  sim_long$time <- as.numeric(sim_long$time)
  sim_long$species <- as.character(sim_long$species)
  sim_long$weight <- as.numeric(sim_long$weight)
  sim_long$N <- as.numeric(sim_long$N)

<<<<<<< HEAD
new_params_4 <- newMultispeciesParams(NS_params@species_params[7,], kappa = 0.5e11)

plot(project(new_params_4, t_max = 40))

export_mizer_inputs_for_python(params = new_params_4,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_7")


new_params_5 <- newMultispeciesParams(NS_params@species_params[9,], kappa = 1e11)

plot(project(new_params_5, t_max = 40))

export_mizer_inputs_for_python(params = new_params_5,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_9")


new_params_6 <- newMultispeciesParams(NS_params@species_params[4,], kappa = 1e11)

plot(project(new_params_6, t_max = 40))

export_mizer_inputs_for_python(params = new_params_6,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_4")
=======
  projection_long[[i]] <- sim_long[, c("time", "species", "weight", "N")]
}

single_species_projection_long <- do.call(rbind, projection_long)
row.names(single_species_projection_long) <- NULL
>>>>>>> 1a2546a321bd7b7154135088d2daf354d91e9557

write.csv(
  single_species_projection_long,
  "final_runs/single_species_projection_long.csv",
  row.names = FALSE
)
