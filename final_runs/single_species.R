# producing the single-species runs

library(mizer)

# Keep the same species-specific resource carrying capacities used previously.
single_species_kappa <- c(
  rep(1.5e11, 5),
  rep(0.5e11, 3),
  rep(1e11, 4)
)

projection_long <- vector("list", nrow(NS_params@species_params))

for (i in seq_len(nrow(NS_params@species_params))) {
  new_params <- newMultispeciesParams(
    NS_params@species_params[i, ],
    kappa = single_species_kappa[i]
  )

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

  sim_long$time <- as.numeric(sim_long$time)
  sim_long$species <- as.character(sim_long$species)
  sim_long$weight <- as.numeric(sim_long$weight)
  sim_long$N <- as.numeric(sim_long$N)

  projection_long[[i]] <- sim_long[, c("time", "species", "weight", "N")]
}

single_species_projection_long <- do.call(rbind, projection_long)
row.names(single_species_projection_long) <- NULL

write.csv(
  single_species_projection_long,
  "final_runs/single_species_projection_long.csv",
  row.names = FALSE
)
