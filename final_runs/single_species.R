#producing the single species runs

library(mizer)

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


new_params_3 <- newMultispeciesParams(NS_params@species_params[5,], kappa = 1.5e11)

plot(project(new_params_3, t_max = 40))

export_mizer_inputs_for_python(params = new_params_3,
                               t_min = 0.1,
                               t_max = 40,
                               outdir = outdir <- "validation/fixtures/pde_single_species/sp_5")


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

