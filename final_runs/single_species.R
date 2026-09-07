#producing the single species runs

library(mizer)

new_params <- newMultispeciesParams(NS_params@species_params[1,], kappa = 1.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[2,], kappa = 1.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[3,], kappa = 1.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[4,], kappa = 1.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[5,], kappa = 1.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[6,], kappa = 0.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[7,], kappa = 0.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[8,], kappa = 0.5e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[9,], kappa = 1e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[10,], kappa = 1e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[11,], kappa = 1e11)

plot(project(new_params, t_max = 30))


new_params <- newMultispeciesParams(NS_params@species_params[12,], kappa = 1e11)

plot(project(new_params, t_max = 30))

