library(mizer)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)

# sim must be a MizerSim object containing:
#   sim@n: time x species x weight abundance density
#   sim@params@w: weight grid
#   sim@params@dw: weight-bin widths
#
# For meaningful annual catch totals, generate sim with subannual saves:
#
# sim <- project(
#   params,
#   t_max = 20,
#   t_save = 0.1,
#   progress_bar = FALSE
# )

gears <- gear_params(mizer::NS_params)

survey_gears <- data.frame(
  gear = rep("survey", 12),
  species = NS_params@species_params$species,
  catchability = rep(1, 12),
  sel_func = rep("knife_edge", 12),
  knife_edge_size = rep(15,12)
)

gear_params(NS_params) <- rbind(gears, survey_gears)
effort <- initial_effort(NS_params)
effort[5] <- 0.02
effort[2] <- 0.6
effort[4] <- 3
initial_effort(NS_params) <- effort

sim <- project(NS_params, t_max = 40, t_save = 0.1, effort  = effort)

make_observation_data <- function(sim, survey_gear_name = "survey") {

  n <- sim@n
  w <- sim@params@w
  dw <- sim@params@dw

  times <- as.numeric(dimnames(n)$time)
  species <- dimnames(n)$sp

  # ------------------------------------------------------------
  # Catch observations
  # getYieldGear(sim) is time x gear x species, in g / year.
  # Annual catch = mean rate over the year * 1 year.
  # ------------------------------------------------------------

  yield <- getYieldGear(sim)

  yield_df <- as.data.frame.table(
    yield,
    responseName = "yield_rate"
  )

  yield_df$time <- as.numeric(as.character(yield_df$time))
  yield_df$year <- floor(yield_df$time)

    catch_obs <- yield_df |>
    dplyr::filter(time >= year, time < year + 1) |>
    dplyr::group_by(year, gear, sp) |>
    dplyr::summarise(
      value = mean(yield_rate, na.rm = TRUE) * 1,
      .groups = "drop"
    ) |>
    dplyr::filter(
      is.finite(value),
      value > 0
    ) |>
    dplyr::rename(species = sp) |>
    dplyr::mutate(
      observation = "catch_biomass"
    ) |>
    dplyr::select(year, species, observation, gear, value)


  gear_levels <- dimnames(getYieldGear(sim))$gear
  species_levels <- dimnames(getYieldGear(sim))$sp

    catch_obs%>%
    dplyr::arrange(year, species, observation, gear)%>%
      dplyr::mutate(
        t_start = year,
        t_end = year + 1,
        species_idx = match(as.character(species), species_levels) - 1L,
        gear_idx = match(as.character(gear), gear_levels) - 1L,
        obs_type = "catch_gear",
        cv = 0.3
      )|>
  dplyr::select(
    obs_type,
    species_idx,
    gear_idx,
    t_start,
    t_end,
    value,
    cv
  )
}

observation_data <- make_observation_data(
  sim = sim
)

write.csv(
  observation_data,
  "observations.csv",
  row.names = FALSE
)


export_mizer_inputs_for_python(NS_params,  outdir = "validation/fixtures/pde_multispecies")
