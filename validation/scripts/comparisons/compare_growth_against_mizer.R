#checking that the growth function works as intended.

#Exporting all the things we need to check against.
export_known_encounter_inputs <- function(params,
                                          n = params@initial_n,
                                          n_pp = params@initial_n_pp,
                                          outdir = "validation/fixtures/pde_single_species") {
  dir.create(outdir, showWarnings = FALSE)

  write_mat <- function(x, name) {
    write.csv(
      as.data.frame(as.matrix(x)),
      file.path(outdir, paste0(name, ".csv")),
      row.names = FALSE
    )
  }

  no_sp <- nrow(params@species_params)
  no_w_full <- length(params@w_full)
  no_w <- length(params@w)

  fish_cols <- (no_w_full - no_w + 1):no_w_full

  prey_full <- matrix(
    n_pp,
    nrow = no_sp,
    ncol = no_w_full,
    byrow = TRUE
  )

  prey_full <- sweep(
    prey_full,
    1,
    params@species_params$interaction_resource,
    "*"
  )

  prey_full[, fish_cols] <- prey_full[, fish_cols] +
    params@interaction %*% n

  prey_known_full <- sweep(
    prey_full,
    2,
    params@w_full * params@dw_full,
    "*"
  )

  encounter <- mizerEncounter(
    params,
    n = n,
    n_pp = n_pp,
    n_other = params@initial_n_other,
    t = 0.1
  )

  feeding_level <- mizerFeedingLevel(
    params,
    n = n,
    n_pp = n_pp,
    n_other = params@initial_n_other,
    t = 0.1,
    encounter = encounter
  )

  e <- mizerEReproAndGrowth(
    params,
    n = n,
    n_pp = n_pp,
    n_other = params@initial_n_other,
    t = 0.1,
    encounter = encounter,
    feeding_level = feeding_level
  )

  e_repro <- mizerERepro(
    params,
    n = n,
    n_pp = n_pp,
    n_other = params@initial_n_other,
    t = 0.1,
    e = e
  )

  e_growth <- mizerEGrowth(
    params,
    n = n,
    n_pp = n_pp,
    n_other = params@initial_n_other,
    t = 0.1,
    e = e,
    e_repro = e_repro
  )

  write_mat(prey_known_full, "prey_known_full")
  write_mat(encounter, "encounter_mizer")
  write_mat(feeding_level, "feeding_level_mizer")
  write_mat(e, "e_mizer")
  write_mat(e_repro, "e_repro_mizer")
  write_mat(e_growth, "e_growth_mizer")
}

dense_kernel <- getPredKernel(NS_params)

params_nofft <- setPredKernel(
  NS_params,
  pred_kernel = dense_kernel
)
export_known_encounter_inputs(params_nofft)

##checking


read_mat <- function(path) {
  as.matrix(read.csv(path, check.names = FALSE))
}

compare_mat <- function(a, b, eps = 1e-14) {
  stopifnot(identical(dim(a), dim(b)))

  diff <- a - b
  abs_err <- abs(diff)
  rel_err <- abs_err / pmax(abs(a), abs(b), eps)

  list(
    diff_range = range(diff, na.rm = TRUE),
    abs_quantiles = quantile(
      abs_err,
      probs = c(0, 0.5, 0.9, 0.99, 1),
      na.rm = TRUE
    ),
    rel_quantiles = quantile(
      rel_err,
      probs = c(0, 0.5, 0.9, 0.99, 1),
      na.rm = TRUE
    )
  )
}

direct_dir <- "py_known_encounter_direct"
mizer_dir <- "validation/fixtures/pde_single_species"

quantities <- c(
  "encounter",
  "feeding_level",
  "e",
  "e_repro",
  "e_growth"
)

results <- lapply(quantities, function(q) {
  direct <- read_mat(file.path(direct_dir, paste0(q, "_direct.csv")))
  mizer <- read_mat(file.path(mizer_dir, paste0(q, "_mizer.csv")))

  compare_mat(direct, mizer)
})

names(results) <- quantities

results
