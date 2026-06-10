outdir <- "validation/fixtures/pde_single_species"

library(mizer)


params_1 <- newMultispeciesParams(NS_params@species_params[1,], kappa = 1.02e11)

params_1 <- steady(params_1)
plot(project(params_1))

export_mizer_inputs_for_python(
  params = params_1,
  n = params_1@initial_n,
  n_pp = params_1@initial_n_pp,
  dt = 0.1,
  outdir = outdir,
  t_max = 40
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

#this is from getRDD - but should it have more w bins? It would be more accurate ?
R_const <- 3711296885

#I now have to calculate g0, then calculate it.

sp <- species_params(params_1)
sp$constant_reproduction <- R_const
species_params(params_1) <- sp
params_1 <- setReproduction(params_1, RDD = "constantRDD")

pro <- project(params_1, effort = 0, t_max = 40, dt = 0.1)

library(ggplot2)

tt <- seq(0, 40, length.out = dim(pro@n)[1])
ii <- unique(round(seq(1, length(tt), length.out = 6)))

w <- params_1@w
x <- log(w)

z <- log10(pmax(pro@n[ii, 1, , drop = FALSE][, 1, ], .Machine$double.xmin))

df <- data.frame(
  time = rep(sprintf("%.2f", tt[ii]), each = length(w)),
  x = rep(x, times = length(ii)),
  log10_N = as.vector(t(z))
)

ggplot(df, aes(x = x, y = log10_N, colour = time, group = time)) +
  geom_line(linewidth = 0.8) +
  labs(
    x = "log weight",
    y = "log10(N)",
    colour = "time",
    title = "mizer log10(N) profiles through time"
  ) +
  theme_bw

##plotting the other stuff on it.

library(readr)
df_pinn_all <- read_csv("runs/pde_only_single_species/20260609_134736/fixed_grid_diagnostics/fixed_grid_fields.csv")

df_r <- data.frame(
  time_num = rep(tt[ii], each = length(w)),
  x = rep(x, times = length(ii)),
  log10_N = as.vector(t(z)),
  source = "R/mizer"
)

target_times <- tt[ii]

nearest_pinn_times <- sapply(target_times, function(t0) {
  df_pinn_all$t_eval[which.min(abs(df_pinn_all$t_eval - t0))]
})

df_pinn <- df_pinn_all %>%
  filter(t_eval %in% nearest_pinn_times) %>%
  transmute(
    time_num = t_eval,
    x = x_eval,
    log10_N = log10_N,
    source = "PINN"
  )

# Use common labels so R and PINN lines share colour by comparable time.
# This maps each PINN time to the nearest selected R time label.
df_pinn <- df_pinn %>%
  rowwise() %>%
  mutate(time_label = sprintf("%.2f", target_times[which.min(abs(target_times - time_num))])) %>%
  ungroup()

df_r <- df_r %>%
  mutate(time_label = sprintf("%.2f", time_num))

df_plot <- bind_rows(df_r, df_pinn)

ggplot(df_plot, aes(x = x, y = log10_N, colour = time_label, linetype = source)) +
  geom_line(linewidth = 0.8) +
  labs(
    x = "log weight",
    y = "log10(N)",
    colour = "time",
    linetype = "source",
    title = "R/mizer and PINN log10(N) profiles through time"
  ) + theme_linedraw()

