#checking the mizer_operator function, which will in turn be checked against continuous functio

library(mizer)

rate_funcs <- lapply(NS_params@rates_funcs, get)

encounter <- rate_funcs$Encounter(NS_params, NS_params@initial_n, NS_params@initial_n_pp,
                                  NS_params@initial_n_other, 0.1)

feeding_level <- rate_funcs$FeedingLevel(NS_params, NS_params@initial_n, NS_params@initial_n_pp,
                                         NS_params@initial_n_other, 0.1, encounter)

pred_rate <- rate_funcs$PredRate(NS_params, NS_params@initial_n, NS_params@initial_n_pp,
                                 NS_params@initial_n_other, 0.1, feeding_level)
pred_mort <- rate_funcs$PredMort(NS_params, NS_params@initial_n, NS_params@initial_n_pp,
                                 NS_params@initial_n_other, 0.1, pred_rate)


read_py_mat <- function(name, dir = "validation/outputs/mizer_ops") {
  as.matrix(
    read.csv(
      file.path(dir, paste0(name, ".csv")),
      check.names = FALSE
    )
  )
}

py_ops <- list(
  encounter = read_py_mat("encounter"),
  feeding_level = read_py_mat("feeding_level"),
  erepog = read_py_mat("erepog"),
  e_repro = read_py_mat("e_repro"),
  e_growth = read_py_mat("e_growth"),
  pred_mort = read_py_mat("pred_mort"),
  resource_mort = read_py_mat("resource_mort"),
  mort = read_py_mat("mort"),
  rdi = read_py_mat("rdi"),
  rdd = read_py_mat("rdd")
)

sum(py_ops$encounter-encounter)
sum(py_ops$pred_mort-pred_mort)
