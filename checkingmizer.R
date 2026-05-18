#################
#write for python input
export_mizer_inputs_for_python <- function(params,
                                           n = params@initial_n,
                                           n_pp = params@initial_n_pp,
                                           dt = 0.1,
                                           outdir = "py_inputs",
                                           t_min = 0,
                                           t_max = 1,
                                           z0_pre = 0.6,
                                           mu_b_allometric = FALSE) {
  dir.create(outdir, showWarnings = FALSE)
  write_vec <- function(x, name) {
    write.csv(
      data.frame(value = as.numeric(x)),
      file.path(outdir, paste0(name, ".csv")),
      row.names = FALSE
    )
  }
  write_mat <- function(x, name) {
    write.csv(
      as.data.frame(as.matrix(x)),
      file.path(outdir, paste0(name, ".csv")),
      row.names = FALSE
    )
  }

  sp <- params@species_params
  get_sp <- function(name, default = NULL, required = FALSE) {
    if (name %in% names(sp)) {
      x <- sp[[name]]
      if (!all(is.na(x))) return(x)
    }

    if (required) {
      stop(sprintf("species_params is missing required column '%s'", name))
    }

    if (is.null(default)) {
      stop(sprintf("species_params is missing '%s' and no default was supplied", name))
    }

    rep(default, nrow(sp))
  }

  get_sp_vec <- function(name, default = NULL, required = FALSE) {
    as.numeric(get_sp(name, default = default, required = required))
  }

  # ----------------------------
  # Continuous analytical params
  # ----------------------------

  gamma <- get_sp_vec("gamma", required = TRUE)
  q <- get_sp_vec("q", required = TRUE)

  h <- get_sp_vec("h", required = TRUE)
  n_exp <- get_sp_vec("n", required = TRUE)

  ks <- get_sp_vec("ks", required = TRUE)
  p_exp <- get_sp_vec("p", required = TRUE)
  k_metab <- get_sp_vec("k", default = 0)

  beta <- get_sp_vec("beta", required = TRUE)
  sigma <- get_sp_vec("sigma", required = TRUE)

  w_max <- get_sp_vec("w_max", required = TRUE)

  w_mat <- if ("w_mat" %in% names(sp) && !all(is.na(sp$w_mat))) {
    as.numeric(sp$w_mat)
  } else {
    0.25 * w_max
  }

  w_repro_max <- if ("w_repro_max" %in% names(sp) && !all(is.na(sp$w_repro_max))) {
    as.numeric(sp$w_repro_max)
  } else {
    w_max
  }

  # Use supplied U if present. Otherwise infer from w_mat25 if possible. Otherwise default to 10.
  if ("U" %in% names(sp) && !all(is.na(sp$U))) {
    U <- as.numeric(sp$U)
  } else if ("w_mat25" %in% names(sp) && !all(is.na(sp$w_mat25))) {
    U <- log(3) / log(w_mat / as.numeric(sp$w_mat25))
  } else {
    U <- rep(10, nrow(sp))
  }

  if (!"z0" %in% names(sp) || any(is.na(sp$z0))) {
    stop("species_params$z0 is required when exporting background mortality.")
  }

  z0 <- as.numeric(sp$z0)

  write_vec(as.integer(mu_b_allometric), "mu_b_allometric")
  write_vec(z0, "z0")

  m_exp <- get_sp_vec("m", default = 1)

  # w_inf is needed for analytical mu_b.
  # If this is not in species_params, do not silently use w_max.
  w_inf <- get_sp_vec("w_inf", required = TRUE)

  z0_pre_vec <- rep(as.numeric(z0_pre), nrow(sp))

  write_vec(gamma, "gamma")
  write_vec(q, "q")

  write_vec(h, "h")
  write_vec(n_exp, "n_exp")

  write_vec(ks, "ks")
  write_vec(p_exp, "p_exp")
  write_vec(k_metab, "k_metab")

  write_vec(beta, "beta")
  write_vec(sigma, "sigma")

  write_vec(w_max, "w_max")
  write_vec(w_mat, "w_mat")
  write_vec(U, "U")
  write_vec(w_repro_max, "w_repro_max")
  write_vec(m_exp, "m_exp")

  write_vec(z0_pre_vec, "z0_pre")
  write_vec(w_inf, "w_inf")

  write_vec(t_min, "t_min")
  write_vec(t_max, "t_max")

  write_vec(params@w_full, "w_full")
  write_vec(params@w, "w")
  write_vec(params@dw_full, "dw_full")
  write_vec(params@dw, "dw")
  write_vec(params@w_min_idx, "w_min_idx")

  write_mat(Re(params@ft_pred_kernel_e), "ft_pred_kernel_e_real")
  write_mat(Im(params@ft_pred_kernel_e), "ft_pred_kernel_e_imag")
  write_mat(Re(params@ft_pred_kernel_p), "ft_pred_kernel_p_real")
  write_mat(Im(params@ft_pred_kernel_p), "ft_pred_kernel_p_imag")
  write_mat(as.numeric(params@ft_mask), "ft_mask")

  # safer version for ft_mask shape
  write_mat(matrix(as.numeric(params@ft_mask),
                   nrow = dim(params@ft_mask)[1],
                   ncol = dim(params@ft_mask)[2]),
            "ft_mask")

  write_mat(params@search_vol, "search_vol")
  write_mat(params@intake_max, "intake_max")
  write_vec(sp$alpha, "alpha")
  write_mat(params@metab, "metab")
  write_mat(params@psi, "psi")
  write_mat(params@mu_b, "mu_b")

  write_vec(sp$interaction_resource, "interaction_resource")
  write_mat(params@interaction, "interaction")

  write_vec(sp$erepro, "erepro")
  write_vec(sp$R_max, "r_max")

  write_vec(params@rr_pp, "rr_pp")
  write_vec(params@cc_pp, "cc_pp")

  write_mat(n, "n")
  write_vec(n_pp, "n_pp")
  write_vec(dt, "dt")

  f_mort <- matrix(0, nrow = nrow(n), ncol = ncol(n))
  write_mat(f_mort, "f_mort")

  invisible(TRUE)
}
n <- params@initial_n
n_pp <- params@initial_n_pp
dt <- 0.1

export_mizer_inputs_for_python(params_1, n, n_pp, dt)

#################
#run python code
read_py_mat <- function(name, dir = "py_steps") {
  as.matrix(read.csv(file.path(dir, paste0(name, ".csv")), check.names = FALSE))
}

py <- list(
  prey = read_py_mat("01_prey"),
  encounter = read_py_mat("02_encounter"),
  feeding_level = read_py_mat("03_feeding_level"),
  erepog = read_py_mat("04_erepog"),
  e_repro = read_py_mat("05_e_repro"),
  e_growth = read_py_mat("06_e_growth"),
  q_matrix = read_py_mat("07_q_matrix"),
  pred_rate = read_py_mat("08_pred_rate"),
  pred_mort = read_py_mat("09_pred_mort"),
  resource_mort = read_py_mat("10_resource_mort"),
  mort = read_py_mat("11_mort"),
  rdi = read_py_mat("12_rdi"),
  rdd = read_py_mat("13_rdd"),
  n_pp_new = read_py_mat("18_step_n_pp_new"),
  n_new = read_py_mat("19_step_n_new")
)

##################
#check against r

rates <- getRates(params, n = n, n_pp = n_pp, effort = 0)

pro <- project(NS_params, effort=0, t_max=0.1)
n_after <- pro@n[2,,]
n_pp_after <- pro@n_pp[2,]
#compare function
cmp <- function(r, p, name, tol = 1e-8) {
  r <- as.matrix(r)
  p <- as.matrix(p)

  if (!all(dim(r) == dim(p))) {
    cat(name, "DIM FAIL",
        "R:", paste(dim(r), collapse = "x"),
        "Python:", paste(dim(p), collapse = "x"), "\n")
    return(invisible(FALSE))
  }

  d <- abs(r - p)
  rel <- d / pmax(abs(r), 1e-12)

  cat(
    sprintf(
      "%-20s max_abs = %.6e   max_rel = %.6e\n",
      name,
      max(d, na.rm = TRUE),
      max(rel, na.rm = TRUE)
    )
  )

  invisible(max(d, na.rm = TRUE) < tol)
}

cmp(rates$encounter,      py$encounter,      "encounter",      tol = 1e-7)
cmp(rates$feeding_level,  py$feeding_level,  "feeding_level",  tol = 1e-8)
cmp(rates$e_repro,        py$e_repro,        "e_repro",        tol = 1e-8)
cmp(rates$e_growth,       py$e_growth,       "e_growth",       tol = 1e-8)
cmp(rates$pred_mort,      py$pred_mort,      "pred_mort",      tol = 1e-7)
cmp(rates$mort,           py$mort,           "mort",           tol = 1e-7)
cmp(rates$rdi,            py$rdi,            "rdi",            tol = 1e-8)
cmp(rates$rdd,            py$rdd,            "rdd",            tol = 1e-8)

cmp(n_pp_after, py$n_pp_new, "n_pp_new", tol = 1e-8)
cmp(n_after,    py$n_new,    "n_new", tol = 1e-8)

