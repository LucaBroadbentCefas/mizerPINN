##Checking the continuous functions - pred_mort - against what mizer gives you

pred_kernels <- (getPredKernel(NS_params))

debug_dir <- "validation/fixtures/pred_mort_debug"

read_py_vec <- function(name, dir = debug_dir) {
  read.csv(
    file.path(dir, paste0(name, ".csv")),
    check.names = FALSE
  )[[1]]
}

read_py_mat <- function(name, dir = debug_dir) {
  as.matrix(
    read.csv(
      file.path(dir, paste0(name, ".csv")),
      check.names = FALSE
    )
  )
}

w <- read_py_vec("w")
dw <- read_py_vec("dw")

gamma_grid <- read_py_mat("gamma_grid")
search_vol_grid <- read_py_mat("search_vol_grid")
gamma_minus_search_vol <- read_py_mat("gamma_minus_search_vol")

feeding_grid <- read_py_mat("feeding_grid")
q_grid <- read_py_mat("q_grid")

pred_rate_direct <- read_py_mat("pred_rate_direct")
pred_mort_direct <- read_py_mat("pred_mort_direct")

phi_pred <- lapply(0:11, function(j) {
  read_py_mat(sprintf("phi_pred_species_%02d", j))
})

dphi_dw_pred <- lapply(0:11, function(j) {
  read_py_mat(sprintf("dphi_dw_pred_species_%02d", j))
})
names(phi_pred) <- sprintf("species_%02d", 0:11)
names(dphi_dw_pred) <- sprintf("species_%02d", 0:11)

#phi_pred has the predation kernels from python. these are 100 100, there is no resource.
#n_pred, n_prey

#pred_kernels is mizer generated, it has the resource spectrum

sp1 <- (pred_kernels[1,,127:226])

py1 <- phi_pred[[1]]

#need to check its within machine tolerance.
range(py1-sp1) # i think it is okay, or close enough


phi_array <- array(dim = dim(pred_kernels[,,127:226]))
for(i in 1:length(phi_pred)){phi_array[i,,] <- phi_pred[[i]]}

range((phi_array-pred_kernels[,,127:226])/(phi_array+1e-14))

#it is correct now


##checking pred_rate / pred_mort
encounter <- rate_funcs$Encounter(NS_params, NS_params@initial_n, NS_params@initial_n_pp, NS_params@initial_n_other, 0.1)
feeding_level <- rate_funcs$FeedingLevel(NS_params, NS_params@initial_n,NS_params@initial_n_pp, NS_params@initial_n_other, 0.1,encounter)
pred_rate <- rate_funcs$PredRate(NS_params, NS_params@initial_n,NS_params@initial_n_pp, NS_params@initial_n_other, 0.1, feeding_level)
pred_mort <- rate_funcs$PredMort(NS_params, NS_params@initial_n,NS_params@initial_n_pp, NS_params@initial_n_other, 0.1, pred_rate)

#checking
range((pred_mort-pred_mort_direct)/(pred_mort+1e-14))
#not very good.
range((pred_rate[,127:226]-pred_rate_direct)/(pred_rate[,127:226]+1e-14))

#is it the feeding_level?
dim(feeding_level);dim(feeding_grid)
range((feeding_level-feeding_grid)/feeding_level)

#feeding_level isnt good.

#checking pred again
q_miz <- (1-feeding_level)*NS_params@initial_n*NS_params@search_vol*NS_params@dw
q_py <- q_grid*dw

range(q_miz-q_py)
range((q_miz-q_py)/(q_miz+1e-14))
#not good.


#Lets try load Q into python and run the pred_Rate code then,to see if we get he right answer

q_known_fish <- sweep(
  (1 - feeding_level) * NS_params@initial_n * NS_params@search_vol,
  2,
  NS_params@dw,
  "*"
)

write.csv(q_known_fish, "validation/fixtures/py_inputs/q_known_fish.csv", row.names = FALSE)


read_py_mat <- function(name, dir = "validation/outputs/py_known_q_direct") {
  as.matrix(read.csv(file.path(dir, paste0(name, ".csv")), check.names = FALSE))
}

pred_rate_direct_known_q <- read_py_mat("pred_rate_direct_known_q")
pred_mort_direct_known_q <- read_py_mat("pred_mort_direct_known_q")

fish_cols <- 127:226

range(pred_rate[, fish_cols] - pred_rate_direct_known_q, na.rm = TRUE)
range((pred_rate[, fish_cols] - pred_rate_direct_known_q) /
        pmax(abs(pred_rate[, fish_cols]), 1e-14), na.rm = TRUE)

range(pred_mort - pred_mort_direct_known_q, na.rm = TRUE)
range((pred_mort - pred_mort_direct_known_q) /
        pmax(abs(pred_mort), 1e-14), na.rm = TRUE)

##checking non-fft rates
nofft <- function (params, n, n_pp, n_other, t, feeding_level, ...) {
  no_sp <- dim(params@interaction)[1]
  no_w <- length(params@w)
  no_w_full <- length(params@w_full)
    n_total_in_size_bins <- sweep(n, 2, params@dw, "*", check.margin = FALSE)
    pred_rate <- sweep(getPredKernel(params), c(1, 2), (1 -feeding_level) * params@search_vol * n_total_in_size_bins,
                       "*", check.margin = FALSE)
    pred_rate <- colSums(aperm(pred_rate, c(2, 1, 3)), dims = 1)
    return(pred_rate)

}
pred_rate_nofft <- nofft(NS_params, NS_params@initial_n,NS_params@initial_n_pp, NS_params@initial_n_other, 0.1, feeding_level)

range((pred_rate_nofft-pred_rate)/(pred_rate_nofft+1e-14))
#the pred calculated here is not even close? So the mizer calculation fft doesnt get really the same


params_fft <- NS_params

pred_rate_fft <- getPredRate(
  params_fft,
  n = params_fft@initial_n,
  n_pp = params_fft@initial_n_pp,
  n_other = params_fft@initial_n_other,
  t = 0.1
)

dense_kernel <- getPredKernel(params_fft)

params_nofft <- setPredKernel(
  params_fft,
  pred_kernel = dense_kernel
)

pred_rate_nofft <- getPredRate(
  params_nofft,
  n = params_nofft@initial_n,
  n_pp = params_nofft@initial_n_pp,
  n_other = params_nofft@initial_n_other,
  t = 0.1
)

range((pred_rate_nofft-pred_rate_fft)/(pred_rate_nofft+1e-14))
range((pred_rate_nofft-pred_rate_fft))

#these are not the same.

#So I need to check my python answer vs this non-fft?

range(pred_rate_direct-pred_rate_nofft[,127:226])

#matches to within machine precision.

#So I need to check mizer against a non-fft version of itself?

#Anyway, it looks fine, so the pred rate is calculated correctly in the non-fft case.
#Need to watch out for using non-fft answers in training, even though its not that serious.

#now using check_biology should probably use a non-fft implementation and take it from R.
# This would give expected outcome.
#Currently, its not that bad, and it is probably good enough, but could be covering up a bug

# (mizer-torch) C:\Users\LB19\OneDrive - CEFAS\Work\PINNs>python -m validation.scripts.check_biology
# encounter            max_abs=2.337496e-05   mean_abs=2.572296e-07   max_rel=4.063562e-05   mean_rel=2.028459e-07
# feeding              max_abs=9.559656e-06   mean_abs=4.770491e-08   max_rel=1.538254e-05   mean_rel=7.672579e-08
# e_growth             max_abs=2.009711e-06   mean_abs=2.208617e-08   max_rel=1.818182e-02   mean_rel=1.528809e-05
# pred_mort            max_abs=6.896374e-09   mean_abs=1.397026e-10   max_rel=1.166861e-08   mean_rel=2.089281e-10
# total_mort           max_abs=6.896374e-09   mean_abs=1.397026e-10   max_rel=1.017920e-08   mean_rel=1.766583e-10

