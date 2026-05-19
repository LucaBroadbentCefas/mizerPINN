#python -m validation.scripts.check_growth_derivatives_finite_difference

summary <- read.csv("validation/fixtures/growth_derivative_fd/finite_difference_summary.csv")

summary[order(summary$quantity, summary$eps_scale), ]

#This is checking the difference between each step in the growth rate or a given w + some small amount
#This approximates the derivative, to see if its in the right area.

#It looks right, the derivatives are coded correctly.
