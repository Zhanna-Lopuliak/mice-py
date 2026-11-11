# L1 fixture for R's BFGS minimiser (vmmin in src/appl/optim.c).
#
#   l1_vmmin.json   optim(par, fn, gr, method = "BFGS", control) on small
#                   functions: the parameters, the value, the numbers of
#                   function and gradient evaluations, the convergence code
#
# Python target: imputation.fitting.vmmin. nnet::multinom (polyreg) and
# MASS::polr (polr) minimise with the same C function, so this fixture pins
# the optimiser independently of the models.
#
# The functions use elementwise arithmetic and sum() only, so that the
# Python side can evaluate them in the same order: sum() adds left to right
# in double precision on this machine (R's long double is a double on
# arm64 macOS). The data of the logistic likelihood are built from integer
# arithmetic and stored.

source(file.path("tests", "reference", "r", "_common.R"))

# ---------------------------------------------------------------------------
# Functions and their gradients
# ---------------------------------------------------------------------------

# A convex quadratic with a cross term, minimum 0 at (1, -2, 0.5)
quad_center <- c(1, -2, 0.5)
quad_scale <- c(1, 10, 0.1)
quadratic <- list(
  fn = function(p) sum(quad_scale * (p - quad_center)^2) + (p[1] - p[2] - 3)^2,
  gr = function(p) {
    g <- 2 * quad_scale * (p - quad_center)
    cross <- 2 * (p[1] - p[2] - 3)
    g + c(cross, -cross, 0)
  }
)

# Rosenbrock's banana function, as in ?optim
rosenbrock <- list(
  fn = function(p) 100 * (p[2] - p[1]^2)^2 + (1 - p[1])^2,
  gr = function(p) {
    c(-400 * p[1] * (p[2] - p[1]^2) - 2 * (1 - p[1]), 200 * (p[2] - p[1]^2))
  }
)

# Negative log-likelihood of a logistic regression on 20 embedded rows
i <- 1:20
logit_data <- list(
  x1 = ((37 * i) %% 101) / 50 - 1,
  x2 = ((61 * i) %% 89) / 20 - 2
)
logit_data$y <- as.numeric((i %% 3 == 0) | (logit_data$x1 > 0.3 & i %% 4 != 1))
logistic <- with(logit_data, list(
  fn = function(p) {
    eta <- p[1] + p[2] * x1 + p[3] * x2
    sum(log(1 + exp(eta)) - y * eta)
  },
  gr = function(p) {
    eta <- p[1] + p[2] * x1 + p[3] * x2
    r <- 1 / (1 + exp(-eta)) - y
    c(sum(r), sum(r * x1), sum(r * x2))
  }
))

# Not finite (NaN) for a negative parameter, so that the line search has to
# shorten steps that leave the domain; minimum at 0.1 in each coordinate
barrier <- list(
  fn = function(p) suppressWarnings(sum(10 * p - log(p))),
  gr = function(p) 10 - 1 / p
)

squares <- list(
  fn = function(p) sum(p^2),
  gr = function(p) 2 * p
)

functions <- list(
  quadratic = quadratic, rosenbrock = rosenbrock, logistic = logistic,
  barrier = barrier, squares = squares
)

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

# Where the relative decrease falls below reltol, vmmin restarts once from the
# new point along the previous gradient, and that last line search can shrink
# its step until the trial value equals the minimum exactly. How many function
# evaluations that takes depends on the last bits of the values: R's C code
# (compiled with fused multiply-adds) and NumPy then differ by a few
# evaluations. The start values below were chosen so that every case ends
# away from that edge: in Python, perturbing every function and gradient
# value by 1e-15 (relative) leaves all counts unchanged. (The logistic
# likelihood from (0, 0, 0) and the barrier from (2, 0.5) do not pass that
# check.)
vmmin_case <- function(name, fun, par, maxit = 100, abstol = -Inf, reltol = sqrt(.Machine$double.eps)) {
  f <- functions[[fun]]
  res <- optim(par, f$fn, f$gr,
    method = "BFGS",
    control = list(maxit = maxit, abstol = abstol, reltol = reltol)
  )
  list(
    name = name,
    fun = fun,
    start = I(par),
    maxit = maxit,
    abstol = abstol,
    reltol = reltol,
    par = I(res$par),
    value = res$value,
    counts = I(unname(res$counts)),
    convergence = res$convergence
  )
}

cases <- list(
  vmmin_case("quadratic", "quadratic", c(0, 0, 0)),
  # stops as soon as the value is at most abstol
  vmmin_case("quadratic_abstol", "quadratic", c(0, 0, 0), abstol = 1e-3),
  vmmin_case("rosenbrock", "rosenbrock", c(-1.2, 1)),
  # stopped by the iteration limit
  vmmin_case("rosenbrock_maxit", "rosenbrock", c(-1.2, 1), maxit = 20),
  vmmin_case("logistic", "logistic", c(-1, 0.5, 0)),
  vmmin_case("logistic_reltol", "logistic", c(0, 0, 0), reltol = 1e-4),
  vmmin_case("logistic_maxit", "logistic", c(0, 0, 0), maxit = 5),
  # nnet's tolerances
  vmmin_case("logistic_nnet_tolerances", "logistic", c(0.5, -0.5, 0.25), abstol = 1e-4, reltol = 1e-8),
  vmmin_case("barrier", "barrier", c(1, 3)),
  # the gradient is 0 at the start: one evaluation of each, converged
  vmmin_case("start_at_minimum", "squares", c(0, 0))
)
by_name <- setNames(cases, vapply(cases, `[[`, "", "name"))
stopifnot(
  by_name$rosenbrock_maxit$convergence == 1L,
  by_name$logistic_maxit$convergence == 1L,
  all(vapply(cases, function(case) case$convergence %in% c(0L, 1L), TRUE)),
  sum(vapply(cases, function(case) case$convergence == 0L, TRUE)) == 8L,
  by_name$quadratic_abstol$value <= 1e-3,
  identical(by_name$start_at_minimum$counts, I(c(1L, 1L))),
  # the barrier's first step leaves the domain and is shortened
  by_name$barrier$counts[1] > by_name$barrier$counts[2]
)

write_fixture("l1_vmmin", list(
  description = paste(
    "optim(start, fn, gr, method = 'BFGS', control = list(maxit, abstol,",
    "reltol)) on the functions named in fun (defined in l1_vmmin.R; the",
    "logistic likelihood on logit_data). par, value, counts (function,",
    "gradient) and convergence (0, or 1 at the iteration limit) as optim",
    "returns them."
  ),
  logit_data = lapply(logit_data, I),
  cases = cases
))
