# L1 fixture for the donor probabilities of midastouch.
#
#   l1_midastouch.json   bootstrap weights, donor probabilities and kappa of
#                        mice.impute.midastouch()
#
# Python target: the planned imputation.midas._donor_probabilities.
#
# The bootstrap weights are the first random draw in mice.impute.midastouch(),
# so set.seed(s); mice:::bootfunc.plain(nobs) reproduces them. A trace() exit
# hook on mice.impute.midastouch() captures the local objects probs (nobs x
# nmis, each column sums to 1), midas.kappa and omega at the end of the call.

source(file.path("tests", "reference", "r", "_common.R"))

capture <- new.env()
invisible(suppressMessages(trace(
  "mice.impute.midastouch",
  exit = bquote(assign(
    "last",
    list(probs = probs, kappa = midas.kappa, omega = omega),
    envir = .(capture)
  )),
  where = asNamespace("mice"),
  print = FALSE
)))

set.seed(20261122)
n <- 30
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- 1 + x1 + 0.5 * x2 + rnorm(n)
ry <- rep(TRUE, n)
ry[sample(n, 8)] <- FALSE
y[!ry] <- NA
x <- cbind(x1, x2)

midas_case <- function(name, seed, outout, kappa = NULL) {
  set.seed(seed)
  omega <- mice:::bootfunc.plain(sum(ry))
  rm(list = ls(capture), envir = capture)
  set.seed(seed)
  # the traced copy lives in the namespace
  imp <- mice:::mice.impute.midastouch(
    y, ry, x,
    midas.kappa = kappa, outout = outout
  )
  got <- get("last", envir = capture)
  stopifnot(
    identical(got$omega, omega),
    all(abs(colSums(got$probs) - 1) < 1e-12),
    dim(got$probs) == c(sum(ry), sum(!ry))
  )
  list(
    name = name,
    seed = seed,
    outout = outout,
    midas_kappa = kappa,
    omega = I(omega),
    probs = unname(got$probs),
    kappa = got$kappa,
    imputed = I(imp)
  )
}

# untrace also when a case fails, so the namespace is left as found
cases <- tryCatch(
  list(
    midas_case("outout_estimated_kappa", 31L, outout = TRUE),
    midas_case("outout_estimated_kappa_seed2", 32L, outout = TRUE),
    midas_case("no_outout_estimated_kappa", 33L, outout = FALSE),
    midas_case("outout_fixed_kappa", 34L, outout = TRUE, kappa = 2)
  ),
  finally = invisible(suppressMessages(
    untrace("mice.impute.midastouch", where = asNamespace("mice"))
  ))
)

write_fixture("l1_midastouch", list(
  description = paste(
    "mice.impute.midastouch(y, ry, x, midas.kappa, outout): omega are the",
    "bootstrap counts (length nobs), probs the donor probabilities",
    "(nobs x nmis, columns sum to 1), kappa the value used. x excludes the",
    "intercept. midas_kappa null means estimated."
  ),
  y = I(y),
  ry = I(ry),
  x = unname(x),
  ridge = 1e-05,
  cases = cases
))
