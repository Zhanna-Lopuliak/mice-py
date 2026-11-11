# Helpers for the layer-2 (distribution parity) fixture generators.
#
# Layer 2 compares the distribution of imputed values, not the values
# themselves, because R and NumPy use different random number generators.
# Each l2_<method>.R script calls one exported mice.impute.<method> function
# many times on fixed inputs (F1 to F4, defined below) and stores sufficient
# statistics per missing cell: donor counts and the moments of the imputed
# value, plus the moments of the OLS coefficients of the completed y.
#
# The inputs are generated here with set.seed and embedded in every fixture,
# so the Python tests never need R.

source(file.path("tests", "reference", "r", "_common.R"))

# Pin the generator so that regenerating reproduces the files byte for byte.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

L2_DIGITS <- 4 # inputs are rounded so that JSON text round-trips exactly

# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------

finish_input <- function(name, y, ry, x, description) {
  perm <- sample(length(ry))
  y <- y[perm]
  ry <- ry[perm]
  x <- x[perm, , drop = FALSE]
  if (is.numeric(y)) y <- round(y, L2_DIGITS)
  x <- round(x, L2_DIGITS)
  y[!ry] <- NA
  list(name = name, description = description, y = y, ry = ry, x = x)
}

# F1: x1 and x2 correlated 0.9, x3 independent, y linear plus noise.
# Half of the missing rows lie near x1 = -x2, away from the bulk of the
# observed rows, so the imputations depend on the poorly determined
# difference of the two slopes and coefficient uncertainty matters.
make_f1 <- function() {
  set.seed(20251)
  bulk <- function(n) {
    z1 <- rnorm(n)
    z2 <- 0.9 * z1 + sqrt(1 - 0.9^2) * rnorm(n)
    cbind(x1 = z1, x2 = z2, x3 = rnorm(n))
  }
  x_obs <- bulk(40)
  x_mis_bulk <- bulk(10)
  s <- runif(10, 1, 2) * sample(c(-1, 1), 10, replace = TRUE)
  x_mis_off <- cbind(x1 = s, x2 = -s + rnorm(10, 0, 0.1), x3 = rnorm(10))
  x <- rbind(x_obs, x_mis_bulk, x_mis_off)
  y <- c(1 + x %*% c(1, 1, 0.5)) + rnorm(nrow(x))
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  finish_input("F1", y, ry, x, paste(
    "40 observed, 20 missing; x1, x2 correlated 0.9, x3 independent;",
    "10 missing rows near x1 = -x2"
  ))
}

# F2: one binary predictor, so the predicted values of the observed rows
# take only two distinct values and PMM must break many ties.
make_f2 <- function() {
  set.seed(20252)
  x1 <- c(rep(0:1, each = 20), rep(0:1, each = 10))
  y <- 2 + 1.5 * x1 + rnorm(60)
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  finish_input("F2", y, ry, cbind(x1 = x1), paste(
    "40 observed, 20 missing; one binary predictor (20/20 observed,",
    "10/10 missing)"
  ))
}

# F3: factor target with three levels, two numeric predictors.
make_f3 <- function() {
  set.seed(20253)
  x <- cbind(x1 = rnorm(60), x2 = rnorm(60))
  eta <- cbind(0, 0.3 + 1.2 * x[, 1], -0.2 + 1.2 * x[, 2])
  p <- exp(eta) / rowSums(exp(eta))
  lev <- c("a", "b", "c")
  y <- factor(apply(p, 1, function(pr) sample(lev, 1, prob = pr)), levels = lev)
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  stopifnot(all(table(y[ry]) >= 8))
  finish_input("F3", y, ry, x, paste(
    "40 observed, 20 missing; factor target with levels a, b, c;",
    "two numeric predictors"
  ))
}

# F4: lognormal target, two numeric predictors.
make_f4 <- function() {
  set.seed(20254)
  x <- cbind(x1 = rnorm(60), x2 = rnorm(60))
  y <- exp(0.5 + 0.6 * x[, 1] + 0.3 * x[, 2] + 0.6 * rnorm(60))
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  finish_input("F4", y, ry, x, paste(
    "40 observed, 20 missing; lognormal target, two numeric predictors"
  ))
}

l2_inputs <- function() {
  list(F1 = make_f1(), F2 = make_f2(), F3 = make_f3(), F4 = make_f4())
}

# The sorted distinct observed values (numeric target) or the levels
# (factor target). Donor counts are stored in this order.
donor_values <- function(inp) {
  if (is.factor(inp$y)) levels(inp$y) else sort(unique(inp$y[inp$ry]))
}

input_payload <- function(inp) {
  is_fac <- is.factor(inp$y)
  list(
    name = inp$name,
    description = inp$description,
    type = if (is_fac) "factor" else "numeric",
    y = if (is_fac) as.character(inp$y) else inp$y,
    ry = inp$ry,
    x = inp$x,
    x_names = I(colnames(inp$x)),
    donor_values = I(donor_values(inp))
  )
}

# ---------------------------------------------------------------------------
# Repeated calls and their sufficient statistics
# ---------------------------------------------------------------------------

# Call `fun(y, ry, x, ...)` n_calls times. Returns an n_calls x n_mis matrix
# of donor indices into donor_values(inp); stops if any imputation is not an
# observed value (or a level).
run_calls <- function(fun, inp, n_calls, ...) {
  dv <- donor_values(inp)
  n_mis <- sum(!inp$ry)
  idx <- matrix(NA_integer_, n_calls, n_mis)
  for (i in seq_len(n_calls)) {
    imp <- fun(inp$y, inp$ry, inp$x, ...)
    if (is.factor(inp$y)) imp <- as.character(imp)
    idx[i, ] <- match(imp, dv)
  }
  stopifnot("every imputation is an observed value" = !anyNA(idx))
  idx
}

# Kurtosis m4 / m2^2 (3 for a normal distribution); NA for a constant.
kurtosis <- function(z) {
  d <- z - mean(z)
  m2 <- mean(d^2)
  if (m2 == 0) NA_real_ else mean(d^4) / m2^2
}

moments <- function(m) {
  # m: draws in rows, quantities in columns
  list(
    mean = I(unname(colMeans(m))),
    var = I(unname(apply(m, 2, var))),
    kurt = I(unname(apply(m, 2, kurtosis)))
  )
}

donor_counts <- function(idx, k) {
  t(apply(idx, 2, tabulate, nbins = k))
}

summarise_calls <- function(idx, inp) {
  dv <- donor_values(inp)
  out <- list(n_calls = nrow(idx), counts = donor_counts(idx, length(dv)))
  if (!is.factor(inp$y)) {
    imp <- matrix(dv[idx], nrow(idx))
    out$cell <- moments(imp)
    # OLS coefficients of the completed y on [1, x], one fit per call.
    x1 <- cbind(1, inp$x)
    ycomp <- matrix(inp$y, nrow(x1), nrow(idx))
    ycomp[!inp$ry, ] <- t(imp)
    coef <- solve(crossprod(x1), crossprod(x1, ycomp))
    out$coef <- moments(t(coef))
    out$coef$names <- I(c("(Intercept)", colnames(inp$x)))
  }
  out
}

# Run one method on the named inputs and write fixture l2_<method>.json.
# `seeds` gives one set.seed value per input; `n_calls` the number of calls.
l2_method_fixture <- function(method, input_names, seeds, n_calls,
                              extra = NULL, ...) {
  fun <- get(paste0("mice.impute.", method), envir = asNamespace("mice"))
  inputs <- l2_inputs()
  out <- list()
  for (j in seq_along(input_names)) {
    inp <- inputs[[input_names[j]]]
    set.seed(seeds[j])
    idx <- run_calls(fun, inp, n_calls, ...)
    res <- c(
      list(input = input_payload(inp), seed = seeds[j]),
      summarise_calls(idx, inp)
    )
    if (!is.null(extra)) res <- c(res, extra(idx, inp))
    out[[inp$name]] <- res
  }
  write_fixture(paste0("l2_", method), list(
    layer = "L2",
    method = method,
    call = paste0("mice.impute.", method, "(y, ry, x)"),
    results = out
  ))
}
