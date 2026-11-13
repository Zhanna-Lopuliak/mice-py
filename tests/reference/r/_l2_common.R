# Helpers for the layer-2 (distribution parity) fixture generators.
#
# Layer 2 compares the distribution of imputed values, not the values
# themselves, because R and NumPy use different random number generators.
# Each l2_<method>.R script calls one exported mice.impute.<method> function
# many times on fixed inputs (F1 to F8, defined below) and stores sufficient
# statistics per missing cell, plus the moments of the OLS coefficients of
# the completed y. What is stored per cell depends on the kind of the method
# (see "Kinds of methods" below): donor counts for a method that imputes
# observed values, decile bins for one that imputes new numbers, level
# counts for one that draws labels from a fitted model; the moments of the
# imputed value for every numeric target.
#
# An input has two masks, as sampler.univ passes them: `ry`, the fit rows,
# and `wy`, the rows to impute (default `!ry`). F8 has general masks: rows
# in both and rows in neither (l2_masks.R). A "cell" below is a row of `wy`.
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

# Shuffles the rows and rounds. `wy` (NULL: `!ry`) marks the rows to impute.
# y is set to NA in the rows to impute that are not fit rows; the other rows
# keep their y, so a row in neither mask can hold an observed y (as a row
# that `ignore` leaves out of the fit does) or NA.
finish_input <- function(name, y, ry, x, description, wy = NULL) {
  if (is.null(wy)) wy <- !ry
  perm <- sample(length(ry))
  y <- y[perm]
  ry <- ry[perm]
  wy <- wy[perm]
  x <- x[perm, , drop = FALSE]
  if (is.numeric(y)) y <- round(y, L2_DIGITS)
  x <- round(x, L2_DIGITS)
  y[wy & !ry] <- NA
  list(name = name, description = description, y = y, ry = ry, wy = wy, x = x)
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

# Plain logistic regression of a two-level factor on [1, x], without the
# augmentation of mice's logreg (used to check F5 and F6).
plain_logit_fit <- function(y, x) {
  suppressWarnings(glm.fit(cbind(1, x), as.integer(y) - 1L, family = binomial()))
}

# F5: binary target (levels no, yes) from a logistic model with clearly
# non-zero slopes, P(yes) = plogis(-0.3 + 1.5 x1 - x2), x1 and x2
# independent. Half of the missing rows lie far out near the line where that
# linear predictor is 0 (x2 = 1.5 x1 - 0.3), on both sides of the bulk: their
# labels depend on the poorly determined slopes, so the shared coefficient
# draw of logreg makes the cells of one call dependent.
make_f5 <- function() {
  set.seed(20255)
  x_bulk <- cbind(x1 = rnorm(50), x2 = rnorm(50))
  s <- runif(10, 1, 2) * sample(c(-1, 1), 10, replace = TRUE)
  x_off <- cbind(x1 = s, x2 = 1.5 * s - 0.3 + rnorm(10, 0, 0.1))
  x <- rbind(x_bulk, x_off)
  p <- plogis(-0.3 + 1.5 * x[, 1] - x[, 2])
  lev <- c("no", "yes")
  y <- factor(lev[1 + (runif(60) < p)], levels = lev)
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  inp <- finish_input("F5", y, ry, x, paste(
    "40 observed, 20 missing; factor target with levels no, yes,",
    "P(yes) = plogis(-0.3 + 1.5 x1 - x2); 10 missing rows far out near",
    "-0.3 + 1.5 x1 - x2 = 0"
  ))
  fit <- plain_logit_fit(inp$y[inp$ry], inp$x[inp$ry, ])
  stopifnot(
    "each level observed at least 12 times" = all(table(inp$y[inp$ry]) >= 12),
    "the observed rows are not separated" =
      fit$converged && all(abs(coef(fit)) < 5)
  )
  inp
}

# F6: binary target (levels no, yes) whose observed rows are perfectly
# separated by x1: yes exactly where x1 > 0; x2 is noise. Of the missing
# rows, 8 lie within 0.3 of the boundary and the others spread out, with
# rows on both sides. Without augmentation the logistic fit of the observed
# rows diverges; the augmentation of logreg decides the result.
make_f6 <- function() {
  set.seed(20256)
  x1 <- c(rnorm(40), runif(8, -0.3, 0.3), rnorm(12))
  x <- cbind(x1 = x1, x2 = rnorm(60))
  lev <- c("no", "yes")
  y <- factor(lev[1 + (x1 > 0)], levels = lev)
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  inp <- finish_input("F6", y, ry, x, paste(
    "40 observed, 20 missing; factor target with levels no, yes,",
    "observed rows separated by x1 (yes where x1 > 0), x2 noise;",
    "8 missing rows within 0.3 of the boundary"
  ))
  x1_obs <- inp$x[inp$ry, "x1"]
  yes_obs <- inp$y[inp$ry] == "yes"
  x1_mis <- inp$x[!inp$ry, "x1"]
  fit <- plain_logit_fit(inp$y[inp$ry], inp$x[inp$ry, ])
  stopifnot(
    "each level observed at least 12 times" = all(table(inp$y[inp$ry]) >= 12),
    "the observed rows are separated by x1" =
      max(x1_obs[!yes_obs]) < min(x1_obs[yes_obs]),
    "missing rows on both sides of the boundary" =
      sum(x1_mis > 0) >= 5 && sum(x1_mis < 0) >= 5,
    "the plain logistic fit diverges" =
      !fit$converged || max(abs(coef(fit))) > 10
  )
  inp
}

# F7: ordered target with the levels a < b < c < d, two numeric predictors,
# drawn from a multinomial model that is not a proportional-odds model:
# x1 moves the outer levels (a, d) against the inner ones (b, c), x2 the
# upper levels against the lower ones. The proportional-odds fit of
# mice.impute.polr and the multinomial fit of mice.impute.polyreg (both on
# mice's augmented data) then predict clearly different probabilities for
# the missing rows.
make_f7 <- function() {
  set.seed(20257)
  x <- cbind(x1 = rnorm(60), x2 = rnorm(60))
  eta <- cbind(0, 0.7 - 1.5 * x[, 1], 0.5 - 1.5 * x[, 1] + 0.9 * x[, 2], 0.9 * x[, 2])
  p <- exp(eta) / rowSums(exp(eta))
  lev <- c("a", "b", "c", "d")
  y <- factor(apply(p, 1, function(pr) sample(lev, 1, prob = pr)), levels = lev, ordered = TRUE)
  ry <- rep(c(TRUE, FALSE), c(40, 20))
  inp <- finish_input("F7", y, ry, x, paste(
    "40 observed, 20 missing; ordered target with levels a < b < c < d from",
    "a multinomial model (x1 moves a and d against b and c, x2 c and d",
    "against a and b); two numeric predictors"
  ))
  aug <- mice:::augment(inp$y, inp$ry, inp$x, !inp$ry)
  xy <- cbind.data.frame(y = aug$y, x = aug$x)
  obs <- xy[aug$ry, , drop = FALSE]
  w <- aug$w[aug$ry]
  prop_odds <- suppressWarnings(MASS::polr(formula(xy), data = obs, weights = w))
  multinomial <- nnet::multinom(formula(xy), data = obs, weights = w, trace = FALSE)
  gap <- max(abs(
    predict(prop_odds, xy[aug$wy, ], type = "probs") - predict(multinomial, xy[aug$wy, ], type = "probs")
  ))
  stopifnot(
    "each level observed at least 6 times" = all(table(inp$y[inp$ry]) >= 6),
    "polr and multinom probabilities differ by at least 0.15" = gap >= 0.15
  )
  inp
}

# F8: general masks, as sampler.univ passes them under `where` and `ignore`.
# Predictors and target as F1. 40 fit rows from the bulk, 8 of them also to
# impute (a row in both masks can be its own donor); 12 rows to impute that
# are not fit rows (6 in the bulk, 6 near x1 = -x2 as in F1); 8 rows in
# neither mask: 4 with an observed decoy y far above the y of the fit rows
# (ignored rows) and 4 with NA in y. Two of the latter have NA in x (in x2
# only, in every predictor); the predictors of the other six lie far outside
# the range of the rows used, so that any use of a row in neither mask
# shows.
make_f8 <- function() {
  set.seed(20258)
  bulk <- function(n) {
    z1 <- rnorm(n)
    z2 <- 0.9 * z1 + sqrt(1 - 0.9^2) * rnorm(n)
    cbind(x1 = z1, x2 = z2, x3 = rnorm(n))
  }
  x_fit <- bulk(40)
  x_imp_bulk <- bulk(6)
  s <- runif(6, 1, 2) * sample(c(-1, 1), 6, replace = TRUE)
  x_imp_off <- cbind(x1 = s, x2 = -s + rnorm(6, 0, 0.1), x3 = rnorm(6))
  x <- rbind(x_fit, x_imp_bulk, x_imp_off)
  y <- c(1 + x %*% c(1, 1, 0.5)) + rnorm(nrow(x))
  k <- 1:8
  x_neither <- cbind(x1 = 7 + k, x2 = -7 - k, x3 = 6 + k)
  x_neither[7, "x2"] <- NA
  x_neither[8, ] <- NA
  x <- rbind(x, x_neither)
  y <- c(y, 50 + 10 * k[1:4], rep(NA, 4))
  ry <- rep(c(TRUE, FALSE, FALSE), c(40, 12, 8))
  wy <- rep(c(TRUE, FALSE, TRUE, FALSE), c(8, 32, 12, 8))
  inp <- finish_input("F8", y, ry, x, paste(
    "40 fit rows, 20 rows to impute of which 8 are fit rows too, 8 rows in",
    "neither mask (4 with an observed decoy y, 4 with NA in y; NA in x2 in",
    "one of them and in every predictor in another, decoy x in the other",
    "six); predictors and target as F1, 6 rows to impute near x1 = -x2"
  ), wy = wy)
  neither <- !inp$ry & !inp$wy
  used <- inp$ry | inp$wy
  na_x <- rowSums(is.na(inp$x))
  y_fit <- inp$y[inp$ry]
  far <- vapply(colnames(inp$x), function(j) {
    v <- inp$x[neither, j]
    v <- v[!is.na(v)]
    r <- range(inp$x[used, j]) + c(-3, 3) * sd(inp$x[used, j])
    all(v < r[1] | v > r[2])
  }, logical(1))
  stopifnot(
    "60 rows" = length(inp$ry) == 60,
    "40 fit rows" = sum(inp$ry) == 40,
    "20 rows to impute, 8 of them fit rows" = sum(inp$wy) == 20 && sum(inp$ry & inp$wy) == 8,
    "8 rows in neither mask, 4 with an observed y" =
      sum(neither) == 8 && sum(neither & !is.na(inp$y)) == 4,
    "y observed in the fit rows, NA in the other rows to impute" =
      !anyNA(y_fit) && all(is.na(inp$y[inp$wy & !inp$ry])),
    "x complete in the rows used" = !anyNA(inp$x[used, ]),
    "NA in x in one predictor of one row and in all of another, both in neither mask with NA in y" =
      identical(sort(unname(na_x[na_x > 0])), c(1, 3)) &&
        all(neither[na_x > 0]) && all(is.na(inp$y[na_x > 0])),
    "decoy y far above the y of the fit rows" =
      all(inp$y[neither & !is.na(inp$y)] > max(y_fit) + 10 * sd(y_fit)),
    "decoy x more than 3 sd outside the range of the rows used" = all(far)
  )
  inp
}

l2_inputs <- function() {
  list(
    F1 = make_f1(), F2 = make_f2(), F3 = make_f3(), F4 = make_f4(),
    F5 = make_f5(), F6 = make_f6(), F7 = make_f7(), F8 = make_f8()
  )
}

# The sorted distinct values of the fit rows (numeric target) or the levels
# (factor target). Donor counts are stored in this order.
donor_values <- function(inp) {
  if (is.factor(inp$y)) levels(inp$y) else sort(unique(inp$y[inp$ry]))
}

# TRUE if the rows to impute are not the rows outside the fit rows.
has_general_masks <- function(inp) !identical(inp$wy, !inp$ry)

# `wy` is written only where it differs from `!ry`, and `ordered` for an
# ordered factor only, so that the payloads of the other inputs keep their
# bytes; the levels of an ordered factor are its order.
input_payload <- function(inp) {
  is_fac <- is.factor(inp$y)
  payload <- list(
    name = inp$name,
    description = inp$description,
    type = if (is_fac) "factor" else "numeric",
    y = if (is_fac) as.character(inp$y) else inp$y,
    ry = inp$ry,
    x = inp$x,
    x_names = I(colnames(inp$x)),
    donor_values = I(donor_values(inp))
  )
  if (has_general_masks(inp)) payload <- append(payload, list(wy = inp$wy), after = 5L)
  if (is.ordered(inp$y)) payload <- append(payload, list(ordered = TRUE), after = 3L)
  payload
}

# ---------------------------------------------------------------------------
# Repeated calls and their sufficient statistics
# ---------------------------------------------------------------------------

# Donor kind: call `fun(y, ry, x, wy = wy, ...)` n_calls times. Returns an
# n_calls x sum(wy) matrix of donor indices into donor_values(inp); stops if
# any imputation is not the value of a fit row (or a level), so a decoy y in
# a row in neither mask is never drawn.
run_calls <- function(fun, inp, n_calls, ...) {
  dv <- donor_values(inp)
  n_imp <- sum(inp$wy)
  idx <- matrix(NA_integer_, n_calls, n_imp)
  for (i in seq_len(n_calls)) {
    imp <- fun(inp$y, inp$ry, inp$x, wy = inp$wy, ...)
    if (is.factor(inp$y)) imp <- as.character(imp)
    idx[i, ] <- match(imp, dv)
  }
  stopifnot("every imputation is the value of a fit row (or a level)" = !anyNA(idx))
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

# Moments of the imputed values per cell (`cell`) and of the OLS
# coefficients of the completed y on [1, x], one fit per call (`coef`).
# imp: n_calls x sum(wy) matrix of imputed values. The completed y holds the
# imputations in the rows to impute, also in a fit row to impute, whose
# observed value they replace (as complete() does under `where`), and the
# observed values in the other fit rows. The regression uses the rows of
# ry | wy only: a row in neither mask is not completed and does not enter.
value_moments <- function(imp, inp) {
  used <- inp$ry | inp$wy
  x1 <- cbind(1, inp$x)[used, , drop = FALSE]
  ycomp <- matrix(inp$y, length(used), nrow(imp))
  ycomp[inp$wy, ] <- t(imp)
  ycomp <- ycomp[used, , drop = FALSE]
  coef <- solve(crossprod(x1), crossprod(x1, ycomp))
  coef_moments <- moments(t(coef))
  coef_moments$names <- I(c("(Intercept)", colnames(inp$x)))
  list(cell = moments(imp), coef = coef_moments)
}

# Donor kind: donor counts per cell; for a numeric target also the moments.
summarise_calls <- function(idx, inp) {
  dv <- donor_values(inp)
  out <- list(n_calls = nrow(idx), counts = donor_counts(idx, length(dv)))
  if (!is.factor(inp$y)) {
    out <- c(out, value_moments(matrix(dv[idx], nrow(idx)), inp))
  }
  out
}

# Continuous kind: call `fun(y, ry, x, wy = wy, ...)` n_calls times and keep
# the imputed values, an n_calls x sum(wy) matrix; stops if one is not
# finite.
run_values <- function(fun, inp, n_calls, ...) {
  n_imp <- sum(inp$wy)
  vals <- matrix(NA_real_, n_calls, n_imp)
  for (i in seq_len(n_calls)) {
    imp <- as.vector(fun(inp$y, inp$ry, inp$x, wy = inp$wy, ...))
    stopifnot("one imputation per cell to impute" = length(imp) == n_imp)
    vals[i, ] <- imp
  }
  stopifnot("every imputation is a finite number" = all(is.finite(vals)))
  vals
}

# Per cell, the nine decile edges of R's draws (quantile type 7) and the
# number of R's draws in each of the ten bins: bin 1 is (-Inf, edge 1),
# bin j is [edge j-1, edge j), bin 10 is [edge 9, Inf), as findInterval()
# counts. The Python side counts its own draws in the same bins.
decile_bins <- function(vals) {
  probs <- (1:9) / 10
  edges <- t(apply(vals, 2, quantile, probs = probs, names = FALSE, type = 7))
  counts <- t(vapply(seq_len(ncol(vals)), function(j) {
    tabulate(findInterval(vals[, j], edges[j, ]) + 1L, nbins = 10L)
  }, integer(10)))
  list(edges = edges, counts = counts)
}

# Continuous kind: moments per cell and of the coefficients, and the decile
# bins per cell.
summarise_values <- function(vals, inp) {
  c(
    list(n_calls = nrow(vals)),
    value_moments(vals, inp),
    list(bins = decile_bins(vals))
  )
}

# Category kind: the level counts per cell (`counts`, as for a donor method
# on a factor target), and the moments of the OLS coefficients of the 0/1
# indicator of each level but the first, in the completed target, on
# [1, x], one fit per call (`coef`, named "<level>:<term>", level by level).
# The target is completed and the rows enter as in value_moments().
# idx: n_calls x sum(wy) matrix of level indices, as run_calls() returns.
summarise_labels <- function(idx, inp) {
  stopifnot("the category kind needs a factor target" = is.factor(inp$y))
  lev <- levels(inp$y)
  used <- inp$ry | inp$wy
  x1 <- cbind(1, inp$x)[used, , drop = FALSE]
  coef <- do.call(rbind, lapply(seq_along(lev)[-1], function(l) {
    ind <- matrix(as.numeric(inp$y == lev[l]), length(used), nrow(idx))
    ind[inp$wy, ] <- t(idx == l)
    solve(crossprod(x1), crossprod(x1, ind[used, , drop = FALSE]))
  }))
  coef_moments <- moments(t(coef))
  terms <- c("(Intercept)", colnames(inp$x))
  coef_moments$names <- I(paste0(rep(lev[-1], each = length(terms)), ":", terms))
  list(
    n_calls = nrow(idx),
    counts = donor_counts(idx, length(lev)),
    coef = coef_moments
  )
}

# The `extra` of the cart fixtures (l2_cart.R, l2_masks.R): per cell, which
# donors were drawn at least once in the first 2,000 calls (`reached_2000`).
# rpart is deterministic, so for a numeric target only the draw within the
# leaf is random; stops unless all calls reach exactly the same donors.
reached_first_2000 <- function(idx, inp) {
  k <- length(donor_values(inp))
  first <- donor_counts(idx[seq_len(2000), , drop = FALSE], k) > 0
  all <- donor_counts(idx, k) > 0
  if (!is.factor(inp$y)) {
    stopifnot("rpart tree is deterministic" = identical(first, all))
  }
  storage.mode(first) <- "integer"
  list(reached_2000 = first)
}

# ---------------------------------------------------------------------------
# Kinds of methods
# ---------------------------------------------------------------------------

# How the imputations of a method are collected (`run`, returning one row
# per call) and summarised (`summarise`, returning the statistics stored per
# input). A new kind is one more entry here with its own pair of functions;
# tests/test_parity/_l2_harness.py dispatches on the same names.
#
# - donor: the imputations are observed values or levels;
# - continuous: they are new numbers;
# - category: they are labels drawn from a fitted model (factor target). The
#   labels are collected as for a donor method; the summary adds the
#   coefficient moments of the level indicators, which see the dependence
#   between the cells of one call that a shared parameter draw creates.
L2_KINDS <- list(
  donor = list(run = run_calls, summarise = summarise_calls),
  continuous = list(run = run_values, summarise = summarise_values),
  category = list(run = run_calls, summarise = summarise_labels)
)

# Fixture name of a method: R's name with dots replaced by underscores, so
# norm.nob writes l2_norm_nob.json. Inputs with general masks (F8) go to
# l2_masks_<method>.json (written by l2_masks.R), so that the fixtures of
# F1 to F7 keep their bytes.
l2_fixture_name <- function(method, masks = FALSE) {
  paste0(if (masks) "l2_masks_" else "l2_", gsub(".", "_", method, fixed = TRUE))
}

# Run one method on the named inputs and write fixture l2_<method>.json, or
# l2_masks_<method>.json for inputs with general masks (all or none of the
# inputs). `seeds` gives one set.seed value per input; `n_calls` the number
# of calls; `extra(draws, inp)` optionally adds entries per input; `...` is
# passed to the method. `kind` (named in full, after `...`) is one of
# names(L2_KINDS); it is written to the fixture unless it is "donor", the
# kind of every fixture written before kinds existed.
l2_method_fixture <- function(method, input_names, seeds, n_calls,
                              extra = NULL, ..., kind = "donor") {
  stopifnot("unknown kind" = kind %in% names(L2_KINDS))
  handler <- L2_KINDS[[kind]]
  fun <- get(paste0("mice.impute.", method), envir = asNamespace("mice"))
  inputs <- l2_inputs()
  general <- vapply(inputs[input_names], has_general_masks, logical(1))
  stopifnot("inputs with general masks go to a fixture of their own" = all(general) || !any(general))
  masks <- all(general)
  out <- list()
  for (j in seq_along(input_names)) {
    inp <- inputs[[input_names[j]]]
    set.seed(seeds[j])
    draws <- handler$run(fun, inp, n_calls, ...)
    res <- c(
      list(input = input_payload(inp), seed = seeds[j]),
      handler$summarise(draws, inp)
    )
    if (!is.null(extra)) res <- c(res, extra(draws, inp))
    out[[inp$name]] <- res
  }
  header <- list(layer = "L2", method = method)
  if (kind != "donor") header$kind <- kind
  write_fixture(l2_fixture_name(method, masks), c(header, list(
    call = paste0("mice.impute.", method, if (masks) "(y, ry, x, wy)" else "(y, ry, x)"),
    results = out
  )))
}
