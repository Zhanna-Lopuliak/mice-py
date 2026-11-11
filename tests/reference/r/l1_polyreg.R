# L1 fixture for imputation of a categorical target by multinomial logistic
# regression.
#
#   l1_polyreg.json   mice.impute.polyreg with its uniform numbers captured,
#                     the nnet::multinom fit (objective, convergence code,
#                     coefficients), the probabilities of every imputed cell
#                     and the labels; and direct nnet::multinom fits for what
#                     polyreg cannot reach through R
#
# Python targets: imputation.polyreg.polyreg (through its step-by-step
# version _polyreg) driven by a strict ScriptedGenerator, and
# imputation.fitting.multinom_fit.
#
# Each case calls mice.impute.polyreg after set.seed(seed) and saves
# .Random.seed. The port's computation is then rebuilt with mice's own
# augment(), nnet::multinom() and predict(), drawing runif(n_imp) after
# set.seed(seed) again (polyreg draws nothing else: the fit starts from zero
# weights). Where mice succeeds, the rebuild must leave the same .Random.seed
# and give identical labels; where mice stops, the draws mice made before are
# checked the same way and the rebuild records what the port does instead
# (see `port` below).
#
# Every case also requires min |u - cumulative probability| > 1e-4 over the
# imputed cells and all their cumulative probabilities, so that no label
# depends on the last digits of a probability.
#
# Level order. The port orders the categories of an ordered factor as
# declared and those of every other target by their sorted labels, R's order
# after augment(). Where R does not augment (no predictor, or exactly one
# missing value), R keeps the declared order of an unordered factor; those
# cases use factors declared in sorted order, so that the same random numbers
# give the same labels. Labels are lowercase ASCII or digits, whose sort
# order does not depend on the locale ("1" < "10" < "2", as Python sorts).

source(file.path("tests", "reference", "r", "_common.R"))

augment <- mice:::augment
NNET_VERSION <- as.character(packageVersion("nnet"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(7401)
n <- 40
x1 <- rnorm(n)
x2 <- rnorm(n)
x12 <- cbind(x1, x2)
x0 <- x12[, 0, drop = FALSE]

# labels drawn from a multinomial logit model; index 1 + number of cumulative
# probabilities below a uniform number
draw_labels <- function(eta, labels) {
  pr <- exp(eta) / rowSums(exp(eta))
  u <- runif(nrow(pr))
  labels[1L + rowSums(u > t(apply(pr, 1, cumsum)))]
}
abc <- draw_labels(cbind(0, 0.4 + 1.3 * x1, -0.3 + 1.1 * x2), c("a", "b", "c"))
yes <- ifelse(runif(n) < plogis(-0.2 + 1.4 * x1 - 0.8 * x2), "yes", "no")
mis <- sort(sample(n, 10))
ry <- !(seq_len(n) %in% mis)
ry_one <- seq_len(n) != mis[1]
stopifnot(all(table(abc[ry]) >= 8), all(table(yes[ry]) >= 8))

# 5 levels, 300 rows, predictors on scales 1000, 1 and 0.001
n5 <- 300
z5 <- cbind(rnorm(n5), rnorm(n5), rnorm(n5))
x5 <- cbind(big = 5000 + 1000 * z5[, 1], unit = z5[, 2], small = 0.001 * z5[, 3])
five <- draw_labels(
  cbind(0, 0.5 + z5[, 1], -0.5 + z5[, 2], 0.2 - z5[, 1] + z5[, 3], 0.8 * z5[, 3]),
  c("a", "b", "c", "d", "e")
)
ry5 <- !(seq_len(n5) %in% sample(n5, 60))
stopifnot(all(table(five[ry5]) >= 15))

# separated by x1: a below -0.5, b up to 0.5, c above
sep <- ifelse(x1 < -0.5, "a", ifelse(x1 < 0.5, "b", "c"))
stopifnot(all(table(sep[ry]) >= 5))

# 51 categories, for the limit of augment()
n51 <- 102
x51 <- cbind(u = rnorm(n51))
many <- sprintf("l%02d", rep(1:51, 2))
ry51 <- seq_len(n51) > 4

blank <- function(y, keep = ry) replace(y, !keep, NA)
fac <- function(v, levels, ...) factor(v, levels = levels, ...)
relabel <- function(v, to) unname(to[v])

inputs <- list(
  base = list(y = blank(fac(abc, c("a", "b", "c"))), ry = ry, x = x12),
  two_levels = list(y = blank(fac(yes, c("no", "yes"))), ry = ry, x = x12),
  # declared (c, a, b); after augment() the levels are sorted
  unsorted = list(y = blank(fac(abc, c("c", "a", "b"))), ry = ry, x = x12),
  # labels that sort differently as text ("1" < "10" < "2") and as numbers
  numeric_labels = list(
    y = blank(fac(relabel(abc, c(a = "1", b = "2", c = "10")), c("1", "2", "10"))), ry = ry, x = x12
  ),
  # "ab" is declared but never occurs; augment() drops it
  unused_level = list(y = blank(fac(abc, c("a", "ab", "b", "c"))), ry = ry, x = x12),
  # declared lo < mid < hi, alphabetically hi < lo < mid
  ordered = list(
    y = blank(fac(relabel(abc, c(a = "lo", b = "mid", c = "hi")), c("lo", "mid", "hi"), ordered = TRUE)),
    ry = ry, x = x12
  ),
  one_category = list(y = blank(fac(rep("b", n), c("a", "b", "c"))), ry = ry, x = x12),
  no_predictor = list(y = blank(fac(abc, c("a", "b", "c"))), ry = ry, x = x0),
  one_missing = list(y = blank(fac(abc, c("a", "b", "c")), ry_one), ry = ry_one, x = x12),
  wy_subset = list(y = blank(fac(abc, c("a", "b", "c"))), ry = ry, x = x12, wy = seq_len(n) %in% mis[c(2, 3, 5, 8, 9)]),
  separated = list(y = blank(fac(sep, c("a", "b", "c"))), ry = ry, x = x12),
  five_levels = list(y = blank(fac(five, c("a", "b", "c", "d", "e")), ry5), ry = ry5, x = x5),
  # where R fails
  two_levels_one_missing = list(y = blank(fac(yes, c("no", "yes")), ry_one), ry = ry_one, x = x12),
  ordered_unused = list(
    y = blank(fac(relabel(abc, c(a = "lo", b = "mid", c = "hi")), c("lo", "mid", "hi", "top"), ordered = TRUE)),
    ry = ry, x = x12
  ),
  # the missing cells hold "d" (inside mice: the current imputations), which
  # no observed row has; without predictors there is no augmentation
  imputed_only_level = list(
    y = fac(ifelse(ry, abc, "d"), c("a", "b", "c", "d")), ry = ry, x = x0
  ),
  many_categories = list(y = blank(fac(many, sort(unique(many))), ry51), ry = ry51, x = x51)
)

# ---------------------------------------------------------------------------
# The port's computation, rebuilt from mice's parts
# ---------------------------------------------------------------------------

# The categories in the port's order: declared (present ones) for an ordered
# factor, the sorted labels present for every other target
port_levels <- function(y) {
  present <- y[!is.na(y)]
  if (is.ordered(y)) {
    levels(y)[levels(y) %in% as.character(present)]
  } else {
    sort(unique(as.character(present)))
  }
}

# Augment, draw and fit as the port does, with mice's augment(), the call of
# nnet::multinom that mice.impute.polyreg makes, and its cumulative rule.
# The fit sees the target with the port's levels; classes without an observed
# row are dropped by multinom and get probability 0. Starts from
# set.seed(seed).
port_steps <- function(y, ry, x, wy, seed, maxit, max_nwts) {
  lev <- port_levels(y)
  aug <- augment(y, ry, as.matrix(x), wy)
  set.seed(seed)
  u <- runif(sum(wy))
  seed_after <- .Random.seed
  yl <- factor(as.character(aug$y), levels = lev)
  observed <- unique(as.character(yl[aug$ry]))
  if (length(observed) == 1L) {
    return(list(u = u, fit = NULL, codes = rep(match(observed, lev), length(u)), seed_after = seed_after))
  }
  xa <- aug$x
  w <- aug$w
  xy <- cbind.data.frame(y = yl, x = xa)
  if (ncol(xa) == 0L) {
    xy <- data.frame(xy, int = 1)
  }
  fit <- suppressWarnings(nnet::multinom(formula(xy),
    data = xy[aug$ry, , drop = FALSE], weights = w[aug$ry],
    maxit = maxit, trace = FALSE, MaxNWts = max_nwts
  ))
  post <- predict(fit, xy[aug$wy, , drop = FALSE], type = "probs")
  n0 <- sum(aug$wy)
  post <- if (length(fit$lev) == 2L) cbind(1 - as.vector(post), as.vector(post)) else matrix(post, nrow = n0)
  prob <- matrix(0, n0, length(lev))
  prob[, match(fit$lev, lev)] <- post
  cum <- matrix(t(apply(prob, 1, cumsum)), nrow = n0)
  stopifnot(min(abs(u - cum)) > 1e-4)
  codes <- pmin(1L + rowSums(u > cum), length(lev))
  list(u = u, fit = fit, prob = prob, codes = codes, seed_after = seed_after)
}

# The weights of a multinom fit as the port's coefficients: one row per
# fitted class but the first, one column per column of the model matrix
# (nnet's bias weights, fixed at 0, left out)
multinom_coef <- function(fit) {
  r <- length(fit$coefnames)
  w <- matrix(fit$wts, nrow = r + 1L)[-1L, , drop = FALSE]
  if (ncol(w) > 1L) w <- w[, -1L, drop = FALSE]
  unname(t(w))
}

# ---------------------------------------------------------------------------
# One polyreg case
# ---------------------------------------------------------------------------

pair <- function(pool, k) list(pool, as.integer(k))

# port: "match" (R's result), "rebuilt" (R stops; the port's computation
# rebuilt) or "value_error" (R and the port stop; r_error holds R's message).
# r_draws: whether mice drew its uniform numbers before it stopped.
polyreg_case <- function(name, input, seed, maxit = 100, max_nwts = 1500, port = "match", r_draws = TRUE) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- if (is.null(inp$wy)) !ry else inp$wy
  x <- inp$x
  n0 <- sum(wy)

  set.seed(seed)
  imp <- tryCatch(
    suppressWarnings(mice.impute.polyreg(y, ry, x, wy = wy, nnet.maxit = maxit, nnet.MaxNWts = max_nwts)),
    error = function(e) e
  )
  saved <- .Random.seed
  failed <- inherits(imp, "error")
  stopifnot(failed == (port != "match"))

  set.seed(seed)
  u <- if (r_draws) runif(n0) else numeric(0)
  stopifnot(identical(.Random.seed, saved))

  steps <- NULL
  if (port != "value_error") {
    steps <- port_steps(y, ry, x, wy, seed, maxit, max_nwts)
    stopifnot(identical(steps$seed_after, saved))
  }
  levels <- port_levels(y)
  imputed <- if (is.null(steps)) NULL else levels[steps$codes]
  if (port == "match") {
    stopifnot(identical(as.character(imp), imputed))
  }
  fit <- steps$fit
  list(
    name = name,
    input = input,
    seed = seed,
    port = port,
    maxit = maxit,
    max_weights = max_nwts,
    n_imp = n0,
    uniforms = I(u),
    order = if (r_draws) list(pair("uniform", n0)) else list(),
    value = fit$value,
    convergence = fit$convergence,
    coef = if (is.null(fit)) NULL else multinom_coef(fit),
    fitted_levels = if (is.null(fit)) NULL else I(fit$lev),
    prob = if (is.null(fit)) NULL else unname(steps$prob),
    levels = I(levels),
    imputed = if (is.null(imputed)) NULL else I(imputed),
    r_error = if (failed) conditionMessage(imp) else NULL
  )
}

cases <- list(
  polyreg_case("polyreg_seed1", "base", 7411L),
  polyreg_case("polyreg_seed2", "base", 7412L),
  polyreg_case("polyreg_two_levels", "two_levels", 7413L),
  polyreg_case("polyreg_unsorted", "unsorted", 7414L),
  polyreg_case("polyreg_numeric_labels", "numeric_labels", 7415L),
  polyreg_case("polyreg_unused_level", "unused_level", 7416L),
  polyreg_case("polyreg_ordered", "ordered", 7417L),
  # one category observed: the uniforms are drawn, nothing is fitted
  polyreg_case("polyreg_one_category", "one_category", 7418L),
  polyreg_case("polyreg_no_predictor", "no_predictor", 7419L),
  polyreg_case("polyreg_one_missing", "one_missing", 7420L),
  polyreg_case("polyreg_wy_subset", "wy_subset", 7421L),
  polyreg_case("polyreg_separated", "separated", 7422L),
  polyreg_case("polyreg_five_levels", "five_levels", 7423L),
  # the fit stops at the iteration limit
  polyreg_case("polyreg_maxit", "base", 7424L, maxit = 5),
  # R stops after its draws: dim(X) must have a positive length
  polyreg_case("polyreg_two_levels_one_missing", "two_levels_one_missing", 7425L, port = "rebuilt"),
  # R stops after its draws: dims do not match; the port drops "top"
  polyreg_case("polyreg_ordered_unused", "ordered_unused", 7426L, port = "rebuilt"),
  # R stops after its draws (multinom drops "d", which then has no column);
  # the port gives "d" probability 0
  polyreg_case("polyreg_imputed_only_level", "imputed_only_level", 7427L, port = "rebuilt"),
  # R stops after its draws: 3 classes and 3 columns need 12 weights
  polyreg_case("polyreg_too_many_weights", "base", 7428L, max_nwts = 10, port = "value_error"),
  # augment() stops before any draw
  polyreg_case("polyreg_many_categories", "many_categories", 7429L, port = "value_error", r_draws = FALSE)
)
by_name <- setNames(cases, vapply(cases, `[[`, "", "name"))

for (case in cases) message(sprintf("%-32s value %-22.17g convergence %s", case$name, if (is.null(case$value)) NA else case$value, if (is.null(case$convergence)) "-" else case$convergence))
stopifnot(
  by_name$polyreg_maxit$convergence == 1L,
  is.null(by_name$polyreg_one_category$value),
  identical(unclass(by_name$polyreg_one_category$imputed), rep("b", 10)),
  identical(unclass(by_name$polyreg_numeric_labels$levels), c("1", "10", "2")),
  identical(unclass(by_name$polyreg_unused_level$levels), c("a", "b", "c")),
  identical(unclass(by_name$polyreg_ordered$levels), c("lo", "mid", "hi")),
  identical(unclass(by_name$polyreg_imputed_only_level$fitted_levels), c("a", "b", "c")),
  all(by_name$polyreg_imputed_only_level$prob[, 4] == 0),
  identical(by_name$polyreg_two_levels_one_missing$r_error, "dim(X) must have a positive length"),
  identical(by_name$polyreg_ordered_unused$r_error, "dims [product 30] do not match the length of object [40]"),
  identical(by_name$polyreg_imputed_only_level$r_error, "dims [product 30] do not match the length of object [40]"),
  identical(by_name$polyreg_too_many_weights$r_error, "too many (12) weights"),
  identical(by_name$polyreg_many_categories$r_error, "Maximum number of categories (50) exceeded")
)

# a factor declared as (c, a, b) is imputed as one declared in sorted order
set.seed(7414L)
stopifnot(identical(
  as.character(mice.impute.polyreg(inputs$base$y, ry, x12)),
  unclass(by_name$polyreg_unsorted$imputed)
))
# the port's result of a target R fails on is what mice computes for an
# equivalent target that it does impute
set.seed(7426L)
stopifnot(identical(
  as.character(mice.impute.polyreg(droplevels(inputs$ordered_unused$y), ry, x12)),
  unclass(by_name$polyreg_ordered_unused$imputed)
))
set.seed(7427L)
stopifnot(identical(
  as.character(suppressWarnings(mice.impute.polyreg(droplevels(blank(inputs$imputed_only_level$y)), ry, x0))),
  unclass(by_name$polyreg_imputed_only_level$imputed)
))

# ---------------------------------------------------------------------------
# Direct nnet::multinom fits
# ---------------------------------------------------------------------------

# Fits that polyreg does not reach through R: the two-class logistic unit on
# separated data (its output is cut at 0 and 1 beyond an activation of 15,
# which polyreg's augmentation prevents), and case weights other than those
# of augment(). y holds 0-based classes; x and new_x are model matrices.
multinom_case <- function(name, x, y, weights, new_x, n_classes = max(y) + 1L) {
  d <- data.frame(y = factor(y), x[, -1L, drop = FALSE])
  stopifnot(all(model.matrix(y ~ ., d) == x))
  fit <- nnet::multinom(y ~ ., data = d, weights = weights, trace = FALSE)
  post <- predict(fit, data.frame(new_x[, -1L, drop = FALSE]), type = "probs")
  if (length(fit$lev) == 2L) post <- cbind(1 - post, post)
  list(
    name = name,
    x = unname(x),
    y = I(as.integer(y)),
    weights = I(weights),
    n_classes = n_classes,
    new_x = unname(new_x),
    value = fit$value,
    convergence = fit$convergence,
    coef = multinom_coef(fit),
    prob = unname(post)
  )
}

# A fit of separated data ends when its objective falls below abstol = 1e-4,
# and how far it is below depends on the whole path; on some data the last
# digits of that path move the objective by 1e-9 (relative). 7431 is the
# first seed from 7430 on whose fit, in Python with every value perturbed by
# 1e-15 (relative), keeps its objective within 1e-11 (7430: 4e-9).
set.seed(7431)
xs <- sort(c(runif(12, -3, -0.2), runif(12, 0.2, 3)))
x_sep <- cbind(1, xs = xs, xn = rnorm(24))
new_sep <- cbind(1, xs = c(-3, -1, -0.1, 0, 0.1, 1, 3), xn = c(0, 0.5, -0.5, 0, 1, -1, 0))
set.seed(7432)
i45 <- 1:45
x_w <- cbind(1, x1 = ((37 * i45) %% 101) / 50 - 1, x2 = ((61 * i45) %% 89) / 20 - 2)
w45 <- ifelse(i45 %% 9 == 0, 0.25, ifelse(i45 %% 7 == 0, 2.5, 1))
y_w3 <- as.integer(cut(x_w[, 2] + 0.5 * x_w[, 3] + rnorm(45), c(-Inf, -0.4, 0.4, Inf))) - 1L
y_w2 <- as.integer(x_w[, 2] - 0.3 * x_w[, 3] + rnorm(45, 0, 0.7) > 0)
new_w <- cbind(1, x1 = c(-1, 0, 0.5, 1), x2 = c(-2, 0, 1, 2.4))

multinom_cases <- list(
  multinom_case("two_classes_separated", x_sep, as.integer(xs > 0), rep(1, 24), new_sep),
  multinom_case("three_classes_weighted", x_w, y_w3, w45, new_w),
  multinom_case("two_classes_weighted", x_w, y_w2, w45, new_w)
)
mc <- setNames(multinom_cases, vapply(multinom_cases, `[[`, "", "name"))
stopifnot(
  # beyond the cut-off the logistic unit returns exactly 0 and 1
  mc$two_classes_separated$prob[1, 2] == 0, mc$two_classes_separated$prob[7, 2] == 1,
  all(table(mc$three_classes_weighted$y) >= 8), all(table(mc$two_classes_weighted$y) >= 15),
  all(vapply(multinom_cases, function(case) case$convergence == 0L, TRUE))
)

encode_input <- function(inp) {
  y <- inp$y
  list(
    kind = if (is.ordered(y)) "ordered" else "factor",
    y = I(as.character(y)),
    levels = I(levels(y)),
    ry = I(inp$ry),
    wy = if (is.null(inp$wy)) NULL else I(inp$wy),
    x = if (ncol(inp$x) > 0L) unname(inp$x) else NULL,
    p = ncol(inp$x)
  )
}

write_fixture("l1_polyreg", list(
  description = paste(
    "mice.impute.polyreg(y, ry, x, wy, nnet.maxit = maxit, nnet.MaxNWts =",
    "max_weights) with its uniform numbers captured: uniforms = runif(n_imp),",
    "order the expected sequence of draws as [pool, count] pairs. value,",
    "convergence and coef come from the nnet::multinom fit (coef: one row per",
    "fitted class but the first, one column per column of [1, x]; nnet's bias",
    "weights left out); fitted_levels are its classes, prob the probabilities",
    "of the port's levels in each imputed cell (0 for a level without an",
    "observed row), levels the port's level order and imputed the port's",
    "labels. Inputs: kind, y (labels; null for NA), levels (declared), ry, wy",
    "(null means !ry), x without intercept (null when p = 0). port: 'match'",
    "(R's result), 'rebuilt' (R stops, r_error; the port's computation rebuilt",
    "with mice's augment and nnet::multinom) or 'value_error' (R and the port",
    "stop; r_error). multinom_cases: nnet::multinom(y ~ ., weights) on the",
    "model matrix x with 0-based classes y; prob the predicted probabilities",
    "of the rows new_x."
  ),
  nnet = NNET_VERSION,
  inputs = lapply(inputs, encode_input),
  cases = cases,
  multinom_cases = multinom_cases
))
