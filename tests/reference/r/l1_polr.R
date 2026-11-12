# L1 fixture for imputation of an ordered categorical target by the
# proportional-odds model.
#
#   l1_polr.json   mice.impute.polr with its uniform numbers captured, the
#                  class of the fit R made (MASS::polr, or nnet::multinom
#                  where polr stops) and polr's error, the fit (coefficients,
#                  cut points, deviance or objective, convergence code), the
#                  probabilities of every imputed cell and the labels; and
#                  direct MASS::polr fits for what the imputer cannot reach
#
# Python targets: imputation.polr.polr (through its step-by-step version
# _polr) driven by a strict ScriptedGenerator, and imputation.fitting.polr_fit.
#
# Each case calls mice.impute.polr after set.seed(seed) and saves
# .Random.seed. The port's computation is then rebuilt with mice's own
# augment(), MASS::polr() or nnet::multinom() and the cumulative rule,
# drawing runif(n_imp) after set.seed(seed) again (polr draws nothing else,
# and draws it after the fit). Where mice succeeds, the rebuild must leave
# the same .Random.seed and give identical labels; where mice stops, the
# draws mice made before are checked the same way and the rebuild records
# what the port does instead (see `port` below).
#
# R's own sequence of fits inside mice.impute.polr (polr, then multinom if
# polr fails) is repeated on mice's augmented data to record the class of the
# fit R made and polr's error message (`fit`, `polr_error`); the rebuild
# records the port's (`port_fit`, `port_fallback`). They agree for every case
# whose port is "match".
#
# Every case also requires min |u - cumulative probability| > 1e-4 over the
# imputed cells and all their cumulative probabilities, so that no label
# depends on the last digits of a probability.
#
# Level order. The port orders the categories of an ordered factor as
# declared, unused levels included (MASS::polr counts every declared level),
# and those of every other target by their sorted labels, R's order after
# augment(); polr then takes that order as the ordinal one. Labels are
# lowercase ASCII, whose sort order does not depend on the locale.

source(file.path("tests", "reference", "r", "_common.R"))

augment <- mice:::augment
MASS_VERSION <- as.character(packageVersion("MASS"))
NNET_VERSION <- as.character(packageVersion("nnet"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(7601)
n <- 40
x1 <- rnorm(n)
x2 <- rnorm(n)
x12 <- cbind(x1 = x1, x2 = x2)
x0 <- x12[, 0, drop = FALSE]
# levels from a proportional-odds model: 1 + number of cut points below
# 1.3 x1 - 0.9 x2 + logistic noise
latent <- 1.3 * x1 - 0.9 * x2 + rlogis(n)
code4 <- 1L + findInterval(latent, c(-1.8, -0.7, 0.6))
code3 <- 1L + findInterval(latent, c(-1.5, 0.3))
mis <- sort(sample(n, 10))
ry <- !(seq_len(n) %in% mis)
ry_one <- seq_len(n) != mis[1]
stopifnot(all(table(factor(code4[ry], 1:4)) >= 6), all(table(factor(code3[ry], 1:3)) >= 8))

abcd <- c("a", "b", "c", "d")
# the middle split (a, b below c, d) separated by x1: a or b where x1 < 0,
# c or d above; within each half the label is drawn at random
low_half <- sample(c("a", "b"), n, replace = TRUE)
high_half <- sample(c("c", "d"), n, replace = TRUE)
sep <- ifelse(x1 < 0, low_half, high_half)
# the same with the second level never used: a below 0, c or d above
sep_unused <- ifelse(x1 < 0, "a", high_half)
stopifnot(
  all(table(factor(sep[ry_one], abcd)) >= 5),
  sum(sep_unused[ry_one] == "a") >= 10,
  max(x1[ry_one][sep[ry_one] %in% c("a", "b")]) < min(x1[ry_one][sep[ry_one] %in% c("c", "d")])
)

# 51 categories, for the limit of augment()
n51 <- 102
x51 <- cbind(u = rnorm(n51))
many <- sprintf("l%02d", rep(1:51, 2))
ry51 <- seq_len(n51) > 4

blank <- function(y, keep = ry) replace(y, !keep, NA)
ord <- function(v, levels) factor(v, levels = levels, ordered = TRUE)

inputs <- list(
  base = list(y = blank(ord(abcd[code4], abcd)), ry = ry, x = x12),
  # declared low < mid < high < top, alphabetically high < low < mid < top
  declared_order = list(y = blank(ord(c("low", "mid", "high", "top")[code4], c("low", "mid", "high", "top"))), ry = ry, x = x12),
  # declared (c, a, d, b); after augment() the levels are sorted, and polr
  # takes a < b < c < d as the order
  unordered = list(y = blank(factor(abcd[code4], c("c", "a", "d", "b"))), ry = ry, x = x12),
  # text; mice-py takes it as an unordered factor (see the case)
  text = list(y = blank(abcd[code4]), ry = ry, x = x12),
  # "b" is declared but never occurs
  unused_middle = list(y = blank(ord(c("a", "c", "d")[code3], abcd)), ry = ry, x = x12),
  # "d" is declared but never occurs
  unused_end = list(y = blank(ord(c("a", "b", "c")[code3], abcd)), ry = ry, x = x12),
  one_missing = list(y = blank(ord(abcd[code4], abcd), ry_one), ry = ry_one, x = x12),
  wy_subset = list(y = blank(ord(abcd[code4], abcd)), ry = ry, x = x12, wy = seq_len(n) %in% mis[c(1, 4, 6, 7, 10)]),
  no_predictor = list(y = blank(ord(abcd[code4], abcd)), ry = ry, x = x0),
  no_predictor_two_levels = list(y = blank(ord(c("lo", "hi")[1L + (code4 > 2L)], c("lo", "hi"))), ry = ry, x = x0),
  # fallbacks
  two_levels = list(y = blank(ord(c("lo", "hi")[1L + (code4 > 2L)], c("lo", "hi"))), ry = ry, x = x12),
  numeric = list(y = blank(as.numeric(code4)), ry = ry, x = x12),
  logical = list(y = blank(code4 > 2L), ry = ry, x = x12),
  # one missing value: no augmentation, and the start fit of the middle
  # split diverges
  separated = list(y = blank(ord(sep, abcd), ry_one), ry = ry_one, x = x12),
  # where R stops or does something else
  one_category = list(y = blank(ord(rep("b", n), abcd)), ry = ry, x = x12),
  one_category_unordered = list(y = blank(factor(rep("b", n), abcd)), ry = ry, x = x12),
  two_levels_one_missing = list(y = blank(ord(c("lo", "hi")[1L + (code4 > 2L)], c("lo", "hi")), ry_one), ry = ry_one, x = x12),
  separated_unused = list(y = blank(ord(sep_unused, abcd), ry_one), ry = ry_one, x = x12),
  # a constant column: aliased with the intercept in the start fit
  constant_column = list(y = blank(ord(abcd[code4], abcd)), ry = ry, x = cbind(x12, k = 1)),
  many_categories = list(y = blank(factor(many, sort(unique(many))), ry51), ry = ry51, x = x51)
)

# ---------------------------------------------------------------------------
# R's fits inside mice.impute.polr, and the port's computation
# ---------------------------------------------------------------------------

# The class of the fit mice.impute.polr makes (NULL if both fits fail) and
# polr's error message (NULL if polr succeeds): the sequence of fits of
# mice.impute.polr on mice's augmented data. The fits draw no random numbers.
r_fits <- function(y, ry, x, wy, maxit, max_nwts) {
  aug <- augment(y, ry, as.matrix(x), wy)
  xy <- cbind.data.frame(y = aug$y, x = aug$x)
  seed <- .Random.seed
  fit <- tryCatch(
    suppressWarnings(MASS::polr(formula(xy), data = xy[aug$ry, , drop = FALSE], weights = aug$w[aug$ry])),
    error = function(e) e
  )
  polr_error <- NULL
  if (inherits(fit, "error")) {
    polr_error <- conditionMessage(fit)
    fit <- tryCatch(
      suppressWarnings(nnet::multinom(formula(xy),
        data = xy[aug$ry, , drop = FALSE], weights = aug$w[aug$ry],
        maxit = maxit, trace = FALSE, MaxNWts = max_nwts
      )),
      error = function(e) e
    )
  }
  stopifnot(identical(.Random.seed, seed))
  list(fit = if (inherits(fit, "error")) NULL else class(fit)[1], polr_error = polr_error)
}

# The categories in the port's order: the declared levels of an ordered
# factor, the sorted labels present of any other target (as text)
port_levels <- function(y) {
  if (is.ordered(y)) {
    return(levels(y))
  }
  present <- y[!is.na(y)]
  if (is.factor(y) || is.character(y)) sort(unique(as.character(present))) else as.character(sort(unique(present)))
}

# Augment, fit, predict and draw as the port does, with mice's augment(), the
# calls of MASS::polr and nnet::multinom that mice.impute.polr makes, and its
# cumulative rule. A text target counts as an unordered factor; a numeric or
# logical one is no factor, so polr is not tried. Columns that polr drops as
# aliased get the coefficient 0. A level without an observed row gets
# probability 0 in the fallback. Without predictors both fits have the
# intercept only (polr: the cut points alone), where R's fits fail. Starts
# from set.seed(seed).
port_steps <- function(y, ry, x, wy, seed, maxit, max_nwts) {
  lev <- port_levels(y)
  aug <- augment(y, ry, as.matrix(x), wy)
  yl <- factor(as.character(aug$y), levels = lev)
  n0 <- sum(aug$wy)
  observed <- unique(as.character(yl[aug$ry]))
  if (length(observed) == 1L) {
    set.seed(seed)
    u <- runif(n0)
    return(list(
      u = u, fit = NULL, fit_class = "none", fallback = NULL,
      codes = rep(match(observed, lev), n0), seed_after = .Random.seed
    ))
  }
  d <- data.frame(y = yl, aug$x)
  xnames <- colnames(d)[-1L]
  model <- if (length(xnames)) y ~ . else y ~ 1
  obs <- d[aug$ry, , drop = FALSE]
  new <- d[aug$wy, , drop = FALSE]
  fallback <- "response must be a factor"
  fit <- NULL
  if (is.factor(y) || is.character(y)) {
    fit <- tryCatch(suppressWarnings(MASS::polr(model, data = obs, weights = aug$w[aug$ry])), error = function(e) e)
    if (inherits(fit, "error")) {
      fallback <- conditionMessage(fit)
      fit <- NULL
    } else {
      fallback <- NULL
    }
  }
  if (is.null(fit)) {
    fit <- suppressWarnings(nnet::multinom(model,
      data = obs, weights = aug$w[aug$ry], maxit = maxit, trace = FALSE, MaxNWts = max_nwts
    ))
    post <- predict(fit, new, type = "probs")
    post <- if (length(fit$lev) == 2L) cbind(1 - as.vector(post), as.vector(post)) else matrix(post, nrow = n0)
    prob <- matrix(0, n0, length(lev))
    prob[, match(fit$lev, lev)] <- post
    out <- list(fit = fit, fit_class = "multinom", value = fit$value, coef = multinom_coef(fit), fitted_levels = I(fit$lev))
  } else {
    coef <- setNames(numeric(length(xnames)), xnames)
    coef[names(fit$coefficients)] <- fit$coefficients
    eta <- if (length(xnames)) drop(as.matrix(new[, xnames, drop = FALSE]) %*% coef) else rep(0, n0)
    cumulative <- plogis(matrix(fit$zeta, n0, length(fit$zeta), byrow = TRUE) - eta)
    prob <- matrix(t(apply(cumulative, 1, function(p) diff(c(0, p, 1)))), nrow = n0)
    out <- list(
      fit = fit, fit_class = "polr", coef = unname(coef), zeta = unname(fit$zeta), deviance = fit$deviance,
      niter = I(unname(fit$niter))
    )
  }
  set.seed(seed)
  u <- runif(n0)
  cum <- matrix(t(apply(prob, 1, cumsum)), nrow = n0)
  stopifnot(min(abs(u - cum)) > 1e-4)
  c(out, list(
    u = u, fallback = fallback, prob = prob, convergence = fit$convergence,
    codes = pmin(1L + rowSums(u > cum), length(lev)), seed_after = .Random.seed
  ))
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
# One polr case
# ---------------------------------------------------------------------------

pair <- function(pool, k) list(pool, as.integer(k))

# port: "match" (R's result), "rebuilt" (R stops or computes something else;
# the port's computation rebuilt) or "value_error" (R and the port stop;
# r_error holds R's message, and neither draws). r_draws: whether mice drew
# its uniform numbers (before it stopped).
polr_case <- function(name, input, seed, maxit = 100, max_nwts = 1500, port = "match", r_draws = TRUE) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- if (is.null(inp$wy)) !ry else inp$wy
  x <- inp$x
  n0 <- sum(wy)

  set.seed(seed)
  imp <- tryCatch(
    suppressWarnings(mice.impute.polr(y, ry, x, wy = wy, nnet.maxit = maxit, nnet.MaxNWts = max_nwts)),
    error = function(e) e
  )
  saved <- .Random.seed
  failed <- inherits(imp, "error")
  stopifnot(!failed || port != "match", failed || port != "value_error")

  set.seed(seed)
  if (r_draws) runif(n0)
  stopifnot(identical(.Random.seed, saved))

  # augment() refuses more than 50 categories before any fit
  too_many <- length(unique(na.omit(as.character(y)))) > 50L
  fits <- if (too_many) list(fit = NULL, polr_error = NULL) else r_fits(y, ry, x, wy, maxit, max_nwts)
  steps <- NULL
  if (port != "value_error") {
    steps <- port_steps(y, ry, x, wy, seed, maxit, max_nwts)
    if (r_draws) stopifnot(identical(steps$seed_after, saved))
  }
  levels <- port_levels(y)
  imputed <- if (is.null(steps)) NULL else levels[steps$codes]
  if (port == "match") {
    stopifnot(
      identical(as.character(imp), imputed),
      identical(fits$fit, steps$fit_class),
      identical(fits$polr_error, steps$fallback)
    )
  }
  list(
    name = name,
    input = input,
    seed = seed,
    port = port,
    maxit = maxit,
    max_weights = max_nwts,
    n_imp = n0,
    uniforms = I(if (port == "value_error") numeric(0) else steps$u),
    order = if (port == "value_error") list() else list(pair("uniform", n0)),
    r_draws = r_draws,
    fit = fits$fit,
    polr_error = fits$polr_error,
    port_fit = steps$fit_class,
    port_fallback = steps$fallback,
    coef = steps$coef,
    zeta = if (is.null(steps$zeta)) NULL else I(steps$zeta),
    deviance = steps$deviance,
    niter = steps$niter,
    value = steps$value,
    fitted_levels = steps$fitted_levels,
    convergence = steps$convergence,
    prob = if (is.null(steps$prob)) NULL else unname(steps$prob),
    levels = I(levels),
    imputed = if (is.null(imputed)) NULL else I(imputed),
    r_imputed = if (failed) NULL else I(as.character(imp)),
    r_error = if (failed) conditionMessage(imp) else NULL
  )
}

cases <- list(
  polr_case("polr_seed1", "base", 7611L),
  polr_case("polr_seed2", "base", 7612L),
  polr_case("polr_declared_order", "declared_order", 7613L),
  polr_case("polr_unordered", "unordered", 7614L),
  # R: polr refuses a character target (not a factor) and multinom imputes;
  # mice() itself drops a character column as constant. mice-py takes text
  # as an unordered factor: the port's result is R's for factor(y)
  polr_case("polr_text", "text", 7615L, port = "rebuilt"),
  polr_case("polr_unused_middle", "unused_middle", 7616L),
  polr_case("polr_unused_end", "unused_end", 7617L),
  polr_case("polr_one_missing", "one_missing", 7618L),
  polr_case("polr_wy_subset", "wy_subset", 7619L),
  # the fallback: fewer than 3 levels, a target that is no factor, a start
  # fit that does not converge; and the fallback stopped at its iteration
  # limit (nnet.maxit = 5)
  polr_case("polr_two_levels", "two_levels", 7620L),
  polr_case("polr_numeric", "numeric", 7621L),
  polr_case("polr_logical", "logical", 7622L),
  polr_case("polr_separated", "separated", 7623L),
  polr_case("polr_fallback_maxit", "numeric", 7624L, maxit = 5),
  # one category: R fits a degenerate polr model (its start fit converges);
  # the port imputes the category without a fit
  polr_case("polr_one_category", "one_category", 7625L, port = "rebuilt"),
  # R stops without a draw: polr has one level, multinom one class
  polr_case("polr_one_category_unordered", "one_category_unordered", 7626L, port = "rebuilt", r_draws = FALSE),
  # R stops after its draw: dim(X) must have a positive length
  polr_case("polr_two_levels_one_missing", "two_levels_one_missing", 7627L, port = "rebuilt"),
  # R stops after its draw: multinom drops "b", whose column is then missing
  polr_case("polr_separated_unused", "separated_unused", 7628L, port = "rebuilt"),
  # R's polr drops the constant column; predict then stops before the draws
  polr_case("polr_constant_column", "constant_column", 7629L, port = "rebuilt", r_draws = FALSE),
  # R stops without a draw: without predictors polr sees no response and
  # multinom no classes; the port fits polr without predictors (cut points
  # only), and where that fails (two levels) multinom on the intercept
  polr_case("polr_no_predictor", "no_predictor", 7630L, port = "rebuilt", r_draws = FALSE),
  polr_case("polr_no_predictor_two_levels", "no_predictor_two_levels", 7633L, port = "rebuilt", r_draws = FALSE),
  # the fallback has 4 classes and 3 columns, 16 weights; R stops before the draws
  polr_case("polr_too_many_weights", "numeric", 7631L, max_nwts = 10, port = "value_error", r_draws = FALSE),
  # augment() stops before any draw
  polr_case("polr_many_categories", "many_categories", 7632L, port = "value_error", r_draws = FALSE)
)
by_name <- setNames(cases, vapply(cases, `[[`, "", "name"))

for (case in cases) {
  message(sprintf(
    "%-30s R %-9s port %-9s polr error: %s", case$name, if (is.null(case$fit)) "-" else case$fit,
    if (is.null(case$port_fit)) "-" else case$port_fit, if (is.null(case$polr_error)) "-" else case$polr_error
  ))
}
stopifnot(
  by_name$polr_seed1$fit == "polr", by_name$polr_unordered$fit == "polr",
  by_name$polr_unused_middle$fit == "polr", by_name$polr_unused_end$fit == "polr",
  identical(unclass(by_name$polr_declared_order$levels), c("low", "mid", "high", "top")),
  identical(unclass(by_name$polr_unordered$levels), abcd),
  identical(unclass(by_name$polr_unused_middle$levels), abcd),
  by_name$polr_two_levels$polr_error == "response must have 3 or more levels",
  by_name$polr_numeric$polr_error == "response must be a factor",
  by_name$polr_logical$polr_error == "response must be a factor",
  by_name$polr_separated$polr_error == "attempt to find suitable starting values failed",
  by_name$polr_fallback_maxit$convergence == 1L,
  # text: R falls back, the port fits polr
  by_name$polr_text$fit == "multinom", by_name$polr_text$port_fit == "polr",
  by_name$polr_text$polr_error == "response must be a factor",
  # one ordered category: R imputes it through polr, the port without a fit
  by_name$polr_one_category$fit == "polr", by_name$polr_one_category$port_fit == "none",
  identical(unclass(by_name$polr_one_category$r_imputed), unclass(by_name$polr_one_category$imputed)),
  identical(by_name$polr_one_category_unordered$r_error, "need two or more classes to fit a multinom model"),
  identical(by_name$polr_two_levels_one_missing$r_error, "dim(X) must have a positive length"),
  by_name$polr_separated_unused$polr_error == "attempt to find suitable starting values failed",
  identical(by_name$polr_separated_unused$r_error, "dims [product 3] do not match the length of object [4]"),
  identical(by_name$polr_constant_column$r_error, "non-conformable arguments"),
  identical(by_name$polr_no_predictor$polr_error, "response must be a factor"),
  identical(by_name$polr_no_predictor$r_error, "need two or more classes to fit a multinom model"),
  by_name$polr_no_predictor$port_fit == "polr", length(by_name$polr_no_predictor$coef) == 0L,
  identical(by_name$polr_no_predictor_two_levels$r_error, "need two or more classes to fit a multinom model"),
  by_name$polr_no_predictor_two_levels$port_fit == "multinom",
  by_name$polr_no_predictor_two_levels$port_fallback == "response must have 3 or more levels",
  identical(by_name$polr_too_many_weights$r_error, "too many (16) weights"),
  identical(by_name$polr_many_categories$r_error, "Maximum number of categories (50) exceeded")
)
# the unused middle level gets a positive probability; the fallback gives
# the unused level of separated_unused probability 0
stopifnot(
  all(by_name$polr_unused_middle$prob[, 2] > 0),
  all(by_name$polr_separated_unused$prob[, 2] == 0)
)

# The port's result of a target R treats otherwise is what mice computes
# for an equivalent target that it imputes the port's way
same_as <- function(case, y, x = x12, r = ry) {
  set.seed(case$seed)
  stopifnot(identical(as.character(suppressWarnings(mice.impute.polr(y, r, x))), unclass(case$imputed)))
}
same_as(by_name$polr_text, factor(inputs$text$y))
same_as(by_name$polr_separated_unused, droplevels(inputs$separated_unused$y), r = ry_one)
same_as(by_name$polr_constant_column, inputs$constant_column$y)
# a factor declared as (c, a, d, b) is imputed as one declared in sorted order
same_as(by_name$polr_unordered, blank(factor(abcd[code4], abcd)))

# ---------------------------------------------------------------------------
# Direct MASS::polr fits
# ---------------------------------------------------------------------------

# Fits that the imputer does not reach through R: case weights other than
# those of augment(), 3 and 5 levels, no predictor, aliased columns; and
# the conditions under which polr stops. y holds 1-based levels; x the
# predictors without intercept; new_x the rows to predict.
polr_direct <- function(name, x, y, n_levels, weights = rep(1, length(y)), new_x = x[1:4, , drop = FALSE]) {
  lev <- letters[seq_len(n_levels)]
  d <- data.frame(y = factor(lev[y], levels = lev, ordered = TRUE), x)
  fit <- tryCatch(
    suppressWarnings(MASS::polr(if (ncol(x)) y ~ . else y ~ 1, data = d, weights = weights)),
    error = function(e) e
  )
  out <- list(
    name = name, x = if (ncol(x)) unname(x) else NULL, p = ncol(x), y = I(as.integer(y) - 1L),
    n_levels = n_levels, weights = I(weights), new_x = if (ncol(x)) unname(new_x) else NULL,
    n_new = nrow(new_x)
  )
  if (inherits(fit, "error")) {
    return(c(out, list(r_error = conditionMessage(fit))))
  }
  xnames <- colnames(d)[-1L]
  coef <- setNames(numeric(length(xnames)), xnames)
  coef[names(fit$coefficients)] <- fit$coefficients
  eta <- if (ncol(x)) drop(new_x %*% coef) else rep(0, nrow(new_x))
  cumulative <- plogis(matrix(fit$zeta, nrow(new_x), n_levels - 1L, byrow = TRUE) - eta)
  c(out, list(
    coef = I(unname(coef)), aliased = I(!(xnames %in% names(fit$coefficients))),
    zeta = I(unname(fit$zeta)), deviance = fit$deviance, convergence = fit$convergence,
    niter = I(unname(fit$niter)),
    prob = unname(t(apply(cumulative, 1, function(p) diff(c(0, p, 1)))))
  ))
}

# 1-based levels from a latent variable and cut points
levels_of <- function(latent, cuts) 1L + findInterval(latent, cuts)

set.seed(7641)
i60 <- 1:60
x_w <- cbind(x1 = ((37 * i60) %% 101) / 50 - 1, x2 = ((61 * i60) %% 89) / 20 - 2)
w60 <- ifelse(i60 %% 9 == 0, 0.25, ifelse(i60 %% 7 == 0, 2.5, 1))
y_w <- levels_of(1.1 * x_w[, 1] - 0.7 * x_w[, 2] + rlogis(60), c(-1, 0.2, 1.3))
set.seed(7642)
x_3 <- cbind(x1 = rnorm(50), x2 = rnorm(50), x3 = rnorm(50))
y_3 <- levels_of(drop(x_3 %*% c(1, -0.5, 0.3)) + rlogis(50), c(-0.6, 0.8))
set.seed(7643)
x_5 <- cbind(x1 = rnorm(90), x2 = rnorm(90))
y_5 <- levels_of(1.2 * x_5[, 1] + 0.6 * x_5[, 2] + rlogis(90), c(-2, -0.6, 0.5, 1.8))
set.seed(7644)
y_0 <- sample(4, 30, replace = TRUE, prob = c(0.2, 0.3, 0.3, 0.2))
x_alias <- cbind(x_3[, 1:2], x12 = x_3[, 1] + x_3[, 2])
# the middle split of four levels separated by x1 (start fit not
# converged); and a row of level 2 far out at x1 = -60, which leaves the start
# fit converged, but at the start values both cut points of level 2 lie
# more than 37 above that row's linear predictor, so its probability is 0
set.seed(7645)
x_sep <- cbind(x1 = sort(c(runif(15, -3, -0.2), runif(15, 0.2, 3))), x2 = rnorm(30))
y_sep <- ifelse(x_sep[, 1] < 0, sample(1:2, 30, TRUE), sample(3:4, 30, TRUE))
x_far <- rbind(x_3[, 1:2], c(-60, 0))
y_far <- c(levels_of(drop(x_3[, 1:2] %*% c(1, -0.5)) + rlogis(50), c(-1, 0, 1)), 2L)

direct <- list(
  polr_direct("weighted", x_w, y_w, 4, weights = w60),
  polr_direct("three_levels", x_3, y_3, 3),
  polr_direct("five_levels", x_5, y_5, 5),
  polr_direct("no_predictor", x0[1:30, , drop = FALSE], y_0, 4, new_x = x0[1:2, , drop = FALSE]),
  polr_direct("aliased_column", x_alias, y_3, 3),
  # where R's polr stops
  polr_direct("two_levels", x_3, pmin(y_3, 2L), 2),
  polr_direct("start_not_converged", x_sep, y_sep, 4),
  polr_direct("start_not_finite", x_far, y_far, 4)
)
dc <- setNames(direct, vapply(direct, `[[`, "", "name"))
stopifnot(
  all(table(factor(y_w, 1:4)) >= 8), all(table(factor(y_3, 1:3)) >= 10), all(table(factor(y_5, 1:5)) >= 10),
  all(vapply(direct[1:5], function(case) case$convergence == 0L, TRUE)),
  identical(unclass(dc$aliased_column$aliased), c(FALSE, FALSE, TRUE)),
  identical(dc$two_levels$r_error, "response must have 3 or more levels"),
  identical(dc$start_not_converged$r_error, "attempt to find suitable starting values failed"),
  identical(dc$start_not_finite$r_error, "initial value in 'vmmin' is not finite")
)

encode_input <- function(inp) {
  y <- inp$y
  kind <- if (is.ordered(y)) "ordered" else if (is.factor(y)) "factor" else if (is.character(y)) "character" else if (is.logical(y)) "logical" else "numeric"
  list(
    kind = kind,
    y = I(if (is.factor(y)) as.character(y) else y),
    levels = if (is.factor(y)) I(levels(y)) else NULL,
    ry = I(inp$ry),
    wy = if (is.null(inp$wy)) NULL else I(inp$wy),
    x = if (ncol(inp$x) > 0L) unname(inp$x) else NULL,
    p = ncol(inp$x)
  )
}

write_fixture("l1_polr", list(
  description = paste(
    "mice.impute.polr(y, ry, x, wy, nnet.maxit = maxit, nnet.MaxNWts =",
    "max_weights) with its uniform numbers captured: uniforms = runif(n_imp)",
    "(the port's; r_draws says whether R drew them), order the expected",
    "sequence of draws as [pool, count] pairs. fit: the class of the fit",
    "mice.impute.polr made ('polr', 'multinom', null if both failed),",
    "polr_error: MASS::polr's error inside it (null if polr succeeded);",
    "port_fit ('polr', 'multinom', 'none') and port_fallback: the same for the",
    "port. A polr fit stores coef (one per column of x, 0 for a column polr",
    "drops as aliased), zeta (cut points), deviance, convergence and niter (R's",
    "function and gradient counts, which the tests do not compare); a multinom",
    "fit value, convergence, coef (one row per fitted class but the first, one",
    "column per column of [1, x]) and fitted_levels. prob: the probabilities",
    "of the port's levels in each imputed cell; levels: the port's level",
    "order; imputed: the port's labels; r_imputed: R's labels where R imputed.",
    "Inputs: kind (ordered, factor, character, numeric, logical), y (null for",
    "NA), levels (declared, factors only), ry, wy (null means !ry), x without",
    "intercept (null when p = 0). port: 'match' (R's result), 'rebuilt' (R",
    "stops or computes something else; the port's computation rebuilt with",
    "mice's augment, MASS::polr and nnet::multinom) or 'value_error' (R and",
    "the port stop before any draw; r_error). direct_cases: MASS::polr(y ~ .,",
    "weights) with 0-based levels y of an ordered factor with n_levels levels;",
    "coef (0 where aliased), zeta, deviance, convergence, prob the predicted",
    "probabilities of the rows new_x; r_error where polr stops."
  ),
  mass = MASS_VERSION,
  nnet = NNET_VERSION,
  inputs = lapply(inputs, encode_input),
  cases = cases,
  direct_cases = direct
))
