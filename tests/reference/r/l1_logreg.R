# L1 fixture for imputation of a two-category target by logistic regression.
#
#   l1_logreg.json   mice.impute.logreg and mice.impute.logreg.boot, with
#                    every random number they draw captured, the fit
#                    (glm.fit), the coefficient draw and the probabilities
#
# Python targets: imputation.logreg.logreg and logreg_boot (through their
# step-by-step versions _logreg and _logreg_boot), driven by a strict
# ScriptedGenerator.
#
# Each case calls the mice function after set.seed(seed) and saves
# .Random.seed. The port's computation is then rebuilt with mice's own
# augment(), glm.fit() and summary.glm(), drawing the expected random numbers
# in the expected order after set.seed(seed) again. Where mice succeeds, the
# rebuild must leave the same .Random.seed and give an identical result;
# where mice stops or returns something else, the draws mice made before
# that are checked the same way and the rebuild records what the port does
# instead (see `port` below).
#
# Every case also requires min |u - p| > 1e-4 over the imputed cells, so that
# no label depends on the last digits of a probability, and refits with
# epsilon times 0.99 and 1.01 to show that the number of iterations is not on
# the edge of glm.fit's convergence rule.
#
# Level order. The port orders the categories of an ordered factor as
# declared and those of every other target by their sorted labels, R's order
# after augment(). Where R does not augment (logreg without predictors or
# with one missing value, and logreg.boot) R takes the declared order of an
# unordered factor; those cases use factors whose declared order is the
# sorted one, so that the same random numbers give the same labels.

source(file.path("tests", "reference", "r", "_common.R"))

augment <- mice:::augment
sym <- mice:::sym

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(7210)
n <- 30
x1 <- rnorm(n)
x2 <- runif(n)
yes <- runif(n) < plogis(-0.3 + 1.2 * x1 - 1.5 * (x2 - 0.5))
mis <- sort(sample(n, 7))
ry <- !(seq_len(n) %in% mis)
ry_one <- seq_len(n) != mis[1]
x12 <- cbind(x1, x2)
x0 <- x12[, 0, drop = FALSE]
lab <- ifelse(yes, "yes", "no")
stopifnot(all(c("no", "yes") %in% lab[ry]))

blank <- function(y, keep = ry) replace(y, !keep, NA)
yn <- function(v, levels = c("no", "yes"), ...) factor(v, levels = levels, ...)

# separated by x1 in every row; a logistic fit without augmentation diverges
sep <- ifelse(x1 > 0, "yes", "no")
# separated by x1 except two observed rows, so that some bootstrap samples
# of the observed rows are separated
near <- sep
rank_obs <- which(ry)[order(x1[ry])]
near[rank_obs[c(10, 14)]] <- ifelse(near[rank_obs[c(10, 14)]] == "yes", "no", "yes")
stopifnot(sort(x1[ry & near == "yes"])[1] < max(x1[ry & near == "no"]))
three <- ifelse(x1 < -0.5, "a", ifelse(x1 < 0.5, "b", "c"))
stopifnot(all(c("a", "b", "c") %in% three[ry]))

inputs <- list(
  base = list(y = blank(yn(lab)), ry = ry, x = x12),
  # the missing cells hold labels (inside mice: the current imputations)
  filled_target = list(y = yn(lab), ry = ry, x = x12),
  one_predictor = list(y = blank(yn(lab)), ry = ry, x = x12[, 1, drop = FALSE]),
  no_predictor = list(y = blank(yn(lab)), ry = ry, x = x0),
  one_missing = list(y = blank(yn(lab), ry_one), ry = ry_one, x = x12),
  wy_subset = list(y = blank(yn(lab)), ry = ry, x = x12, wy = seq_len(n) %in% mis[c(1, 3, 4, 6)]),
  numeric01 = list(y = blank(as.numeric(yes)), ry = ry, x = x12),
  logical = list(y = blank(yes), ry = ry, x = x12),
  declared_yes_no = list(y = blank(yn(lab, c("yes", "no"))), ry = ry, x = x12),
  # declared order lo < hi, alphabetical order hi < lo
  ordered = list(y = blank(yn(ifelse(yes, "hi", "lo"), c("lo", "hi"), ordered = TRUE)), ry = ry, x = x12),
  separated = list(y = blank(yn(sep)), ry = ry, x = x12),
  near_separated = list(y = blank(yn(near)), ry = ry, x = x12),
  # where R fails or misbehaves
  three_levels = list(y = blank(factor(three)), ry = ry, x = x12),
  numeric12 = list(y = blank(as.numeric(yes) + 1), ry = ry, x = x12),
  one_category = list(y = blank(yn(rep("no", n))), ry = ry, x = x12),
  constant_column = list(y = blank(yn(lab)), ry = ry, x = cbind(x12, one = 1)),
  ordered_unused = list(
    y = blank(yn(ifelse(yes, "hi", "lo"), c("lo", "mid", "hi"), ordered = TRUE)), ry = ry, x = x12
  ),
  duplicate_column = list(y = blank(yn(lab)), ry = ry, x = cbind(x12, x1))
)

# ---------------------------------------------------------------------------
# The port's computation, rebuilt from mice's parts
# ---------------------------------------------------------------------------

pair <- function(pool, k) list(pool, as.integer(k))

# The categories in the port's order: declared (present ones) for an ordered
# factor, the sorted labels present for every other target
port_levels <- function(y) {
  present <- y[!is.na(y)]
  if (is.ordered(y)) {
    levels(y)[levels(y) %in% as.character(present)]
  } else if (is.factor(y)) {
    sort(unique(as.character(present)))
  } else {
    sort(unique(present))
  }
}

# R's result in the port's form: labels, booleans for a logical target
as_port <- function(imp, y) {
  if (is.factor(imp)) {
    return(as.character(imp))
  }
  v <- as.vector(imp)
  if (is.logical(y)) as.logical(v) else v
}

refit_iterations <- function(fit_call, factor) {
  fit_call(list(epsilon = 1e-8 * factor))$iter
}

# Draw and compute as the port does. `y` is the target the fit sees (the
# original one, or an equivalent one where R would fail). Starts from
# set.seed(seed) and returns the draws, the fit and the result.
port_steps <- function(method, y, ry, x, wy, seed) {
  set.seed(seed)
  n1 <- sum(ry)
  q <- ncol(x) + 1L
  s <- NULL
  z <- numeric(0)
  if (method == "logreg") {
    aug <- augment(y, ry, x, wy)
    X <- cbind(1, as.matrix(aug$x))
    fit_call <- function(control) {
      glm.fit(
        x = X[aug$ry, , drop = FALSE], y = aug$y[aug$ry],
        family = quasibinomial(link = logit), weights = aug$w[aug$ry], control = control
      )
    }
    fit <- fit_call(list())
    cov_unscaled <- summary.glm(fit)$cov.unscaled
    kept <- !is.na(coef(fit))
    rv <- t(chol(sym(cov_unscaled)))
    z <- rnorm(ncol(rv))
    if (all(kept)) {
      beta_star <- coef(fit) + rv %*% z
    } else {
      # the port: aliased coefficients 0, the others drawn
      beta_star <- matrix(0, q, 1)
      beta_star[kept] <- coef(fit)[kept] + rv %*% z
    }
    Xm <- X[aug$wy, , drop = FALSE]
    y_out <- aug$y
  } else {
    s <- sample(n1, n1, replace = TRUE)
    doty <- y
    doty[ry] <- y[ry][s]
    dotx <- x
    dotx[ry, ] <- x[ry, , drop = FALSE][s, , drop = FALSE]
    X <- cbind(1, as.matrix(dotx))
    fit_call <- function(control) {
      suppressWarnings(glm.fit(
        x = X[ry, , drop = FALSE], y = doty[ry], family = binomial(link = logit), control = control
      ))
    }
    fit <- fit_call(list())
    cov_unscaled <- summary.glm(fit)$cov.unscaled
    beta_star <- coef(fit)
    beta_star[is.na(beta_star)] <- 0 # the port; R predicts NA
    # mice's x[wy, ] (it drops to a vector), except where that fails in R:
    # no predictor and more than one cell to impute
    Xm <- if (q > 1L || sum(wy) == 1L) X[wy, ] else X[wy, , drop = FALSE]
    y_out <- doty
  }
  stopifnot(
    refit_iterations(fit_call, 0.99) == fit$iter,
    refit_iterations(fit_call, 1.01) == fit$iter
  )
  p <- 1 / (1 + exp(-(Xm %*% beta_star)))
  u <- runif(nrow(p))
  stopifnot(min(abs(u - p)) > 1e-4)
  vec <- (u <= p)
  vec[vec] <- 1
  if (is.factor(y_out)) {
    vec <- factor(vec, c(0, 1), levels(y_out))
  }
  list(
    s = s, z = z, u = u, fit = fit, cov_unscaled = cov_unscaled, beta_star = as.vector(beta_star),
    p = as.vector(p), codes = as.integer(as.vector(u <= p)), result = vec, seed_after = .Random.seed
  )
}

# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

# port: "match" (R's result), "rebuilt" (R stops or returns something else;
# the port's result is rebuilt, from `fit_y` where given), "one_category"
# (one category present: the port imputes it without fit or draws) or
# "value_error" (more than two categories). r_draws: what mice draws before
# it returns or stops, where that differs from the port's draws.
logreg_case <- function(name, method, input, seed, port = "match", fit_y = NULL, r_draws = NULL) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- if (is.null(inp$wy)) !ry else inp$wy
  x <- inp$x
  n0 <- sum(wy)
  n1 <- sum(ry)
  q <- ncol(x) + 1L
  fun <- if (method == "logreg") mice.impute.logreg else mice.impute.logreg.boot

  set.seed(seed)
  imp <- tryCatch(suppressWarnings(fun(y, ry, x, wy = wy)), error = function(e) e)
  saved <- .Random.seed
  failed <- inherits(imp, "error")
  stopifnot(failed == (port == "value_error") || port == "rebuilt")

  if (!is.null(r_draws)) {
    set.seed(seed)
    r_draws(q, n0, n1)
    stopifnot(identical(.Random.seed, saved))
  }
  steps <- NULL
  if (port %in% c("match", "rebuilt")) {
    steps <- port_steps(method, if (is.null(fit_y)) y else fit_y, ry, x, wy, seed)
    if (port == "match") {
      stopifnot(!failed, is.null(r_draws), identical(steps$seed_after, saved), identical(steps$result, imp))
    }
  }

  levels <- port_levels(y)
  imputed <- switch(port,
    match = ,
    rebuilt = levels[steps$codes + 1L],
    one_category = rep(levels, n0),
    value_error = NULL
  )
  if (port == "match") {
    # R's labels are the port's: the level orders agree
    stopifnot(identical(as_port(imp, y), imputed))
  }
  fit <- steps$fit
  order <- list()
  if (!is.null(steps)) {
    order <- if (method == "logreg") {
      list(pair("normal", length(steps$z)), pair("uniform", n0))
    } else {
      list(pair("integer", n1), pair("uniform", n0))
    }
  }

  list(
    name = name,
    method = method,
    input = input,
    seed = seed,
    port = port,
    n_obs = n1,
    n_coef = q,
    n_imp = n0,
    normals = I(if (is.null(steps)) numeric(0) else steps$z),
    uniforms = I(if (is.null(steps)) numeric(0) else steps$u),
    integers = if (is.null(steps$s)) NULL else I(steps$s - 1L),
    order = order,
    coef = if (is.null(fit)) NULL else I(unname(coef(fit))),
    cov_unscaled = if (is.null(fit)) NULL else unname(steps$cov_unscaled),
    iter = fit$iter,
    converged = fit$converged,
    beta_star = if (is.null(fit)) NULL else I(steps$beta_star),
    prob = if (is.null(fit)) NULL else I(steps$p),
    levels = I(levels),
    imputed = if (is.null(imputed)) NULL else I(imputed),
    r_imputed = if (port %in% c("rebuilt", "one_category") && !failed) I(as_port(imp, y)) else NULL,
    r_error = if (failed) conditionMessage(imp) else NULL
  )
}

draw_normal_uniform <- function(q, n0, n1) {
  rnorm(q)
  runif(n0)
}
draw_sample_uniform <- function(q, n0, n1) {
  sample(n1, n1, replace = TRUE)
  runif(n0)
}

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

cases <- list(
  logreg_case("logreg_seed1", "logreg", "base", 7211L),
  logreg_case("logreg_seed2", "logreg", "base", 7212L),
  logreg_case("logreg_filled_target", "logreg", "filled_target", 7211L),
  logreg_case("logreg_numeric", "logreg", "numeric01", 7213L),
  logreg_case("logreg_logical", "logreg", "logical", 7214L),
  logreg_case("logreg_declared_yes_no", "logreg", "declared_yes_no", 7215L),
  logreg_case("logreg_ordered", "logreg", "ordered", 7216L),
  logreg_case("logreg_one_missing", "logreg", "one_missing", 7217L),
  logreg_case("logreg_no_predictor", "logreg", "no_predictor", 7218L),
  logreg_case("logreg_wy_subset", "logreg", "wy_subset", 7219L),
  logreg_case("logreg_separated", "logreg", "separated", 7220L),
  # R draws everything, then stops: invalid 'labels'
  logreg_case("logreg_three_levels", "logreg", "three_levels", 7221L,
    port = "value_error", r_draws = draw_normal_uniform
  ),
  # R stops in glm.fit before drawing; the port fits y - 1
  logreg_case("logreg_numeric_12", "logreg", "numeric12", 7222L,
    port = "rebuilt", fit_y = inputs$numeric12$y - 1, r_draws = function(q, n0, n1) NULL
  ),
  # R draws and returns invented labels; the port draws nothing
  logreg_case("logreg_one_category", "logreg", "one_category", 7223L,
    port = "one_category", r_draws = draw_normal_uniform
  ),
  # R draws one normal per coefficient that is not aliased, then stops
  logreg_case("logreg_constant_column", "logreg", "constant_column", 7224L,
    port = "rebuilt", r_draws = function(q, n0, n1) rnorm(q - 1L)
  ),
  # R draws everything, then stops: invalid 'labels'; the port fits the
  # factor without its unused level
  logreg_case("logreg_ordered_unused", "logreg", "ordered_unused", 7225L,
    port = "rebuilt", fit_y = droplevels(inputs$ordered_unused$y), r_draws = draw_normal_uniform
  ),
  logreg_case("logreg_boot_seed1", "logreg.boot", "base", 7231L),
  logreg_case("logreg_boot_seed2", "logreg.boot", "base", 7232L),
  logreg_case("logreg_boot_one_predictor", "logreg.boot", "one_predictor", 7233L),
  # R stops in x[wy, ] %*% beta after the bootstrap draw
  logreg_case("logreg_boot_no_predictor", "logreg.boot", "no_predictor", 7234L,
    port = "rebuilt", r_draws = function(q, n0, n1) sample(n1, n1, replace = TRUE)
  ),
  logreg_case("logreg_boot_one_missing", "logreg.boot", "one_missing", 7235L),
  # 7237: the first seed from 7236 on whose bootstrap sample is separated, so
  # that glm.fit stops at its iteration limit
  logreg_case("logreg_boot_separated", "logreg.boot", "near_separated", 7237L),
  # R draws and returns the category; the port draws nothing
  logreg_case("logreg_boot_one_category", "logreg.boot", "one_category", 7239L,
    port = "one_category", r_draws = draw_sample_uniform
  ),
  # the copy of x1 is aliased: R imputes NA, the port uses coefficient 0
  logreg_case("logreg_boot_duplicate_column", "logreg.boot", "duplicate_column", 7238L,
    port = "rebuilt", r_draws = draw_sample_uniform
  )
)
by_name <- setNames(cases, vapply(cases, `[[`, "", "name"))

# The values at the missing cells do not matter
stopifnot(identical(by_name$logreg_filled_target$imputed, by_name$logreg_seed1$imputed))
# Augmentation keeps the separated fit finite and converged
plain <- suppressWarnings(glm.fit(cbind(1, x12)[ry, ], inputs$separated$y[ry], family = binomial()))
stopifnot(
  by_name$logreg_separated$converged,
  max(abs(by_name$logreg_separated$coef)) < 10,
  max(abs(coef(plain))) > 10 * max(abs(by_name$logreg_separated$coef))
)
# The separated bootstrap sample: glm.fit runs into its iteration limit
stopifnot(!by_name$logreg_boot_separated$converged, by_name$logreg_boot_separated$iter == 25L)
# R's failures, as documented
stopifnot(
  identical(by_name$logreg_three_levels$r_error, "invalid 'labels'; length 3 should be 1 or 2"),
  identical(by_name$logreg_numeric_12$r_error, "y values must be 0 <= y <= 1"),
  identical(by_name$logreg_ordered_unused$r_error, "invalid 'labels'; length 3 should be 1 or 2"),
  identical(by_name$logreg_boot_no_predictor$r_error, "non-conformable arguments"),
  all(is.na(by_name$logreg_boot_duplicate_column$r_imputed)),
  all(by_name$logreg_one_category$r_imputed %in% c("no1", "no2"))
)
stopifnot(is.na(by_name$logreg_constant_column$coef[4]))
# the port's fit of an equivalent target is what mice computes for it
set.seed(7222L)
stopifnot(identical(
  as.vector(mice.impute.logreg(inputs$numeric12$y - 1, ry, x12)) + 1,
  unclass(by_name$logreg_numeric_12$imputed)
))
set.seed(7225L)
stopifnot(identical(
  as.character(mice.impute.logreg(droplevels(inputs$ordered_unused$y), ry, x12)),
  unclass(by_name$logreg_ordered_unused$imputed)
))

encode_input <- function(inp) {
  y <- inp$y
  list(
    kind = if (is.ordered(y)) "ordered" else if (is.factor(y)) "factor" else if (is.logical(y)) "logical" else "numeric",
    y = if (is.factor(y)) I(as.character(y)) else I(y),
    levels = if (is.factor(y)) I(levels(y)) else NULL,
    ry = I(inp$ry),
    wy = if (is.null(inp$wy)) NULL else I(inp$wy),
    x = if (ncol(inp$x) > 0L) unname(inp$x) else NULL,
    p = ncol(inp$x)
  )
}

write_fixture("l1_logreg", list(
  description = paste(
    "mice.impute.logreg(y, ry, x, wy) and mice.impute.logreg.boot(y, ry, x, wy)",
    "with every random number captured: normals = rnorm(q) (logreg; one per",
    "coefficient that is not aliased), integers = sample(n_obs, n_obs, TRUE) - 1",
    "(logreg.boot), uniforms = runif(n_imp). order is the expected sequence of",
    "draws as [pool, count] pairs. coef (null where aliased), cov_unscaled",
    "(summary.glm, the coefficients that are not aliased), iter and converged",
    "come from glm.fit (quasibinomial on augmented data for logreg, binomial on",
    "the bootstrap sample for logreg.boot); beta_star are the coefficients of",
    "the prediction (aliased ones 0), prob the probabilities of the second",
    "level, levels the port's level order and imputed the port's labels",
    "(booleans for a logical target). Inputs: kind, y (labels; null for NA),",
    "levels (declared), ry, wy (null means !ry), x without intercept (null",
    "when p = 0). port: 'match' (R's result), 'rebuilt' (R stops, r_error, or",
    "returns something else, r_imputed; the port's computation rebuilt with",
    "mice's augment and glm.fit), 'one_category' (the port imputes the single",
    "category without fit or draws; R's output in r_imputed) or 'value_error'",
    "(more than two categories)."
  ),
  inputs = lapply(inputs, encode_input),
  cases = cases
))
