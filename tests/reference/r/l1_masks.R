# L1 fixture for the univariate imputers called with the masks of
# sampler.univ.
#
#   l1_masks.json   direct calls of mice's imputers with every row: the
#                   rows to impute overlapping the fit rows (where = "all"),
#                   rows in neither mask with an observed y (ignore) or a
#                   missing one (left out by where), and such rows with NA
#                   in x; and direct calls of mice:::augment with such masks
#
# Python targets: imputation.norm.norm, norm_nob, norm_boot, norm_predict,
# imputation.mean.mean, imputation.logreg._logreg and _logreg_boot,
# imputation.polyreg._polyreg and imputation.polr._polr (driven by a strict
# ScriptedGenerator), imputation.PMM.pmm with one donor,
# imputation.midas._donor_probabilities and imputation.logreg.augment. The
# cases have the format of the fixtures l1_norm, l1_logreg, l1_polyreg,
# l1_polr, l1_pmm_one_donor, l1_midastouch and l1_augment; the tests replay
# them with the helpers of those fixtures' tests.
#
# Inside mice(), sampler.univ passes every row: y, ry = complete.cases(x, y)
# & r & !ignore (the fit rows), x and wy = complete.cases(x) & where (the
# rows to impute). The rows in neither mask carry decoys: x far outside its
# range where it is not NA, and an extreme y, or a label that no fit row
# has. Only mice:::augment reads those rows (logreg, polyreg, polr): it takes
# its means, standard deviations and ranges over all rows, column by column
# without NA, and its categories from all of y. The script checks that every
# other method gives the same result without those rows, and that the rows
# augment adds move far without them.
#
# Every random number is captured as in the fixture of the method: the mice
# function is called after set.seed(seed) and .Random.seed is saved; the
# expected draws are made after set.seed(seed) again, in the expected order,
# and must leave the same .Random.seed; the result is rebuilt from them with
# mice's own parts (estimice, augment, glm.fit, nnet::multinom, MASS::polr).
# Labels must not depend on the last digits of a probability: every uniform
# number is at least 1e-4 away from every probability it is compared with.

source(file.path("tests", "reference", "r", "_common.R"))

augment <- mice:::augment
sym <- mice:::sym
MASS_VERSION <- as.character(packageVersion("MASS"))
NNET_VERSION <- as.character(packageVersion("nnet"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(7801)
n <- 30
x1 <- rnorm(n)
x2 <- runif(n)
fill <- c(3L, 9L, 16L, 24L) # rows to impute (outside, na_rows)
ignored <- c(6L, 13L, 27L) # observed y, left out of the fit (R's ignore)
left_out <- c(11L, 20L, 29L) # missing y that where leaves out
neither <- sort(c(ignored, left_out))
fit <- setdiff(seq_len(n), c(fill, neither))
rows <- function(i) seq_len(n) %in% i

# Three mask patterns. outside and na_rows: 20 fit rows, 4 rows to impute,
# 6 rows in neither mask. overlap: R's where = "all" without ignore, so the
# 23 observed rows are fitted and every row is imputed.
masks <- list(
  overlap = list(ry = rows(c(fit, ignored)), wy = rep(TRUE, n)),
  outside = list(ry = rows(fit), wy = rows(fill)),
  na_rows = list(ry = rows(fit), wy = rows(fill))
)

x_plain <- cbind(x1, x2)
# decoys: the rows in neither mask lie far outside the range of x1 and x2
x_decoy <- x_plain
x_decoy[neither, "x1"] <- 6 + seq_along(neither)
x_decoy[neither, "x2"] <- -3 - seq_along(neither) / 2
# NA in one predictor (rows 6 and 29) and in both (row 11)
x_na <- x_decoy
x_na[6L, "x1"] <- NA
x_na[29L, "x2"] <- NA
x_na[11L, ] <- NA
x_of <- list(overlap = x_plain, outside = x_decoy, na_rows = x_na)

# The targets. Fit rows: drawn from a model in x1 and x2. Rows to impute
# that are not fit rows: values as the current imputations inside mice
# (decoys). Ignored rows: observed decoys. Rows left out by where: NA.
y_num <- 4 + 2 * x1 - 3 * x2 + rnorm(n)
yes <- runif(n) < plogis(-0.3 + 1.4 * x1 - 1.6 * (x2 - 0.5))
latent <- 1.2 * x1 - 1.5 * (x2 - 0.5) + rlogis(n)
abc <- c("a", "b", "c")[1L + findInterval(latent, c(-0.7, 0.7))]
stopifnot(
  all(table(factor(yes[fit])) >= 6),
  all(table(factor(abc[fit], c("a", "b", "c"))) >= 5),
  all(table(factor(abc[c(fit, ignored)], c("a", "b", "c"))) >= 5)
)

target <- function(pattern, values, current, decoy) {
  y <- values
  if (pattern == "overlap") {
    y[c(fill, left_out)] <- current[seq_along(c(fill, left_out))]
  } else {
    y[fill] <- current[seq_along(fill)]
    y[ignored] <- decoy
    y[left_out] <- NA
  }
  y
}
# a model in x1 predicts "yes" in the decoy rows, so the decoy label is "no";
# "d" is held by no fit row
bin_of <- function(pattern) {
  factor(target(pattern, ifelse(yes, "yes", "no"), c("no", "yes", "no", "no", "yes", "no", "yes"), "no"),
    levels = c("no", "yes")
  )
}
cat_of <- function(pattern, ordered = FALSE) {
  factor(target(pattern, abc, c("d", "b", "a", "c", "d", "a", "b"), c("d", "a", "d")),
    levels = c("a", "b", "c", "d"), ordered = ordered
  )
}

inputs <- list()
for (pattern in names(masks)) {
  m <- masks[[pattern]]
  x <- x_of[[pattern]]
  inputs[[paste0("num_", pattern)]] <- list(
    y = target(pattern, y_num, 500 + 10 * seq_len(7), 700 + 10 * seq_along(ignored)), ry = m$ry, wy = m$wy, x = x
  )
  inputs[[paste0("bin_", pattern)]] <- list(y = bin_of(pattern), ry = m$ry, wy = m$wy, x = x)
  inputs[[paste0("cat_", pattern)]] <- list(y = cat_of(pattern), ry = m$ry, wy = m$wy, x = x)
  inputs[[paste0("ord_", pattern)]] <- list(y = cat_of(pattern, ordered = TRUE), ry = m$ry, wy = m$wy, x = x)
}
# logreg: "yes" only outside the fit rows (an ignored row and a row to
# impute); augment adds rows of both categories
inputs$bin_outside_category <- local({
  y <- factor(rep("no", n), levels = c("no", "yes"))
  y[c(ignored[2L], fill[3L])] <- "yes"
  y[left_out] <- NA
  list(y = y, ry = masks$outside$ry, wy = masks$outside$wy, x = x_na)
})
# a 0/1 target with the value 2 in an ignored row: augment counts it as a
# third category (logreg stops), logreg.boot does not read the row
inputs$num01_third_value <- local({
  y <- as.numeric(yes)
  y[ignored[1L]] <- 2
  y[left_out] <- NA
  list(y = y, ry = masks$outside$ry, wy = masks$outside$wy, x = x_na)
})

# R's ry and wy select the rows the imputers use; dropping every other row
# must not change the result of a method that does not augment
drop_neither <- function(inp) {
  keep <- inp$ry | inp$wy
  list(y = inp$y[keep], ry = inp$ry[keep], wy = inp$wy[keep], x = inp$x[keep, , drop = FALSE])
}

pair <- function(pool, k) list(pool, as.integer(k))
kind_of <- function(y) {
  if (is.ordered(y)) "ordered" else if (is.factor(y)) "factor" else if (is.logical(y)) "logical" else "numeric"
}

# ---------------------------------------------------------------------------
# norm, norm.nob, norm.boot, norm.predict, mean (as l1_norm.R)
# ---------------------------------------------------------------------------

norm_case <- function(name, method, input, seed, ls_meth = "qr", ridge = 1e-05) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- inp$wy
  x <- inp$x
  X <- cbind(1, x)
  q <- ncol(X)
  n1 <- sum(ry)
  n0 <- sum(wy)
  regression <- method != "mean"

  fun <- get(paste0("mice.impute.", method), envir = asNamespace("mice"))
  extra <- if (regression) list(ls.meth = ls_meth, ridge = ridge) else list()
  set.seed(seed)
  imp <- do.call(fun, c(list(y, ry, x, wy = wy), extra))
  saved <- .Random.seed
  # without the rows in neither mask, the same draws give the same result
  small <- drop_neither(inp)
  set.seed(seed)
  stopifnot(identical(do.call(fun, c(list(small$y, small$ry, small$x, wy = small$wy), extra)), imp))

  # the expected draws, in the expected order
  set.seed(seed)
  chi <- NULL
  z <- numeric(0)
  s <- NULL
  df <- NULL
  if (method == "norm") {
    df <- max(n1 - q, 1)
    chi <- rchisq(1, df)
    z <- rnorm(q)
  }
  if (method == "norm.boot") {
    s <- sample(n1, n1, replace = TRUE)
  }
  e <- if (method %in% c("norm", "norm.nob", "norm.boot")) rnorm(n0) else numeric(0)
  stopifnot(identical(.Random.seed, saved))

  # rebuild the imputations from the draws
  Xo <- X[ry, , drop = FALSE]
  Xm <- X[wy, , drop = FALSE]
  yo <- y[ry]
  if (method == "norm") {
    p <- mice:::estimice(Xo, yo, ls.meth = ls_meth, ridge = ridge)
    sigma <- sqrt(sum(p$r^2) / chi)
    beta <- p$c + (t(chol(sym(p$v))) %*% z) * sigma
    rebuilt <- Xm %*% beta + e * sigma
  } else if (method == "norm.nob") {
    p <- mice:::estimice(Xo, yo, ls.meth = ls_meth, ridge = ridge)
    rebuilt <- Xm %*% p$c + e * sqrt(sum(p$r^2) / (n1 - q - 1))
  } else if (method == "norm.boot") {
    p <- mice:::estimice(Xo[s, , drop = FALSE], yo[s], ls.meth = ls_meth, ridge = ridge)
    rebuilt <- Xm %*% p$c + e * sqrt(sum(p$r^2) / (n1 - q - 1))
  } else if (method == "norm.predict") {
    p <- mice:::estimice(Xo, yo, ls.meth = ls_meth, ridge = ridge)
    rebuilt <- Xm %*% p$c
  } else {
    rebuilt <- rep.int(mean(yo), n0)
  }
  rebuilt <- as.vector(rebuilt)
  stopifnot(identical(rebuilt, as.vector(imp)), all(is.finite(rebuilt)))

  order <- list()
  if (!is.null(chi)) order <- c(order, list(pair("chisq", 1)))
  if (!is.null(s)) order <- c(order, list(pair("integer", n1)))
  if (length(z) + length(e) > 0) order <- c(order, list(pair("normal", length(z) + length(e))))

  list(
    name = name,
    method = method,
    input = input,
    seed = seed,
    ls_meth = if (regression) ls_meth else NULL,
    ridge = if (regression) ridge else NULL,
    n_obs = n1,
    n_coef = q,
    df = df,
    chisq = if (is.null(chi)) I(numeric(0)) else I(chi),
    normals = I(c(z, e)),
    integers = if (is.null(s)) NULL else I(s - 1L),
    order = order,
    port = "match",
    imputed = I(rebuilt),
    r_error = NULL,
    port_imputed = NULL
  )
}

NORM_SEEDS <- c(norm = 7811L, norm.nob = 7821L, norm.boot = 7831L, norm.predict = 7841L, mean = 7851L)
label <- function(method) gsub(".", "_", method, fixed = TRUE)
norm_cases <- list()
for (method in names(NORM_SEEDS)) {
  for (k in seq_along(masks)) {
    pattern <- names(masks)[k]
    norm_cases[[length(norm_cases) + 1L]] <- norm_case(
      paste(label(method), pattern, sep = "_"), method, paste0("num_", pattern), NORM_SEEDS[[method]] + k
    )
  }
}

# ---------------------------------------------------------------------------
# logreg and logreg.boot (as l1_logreg.R)
# ---------------------------------------------------------------------------

# The categories in the port's order: declared (present ones) for an
# ordered factor, the sorted labels present for every other target
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

# Draw and compute as the port does, from set.seed(seed). logreg augments
# all rows; logreg.boot reads the rows of the masks only, and predicts every
# row to impute from its own predictors (see the case logreg_boot_overlap).
logreg_steps <- function(method, y, ry, x, wy, seed) {
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
    rv <- t(chol(sym(cov_unscaled)))
    z <- rnorm(ncol(rv))
    beta_star <- coef(fit) + rv %*% z
    Xm <- X[aug$wy, , drop = FALSE]
    y_out <- aug$y
  } else {
    s <- sample(n1, n1, replace = TRUE)
    X <- cbind(1, as.matrix(x))
    Xo <- X[ry, , drop = FALSE][s, , drop = FALSE]
    yo <- y[ry][s]
    fit_call <- function(control) {
      suppressWarnings(glm.fit(x = Xo, y = yo, family = binomial(link = logit), control = control))
    }
    fit <- fit_call(list())
    cov_unscaled <- summary.glm(fit)$cov.unscaled
    beta_star <- coef(fit)
    Xm <- X[wy, , drop = FALSE]
    y_out <- y
  }
  stopifnot(
    all(!is.na(coef(fit))),
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

# port: "match" (R's result), "rebuilt" (R computes something else, r_imputed
# and r_prob; the port's computation rebuilt) or "value_error" (R stops,
# r_error; so does the port).
logreg_case <- function(name, method, input, seed, port = "match") {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- inp$wy
  x <- inp$x
  n0 <- sum(wy)
  n1 <- sum(ry)
  q <- ncol(x) + 1L
  fun <- if (method == "logreg") mice.impute.logreg else mice.impute.logreg.boot

  set.seed(seed)
  imp <- tryCatch(suppressWarnings(fun(y, ry, x, wy = wy)), error = function(e) e)
  saved <- .Random.seed
  failed <- inherits(imp, "error")
  stopifnot(failed == (port == "value_error"))
  # logreg.boot gives the same result without the rows in neither mask (for
  # logreg the decoys move the augmented rows, see below)
  if (method == "logreg.boot") {
    small <- drop_neither(inp)
    set.seed(seed)
    stopifnot(identical(suppressWarnings(fun(small$y, small$ry, small$x, wy = small$wy)), imp))
  }

  steps <- NULL
  if (port == "value_error") {
    # R stops in glm.fit, before any draw
    set.seed(seed)
    stopifnot(identical(.Random.seed, saved))
  } else {
    used <- ry | wy
    steps <- if (method == "logreg") {
      logreg_steps(method, y, ry, x, wy, seed)
    } else {
      logreg_steps(method, y[used], ry[used], x[used, , drop = FALSE], wy[used], seed)
    }
    stopifnot(identical(steps$seed_after, saved))
    if (port == "match") {
      stopifnot(identical(steps$result, imp))
    }
  }

  levels <- port_levels(if (method == "logreg") y else y[ry | wy])
  imputed <- if (is.null(steps)) NULL else levels[steps$codes + 1L]
  if (port == "match") {
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
    r_imputed = if (port == "rebuilt") I(as_port(imp, y)) else NULL,
    r_error = if (failed) conditionMessage(imp) else NULL
  )
}

LOGREG_SEEDS <- c(overlap = 7861L, outside = 7862L, na_rows = 7863L)
# 7872 draws a bootstrap sample of the outside input whose categories are
# separated (25 iterations, no convergence); 7875 is the next seed whose fit
# converges
BOOT_SEEDS <- c(overlap = 7871L, outside = 7875L, na_rows = 7873L)
logreg_cases <- list()
for (pattern in names(masks)) {
  logreg_cases[[length(logreg_cases) + 1L]] <- logreg_case(
    paste0("logreg_", pattern), "logreg", paste0("bin_", pattern), LOGREG_SEEDS[[pattern]]
  )
  # R's logreg.boot predicts a row that is also a fit row from the
  # predictors of the bootstrap row that replaced it (dotx); the port from
  # its own
  logreg_cases[[length(logreg_cases) + 1L]] <- logreg_case(
    paste0("logreg_boot_", pattern), "logreg.boot", paste0("bin_", pattern), BOOT_SEEDS[[pattern]],
    port = if (pattern == "overlap") "rebuilt" else "match"
  )
}
logreg_cases <- c(logreg_cases, list(
  logreg_case("logreg_outside_category", "logreg", "bin_outside_category", 7864L),
  logreg_case("logreg_third_value", "logreg", "num01_third_value", 7865L, port = "value_error"),
  logreg_case("logreg_boot_third_value", "logreg.boot", "num01_third_value", 7874L)
))

# ---------------------------------------------------------------------------
# polyreg (as l1_polyreg.R)
# ---------------------------------------------------------------------------

# The weights of a multinom fit as the port's coefficients: one row per
# fitted class but the first, one column per column of the model matrix
# (nnet's bias weights, fixed at 0, left out)
multinom_coef <- function(fit) {
  r <- length(fit$coefnames)
  w <- matrix(fit$wts, nrow = r + 1L)[-1L, , drop = FALSE]
  if (ncol(w) > 1L) w <- w[, -1L, drop = FALSE]
  unname(t(w))
}

polyreg_steps <- function(y, ry, x, wy, seed, maxit, max_nwts) {
  lev <- port_levels(y)
  aug <- augment(y, ry, as.matrix(x), wy)
  set.seed(seed)
  u <- runif(sum(wy))
  seed_after <- .Random.seed
  yl <- factor(as.character(aug$y), levels = lev)
  xa <- aug$x
  w <- aug$w
  xy <- cbind.data.frame(y = yl, x = xa)
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

polyreg_case <- function(name, input, seed, maxit = 100, max_nwts = 1500) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- inp$wy
  x <- inp$x
  n0 <- sum(wy)

  set.seed(seed)
  imp <- suppressWarnings(mice.impute.polyreg(y, ry, x, wy = wy, nnet.maxit = maxit, nnet.MaxNWts = max_nwts))
  saved <- .Random.seed
  set.seed(seed)
  u <- runif(n0)
  stopifnot(identical(.Random.seed, saved))

  steps <- polyreg_steps(y, ry, x, wy, seed, maxit, max_nwts)
  stopifnot(identical(steps$seed_after, saved))
  levels <- port_levels(y)
  imputed <- levels[steps$codes]
  stopifnot(identical(as.character(imp), imputed))
  fit <- steps$fit
  list(
    name = name,
    input = input,
    seed = seed,
    port = "match",
    maxit = maxit,
    max_weights = max_nwts,
    n_imp = n0,
    uniforms = I(u),
    order = list(pair("uniform", n0)),
    value = fit$value,
    convergence = fit$convergence,
    coef = multinom_coef(fit),
    fitted_levels = I(fit$lev),
    prob = unname(steps$prob),
    levels = I(levels),
    imputed = I(imputed),
    r_error = NULL
  )
}

polyreg_cases <- lapply(seq_along(masks), function(k) {
  pattern <- names(masks)[k]
  polyreg_case(paste0("polyreg_", pattern), paste0("cat_", pattern), 7880L + k)
})

# ---------------------------------------------------------------------------
# polr (as l1_polr.R)
# ---------------------------------------------------------------------------

# The class of the fit mice.impute.polr makes and polr's error message (NULL
# if polr succeeds), from the sequence of fits of mice.impute.polr on mice's
# augmented data
r_polr_fits <- function(y, ry, x, wy, maxit, max_nwts) {
  aug <- augment(y, ry, as.matrix(x), wy)
  xy <- cbind.data.frame(y = aug$y, x = aug$x)
  fit <- tryCatch(
    suppressWarnings(MASS::polr(formula(xy), data = xy[aug$ry, , drop = FALSE], weights = aug$w[aug$ry])),
    error = function(e) e
  )
  polr_error <- NULL
  if (inherits(fit, "error")) {
    polr_error <- conditionMessage(fit)
    fit <- suppressWarnings(nnet::multinom(formula(xy),
      data = xy[aug$ry, , drop = FALSE], weights = aug$w[aug$ry],
      maxit = maxit, trace = FALSE, MaxNWts = max_nwts
    ))
  }
  list(fit = class(fit)[1], polr_error = polr_error)
}

# Augment, fit, predict and draw as the port does, for an ordered factor
# (the declared levels; polr, the fallback where it stops), from
# set.seed(seed)
polr_steps <- function(y, ry, x, wy, seed, maxit, max_nwts) {
  lev <- levels(y)
  aug <- augment(y, ry, as.matrix(x), wy)
  yl <- factor(as.character(aug$y), levels = lev)
  n0 <- sum(aug$wy)
  d <- data.frame(y = yl, aug$x)
  xnames <- colnames(d)[-1L]
  obs <- d[aug$ry, , drop = FALSE]
  new <- d[aug$wy, , drop = FALSE]
  fit <- tryCatch(suppressWarnings(MASS::polr(y ~ ., data = obs, weights = aug$w[aug$ry])), error = function(e) e)
  fallback <- NULL
  if (inherits(fit, "error")) {
    fallback <- conditionMessage(fit)
    fit <- suppressWarnings(nnet::multinom(y ~ .,
      data = obs, weights = aug$w[aug$ry], maxit = maxit, trace = FALSE, MaxNWts = max_nwts
    ))
    post <- predict(fit, new, type = "probs")
    prob <- matrix(0, n0, length(lev))
    prob[, match(fit$lev, lev)] <- matrix(post, nrow = n0)
    out <- list(fit_class = "multinom", value = fit$value, coef = multinom_coef(fit), fitted_levels = I(fit$lev))
  } else {
    coef <- setNames(numeric(length(xnames)), xnames)
    coef[names(fit$coefficients)] <- fit$coefficients
    eta <- drop(as.matrix(new[, xnames, drop = FALSE]) %*% coef)
    cumulative <- plogis(matrix(fit$zeta, n0, length(fit$zeta), byrow = TRUE) - eta)
    prob <- matrix(t(apply(cumulative, 1, function(p) diff(c(0, p, 1)))), nrow = n0)
    out <- list(
      fit_class = "polr", coef = unname(coef), zeta = unname(fit$zeta), deviance = fit$deviance,
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

polr_case <- function(name, input, seed, maxit = 100, max_nwts = 1500) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- inp$wy
  x <- inp$x
  n0 <- sum(wy)

  set.seed(seed)
  imp <- suppressWarnings(mice.impute.polr(y, ry, x, wy = wy, nnet.maxit = maxit, nnet.MaxNWts = max_nwts))
  saved <- .Random.seed
  set.seed(seed)
  runif(n0)
  stopifnot(identical(.Random.seed, saved))

  fits <- r_polr_fits(y, ry, x, wy, maxit, max_nwts)
  steps <- polr_steps(y, ry, x, wy, seed, maxit, max_nwts)
  stopifnot(identical(steps$seed_after, saved))
  levels <- levels(y)
  imputed <- levels[steps$codes]
  stopifnot(
    identical(as.character(imp), imputed),
    identical(fits$fit, steps$fit_class),
    identical(fits$polr_error, steps$fallback)
  )
  list(
    name = name,
    input = input,
    seed = seed,
    port = "match",
    maxit = maxit,
    max_weights = max_nwts,
    n_imp = n0,
    uniforms = I(steps$u),
    order = list(pair("uniform", n0)),
    r_draws = TRUE,
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
    prob = unname(steps$prob),
    levels = I(levels),
    imputed = I(imputed),
    r_imputed = I(as.character(imp)),
    r_error = NULL
  )
}

polr_cases <- lapply(seq_along(masks), function(k) {
  pattern <- names(masks)[k]
  polr_case(paste0("polr_", pattern), paste0("ord_", pattern), 7890L + k)
})

# ---------------------------------------------------------------------------
# pmm with one donor (as l1_pmm.R)
# ---------------------------------------------------------------------------

nearest_donors <- function(yhatobs, yhatmis) {
  vapply(yhatmis, function(t) {
    d <- abs(yhatobs - t)
    o <- order(d)
    # no ties, and no near-ties that rounding could flip
    stopifnot(d[o[2]] - d[o[1]] > 1e-6)
    o[1]
  }, integer(1))
}

pmm_case <- function(name, input, matchtype, seed) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- inp$wy
  x <- inp$x
  xi <- cbind(1, x)
  set.seed(seed)
  imp <- mice.impute.pmm(y, ry, x, wy = wy, donors = 1L, matchtype = matchtype)
  small <- drop_neither(inp)
  set.seed(seed)
  stopifnot(identical(
    mice.impute.pmm(small$y, small$ry, small$x, wy = small$wy, donors = 1L, matchtype = matchtype), imp
  ))

  df <- max(sum(ry) - ncol(xi), 1)
  set.seed(seed)
  chi <- rchisq(1, df)
  z <- rnorm(ncol(xi))
  p <- mice:::estimice(xi[ry, , drop = FALSE], y[ry])
  sigma <- sqrt(sum(p$r^2) / chi)
  beta <- as.vector(p$c + (t(chol(sym(p$v))) %*% z) * sigma)
  coef <- as.vector(p$c)
  bobs <- if (matchtype == 2L) beta else coef
  bmis <- if (matchtype == 0L) coef else beta
  idx <- nearest_donors(xi[ry, , drop = FALSE] %*% bobs, xi[wy, , drop = FALSE] %*% bmis)
  stopifnot(identical(imp, y[ry][idx]))

  list(
    name = name,
    input = input,
    seed = seed,
    matchtype = matchtype,
    donors = 1L,
    chisq = chi,
    normals = I(z),
    donor_index = I(idx - 1L),
    imputed = I(imp)
  )
}

pmm_cases <- c(
  lapply(seq_along(masks), function(k) {
    pattern <- names(masks)[k]
    pmm_case(paste0("pmm_", pattern), paste0("num_", pattern), 1L, 7900L + k)
  }),
  # matchtype 0: a row that is also a fit row is its own nearest donor
  list(pmm_case("pmm_overlap_matchtype0", "num_overlap", 0L, 7904L))
)
own <- pmm_cases[[4L]]
ry_all <- masks$overlap$ry
stopifnot(identical(as.integer(own$donor_index[ry_all] + 1L), seq_len(sum(ry_all))))

# ---------------------------------------------------------------------------
# midastouch donor probabilities (as l1_midastouch.R)
# ---------------------------------------------------------------------------

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

midas_case <- function(name, input, seed, outout) {
  inp <- inputs[[input]]
  ry <- inp$ry
  wy <- inp$wy
  set.seed(seed)
  omega <- mice:::bootfunc.plain(sum(ry))
  rm(list = ls(capture), envir = capture)
  set.seed(seed)
  # the traced copy lives in the namespace
  imp <- mice:::mice.impute.midastouch(inp$y, ry, inp$x, wy = wy, outout = outout)
  got <- get("last", envir = capture)
  small <- drop_neither(inp)
  set.seed(seed)
  stopifnot(identical(
    mice:::mice.impute.midastouch(small$y, small$ry, small$x, wy = small$wy, outout = outout), imp
  ))
  # R's sample() rescales a column that sums to more than 1 (donors at
  # distance 0 hit the upper bound of minmax); so does the port
  probs <- t(t(got$probs) / colSums(got$probs))
  stopifnot(
    identical(got$omega, omega),
    all(abs(colSums(probs) - 1) < 1e-12),
    dim(probs) == c(sum(ry), sum(wy))
  )
  list(
    name = name,
    input = input,
    seed = seed,
    outout = outout,
    midas_kappa = NULL,
    omega = I(omega),
    probs = unname(probs),
    kappa = got$kappa,
    imputed = I(imp)
  )
}

# untrace also when a case fails, so the namespace is left as found
midas_cases <- tryCatch(
  c(
    lapply(seq_along(masks), function(k) {
      pattern <- names(masks)[k]
      midas_case(paste0("midastouch_", pattern), paste0("num_", pattern), 7910L + k, outout = TRUE)
    }),
    list(midas_case("midastouch_overlap_no_outout", "num_overlap", 7914L, outout = FALSE))
  ),
  finally = invisible(suppressMessages(
    untrace("mice.impute.midastouch", where = asNamespace("mice"))
  ))
)

# ---------------------------------------------------------------------------
# augment (as l1_augment.R)
# ---------------------------------------------------------------------------

set.seed(7921)
na <- 12
a1 <- rnorm(na)
a2 <- runif(na)
a12 <- cbind(a1, a2)
lab <- c("a", "b", "c")[c(1, 2, 3, 1, 2, 3, 2, 1, 3, 1, 2, 2)]
rows12 <- function(i) seq_len(na) %in% i

aug_inputs <- list(
  # rows 11 and 12 in neither mask; row 11 (ignored, y observed) lacks a1
  # and has an extreme a2, row 12 (left out, y missing) lacks both: a1
  # counts 10 rows, a2 11
  na_one_predictor = list(
    y = factor(replace(lab, 12, NA), levels = c("a", "b", "c")),
    ry = rows12(1:8), wy = rows12(9:10),
    x = local({
      x <- a12
      x[11, ] <- c(NA, 9)
      x[12, ] <- NA
      x
    })
  ),
  # "c" only in an ignored row, which also has extreme predictors
  outside_only_category = list(
    y = factor(replace(ifelse(lab == "c", "a", lab), 12, "c"), levels = c("a", "b", "c")),
    ry = rows12(1:9), wy = rows12(10:11),
    x = local({
      x <- a12
      x[12, ] <- c(-8, 7)
      x
    })
  ),
  # exactly one row outside the fit rows, not to be imputed; two fit rows
  # are imputed (where): skipped
  one_outside_not_imputed = list(
    y = factor(replace(lab, 7, NA), levels = c("a", "b", "c")),
    ry = !rows12(7), wy = rows12(c(2, 5)), x = a12
  ),
  # one row to impute and two ignored rows: not skipped
  one_imputed_plus_ignored = list(
    y = factor(replace(lab, 4, NA), levels = c("a", "b", "c")),
    ry = !rows12(c(4, 8, 11)), wy = rows12(4), x = a12
  ),
  # nothing missing, every row imputed (where = "all"): not skipped
  all_observed_all_imputed = list(
    y = factor(lab, levels = c("a", "b", "c")),
    ry = rep(TRUE, na), wy = rep(TRUE, na), x = a12
  )
)

as_labels <- function(y) if (is.factor(y)) I(as.character(y)) else I(as.vector(y))

augment_case <- function(name) {
  inp <- aug_inputs[[name]]
  y <- inp$y
  x <- inp$x
  ry <- inp$ry
  wy <- inp$wy
  p <- ncol(x)
  n_rows <- length(y)
  aug <- augment(y, ry, x, wy)
  k <- length(sort(unique(unclass(y))))
  skipped <- sum(!ry) == 1
  n_added <- if (skipped) 0L else 2L * p * k
  stopifnot(
    length(aug$y) == n_rows + n_added, nrow(aug$x) == n_rows + n_added,
    identical(aug$ry, c(ry, rep(TRUE, n_added))), identical(aug$wy, c(wy, rep(FALSE, n_added))),
    identical(aug$w, c(rep(1, n_rows), rep((p + 1) / max(n_added, 1L), n_added)))
  )
  if (!skipped) {
    # the statistics are those of each column without its NA, over all rows
    added <- as.matrix(aug$x[-seq_len(n_rows), , drop = FALSE])
    for (j in seq_len(p)) {
      v <- x[!is.na(x[, j]), j]
      expected <- pmin(pmax(mean(v) + c(0.5, -0.5) * sd(v), min(v)), max(v))
      stopifnot(isTRUE(all.equal(unname(added[2 * k * (j - 1) + 1:2, j]), expected, tolerance = 1e-14)))
    }
  }
  list(
    name = name,
    kind = kind_of(y),
    y = as_labels(y),
    levels = I(levels(y)),
    ry = I(ry),
    wy = I(wy),
    x = unname(x),
    p = p,
    augmented = list(
      y = as_labels(aug$y),
      levels = if (is.factor(aug$y)) I(levels(aug$y)) else NULL,
      ry = I(aug$ry),
      wy = I(aug$wy),
      x = unname(as.matrix(aug$x)),
      w = I(aug$w)
    ),
    r_error = NULL
  )
}

augment_cases <- lapply(names(aug_inputs), augment_case)
names(augment_cases) <- names(aug_inputs)

# ---------------------------------------------------------------------------
# What the cases show
# ---------------------------------------------------------------------------

by_name <- function(cases) setNames(cases, vapply(cases, `[[`, "", "name"))
lr <- by_name(logreg_cases)

# augment uses the rows in neither mask: without them its rows move far
for (pattern in c("outside", "na_rows")) {
  inp <- inputs[[paste0("bin_", pattern)]]
  small <- drop_neither(inp)
  a <- augment(inp$y, inp$ry, inp$x, inp$wy)
  b <- augment(small$y, small$ry, small$x, small$wy)
  stopifnot(max(abs(as.matrix(a$x[-seq_len(n), ]) - as.matrix(b$x[-seq_len(nrow(small$x)), ]))) > 1)
}
# a category held only outside the fit rows is augmented and fitted
oc <- inputs$bin_outside_category
stopifnot(
  identical(unclass(lr$logreg_outside_category$levels), c("no", "yes")),
  lr$logreg_outside_category$converged,
  all(oc$y[oc$ry] == "no"), any(oc$y[!oc$ry] == "yes", na.rm = TRUE)
)
# logreg stops on the third value, logreg.boot never reads it
stopifnot(
  identical(lr$logreg_third_value$r_error, "y values must be 0 <= y <= 1"),
  identical(unclass(lr$logreg_boot_third_value$levels), c(0, 1))
)
# logreg.boot with overlapping masks: R predicts the rows that are fit rows
# from the bootstrapped predictors; the rows that are not fit rows agree
boot <- lr$logreg_boot_overlap
inp <- inputs$bin_overlap
set.seed(boot$seed)
s <- sample(sum(inp$ry), sum(inp$ry), replace = TRUE)
dotx <- inp$x
dotx[inp$ry, ] <- inp$x[inp$ry, , drop = FALSE][s, , drop = FALSE]
r_prob <- as.vector(1 / (1 + exp(-(cbind(1, dotx) %*% unclass(boot$beta_star)))))
port_prob <- unclass(boot$prob)
set.seed(boot$seed)
stopifnot(identical(
  as.character(suppressWarnings(mice.impute.logreg.boot(inp$y, inp$ry, inp$x, wy = inp$wy))),
  c("no", "yes")[1L + (unclass(boot$uniforms) <= r_prob)]
))
stopifnot(
  isTRUE(all.equal(r_prob[!inp$ry], port_prob[!inp$ry], tolerance = 1e-14)),
  max(abs(r_prob[inp$ry] - port_prob[inp$ry])) > 0.1,
  !identical(unclass(boot$r_imputed), unclass(boot$imputed))
)
lr$logreg_boot_overlap$r_prob <- I(r_prob)
logreg_cases <- unname(lr)
# polyreg and polr: "d" occurs only outside the fit rows and is fitted
for (case in c(polyreg_cases, polr_cases)) {
  stopifnot("d" %in% case$levels)
}
stopifnot(all(vapply(polyreg_cases, function(case) "d" %in% case$fitted_levels, TRUE)))
# augment: R's skip rule counts the rows outside the fit rows, not the rows
# to impute
au <- augment_cases
stopifnot(
  length(au$one_outside_not_imputed$augmented$w) == na, sum(aug_inputs$one_outside_not_imputed$wy) == 2,
  length(au$one_imputed_plus_ignored$augmented$w) > na, sum(aug_inputs$one_imputed_plus_ignored$wy) == 1,
  length(au$all_observed_all_imputed$augmented$w) > na,
  identical(unclass(au$outside_only_category$augmented$levels), c("a", "b", "c"))
)
# midastouch with overlapping masks: a fit row in the bootstrap sample is
# its own donor with a probability near 1
md <- by_name(midas_cases)$midastouch_overlap
own_rows <- which(masks$overlap$ry)
own_prob <- md$probs[cbind(seq_along(own_rows), own_rows)]
stopifnot(all(own_prob[md$omega > 0] > 1 - 1e-6), all(own_prob[md$omega == 0] == 0))

stopifnot(all(vapply(logreg_cases, function(case) is.null(case$converged) || case$converged, TRUE)))
for (case in polr_cases) message(sprintf("%-28s R %s port %s", case$name, case$fit, case$port_fit))

encode_input <- function(inp) {
  y <- inp$y
  list(
    kind = kind_of(y),
    y = if (is.factor(y)) I(as.character(y)) else I(y),
    levels = if (is.factor(y)) I(levels(y)) else NULL,
    ry = I(inp$ry),
    wy = I(inp$wy),
    x = unname(inp$x),
    p = ncol(inp$x)
  )
}

write_fixture("l1_masks", list(
  description = paste(
    "Direct calls of mice's imputers with the masks sampler.univ produces:",
    "wy all true (overlap), rows in neither mask with an observed y (ignored)",
    "or NA (left out by where), and such rows with NA in x (na_rows). The",
    "rows in neither mask hold decoys (extreme x, extreme y or a label no fit",
    "row has). inputs: kind, y (labels or numbers; null for NA), levels",
    "(declared), ry, wy, x without intercept (rows; null for NA), p. The",
    "cases have the formats of l1_norm (norm_cases), l1_logreg",
    "(logreg_cases; logreg_boot_overlap adds r_prob, R's probabilities),",
    "l1_polyreg (polyreg_cases), l1_polr (polr_cases), l1_pmm_one_donor",
    "(pmm_cases, with input), l1_midastouch (midastouch_cases, with input;",
    "probs rescaled to sum to 1 as R's sample() does) and l1_augment",
    "(augment_cases). port 'rebuilt' (logreg.boot with overlapping masks):",
    "R predicts a row that is also a fit row from the bootstrapped",
    "predictors that replaced it (r_imputed, r_prob); the port from its own",
    "predictors. port 'value_error': R stops (r_error), and so does the port."
  ),
  versions = list(mass = MASS_VERSION, nnet = NNET_VERSION),
  inputs = lapply(inputs, encode_input),
  norm_cases = norm_cases,
  logreg_cases = logreg_cases,
  polyreg_cases = polyreg_cases,
  polr_cases = polr_cases,
  pmm_cases = pmm_cases,
  midastouch_cases = midas_cases,
  augment_cases = unname(augment_cases)
))
