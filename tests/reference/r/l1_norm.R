# L1 fixture for imputation by linear regression and by the mean.
#
#   l1_norm.json   mice.impute.norm, mice.impute.norm.nob, mice.impute.norm.boot,
#                  mice.impute.norm.predict and mice.impute.mean, with every
#                  random number they draw captured
#
# Python targets: imputation.norm.norm, norm_nob, norm_boot, norm_predict and
# imputation.mean.mean, driven by a strict ScriptedGenerator.
#
# Each case calls the mice function after set.seed(seed) and saves
# .Random.seed. Then it seeds again and draws what the function is expected to
# draw, in the expected order; the same .Random.seed afterwards proves the
# kind, number and order of the draws, and that nothing else was drawn. The
# imputations are rebuilt from those draws with mice's own estimice() and must
# be identical to the function's result.

source(file.path("tests", "reference", "r", "_common.R"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(20261124)
n <- 20
x1 <- rnorm(n)
x2 <- runif(n)
y_full <- 4 + 2 * x1 - 3 * x2 + rnorm(n)
mis <- sort(sample(n, 6))
ry <- !(seq_len(n) %in% mis)
ry_one <- seq_len(n) != mis[1]
ry_tiny <- seq_len(n) %in% sort(sample(n, 4)) # 4 observed: n1 = q + 1 with x1, x2
x12 <- cbind(x1, x2)

inputs <- list(
  base = list(y = replace(y_full, !ry, NA), ry = ry, x = x12),
  one_predictor = list(y = replace(y_full, !ry, NA), ry = ry, x = cbind(x1)),
  no_predictor = list(y = replace(y_full, !ry, NA), ry = ry, x = x12[, 0, drop = FALSE]),
  one_missing = list(y = replace(y_full, !ry_one, NA), ry = ry_one, x = x12),
  # impute only three of the six missing cells (R's wy)
  wy_subset = list(
    y = replace(y_full, !ry, NA), ry = ry, x = x12,
    wy = seq_len(n) %in% mis[c(1, 3, 4)]
  ),
  # the missing cells hold values (inside mice: the current imputations)
  filled_target = list(y = replace(y_full, !ry, 1000 + seq_along(mis)), ry = ry, x = x12),
  tiny = list(y = replace(y_full, !ry_tiny, NA), ry = ry_tiny, x = x12)
)

# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

pair <- function(pool, k) list(pool, as.integer(k))

norm_case <- function(name, method, input, seed, ls_meth = "qr", ridge = 1e-05) {
  inp <- inputs[[input]]
  y <- inp$y
  ry <- inp$ry
  wy <- if (is.null(inp$wy)) !ry else inp$wy
  x <- inp$x
  X <- cbind(1, x)
  q <- ncol(X)
  n1 <- sum(ry)
  n0 <- sum(wy)
  regression <- method != "mean"

  fun <- get(paste0("mice.impute.", method), envir = asNamespace("mice"))
  args <- list(y, ry, x, wy = wy)
  if (regression) args <- c(args, list(ls.meth = ls_meth, ridge = ridge))
  set.seed(seed)
  imp <- tryCatch(do.call(fun, args), error = function(e) e)
  saved <- .Random.seed
  failed <- inherits(imp, "error")

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
  if (failed) {
    # R stops after the bootstrap draw; the port goes on and draws the residuals
    stopifnot(identical(.Random.seed, saved))
  }
  e <- if (method %in% c("norm", "norm.nob", "norm.boot")) rnorm(n0) else numeric(0)
  if (!failed) {
    stopifnot(identical(.Random.seed, saved))
  }

  # rebuild the imputations from the draws
  Xo <- X[ry, , drop = FALSE]
  Xm <- X[wy, , drop = FALSE]
  yo <- y[ry]
  if (method == "norm") {
    p <- mice:::estimice(Xo, yo, ls.meth = ls_meth, ridge = ridge)
    sigma <- sqrt(sum(p$r^2) / chi)
    beta <- p$c + (t(chol(mice:::sym(p$v))) %*% z) * sigma
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
  if (!failed) {
    stopifnot(identical(rebuilt, as.vector(imp)))
  }

  # what the port does: R's result, the intercept-only fit where R fails,
  # an error where R's imputations are not finite
  port <- if (failed) {
    "intercept_only"
  } else if (!all(is.finite(rebuilt))) {
    "value_error"
  } else {
    "match"
  }
  stopifnot(!failed || (method == "norm.boot" && q == 1L))

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
    port = port,
    imputed = if (failed) NULL else I(rebuilt),
    r_error = if (failed) conditionMessage(imp) else NULL,
    port_imputed = if (failed) I(rebuilt) else NULL
  )
}

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

SEEDS <- c(
  norm = 7101L, norm.nob = 7105L, norm.boot = 7106L,
  norm.predict = 7108L, mean = 7109L
)
VARIANTS <- c("one_predictor", "no_predictor", "one_missing", "wy_subset", "filled_target")
label <- function(method) gsub(".", "_", method, fixed = TRUE)

cases <- list(
  norm_case("norm_qr_seed1", "norm", "base", 7101L),
  norm_case("norm_qr_seed2", "norm", "base", 7102L),
  norm_case("norm_ridge_seed1", "norm", "base", 7103L, ls_meth = "ridge"),
  norm_case("norm_ridge_seed2", "norm", "base", 7104L, ls_meth = "ridge"),
  norm_case("norm_boot_seed2", "norm.boot", "base", 7107L)
)
for (method in names(SEEDS)) {
  if (method != "norm") {
    cases[[length(cases) + 1L]] <- norm_case(label(method), method, "base", SEEDS[[method]])
  }
  if (method %in% c("norm.nob", "norm.boot", "norm.predict")) {
    # a large penalty, so that a lost ridge argument shows
    cases[[length(cases) + 1L]] <- norm_case(
      paste0(label(method), "_ridge"), method, "base", SEEDS[[method]],
      ls_meth = "ridge", ridge = 0.1
    )
  }
  for (variant in VARIANTS) {
    cases[[length(cases) + 1L]] <- norm_case(
      paste(label(method), variant, sep = "_"), method, variant, SEEDS[[method]]
    )
  }
}
cases[[length(cases) + 1L]] <- norm_case("norm_nob_tiny", "norm.nob", "tiny", SEEDS[["norm.nob"]])

# The values at the missing cells must not matter
base_of <- function(method) if (method == "norm") "norm_qr_seed1" else label(method)
by_name <- setNames(cases, vapply(cases, `[[`, "", "name"))
for (method in names(SEEDS)) {
  stopifnot(identical(
    by_name[[paste(label(method), "filled_target", sep = "_")]]$imputed,
    by_name[[base_of(method)]]$imputed
  ))
}
stopifnot(by_name$norm_boot_no_predictor$port == "intercept_only")
stopifnot(by_name$norm_nob_tiny$port == "value_error")
stopifnot(sum(vapply(cases, function(case) case$port != "match", logical(1))) == 2L)

encode_input <- function(inp) {
  list(
    y = I(inp$y),
    ry = I(inp$ry),
    wy = if (is.null(inp$wy)) NULL else I(inp$wy),
    x = if (ncol(inp$x) > 0L) unname(inp$x) else NULL,
    p = ncol(inp$x)
  )
}

write_fixture("l1_norm", list(
  description = paste(
    "mice.impute.<method>(y, ry, x, wy, ls.meth, ridge) for norm, norm.nob,",
    "norm.boot, norm.predict and mean (no ls.meth or ridge), with every random",
    "number captured: chisq = rchisq(1, df) (norm only), integers =",
    "sample(n_obs, n_obs, TRUE) - 1 (norm.boot only), normals = rnorm(n_coef)",
    "(norm only) followed by rnorm(n_imputed). order is the expected sequence",
    "of draws as [pool, count] pairs. Inputs: y is null where ry is false",
    "(except filled_target), wy null means !ry, x excludes the intercept and",
    "has p columns (null when p = 0). port says what mice-py does: 'match'",
    "(R's imputed), 'intercept_only' (R errors, r_error; mice-py fits the",
    "intercept-only model, port_imputed, rebuilt in R with estimice and the",
    "draws R would make next) or 'value_error' (R's imputed are not finite)."
  ),
  inputs = lapply(inputs, encode_input),
  cases = cases
))
