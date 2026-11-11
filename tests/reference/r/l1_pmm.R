# L1 fixtures for predictive mean matching.
#
#   l1_quantify.json        mice:::quantify() for a 3-level and a 2-level factor
#   l1_pmm_one_donor.json   mice.impute.pmm(donors = 1) with captured draws
#   l1_matchindex.json      donors reached by matchindex(d, t, k) in 2000 calls
#
# Python targets: imputation.PMM.quantify, imputation.PMM.pmm,
# imputation.PMM.matchindex.

source(file.path("tests", "reference", "r", "_common.R"))

# ---------------------------------------------------------------------------
# quantify
# ---------------------------------------------------------------------------
# mice.impute.pmm() calls quantify(y, ry, x) after adding the intercept
# column to x, so the stored x includes it. The canonical variate is defined
# up to its sign only.

set.seed(20261113)
n <- 40
x1 <- rnorm(n)
x2 <- rnorm(n)
x <- unname(cbind(1, x1, x2))
ry <- rep(TRUE, n)
ry[sample(n, 8)] <- FALSE

lin <- x1 - 0.7 * x2 + rnorm(n, sd = 0.8)
y3 <- cut(lin, c(-Inf, -0.6, 0.6, Inf), labels = c("low", "mid", "high"))
y2 <- factor(ifelse(x2 + rnorm(n) > 0, "yes", "no"), levels = c("no", "yes"))

quantify_cases <- list()
for (nm in c("three_levels", "two_levels")) {
  y <- if (nm == "three_levels") y3 else y2
  y[!ry] <- NA
  ynum <- mice:::quantify(y, ry, x)
  stopifnot(all(is.na(ynum[!ry])), !anyNA(ynum[ry]))
  quantify_cases[[length(quantify_cases) + 1L]] <- list(
    name = nm,
    y = I(as.character(y)),
    levels = I(levels(y)),
    ry = I(ry),
    x = x,
    ynum = I(as.vector(ynum))
  )
}

write_fixture("l1_quantify", list(
  description = paste(
    "mice:::quantify(y, ry, x): optimal scaling of factor y. x includes the",
    "intercept column, as in mice.impute.pmm(). ynum is null where ry is",
    "false; observed entries are defined up to a global sign."
  ),
  cases = quantify_cases
))

# ---------------------------------------------------------------------------
# PMM with one donor and captured draws
# ---------------------------------------------------------------------------
# With donors = 1, matchindex() still calls sample() twice, but the nearest
# donor is unique when there are no ties, so the result depends only on the
# draws of .norm.draw(): rchisq(1, df) and rnorm(ncol(x) + 1). These are
# captured by re-seeding. The check rebuilds the imputations from them.

nearest_donors <- function(yhatobs, yhatmis) {
  vapply(yhatmis, function(t) {
    d <- abs(yhatobs - t)
    o <- order(d)
    # no ties, and no near-ties that rounding could flip
    stopifnot(d[o[2]] - d[o[1]] > 1e-6)
    o[1]
  }, integer(1))
}

pmm_case <- function(name, y, ry, x, matchtype, seed) {
  xi <- cbind(1, x)
  set.seed(seed)
  imp <- mice.impute.pmm(y, ry, x, donors = 1L, matchtype = matchtype)

  ynum <- if (is.factor(y)) mice:::quantify(y, ry, xi) else y
  df <- max(sum(ry) - ncol(xi), 1)
  set.seed(seed)
  chi <- rchisq(1, df)
  z <- rnorm(ncol(xi))

  p <- mice:::estimice(xi[ry, , drop = FALSE], ynum[ry])
  sigma <- sqrt(sum(p$r^2) / chi)
  beta <- as.vector(p$c + (t(chol(mice:::sym(p$v))) %*% z) * sigma)
  coef <- as.vector(p$c)
  bobs <- if (matchtype == 2L) beta else coef
  bmis <- if (matchtype == 0L) coef else beta
  idx <- nearest_donors(xi[ry, ] %*% bobs, xi[!ry, ] %*% bmis)
  stopifnot(identical(imp, y[ry][idx]))

  list(
    name = name,
    seed = seed,
    matchtype = matchtype,
    donors = 1L,
    y = I(if (is.factor(y)) as.character(y) else y),
    levels = if (is.factor(y)) I(levels(y)) else NULL,
    ry = I(ry),
    x = unname(x),
    chisq = chi,
    normals = I(z),
    donor_index = I(idx - 1L),
    imputed = I(if (is.factor(imp)) as.character(imp) else imp)
  )
}

set.seed(20261114)
n <- 30
x1 <- rnorm(n)
x2 <- runif(n)
yn <- 10 + 2 * x1 - 3 * x2 + rnorm(n)
ry <- rep(TRUE, n)
ry[sample(n, 10)] <- FALSE
yn[!ry] <- NA
stopifnot(!anyDuplicated(yn[ry]))
xn <- cbind(x1, x2)

pmm_cases <- list(
  pmm_case("numeric_matchtype0", yn, ry, xn, 0L, 11L),
  pmm_case("numeric_matchtype1", yn, ry, xn, 1L, 12L),
  pmm_case("numeric_matchtype2", yn, ry, xn, 2L, 13L),
  pmm_case("numeric_matchtype1_seed2", yn, ry, xn, 1L, 14L)
)

# Factor target: the quantified y is defined up to its sign. With
# matchtype = 0 only the least-squares fit is used, so the donor is the same
# for either sign. (With matchtype 1 or 2 the captured normals would act on
# coefficients of the opposite sign.)
set.seed(20261115)
lin <- x1 - 2 * x2 + rnorm(n, sd = 0.5)
yf <- cut(lin, quantile(lin, c(0, 1 / 3, 2 / 3, 1)),
  include.lowest = TRUE, labels = c("a", "b", "c")
)
ryf <- rep(TRUE, n)
ryf[sample(n, 8)] <- FALSE
yf[!ryf] <- NA
stopifnot(all(table(yf[ryf]) >= 1))
pmm_cases[[length(pmm_cases) + 1L]] <-
  pmm_case("factor_matchtype0", yf, ryf, xn, 0L, 15L)

write_fixture("l1_pmm_one_donor", list(
  description = paste(
    "mice.impute.pmm(y, ry, x, donors = 1, matchtype) with the draws of",
    ".norm.draw(): chisq = rchisq(1, df), normals = rnorm(ncol(x) + 1).",
    "x excludes the intercept. donor_index is 0-based into y[ry].",
    "Factor y is stored as strings plus levels."
  ),
  cases = pmm_cases
))

# ---------------------------------------------------------------------------
# matchindex with k > 1
# ---------------------------------------------------------------------------
# For each target, record which donors 2000 calls of matchindex(d, t, k) reach.
# Without ties that set is the k nearest donors (k clamped to [1, length(d)]).
# The probability that 2000 calls miss one of k <= 12 donors is below 1e-70.

set.seed(20261116)
d <- rnorm(12)
t <- c(rnorm(5), min(d) - 1, max(d) + 1, mean(d))
stopifnot(!anyDuplicated(d))

match_cases <- list()
for (k in c(1L, 3L, 5L, 12L, 20L)) {
  set.seed(1000L + k)
  reached <- matrix(FALSE, length(t), length(d))
  for (it in seq_len(2000L)) {
    idx <- matchindex(d, t, k)
    reached[cbind(seq_along(t), idx)] <- TRUE
  }
  keff <- min(max(k, 1L), length(d))
  nearest <- t(vapply(t, function(v) {
    dist <- abs(d - v)
    stopifnot(!anyDuplicated(dist))
    rank(dist) <= keff
  }, logical(length(d))))
  stopifnot(identical(reached, nearest))
  match_cases[[length(match_cases) + 1L]] <- list(
    name = paste0("k", k),
    k = k,
    reached = lapply(seq_along(t), function(i) I(which(reached[i, ]) - 1L))
  )
}

write_fixture("l1_matchindex", list(
  description = paste(
    "Donors reached by 2000 calls of matchindex(d, t, k), per target, as",
    "0-based indices into d. Equal to the min(k, length(d)) nearest donors."
  ),
  d = I(d),
  t = I(t),
  calls = 2000L,
  cases = match_cases
))
