# L1 fixtures for the least-squares step and the Bayesian coefficient draw.
#
#   l1_estimice.json   mice:::estimice() for ls.meth = "qr", "ridge", "svd"
#   l1_norm_draw.json  mice:::.norm.draw() with its random numbers captured
#
# Python targets: imputation.sampler.estimice and imputation.sampler.norm_draw.

source(file.path("tests", "reference", "r", "_common.R"))

# ---------------------------------------------------------------------------
# estimice
# ---------------------------------------------------------------------------
# Inputs are design matrices as .norm.draw() passes them to estimice(), that is
# with the intercept column already added (except "one_column").

set.seed(20261111)
n <- 30
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- runif(n)
y <- 1 + 0.5 * x1 - x2 + 2 * x3 + rnorm(n)
# third predictor almost equal to x1 + x2: correlation above 0.9999, but X'X
# still invertible without the ridge fallback (which needs a running mice())
x_near <- x1 + x2 + rnorm(n, sd = 0.01)

designs <- list(
  well = cbind(1, x1, x2, x3),
  single = cbind(1, x1),
  one_column = cbind(x1),
  near_collinear = cbind(1, x1, x2, x_near)
)
stopifnot(cor(x_near, x1 + x2) > 0.9999)

estimice_cases <- list()
for (design in names(designs)) {
  x <- unname(designs[[design]])
  for (meth in c("qr", "ridge", "svd")) {
    # R's svd branch fails for a one-column x: diag() of a scalar singular
    # value builds an identity matrix instead of a 1 x 1 matrix.
    if (meth == "svd" && ncol(x) == 1L) next
    p <- mice:::estimice(x, y, ls.meth = meth)
    estimice_cases[[length(estimice_cases) + 1L]] <- list(
      name = paste(design, meth, sep = "_"),
      design = design,
      ls_meth = meth,
      ridge = 1e-05,
      x = x,
      y = I(y),
      c = I(as.vector(p$c)),
      r = I(as.vector(p$r)),
      v = p$v,
      df = p$df,
      kappa_xtx = kappa(crossprod(x), exact = TRUE)
    )
  }
}

write_fixture("l1_estimice", list(
  description = paste(
    "mice:::estimice(x, y, ls.meth) on fixed inputs. x already holds the",
    "intercept column where .norm.draw() would add it. For ls.meth = 'svd'",
    "R returns residuals as fitted minus observed."
  ),
  cases = estimice_cases
))

# ---------------------------------------------------------------------------
# .norm.draw with captured random numbers
# ---------------------------------------------------------------------------
# .norm.draw() draws rchisq(1, df) and then rnorm(ncol(x)), nothing else.
# Re-seeding and drawing the same two calls captures these numbers. The check
# below rebuilds sigma and beta from them, which proves the capture is right.

set.seed(20261112)
n <- 40
x1 <- rnorm(n)
x2 <- rnorm(n, sd = 2)
y <- 2 + x1 - 0.5 * x2 + rnorm(n)
ry <- rep(TRUE, n)
ry[sample(n, 10)] <- FALSE
y[!ry] <- NA
x <- unname(cbind(1, x1, x2))

draw_cases <- list()
for (meth in c("qr", "ridge")) {
  for (s in c(1L, 2L, 3L)) {
    set.seed(s)
    parm <- mice:::.norm.draw(y, ry, x, ls.meth = meth)
    df <- max(sum(ry) - ncol(x), 1)
    set.seed(s)
    chi <- rchisq(1, df)
    z <- rnorm(ncol(x))

    p <- mice:::estimice(x[ry, , drop = FALSE], y[ry], ls.meth = meth)
    sigma <- sqrt(sum(p$r^2) / chi)
    beta <- p$c + (t(chol(mice:::sym(p$v))) %*% z) * sigma
    stopifnot(
      identical(sigma, parm$sigma),
      identical(as.vector(beta), as.vector(parm$beta)),
      identical(as.vector(p$c), as.vector(parm$coef))
    )

    draw_cases[[length(draw_cases) + 1L]] <- list(
      name = paste(meth, "seed", s, sep = "_"),
      ls_meth = meth,
      seed = s,
      chisq = chi,
      normals = I(z),
      coef = I(as.vector(parm$coef)),
      beta = I(as.vector(parm$beta)),
      sigma = parm$sigma
    )
  }
}

write_fixture("l1_norm_draw", list(
  description = paste(
    "mice:::.norm.draw(y, ry, x, ls.meth) with the random numbers it uses:",
    "chisq = rchisq(1, df), then normals = rnorm(ncol(x)). x holds the",
    "intercept column. y is null where ry is false."
  ),
  y = I(y),
  ry = I(ry),
  x = x,
  cases = draw_cases
))
