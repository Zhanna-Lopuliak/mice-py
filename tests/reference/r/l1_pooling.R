# L1 fixtures for pooling with Rubin's rules and Barnard-Rubin df.
#
#   l1_pooling.json   mice(nhanes) completed datasets; per-imputation fits of
#                     a linear and a logistic model; pool(); summary(pool());
#                     pool.table() edge cases; barnard.rubin() on a grid
#
# Python targets: imputation.pooling.barnard_rubin, pool_estimates, and the
# end-to-end path MICE.fit / MICE.pool on R's completed datasets.

source(file.path("tests", "reference", "r", "_common.R"))

columns_of <- function(df, drop = character(0)) {
  df <- as.data.frame(df)
  df <- df[, setdiff(names(df), drop), drop = FALSE]
  out <- lapply(df, function(col) I(if (is.factor(col)) as.character(col) else col))
  names(out) <- names(df)
  out
}

# summary(pool(fit), type = "all", conf.int = TRUE) also has the columns
# "2.5 %" and "97.5 %", which duplicate conf.low and conf.high.
summary_columns <- function(s) columns_of(s, drop = c("2.5 %", "97.5 %"))

imp <- mice(nhanes, m = 5, seed = 20261120L, printFlag = FALSE)
stopifnot(is.null(imp$loggedEvents))
completed <- lapply(seq_len(imp$m), function(i) encode_frame(complete(imp, i)))

model_case <- function(name, formula_r, formula_py, family, fit) {
  per_imp <- lapply(fit$analyses, function(f) {
    list(
      estimate = I(unname(coef(f))),
      std_error = I(unname(sqrt(diag(vcov(f))))),
      df_residual = df.residual(f)
    )
  })
  pooled <- pool(fit)
  list(
    name = name,
    formula_r = formula_r,
    formula_py = formula_py,
    family = family,
    terms = I(names(coef(fit$analyses[[1]]))),
    per_imputation = per_imp,
    pooled = columns_of(pooled$pooled),
    summary = summary_columns(summary(pooled, type = "all", conf.int = TRUE))
  )
}

fit_lm <- with(imp, lm(bmi ~ age + hyp + chl))
# Tight convergence: vcov.glm() uses the IRLS weights of the last iteration,
# which with the default epsilon = 1e-8 are those of the previous iterate.
# The standard errors then differ from the inverse information at the MLE
# (what statsmodels reports) by about 1e-5 relative here; with epsilon =
# 1e-14 the two agree to about 1e-10.
fit_glm <- with(imp, glm(I(hyp - 1) ~ age + chl,
  family = binomial,
  control = glm.control(epsilon = 1e-14, maxit = 100)
))
stopifnot(all(vapply(fit_glm$analyses, function(f) f$converged, logical(1))))

models <- list(
  model_case("lm", "bmi ~ age + hyp + chl", "bmi ~ age + hyp + chl",
    "gaussian", fit_lm),
  model_case("logistic", "I(hyp - 1) ~ age + chl", "I(hyp - 1) ~ age + chl",
    "binomial", fit_glm)
)

# ---------------------------------------------------------------------------
# pool.table edge cases on hand-made estimates
# ---------------------------------------------------------------------------

edge_case <- function(name, estimates, std_errors, dfcom) {
  m <- nrow(estimates)
  terms <- c("a", "b")
  w <- data.frame(
    term = rep(terms, times = m),
    estimate = as.vector(t(estimates)),
    std.error = as.vector(t(std_errors))
  )
  s <- suppressWarnings( # "Large sample assumed." for infinite dfcom
    pool.table(w, type = "all", conf.int = TRUE, dfcom = dfcom)
  )
  list(
    name = name,
    terms = I(terms),
    estimates = estimates,
    std_errors = std_errors,
    dfcom = dfcom,
    summary = summary_columns(s)
  )
}

set.seed(20261121)
est5 <- cbind(rnorm(5, 1, 0.3), rnorm(5, -2, 0.1))
se5 <- cbind(runif(5, 0.2, 0.4), runif(5, 0.05, 0.1))
edges <- list(
  # identical estimates: b = 0, lambda = 0, riv = 0
  edge_case("b_zero", cbind(rep(1.5, 5), rep(-0.3, 5)), se5, 20),
  edge_case("m_two", est5[1:2, ], se5[1:2, ], 30),
  edge_case("dfcom_infinite", est5, se5, Inf),
  # b = 0 and infinite dfcom: df is infinite, fmi is 0
  edge_case("b_zero_dfcom_infinite", cbind(rep(1.5, 5), rep(-0.3, 5)), se5, Inf)
)

# ---------------------------------------------------------------------------
# barnard.rubin on a grid
# ---------------------------------------------------------------------------

grid <- expand.grid(
  m = c(2, 3, 5, 20),
  b = c(0, 0.05, 0.5, 2),
  ubar = c(0.1, 1),
  dfcom = c(Inf, 5, 30, 1000)
)
grid$t <- grid$ubar + (1 + 1 / grid$m) * grid$b
grid$df <- mapply(mice:::barnard.rubin, grid$m, grid$b, grid$t, grid$dfcom)

write_fixture("l1_pooling", list(
  description = paste(
    "Pooling reference. 'completed' are the m = 5 datasets of",
    "mice(nhanes, m = 5, seed = 20261120). For each model: per-imputation",
    "estimates and standard errors, pool(fit)$pooled and",
    "summary(pool(fit), type = 'all', conf.int = TRUE). 'edges' run",
    "pool.table() on hand-made estimates. 'barnard_rubin' evaluates",
    "mice:::barnard.rubin(m, b, t, dfcom) on a grid."
  ),
  data = encode_frame(nhanes),
  completed = completed,
  models = models,
  edges = edges,
  barnard_rubin = columns_of(grid)
))
