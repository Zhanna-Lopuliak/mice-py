# L1 fixtures for the setup step of mice().
#
#   l1_setup.json   mice(data, maxit = 0): method, predictorMatrix,
#                   visitSequence for each visit order, loggedEvents
#   l1_design.json  mice:::obtain.design, mice:::remove.lindep,
#                   mice:::find.collinear
#
# Python targets: MICE.impute (.method, .predictor_matrix, .visit_sequence)
# and the planned module imputation.design.

source(file.path("tests", "reference", "r", "_common.R"))

VISITS <- c("roman", "arabic", "monotone", "revmonotone")

named_matrix <- function(m) {
  list(rows = I(rownames(m)), cols = I(colnames(m)), values = unname(m))
}

logged_events <- function(imp) {
  le <- imp$loggedEvents
  if (is.null(le)) {
    return(NULL)
  }
  list(
    it = I(le$it), im = I(le$im), dep = I(le$dep),
    meth = I(le$meth), out = I(le$out)
  )
}

setup_case <- function(name, data, expect_log = FALSE) {
  run <- function(...) {
    imp <- withCallingHandlers(
      mice(data, maxit = 0, printFlag = FALSE, seed = 1L, ...),
      warning = function(w) {
        # the only warning allowed is the count of logged events
        if (expect_log && grepl("logged events", conditionMessage(w))) {
          invokeRestart("muffleWarning")
        }
      }
    )
    if (!expect_log) stopifnot(is.null(imp$loggedEvents))
    imp
  }
  imp <- run()
  visits <- lapply(VISITS, function(v) I(run(visitSequence = v)$visitSequence))
  names(visits) <- VISITS

  # Effective imputation order: R's visitSequence can list variables that
  # are never imputed (complete ones for "monotone", "revmonotone" and the
  # default; variables whose method mice.edit.setup set to ""). The sampler
  # skips those, so the order in which R imputes is the sequence restricted
  # to variables with a non-empty method.
  imputed <- names(imp$method)[imp$method != ""]
  effective <- lapply(visits, function(v) I(v[v %in% imputed]))

  # Variables left incomplete after one iteration (mice leaves a variable
  # with method "" untouched).
  imp1 <- withCallingHandlers(
    mice(data, m = 1, maxit = 1, printFlag = FALSE, seed = 1L),
    warning = function(w) {
      if (expect_log && grepl("logged events", conditionMessage(w))) {
        invokeRestart("muffleWarning")
      }
    }
  )
  still_missing <- names(which(colSums(is.na(complete(imp1))) > 0))

  list(
    name = name,
    data = encode_frame(data),
    nmis = as.list(colSums(is.na(data))),
    method = as.list(imp$method),
    predictor_matrix = named_matrix(imp$predictorMatrix),
    visit_sequence_default = I(imp$visitSequence),
    visit_sequence = visits,
    imputed_variables = I(imputed),
    effective_visit_sequence = effective,
    effective_visit_sequence_default =
      I(imp$visitSequence[imp$visitSequence %in% imputed]),
    logged_events = logged_events(imp),
    incomplete_after_imputation = I(still_missing)
  )
}

# Synthetic data: a constant column and an exact copy of a complete column.
set.seed(20261117)
n <- 50
a <- rnorm(n)
b <- a + rnorm(n)
b[sample(n, 10)] <- NA
e <- 0.5 * a + rnorm(n)
e[sample(n, 8)] <- NA
synthetic <- data.frame(a = a, b = b, const = 3, dup = a, e = e)

# Same, but the copy is of the incomplete column b. R then also stops
# imputing the copy (method ""), and leaves it incomplete.
synthetic_dup_target <- data.frame(a = a, b = b, const = 3, dup = b, e = e)

setup_cases <- list(
  setup_case("nhanes", nhanes),
  setup_case("nhanes2", nhanes2),
  setup_case("boys", boys),
  setup_case("synthetic", synthetic, expect_log = TRUE),
  setup_case("synthetic_dup_target", synthetic_dup_target, expect_log = TRUE)
)

write_fixture("l1_setup", list(
  description = paste(
    "mice(data, maxit = 0) with default arguments and with each visitSequence",
    "code. effective_visit_sequence keeps only variables with a non-empty",
    "method, in R's order: the order in which R actually imputes."
  ),
  cases = setup_cases
))

# ---------------------------------------------------------------------------
# Design matrix, linear dependencies, collinear variables
# ---------------------------------------------------------------------------

# remove.lindep() and find.collinear() write to the event log of a running
# mice() call. Outside mice() the log objects must exist in the global
# environment.
with_log <- function(expr) {
  assign("state", list(it = 0, im = 0, dep = "", meth = "", log = FALSE),
    envir = globalenv()
  )
  assign("loggedEvents",
    data.frame(it = 0, im = 0, dep = "", meth = "", out = ""),
    envir = globalenv()
  )
  on.exit(rm("state", "loggedEvents", envir = globalenv()))
  force(expr)
}

# obtain.design: what sampler.univ() builds for one target, intercept removed.
design_case <- function(name, data, target, predictors) {
  f <- reformulate(predictors, response = target)
  x <- mice:::obtain.design(data, f)
  x <- x[, -1L, drop = FALSE]
  list(
    name = name,
    data = encode_frame(data),
    target = target,
    predictors = I(predictors),
    columns = I(colnames(x)),
    x = unname(x)
  )
}

imp_n2 <- mice(nhanes2, m = 1, maxit = 2, seed = 7L, printFlag = FALSE)
stopifnot(is.null(imp_n2$loggedEvents))
nhanes2_completed <- complete(imp_n2, 1)
boys_cc <- boys[complete.cases(boys), ]
rownames(boys_cc) <- NULL

design_cases <- list(
  design_case("nhanes2_factors", nhanes2_completed, "chl", c("age", "bmi", "hyp")),
  design_case("boys_unordered_factor", boys_cc, "hgt", c("age", "reg")),
  design_case("boys_ordered_factor", boys_cc, "hgt", c("age", "gen"))
)

# remove.lindep(x, y, ry): x without intercept, as in sampler.univ().
set.seed(20261118)
n <- 40
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
y <- x1 - x2 + rnorm(n)
ry <- rep(TRUE, n)
ry[sample(n, 8)] <- FALSE
const_obs <- ifelse(ry, 2, rnorm(n)) # constant on the observed rows only
y_copy <- y + rnorm(n, sd = 1e-3) # cor with y above 0.99
y_neg <- -y + rnorm(n, sd = 1e-3) # cor with y below -0.99: R keeps it
near <- x1 + 0.5 * x2 + rnorm(n, sd = 1e-4) # near-collinear with x1, x2

lindep_case <- function(name, x, y, ry, eps = 1e-04, maxcor = 0.99,
                        several = FALSE) {
  keep <- with_log(mice:::remove.lindep(x, y, ry, eps = eps, maxcor = maxcor))
  list(
    name = name,
    several_dependencies = several,
    x = unname(x),
    columns = I(colnames(x)),
    y = I(y),
    ry = I(ry),
    eps = eps,
    maxcor = maxcor,
    keep = I(unname(keep))
  )
}

lindep_cases <- list(
  lindep_case("independent", cbind(x1, x2, x3), y, ry),
  lindep_case("constant_on_observed", cbind(x1, const_obs, x2), y, ry),
  lindep_case("high_cor_with_y", cbind(x1, y_copy, x2), y, ry),
  lindep_case("high_negative_cor_with_y", cbind(x1, y_neg, x2), y, ry),
  lindep_case("near_collinear", cbind(x1, x2, x3, near), y, ry),
  lindep_case("constant_y", cbind(x1, x2), ifelse(ry, 5, y), ry),
  lindep_case("strict_eps", cbind(x1, x2, n3 = x1 + 0.5 * x2 + rnorm(n, sd = 1e-2)),
    y, ry,
    eps = 1e-2
  ),
  # cor(x1, y) is about 0.19 on the observed rows: dropped with maxcor 0.15
  lindep_case("low_maxcor", cbind(x1, x2, x3), y, ry, maxcor = 0.15)
)

# Several linear dependencies at once. After the first drop, remove.lindep()
# subsets the correlation matrix with keep[keep], an all-TRUE vector that is
# recycled, so later steps reuse the eigen decomposition of all columns. The
# result depends on column order and can keep a column that is still
# collinear (n2 = x3 - x4 + noise survives in all three "two_dependencies"
# cases; "three_near_copies" drops the unrelated x2 and keeps d1 and d2,
# which are near copies of each other). These cases
# pin R's actual output; a port that removes every dependency differs here.
set.seed(20261123)
x4 <- rnorm(n)
n1 <- x1 + x2 + rnorm(n, sd = 1e-4)
n2 <- x3 - x4 + rnorm(n, sd = 1e-4)
d1 <- x1 + rnorm(n, sd = 1e-5)
d2 <- x1 + rnorm(n, sd = 1e-5)
lindep_cases <- c(lindep_cases, list(
  lindep_case("two_dependencies_a", cbind(x1, x2, x3, x4, n1, n2), y, ry,
    several = TRUE
  ),
  lindep_case("two_dependencies_b", cbind(x1, x2, n1, x3, x4, n2), y, ry,
    several = TRUE
  ),
  lindep_case("two_dependencies_c", cbind(n1, n2, x1, x2, x3, x4), y, ry,
    several = TRUE
  ),
  lindep_case("three_near_copies", cbind(x1, d1, d2, x2), y, ry,
    several = TRUE
  )
))

collinear_case <- function(name, data, threshold = 0.999) {
  list(
    name = name,
    data = encode_frame(data),
    threshold = threshold,
    collinear = I(mice:::find.collinear(data, threshold = threshold))
  )
}

collinear_cases <- list(
  collinear_case("nhanes", nhanes),
  collinear_case("synthetic", synthetic),
  collinear_case("synthetic_dup_target", synthetic_dup_target),
  collinear_case("synthetic_threshold_0.5", synthetic, threshold = 0.5)
)

write_fixture("l1_design", list(
  description = paste(
    "obtain_design: mice:::obtain.design(data, target ~ predictors) without",
    "the intercept column. remove_lindep: keep-mask of",
    "mice:::remove.lindep(x, y, ry, eps, maxcor); cases with",
    "several_dependencies = true pin R's order-dependent result when more",
    "than one column has to go. find_collinear:",
    "mice:::find.collinear(data, threshold)."
  ),
  obtain_design = design_cases,
  remove_lindep = lindep_cases,
  find_collinear = collinear_cases
))
