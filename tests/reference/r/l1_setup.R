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

# ---------------------------------------------------------------------------
# where: methods and visit sequence; the monotone edit of predictorMatrix
# ---------------------------------------------------------------------------

# Evaluate `expr`; any warning stops the script
quietly <- function(expr) {
  withCallingHandlers(expr, warning = function(w) stop("unexpected warning: ", conditionMessage(w)))
}

# A where keyword as given, a matrix with its column names (values a list
# of rows)
encode_where <- function(where) {
  if (is.character(where)) where else list(cols = I(colnames(where)), values = unname(where))
}

where_matrix <- function(data, where) {
  if (is.character(where)) make.where(data, where) else where
}

# make.method(data, where = ) and mice(data, maxit = 0, where = ), with
# `method` when given, for the default visit sequence and each keyword
where_case <- function(name, data, where, method = NULL) {
  args <- list(where = where)
  if (!is.null(method)) args$method <- method
  run <- function(...) {
    quietly(do.call(mice, c(list(data, maxit = 0, printFlag = FALSE, seed = 1L), args, list(...))))
  }
  imp <- run()
  stopifnot(is.null(imp$loggedEvents))
  visits <- lapply(VISITS, function(v) I(run(visitSequence = v)$visitSequence))
  names(visits) <- VISITS
  imputed <- names(imp$method)[imp$method != ""]
  list(
    name = name,
    data = encode_frame(data),
    where = encode_where(where),
    method_arg = if (is.null(method)) NULL else as.list(method),
    nmis = as.list(colSums(is.na(data))),
    nimp = as.list(colSums(where_matrix(data, where))),
    make_method = as.list(make.method(data, where = where_matrix(data, where))),
    method = as.list(imp$method),
    predictor_matrix = named_matrix(imp$predictorMatrix),
    visit_sequence_default = I(imp$visitSequence),
    visit_sequence = visits,
    imputed_variables = I(imputed),
    effective_visit_sequence = lapply(visits, function(v) I(v[v %in% imputed])),
    effective_visit_sequence_default = I(imp$visitSequence[imp$visitSequence %in% imputed])
  )
}

# The missing cells of `data`, with the cells of `add` (variable = rows) put
# in and those of `drop` left out
where_cells <- function(data, add = list(), drop = list()) {
  where <- is.na(data)
  for (j in names(add)) where[add[[j]], j] <- TRUE
  for (j in names(drop)) where[drop[[j]], j] <- FALSE
  where
}

# Where-counts against missing counts (age 0, bmi 9, hyp 8, chl 10): age 6
# (complete, with where-cells), bmi 0 (incomplete, without where-cells),
# hyp 5, chl 12 (two observed where-cells). "monotone" by where-counts is
# bmi hyp age chl, by missing counts age hyp bmi chl.
nhanes2_where <- where_cells(nhanes2,
  add = list(age = c(2, 5, 7, 8, 9, 13), chl = c(2, 5)),
  drop = list(bmi = seq_len(nrow(nhanes2)), hyp = c(1, 4, 6))
)
stopifnot(identical(unname(colSums(nhanes2_where)), c(6, 0, 5, 12)))

where_cases <- list(
  where_case("nhanes2_missing", nhanes2, "missing"),
  where_case("nhanes2_all", nhanes2, "all"),
  where_case("nhanes2_observed", nhanes2, "observed"),
  where_case("nhanes2_none", nhanes2, "none"),
  where_case("nhanes2_matrix", nhanes2, nhanes2_where),
  # bmi has no where-cells: R gives it "" although the user names a method
  where_case("nhanes2_matrix_user_method", nhanes2, nhanes2_where,
    method = c(age = "polyreg", bmi = "norm", hyp = "logreg", chl = "pmm")
  )
)

# mice(data, m = 1, maxit, visitSequence, method = "norm.predict"): with
# maxit = 1 and "monotone", mice.edit.predictorMatrix() keeps for each
# variable only the variables before it in R's monotone order (all
# variables by number of where-cells). One iteration then never uses a start
# value, so its completed data are stored too (the script checks that they
# do not depend on the seed). maxit = 2 and "revmonotone" leave the matrix
# as it is.
monotone_case <- function(name, data, maxit, visit, where = NULL) {
  run <- function(seed) {
    args <- list(m = 1, maxit = maxit, visitSequence = visit, method = "norm.predict", seed = seed, printFlag = FALSE)
    if (!is.null(where)) args$where <- where
    quietly(do.call(mice, c(list(data), args)))
  }
  imp <- run(1L)
  edited <- maxit == 1L && visit == "monotone"
  stopifnot(edited != identical(imp$predictorMatrix, make.predictorMatrix(data)))
  if (edited) {
    stopifnot(identical(complete(imp), complete(run(2L))))
  }
  imputed <- names(imp$method)[imp$method != ""]
  list(
    name = name,
    data = encode_frame(data),
    where = if (is.null(where)) NULL else encode_where(where),
    maxit = maxit,
    visit_sequence_arg = visit,
    method = as.list(imp$method),
    predictor_matrix = named_matrix(imp$predictorMatrix),
    visit_sequence = I(imp$visitSequence),
    effective_visit_sequence = I(imp$visitSequence[imp$visitSequence %in% imputed]),
    completed = if (edited) encode_frame(complete(imp)) else NULL
  )
}

# hyp with six observed where-cells (14): "monotone" is age bmi chl hyp, by
# missing counts age hyp bmi chl
nhanes_where <- where_cells(nhanes, add = list(hyp = c(2, 5, 7, 8, 9, 13)))

monotone_cases <- list(
  monotone_case("nhanes_monotone", nhanes, 1L, "monotone"),
  monotone_case("nhanes_monotone_where", nhanes, 1L, "monotone", where = nhanes_where),
  monotone_case("nhanes_monotone_maxit_2", nhanes, 2L, "monotone"),
  monotone_case("nhanes_revmonotone", nhanes, 1L, "revmonotone")
)

# ---------------------------------------------------------------------------
# passive variables: methods and visit sequence
# ---------------------------------------------------------------------------

# mice(data, maxit = 0, method, where) with the passive variables of
# `passive` (recipes, see _common.R; their formulas go into method), for the
# default visit sequence, each keyword and the vector `visit`, which names a
# variable twice. Without visitSequence, mice.edit.setup() moves the passive
# variables to the end; a keyword or a vector places them as it says. The
# only warning allowed is the count of logged events
passive_case <- function(name, data, method, passive, visit, where = NULL) {
  args <- with_recipes(list(method = method, passive = passive), data)
  if (!is.null(where)) args$where <- where
  run <- function(...) {
    withCallingHandlers(
      do.call(mice, c(list(data, maxit = 0, printFlag = FALSE, seed = 1L), args, list(...))),
      warning = function(w) {
        if (!startsWith(conditionMessage(w), "Number of logged events")) {
          stop("unexpected warning: ", conditionMessage(w))
        }
        invokeRestart("muffleWarning")
      }
    )
  }
  imp <- run()
  visits <- lapply(VISITS, function(v) I(run(visitSequence = v)$visitSequence))
  names(visits) <- VISITS
  visit_vector <- run(visitSequence = visit)$visitSequence
  imputed <- names(imp$method)[imp$method != ""]
  effective <- function(v) I(v[v %in% imputed])
  list(
    name = name,
    data = encode_frame(data),
    where = if (is.null(where)) NULL else encode_where(where),
    method_arg = as.list(method),
    passive = passive,
    visit_sequence_arg = I(visit),
    method = as.list(imp$method),
    predictor_matrix = named_matrix(imp$predictorMatrix),
    logged_events = logged_events(imp),
    visit_sequence_default = I(imp$visitSequence),
    visit_sequence = visits,
    visit_sequence_vector = I(visit_vector),
    imputed_variables = I(imputed),
    effective_visit_sequence_default = effective(imp$visitSequence),
    effective_visit_sequence = lapply(visits, effective),
    effective_visit_sequence_vector = effective(visit_vector)
  )
}

# nhanes with s, the passive sum of bmi and chl (rounded where both are
# observed), placed before them: data order and "roman" visit s first, the
# default last
nhanes_sum <- data.frame(age = nhanes$age, s = round(nhanes$bmi + nhanes$chl), nhanes[c("bmi", "hyp", "chl")])
pmm_methods <- c(age = "", bmi = "pmm", hyp = "pmm", chl = "pmm")

# where: s has two observed where-cells and one missing one (the others
# left out), hyp none, so "monotone" puts s first and hyp gets ""
nhanes_sum_where <- where_cells(nhanes_sum,
  add = list(s = c(2, 5)), drop = list(s = setdiff(which(is.na(nhanes_sum$s)), 1L), hyp = seq_len(nrow(nhanes)))
)
stopifnot(identical(unname(colSums(nhanes_sum_where)), c(0, 3, 9, 0, 10)))

passive_cases <- list(
  passive_case("nhanes_sum", nhanes_sum, pmm_methods, list(s = passive_sum("bmi", "chl")),
    visit = c("bmi", "s", "hyp", "chl", "s")
  ),
  passive_case("nhanes_sum_where", nhanes_sum, pmm_methods, list(s = passive_sum("bmi", "chl")),
    visit = c("s", "bmi", "chl", "s"), where = nhanes_sum_where
  ),
  # c2, a passive copy of chl: collinear, method "" (logged)
  passive_case("nhanes_copy", cbind(nhanes, c2 = 2 * nhanes$chl), pmm_methods, list(c2 = passive_multiple("chl", 2)),
    visit = c("chl", "bmi", "hyp", "chl")
  ),
  # a2, a complete passive variable: method ""
  passive_case("nhanes_complete", cbind(nhanes, a2 = nhanes$age * nhanes$age), pmm_methods,
    list(a2 = passive_product("age", "age")),
    visit = c("chl", "bmi", "hyp", "chl")
  ),
  # s missing in every row: the setup step does not check a passive
  # variable for being constant, so it keeps its formula
  passive_case("nhanes_all_missing", cbind(nhanes, s = NA_real_), pmm_methods, list(s = passive_sum("bmi", "chl")),
    visit = c("s", "bmi", "hyp", "chl", "s")
  )
)

write_fixture("l1_setup", list(
  description = paste(
    "mice(data, maxit = 0) with default arguments and with each visitSequence",
    "code. effective_visit_sequence keeps only variables with a non-empty",
    "method, in R's order: the order in which R actually imputes.",
    "where_cases: make.method(data, where = ) and mice(data, maxit = 0, where",
    "= ), where a keyword or cols and values (a list of rows), method_arg the",
    "method given to mice(). monotone_cases: mice(data, m = 1, maxit,",
    "visitSequence = visit_sequence_arg, method = 'norm.predict', where): the",
    "predictorMatrix that mice.edit.predictorMatrix() leaves, and for maxit =",
    "1 with 'monotone' the completed data, which use no start value.",
    "passive_cases: mice(data, maxit = 0, method, where) with method_arg and",
    "the passive variables of `passive` (recipes, see _common.R, whose",
    "formulas R's method holds), for the default visit sequence, each",
    "keyword and the vector visit_sequence_arg (visit_sequence_vector)."
  ),
  cases = setup_cases,
  where_cases = where_cases,
  monotone_cases = monotone_cases,
  passive_cases = passive_cases
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
