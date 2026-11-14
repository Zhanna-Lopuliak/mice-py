# L1 fixture for whole chains of mice(), compared exactly.
#
#   l1_chain.json   mice(data, m, maxit = 0) for the start of each chain and
#                   the generator state after it; mice(data, m, maxit) (and
#                   mice.mids() for a continued run) for the run: imp, the
#                   completed data, chainMean, chainVar, method,
#                   visitSequence, predictorMatrix, loggedEvents, iteration;
#                   the draws or the imputer calls of the run. r_errors:
#                   calls of mice() that stop in R, with the error and the
#                   warnings before it
#
# Python target: MICE.impute(maxit=0), then MICE.continue_chains, with R's
# start values written into the chain states and R's draws served by their
# generators (tests/test_parity/_l1_chain.py).
#
# R and Python draw different random numbers, so a whole chain can be
# compared exactly only from the same start and with the same draws.
# set.seed(s); start <- mice(data, m, maxit = 0, ...) gives the start
# (start$imp) and the generator state after it (start$lastSeedValue);
# mice(data, m, maxit = k, ..., seed = s) gives the run, which starts from
# the same values. Three kinds of case:
#
# - deterministic: methods that draw nothing (norm.predict, mean), m = 2 with
#   different start values per chain. The run must leave the generator where
#   the start left it.
# - scripted: methods that draw (norm, logreg, polr), m = 1, so that R's
#   order of the draws (iterations outside, imputations inside) is mice-py's
#   (one chain after the other). replay_chain() goes through the chain as
#   sampler() and sampler.univ() do, from start$lastSeedValue: before each
#   imputer call it saves .Random.seed, calls the imputer, saves the state
#   after, restores the state before and draws the expected numbers in the
#   expected order (norm: rchisq(1, df), rnorm(q), rnorm(n0); logreg:
#   rnorm(q), runif(n0); polr: runif(n0)); the same state afterwards proves
#   kind, number and order of the draws. After each step of the run (see
#   below) the replay must have its data and its generator state.
# - spy: the method "spy" (mice.impute.spy below) records at every call what
#   sampler.univ() hands an imputer and imputes as norm.predict; m = 1. As in
#   a deterministic case, nothing is drawn after the start.
#
# A case with several values of maxit runs mice() with the first and
# mice.mids() with each further one, and requires that the result equals
# mice() with their sum. The cases take the arguments of mice() through
# `...` (encode_args() lists the ones it knows).
#
# The cases after the first ten pass `where`. complete() writes NA into a
# where-cell without an imputation, also over an observed value; such a case
# names the variables where that happens (`lost`), and the fixture holds,
# after the start and after each step, the number of observed where-cells
# of each that are NA in the completed data of some chain (mice-py warns
# with these numbers). The other cases lose no observed value.
#
# The cases after the first 24 pass `ignore`. A case that expects a warning
# of mice() other than "Number of logged events" names it (`warnings`); the
# fixture holds it. r_errors holds calls of mice() that stop in R, for tests
# of what mice-py does there (stop_case()).

source(file.path("tests", "reference", "r", "_common.R"))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

n_rows <- nrow(nhanes)

# nhanes with w, a complete column that is a near copy of chl where chl is
# observed: remove.lindep() drops it from the model of chl in every
# iteration (correlation above 0.99), find.collinear() keeps it (below 0.999)
set.seed(7400)
chl_observed <- !is.na(nhanes$chl)
w <- numeric(n_rows)
w[chl_observed] <- nhanes$chl[chl_observed] + rnorm(sum(chl_observed), sd = 4)
w[!chl_observed] <- round(rnorm(sum(!chl_observed), 190, 30))
nhanes_w <- cbind(nhanes, w = w)
stopifnot(
  cor(w[chl_observed], nhanes$chl[chl_observed]) > 0.99,
  length(mice:::find.collinear(nhanes_w)) == 0L
)

# nhanes with v, observed in three rows only and predicted by age and chl:
# 3 fit rows for 3 coefficients, so check.df() logs "df set to 1"
v <- replace(rep(NA_real_, n_rows), c(2L, 9L, 17L), c(7.5, 14.2, 6.2))
nhanes_v <- cbind(nhanes, v = v)
nhanes_wv <- cbind(nhanes_w, v = v)

# The default predictor matrix, except that v is predicted by age and chl
# only and predicts nothing
predictors_of_v <- function(data) {
  pred <- make.predictorMatrix(data)
  pred[, "v"] <- 0
  pred["v", ] <- 0
  pred["v", c("age", "chl")] <- 1
  pred
}

# The numeric columns of boys, 60 rows of boys older than 8 (tv is measured
# from about that age): four rows with missing body measures, the others
# drawn from the rows where they are complete
body <- c("hgt", "wgt", "bmi", "hc")
older <- rownames(boys)[boys$age > 8]
set.seed(7401)
picked <- c(
  c("4254", "4268", "4501", "7192"),
  sample(older[complete.cases(boys[older, body])], 56)
)
boys60 <- boys[sort(match(picked, rownames(boys))), c("age", body, "tv")]
rownames(boys60) <- NULL
stopifnot(all(colSums(is.na(boys60[body])) > 0), sum(is.na(boys60$tv)) > 10)

# A two-category target g and a predictor z that is not imputed (method "")
# and keeps its missing values, also in four rows where g is missing: those
# cells of g cannot be imputed, and the rows where only z is missing are in
# neither mask, but augment() takes its statistics over them. g is separated
# by x1 (yes where x1 > 0), so that the slope of the fit comes from the rows
# augment() adds; z is missing where x1 is farthest from 0, so that those
# rows move the standard deviation of x1 and with it the added rows; g is
# missing where x1 is nearest to 0, where the labels depend on the slope.
# 7403 is the first seed from 7402 on under which the rows MICE 0.3 passed
# (those with complete predictors only) change a label in the first
# iteration (test_l1_chain_mutants).
set.seed(7403)
n_syn <- 40
x1 <- rnorm(n_syn)
x2 <- runif(n_syn)
z <- 0.5 * x1 + rnorm(n_syn)
g <- ifelse(x1 > 0, "yes", "no")
z_missing <- sort(order(abs(x1), decreasing = TRUE)[1:8])
g_missing <- sort(c(z_missing[c(1, 3, 5, 7)], order(abs(x1))[1:8]))
z[z_missing] <- NA
g[g_missing] <- NA
syn <- data.frame(x1 = x1, x2 = x2, z = z, g = factor(g, levels = c("no", "yes")))
stopifnot(all(c("no", "yes") %in% syn$g))

# nhanes with k, a constant column: the setup step removes its method
nhanes_k <- cbind(nhanes, k = 1)

# The rows of nhanes that the cases with ignore leave out of the fits: 1
# (all missing), 6 (bmi and hyp missing, chl observed), 14, 17 and 22 (all
# observed; chl 284 in row 17 is its largest value) and 24 (chl missing)
ignored_nhanes <- seq_len(n_rows) %in% c(1L, 6L, 14L, 17L, 22L, 24L)

# nhanes with u, observed in rows 2, 5, 9, 14 and 17 and predicted by age
# and chl only: without the ignored rows 14 and 17, 3 fit rows for 3
# coefficients, so check.df() logs "df set to 1" (with them it would not).
# And p, complete, 1 in every row but the ignored rows 14 and 17: constant
# in the fit rows, so remove.lindep() drops it from the models of bmi, hyp
# and chl in every iteration, while the setup step, which uses all rows,
# keeps it
nhanes_up <- cbind(
  nhanes,
  u = replace(rep(NA_real_, n_rows), c(2L, 5L, 9L, 14L, 17L), c(3.1, 2.4, 4.0, 3.3, 5.2)),
  p = replace(rep(1, n_rows), c(14L, 17L), c(2, 3))
)
stopifnot(length(mice:::find.collinear(nhanes_up)) == 0L)

# The default predictor matrix, except that u is predicted by age and chl
# only and predicts nothing
predictors_of_u <- function(data) {
  pred <- make.predictorMatrix(data)
  pred[, "u"] <- 0
  pred["u", ] <- 0
  pred["u", c("age", "chl")] <- 1
  pred
}

# Two-category targets g with ignored rows, imputed by logreg from x1, x2.
# syn_one: one missing cell (where x1 is nearest 0) and four ignored rows
# with observed labels (the largest |x1|): augment() is not skipped, because
# sum(!ry) counts the ignored rows. 7441 is a seed under which the label of
# the first iteration changes when the ignored rows are dropped before the
# imputer.
set.seed(7441)
n_syn2 <- 20
x1 <- rnorm(n_syn2)
x2 <- runif(n_syn2)
g <- ifelse(runif(n_syn2) < plogis(0.3 + 1.5 * x1 - x2), "yes", "no")
g[which.min(abs(x1))] <- NA
syn_one <- data.frame(x1 = round(x1, 4), x2 = round(x2, 4), g = factor(g, levels = c("no", "yes")))
ignored_one <- seq_len(n_syn2) %in% order(abs(x1), decreasing = TRUE)[1:4]
stopifnot(sum(is.na(syn_one$g)) == 1L, all(table(syn_one$g[!ignored_one]) >= 4))

# syn_only: "yes" only in the three ignored rows (the largest x1), six
# missing cells. g is constant in the fit rows, so remove.lindep() drops
# every predictor (without logging it), augment() is skipped for want of a
# predictor, and the intercept is fitted on one category: its estimate and
# standard error grow until the fit stops, and every cell gets the label
# that the sign of the normal draw gives (probabilities 0 or 1)
set.seed(7450)
x1 <- rnorm(n_syn2)
x2 <- runif(n_syn2)
g <- rep("no", n_syn2)
g[order(x1, decreasing = TRUE)[1:3]] <- "yes"
g[order(x1, decreasing = TRUE)[4:9]] <- NA
syn_only <- data.frame(x1 = round(x1, 4), x2 = round(x2, 4), g = factor(g, levels = c("no", "yes")))
ignored_only <- !is.na(syn_only$g) & syn_only$g == "yes"

# syn_clamp: x2 is 1 in 19 of the 23 rows that are not ignored, so that
# mean + sd / 2 lies above its maximum there and augment() would clamp the
# added rows to it; the ignored row holds x2 = 2.5, which lifts the maximum
# (and moves the mean and the standard deviation)
set.seed(7460)
n_clamp <- 24
x1 <- rnorm(n_clamp)
x2 <- rep(1, n_clamp)
low <- sample(n_clamp, 4)
x2[low] <- round(runif(4, -2, 0), 4)
g <- ifelse(runif(n_clamp) < plogis(0.2 + 1.2 * x1 + 0.8 * x2), "yes", "no")
ignored_clamp <- seq_len(n_clamp) == setdiff(seq_len(n_clamp), low)[1]
x2[ignored_clamp] <- 2.5
g[setdiff(order(abs(x1)), which(ignored_clamp))[1:5]] <- NA
syn_clamp <- data.frame(x1 = round(x1, 4), x2 = x2, g = factor(g, levels = c("no", "yes")))
kept_x2 <- syn_clamp$x2[!ignored_clamp]
stopifnot(mean(kept_x2) + sd(kept_x2) / 2 > max(kept_x2), mean(x2) + sd(x2) / 2 < 2.5)

# A factor g (a, b, c) and an ordered factor o (p < q < r < s) predicted by
# x; c and s occur only in row 3, which is ignored. Where augment() is
# skipped (no predictor, or row 3 the only row outside the fit rows),
# polyreg's multinom() drops the empty level c and R stops (stop_case()),
# while polr counts the empty level s and gives it a small probability
set.seed(7470)
n_lab <- 20
x <- round(rnorm(n_lab), 4)
g <- factor(ifelse(x + rnorm(n_lab) > 0, "b", "a"), levels = c("a", "b", "c"))
g[3] <- "c"
o <- cut(x + rnorm(n_lab, sd = 0.7), c(-Inf, -0.5, 0.5, Inf), labels = c("p", "q", "r"))
o <- factor(as.character(o), levels = c("p", "q", "r", "s"), ordered = TRUE)
o[3] <- "s"
ignored_lab <- seq_len(n_lab) == 3L
labels_g <- data.frame(x = x, g = g)
labels_g_missing <- data.frame(x = x, g = replace(g, c(5L, 9L, 14L), NA))
labels_o <- data.frame(x = x, o = o)
no_predictor <- matrix(0, 2, 2, dimnames = list(c("x", "g"), c("x", "g")))

# The missing cells of `data`, with the cells of `add` (variable = rows) put
# in and those of `drop` left out
where_cells <- function(data, add = list(), drop = list()) {
  where <- is.na(data)
  for (j in names(add)) where[add[[j]], j] <- TRUE
  for (j in names(drop)) where[drop[[j]], j] <- FALSE
  where
}

# ---------------------------------------------------------------------------
# The spy and the capture of logreg's probabilities
# ---------------------------------------------------------------------------

# sampler.univ() finds an imputer by its name, also in the global
# environment; the method "spy" records its arguments and the iteration,
# imputation and column of the call (from mice()'s `state`)
spy_log <- new.env()
spy_log$calls <- list()
spy <- function(y, ry, x, wy = NULL, ...) {
  s <- dynGet("state")
  spy_log$calls[[length(spy_log$calls) + 1L]] <- list(
    it = s$it,
    im = s$im,
    dep = s$dep,
    y = I(y),
    ry = I(ry),
    wy = I(wy),
    x_names = I(colnames(x)),
    x = unname(x)
  )
  mice.impute.norm.predict(y, ry, x, wy, ...)
}

# The probabilities of mice.impute.logreg() and the number of normals it
# draws, and the probabilities of mice.impute.polr(), from trace() exit
# hooks, so that the replay can require that no label depends on the last
# digits of a probability
logreg_capture <- new.env()
polr_capture <- new.env()

# ---------------------------------------------------------------------------
# Running mice()
# ---------------------------------------------------------------------------

# The warnings other than "Number of logged events" that the running case
# expects (chain_case(warnings = )), and those seen since `seen` was emptied
warning_log <- new.env()
warning_log$expected <- character(0)
warning_log$seen <- character(0)

# Evaluate `expr`; the warning "Number of logged events" is expected and
# muffled, and so are the warnings the case expects, which are recorded;
# any other warning stops the script
quietly <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    message <- conditionMessage(w)
    if (message %in% warning_log$expected) {
      warning_log$seen <- c(warning_log$seen, message)
    } else if (!startsWith(message, "Number of logged events")) {
      stop("unexpected warning: ", message)
    }
    invokeRestart("muffleWarning")
  })
}

run_mice <- function(data, m, maxit, seed, args) {
  quietly(do.call(mice, c(list(data, m = m, maxit = maxit, seed = seed, printFlag = FALSE), args)))
}

# The run: mice() with maxit[1], then mice.mids() with each further value
run_steps <- function(data, m, maxit, seed, args) {
  runs <- list(run_mice(data, m, maxit[1], seed, args))
  for (k in maxit[-1]) {
    runs[[length(runs) + 1L]] <- quietly(mice.mids(runs[[length(runs)]], maxit = k, printFlag = FALSE))
  }
  runs
}

same_run <- function(a, b) {
  identical(a$imp, b$imp) &&
    identical(unname(a$chainMean), unname(b$chainMean)) &&
    identical(unname(a$chainVar), unname(b$chainVar)) &&
    identical(logged_events(a), logged_events(b)) &&
    identical(a$lastSeedValue, b$lastSeedValue) &&
    a$iteration == b$iteration
}

# The variables R imputes, in its order: visitSequence restricted to the
# variables with a method (it may list complete variables and variables
# with method "")
effective_visits <- function(obj) {
  vis <- obj$visitSequence
  vis[vis %in% names(obj$method)[obj$method != ""]]
}

# ---------------------------------------------------------------------------
# Replay of a scripted chain
# ---------------------------------------------------------------------------

# The working data of chain i, as sampler() fills them from imp: the missing
# where-cells of each visited variable
working_data <- function(obj, i = 1L) {
  data <- obj$data
  r <- !is.na(data)
  for (j in obj$visitSequence) {
    wy <- obj$where[, j]
    data[!r[, j] & wy, j] <- obj$imp[[j]][!r[wy, j], i]
  }
  data
}

# The completed data of the chain: `data` with every where-cell replaced by
# its value in `imp`, as complete() does
completed_data <- function(data, imp, where) {
  for (j in colnames(where)[colSums(where) > 0]) {
    data[where[, j], j] <- if (is.null(imp[[j]])) NA else imp[[j]]
  }
  data
}

# Go through the chain of `start` (m = 1) for sum(maxit) iterations as
# sampler() and sampler.univ() do, from the generator state after the start,
# and draw what each imputer is expected to draw (see the top of the file).
# Returns the draws, their order as [pool, count] pairs, and at the end of
# each step the completed data, the generator state and the number of values
# drawn from each pool so far.
replay_chain <- function(start, maxit) {
  stopifnot(start$m == 1L)
  data <- working_data(start)
  r <- !is.na(start$data)
  where <- start$where
  ignore <- start$ignore
  # imp of the chain: the values of the where-cells of each variable
  imp <- lapply(start$imp, function(imp_j) if (is.null(imp_j)) NULL else imp_j[, 1])
  # remove.lindep() writes to the log of the mice() call that runs it; here
  # it finds these
  state <- list(it = 0, im = 0, dep = "", meth = "", log = FALSE)
  loggedEvents <- data.frame(it = 0, im = 0, dep = "", meth = "", out = "")
  pools <- list(chisq = numeric(0), normal = numeric(0), uniform = numeric(0))
  chisq_df <- numeric(0)
  order <- list()
  draw <- function(pool, values) {
    pools[[pool]] <<- c(pools[[pool]], values)
    last <- length(order)
    if (last > 0L && order[[last]][[1]] == pool) {
      order[[last]][[2]] <<- order[[last]][[2]] + length(values)
    } else {
      order[[last + 1L]] <<- list(pool, length(values))
    }
  }
  steps <- list()
  assign(".Random.seed", start$lastSeedValue, envir = globalenv())
  for (k in seq_len(sum(maxit))) {
    for (h in start$visitSequence) {
      meth <- start$method[[h]]
      if (meth == "") next
      xnames <- setdiff(colnames(data)[start$predictorMatrix[h, ] != 0], h)
      formula <- if (length(xnames) > 0L) reformulate(xnames, response = h) else as.formula(paste(h, "~ 1"))
      x <- mice:::obtain.design(data, formula)[, -1L, drop = FALSE]
      y <- data[, h]
      ry <- complete.cases(x, y) & r[, h] & !ignore
      wy <- complete.cases(x) & where[, h]
      fill <- !r[, h] & where[, h]
      if (!any(wy)) {
        data[fill, h] <- NA
        imp[[h]][] <- NA
        next
      }
      keep <- mice:::remove.lindep(x, y, ry)
      x <- x[, keep, drop = FALSE]
      imputer <- get(paste0("mice.impute.", meth), envir = asNamespace("mice"))
      before <- .Random.seed
      imputes <- imputer(y, ry, x, wy = wy)
      after <- .Random.seed
      assign(".Random.seed", before, envir = globalenv())
      if (meth == "norm") {
        q <- ncol(x) + 1L
        df <- max(sum(ry) - q, 1)
        draw("chisq", rchisq(1, df))
        chisq_df <- c(chisq_df, df)
        draw("normal", rnorm(q))
        draw("normal", rnorm(sum(wy)))
      } else if (meth == "logreg") {
        draw("normal", rnorm(logreg_capture$last$q))
        u <- runif(sum(wy))
        stopifnot(min(abs(u - logreg_capture$last$p)) > 1e-4)
        draw("uniform", u)
      } else if (meth == "polr") {
        u <- runif(sum(wy))
        # the cumulative probabilities of each cell but the last (1), one row per cell
        cumulative <- t(apply(polr_capture$last$post, 1, cumsum))
        stopifnot(min(abs(u - cumulative[, -ncol(cumulative), drop = FALSE])) > 1e-4)
        draw("uniform", u)
      } else {
        stopifnot(meth %in% c("norm.predict", "mean"))
      }
      stopifnot(identical(.Random.seed, after))
      # imp of the where-cells: NA where the predictors are incomplete
      new <- data[where[, h], h]
      new[!wy[where[, h]]] <- NA
      new[wy[where[, h]]] <- imputes
      imp[[h]] <- new
      data[fill, h] <- new[fill[where[, h]]]
    }
    if (k %in% cumsum(maxit)) {
      steps[[length(steps) + 1L]] <- list(
        completed = completed_data(data, imp, where),
        seed = .Random.seed,
        used = lapply(pools, length)
      )
    }
  }
  list(pools = pools, chisq_df = chisq_df, order = order, steps = steps)
}

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

named_matrix <- function(m) {
  list(rows = I(rownames(m)), cols = I(colnames(m)), values = unname(m))
}

logged_events <- function(obj) {
  le <- obj$loggedEvents
  if (is.null(le)) {
    return(NULL)
  }
  list(it = I(le$it), im = I(le$im), dep = I(le$dep), meth = I(le$meth), out = I(le$out))
}

# The arguments of mice() a case passes, for the Python side. `where` is
# written only when given: a keyword, or the matrix with its column names
# (values a list of rows). `ignore` too, as given (one value per row, also
# where it is not logical)
encode_args <- function(args) {
  stopifnot(
    "encode_args() knows the arguments of mice() a case passes" =
      all(names(args) %in% c("method", "predictorMatrix", "visitSequence", "where", "ignore"))
  )
  method <- args$method
  vis <- args$visitSequence
  keyword <- length(vis) == 1L && vis %in% c("roman", "arabic", "monotone", "revmonotone")
  out <- list(
    method = if (is.null(names(method))) method else as.list(method),
    predictor_matrix = if (is.null(args$predictorMatrix)) NULL else named_matrix(args$predictorMatrix),
    visit_sequence = if (is.null(vis) || keyword) vis else I(vis)
  )
  where <- args$where
  if (!is.null(where)) {
    out$where <- if (is.character(where)) where else list(cols = I(colnames(where)), values = unname(where))
  }
  if (!is.null(args$ignore)) {
    out$ignore <- I(args$ignore)
  }
  out
}

values_of <- function(v) if (is.factor(v)) as.character(v) else as.vector(v)

# imp of every variable with where-cells: their rows (0-based) and, per
# chain, the value of each
encode_imp <- function(obj) {
  has_cells <- vapply(obj$imp, function(imp_j) !is.null(imp_j) && nrow(imp_j) > 0L, logical(1))
  cols <- names(obj$imp)[has_cells]
  out <- lapply(cols, function(j) {
    list(
      rows = I(which(obj$where[, j]) - 1L),
      values = lapply(seq_len(obj$m), function(i) I(values_of(obj$imp[[j]][[i]])))
    )
  })
  setNames(out, cols)
}

# chainMean or chainVar of the numeric variables R imputes, iteration x chain
encode_chain <- function(obj, stat) {
  vis <- effective_visits(obj)
  cols <- vis[!vapply(obj$data[vis], is.factor, logical(1))]
  out <- lapply(cols, function(j) matrix(obj[[stat]][j, , ], nrow = obj$iteration, ncol = obj$m))
  setNames(out, cols)
}

# Per variable, the number of observed where-cells that are NA in the
# completed data of some chain; only the variables with such cells
lost_observed <- function(obj) {
  r <- !is.na(obj$data)
  lost <- vapply(names(obj$data), function(j) {
    na <- Reduce(`|`, lapply(seq_len(obj$m), function(i) is.na(complete(obj, i)[, j])))
    sum(obj$where[, j] & r[, j] & na)
  }, numeric(1))
  as.list(lost[lost > 0])
}

encode_step <- function(obj, maxit, used) {
  list(
    maxit = maxit,
    iteration = obj$iteration,
    imp = encode_imp(obj),
    completed = lapply(seq_len(obj$m), function(i) encode_frame(complete(obj, i))),
    chain_mean = encode_chain(obj, "chainMean"),
    chain_var = encode_chain(obj, "chainVar"),
    logged_events = logged_events(obj),
    draws_used = used
  )
}

# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

chain_case <- function(name, kind, data, m, maxit, seed, ..., lost = NULL, warnings = NULL) {
  args <- list(...)
  stopifnot(kind %in% c("deterministic", "scripted", "spy"), kind == "deterministic" || m == 1L)
  # each call of mice() gives the expected warnings once; mice.mids() does
  # not check its arguments again and gives none
  warning_log$expected <- as.character(warnings)
  warning_log$seen <- character(0)
  start <- run_mice(data, m, 0, seed, args)
  stopifnot(identical(warning_log$seen, as.character(warnings)))
  warning_log$seen <- character(0)
  spy_log$calls <- list()
  runs <- run_steps(data, m, maxit, seed, args)
  stopifnot(identical(warning_log$seen, as.character(warnings)))
  calls <- spy_log$calls
  last <- runs[[length(runs)]]
  if (length(maxit) > 1L) {
    stopifnot(same_run(last, run_mice(data, m, sum(maxit), seed, args)))
  }
  # the variables named in `lost` are those whose completed data lose
  # observed values, after the start or after a step
  lost_start <- lost_observed(start)
  lost_steps <- lapply(runs, lost_observed)
  stopifnot(setequal(unique(c(names(lost_start), unlist(lapply(lost_steps, names)))), as.character(lost)))

  draws <- NULL
  used <- rep(list(NULL), length(runs))
  if (kind == "scripted") {
    replay <- replay_chain(start, maxit)
    for (s in seq_along(runs)) {
      stopifnot(
        identical(replay$steps[[s]]$seed, runs[[s]]$lastSeedValue),
        isTRUE(all.equal(replay$steps[[s]]$completed, complete(runs[[s]], 1), tolerance = 0))
      )
    }
    draws <- list(
      chisq = I(replay$pools$chisq),
      chisq_df = I(replay$chisq_df),
      normals = I(replay$pools$normal),
      uniforms = I(replay$pools$uniform),
      order = replay$order
    )
    used <- lapply(replay$steps, `[[`, "used")
  } else {
    # nothing is drawn after the start
    for (run in runs) stopifnot(identical(run$lastSeedValue, start$lastSeedValue))
  }
  stopifnot((kind == "spy") == (length(calls) > 0L))

  case <- list(
    name = name,
    kind = kind,
    data = encode_frame(data),
    m = m,
    maxit = I(maxit),
    seed = seed,
    args = encode_args(args),
    method = as.list(last$method),
    visit_sequence = I(last$visitSequence),
    effective_visit_sequence = I(effective_visits(last)),
    predictor_matrix = named_matrix(last$predictorMatrix),
    start = encode_imp(start),
    draws = draws,
    calls = if (kind == "spy") calls else NULL,
    steps = lapply(seq_along(runs), function(s) encode_step(runs[[s]], maxit[s], used[[s]]))
  )
  if (!is.null(lost)) {
    case$lost_observed <- list(start = lost_start, steps = lost_steps)
  }
  if (!is.null(warnings)) {
    case$warnings <- I(warnings)
  }
  warning_log$expected <- character(0)
  case
}

# A call of mice() that stops in R: its arguments, its error and the
# warnings before it (other than "Number of logged events"), for a test of
# what mice-py does instead
stop_case <- function(name, data, m, maxit, seed, ...) {
  args <- list(...)
  warnings <- character(0)
  error <- tryCatch(
    withCallingHandlers(
      {
        do.call(mice, c(list(data, m = m, maxit = maxit, seed = seed, printFlag = FALSE), args))
        NULL
      },
      warning = function(w) {
        if (!startsWith(conditionMessage(w), "Number of logged events")) {
          warnings <<- c(warnings, conditionMessage(w))
        }
        invokeRestart("muffleWarning")
      }
    ),
    error = conditionMessage
  )
  stopifnot("the call stops in R" = !is.null(error))
  list(
    name = name,
    data = encode_frame(data),
    m = m,
    maxit = maxit,
    seed = seed,
    args = encode_args(args),
    error = error,
    warnings = I(warnings)
  )
}

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

empty_method <- function(data, ...) {
  method <- setNames(rep("", ncol(data)), names(data))
  given <- c(...)
  method[names(given)] <- given
  method
}

assign("mice.impute.spy", spy, envir = globalenv())
invisible(suppressMessages(trace(
  "mice.impute.logreg",
  exit = bquote(assign("last", list(p = as.vector(p), q = ncol(rv)), envir = .(logreg_capture))),
  where = asNamespace("mice"),
  print = FALSE
)))
invisible(suppressMessages(trace(
  "mice.impute.polr",
  exit = bquote(assign("last", list(post = post), envir = .(polr_capture))),
  where = asNamespace("mice"),
  print = FALSE
)))
# remove the spy and the traces also when a case fails
cases <- tryCatch(
  list(
    chain_case("norm_nhanes", "scripted", nhanes, 1L, 3L, 7410L, method = "norm"),
    chain_case("norm_predict_nhanes", "deterministic", nhanes, 2L, 2L, 7411L, method = "norm.predict"),
    chain_case("boys_mean_norm_predict", "deterministic", boys60, 2L, 3L, 7412L,
      method = empty_method(boys60,
        hgt = "norm.predict", wgt = "mean", bmi = "norm.predict", hc = "mean", tv = "norm.predict"
      )
    ),
    # hyp keeps its missing values and predicts bmi and chl
    chain_case("spy_empty_method_predictor", "spy", nhanes, 1L, 2L, 7413L,
      method = empty_method(nhanes, bmi = "spy", chl = "spy")
    ),
    # one step per iteration: g predicts nothing, so a label of an earlier
    # iteration shows only in the result of that iteration
    chain_case("logreg_empty_method_predictor", "scripted", syn, 1L, c(1L, 1L, 1L), 7414L,
      method = empty_method(syn, g = "logreg")
    ),
    chain_case("lindep_every_iteration", "deterministic", nhanes_w, 2L, 2L, 7415L, method = "norm.predict"),
    chain_case("spy_df_set_to_1", "spy", nhanes_v, 1L, 2L, 7416L,
      method = c(age = "", bmi = "norm.predict", hyp = "norm.predict", chl = "norm.predict", v = "spy"),
      predictorMatrix = predictors_of_v(nhanes_v)
    ),
    chain_case("spy_arabic_predictor_matrix", "spy", nhanes, 1L, 2L, 7417L,
      method = "spy", visitSequence = "arabic",
      predictorMatrix = matrix(
        c(
          0, 0, 0, 0,
          1, 0, 0, 1,
          1, 1, 0, 0,
          1, 1, 1, 0
        ),
        nrow = 4, byrow = TRUE, dimnames = list(names(nhanes), names(nhanes))
      )
    ),
    chain_case("continue_norm", "scripted", nhanes, 1L, c(2L, 3L), 7418L, method = "norm"),
    chain_case("continue_events", "deterministic", nhanes_wv, 2L, c(2L, 3L), 7419L,
      method = "norm.predict", predictorMatrix = predictors_of_v(nhanes_wv)
    ),
    # where: every cell. age is complete and gets a method; the chain
    # statistics are taken over all 25 where-cells of each variable, so those
    # of a variable imputed by mean are its observed mean and variance 0
    chain_case("all_mean_norm_predict", "deterministic", nhanes, 2L, 2L, 7420L,
      where = "all", method = c(age = "mean", bmi = "norm.predict", hyp = "mean", chl = "norm.predict")
    ),
    # every call fits on the observed rows and fills all 25 rows; the next
    # variable is predicted from the observed values, not the imputations
    chain_case("spy_all", "spy", nhanes, 1L, 2L, 7421L, where = "all", method = "spy"),
    # 25 normals for the cells of each call
    chain_case("norm_all", "scripted", nhanes, 1L, 2L, 7422L, where = "all", method = "norm"),
    # the missing cells stay NA, so the observed where-cells of a row with a
    # missing predictor become NA in the completed data. Nothing is written
    # to the working data, so one iteration shows it all
    chain_case("spy_observed", "spy", nhanes, 1L, 1L, 7423L,
      where = "observed", method = "spy", lost = c("age", "bmi", "hyp", "chl")
    ),
    chain_case("none", "deterministic", nhanes, 2L, 1L, 7424L, where = "none"),
    # bmi left out in rows 1 and 3 and hyp in row 6: they stay NA. Rows 1 and
    # 3 are in neither mask of bmi with complete predictors, row 6 because
    # hyp is missing; the where-cells of hyp and chl in row 1 and of bmi in
    # row 6 get NA, and row 6 drops out of the fit rows of chl
    chain_case("spy_left_out", "spy", nhanes, 1L, 2L, 7425L,
      where = where_cells(nhanes, drop = list(bmi = c(1, 3), hyp = 6)), method = "spy"
    ),
    # the complete age with three where-cells
    chain_case("complete_column", "deterministic", nhanes, 2L, 2L, 7426L,
      where = where_cells(nhanes, add = list(age = c(3, 10, 20))), method = "norm.predict"
    ),
    # observed where-cells in age and bmi: bmi is predicted from the
    # observed age, hyp from the observed bmi (x of the next calls), while
    # the completed data hold their imputations
    chain_case("spy_observed_where_cells", "spy", nhanes, 1L, 2L, 7427L,
      where = where_cells(nhanes, add = list(age = c(3, 10, 20), bmi = c(2, 5))), method = "spy"
    ),
    # chl left out in row 15 and hyp in row 1: the where-cells of bmi in row
    # 15 (observed) and row 1 (missing) and of chl in row 1 have an
    # incomplete predictor at their visit and end as NA
    chain_case("incomplete_predictors", "deterministic", nhanes, 2L, 2L, 7428L,
      where = where_cells(nhanes, add = list(bmi = 15), drop = list(chl = 15, hyp = 1)),
      method = "norm.predict", lost = "bmi"
    ),
    # hyp left out entirely (method ""), bmi left out in row 3: every
    # where-cell of bmi lies in a row where hyp is missing
    chain_case("no_complete_predictors", "deterministic", nhanes, 2L, 2L, 7429L,
      where = where_cells(nhanes, drop = list(hyp = seq_len(n_rows), bmi = 3)), method = "norm.predict"
    ),
    # where = "all" with method "" for age: all 25 cells of age are NA in the
    # completed data, and age still predicts with its observed values
    chain_case("user_empty_method", "deterministic", nhanes, 2L, 2L, 7430L,
      where = "all", method = c(age = "", bmi = "norm.predict", hyp = "norm.predict", chl = "norm.predict"),
      lost = "age"
    ),
    # k has three where-cells and is removed as constant: they are NA
    chain_case("constant_removed", "deterministic", nhanes_k, 2L, 2L, 7431L,
      where = where_cells(nhanes_k, add = list(k = c(2, 5, 7))), method = "norm.predict", lost = "k"
    ),
    # observed where-cells, missing cells left out (the observed where-cell
    # of age in row 3 ends as NA), continued with mice.mids()
    chain_case("continue_where", "scripted", nhanes, 1L, c(2L, 3L), 7432L,
      where = where_cells(nhanes, add = list(age = c(3, 10, 20), bmi = c(2, 5)), drop = list(bmi = c(1, 3))),
      method = "norm", lost = "age"
    ),
    # where = "all": g is fitted on its observed rows with complete z and
    # fills its 32 cells with complete z, overlapping; x1, x2, z have no
    # method, so all their cells are NA in the completed data
    chain_case("logreg_all", "scripted", syn, 1L, c(1L, 1L), 7433L,
      where = "all", method = empty_method(syn, g = "logreg"), lost = c("x1", "x2", "z", "g")
    ),
    # ignore: the missing cells of the ignored rows 1, 6 and 24 are imputed
    # (in wy, not in ry); their observed cells and those of rows 14, 17 and
    # 22 are in neither mask; the imputer gets them all
    chain_case("spy_ignore", "spy", nhanes, 1L, 2L, 7434L, method = "spy", ignore = ignored_nhanes),
    # mean imputes the mean of the rows that are not ignored
    chain_case("ignore_norm_predict_mean", "deterministic", nhanes, 2L, 2L, 7435L,
      method = c(age = "", bmi = "norm.predict", hyp = "mean", chl = "norm.predict"), ignore = ignored_nhanes
    ),
    # the degrees of freedom of the chi-square draw count the fit rows
    chain_case("norm_ignore", "scripted", nhanes, 1L, 3L, 7436L, method = "norm", ignore = ignored_nhanes),
    # check.df() counts the 3 fit rows of u; remove.lindep() drops p, constant
    # in the fit rows, from the models of bmi, hyp and chl
    chain_case("spy_ignore_few_fit_rows", "spy", nhanes_up, 1L, 2L, 7437L,
      method = empty_method(nhanes_up, bmi = "spy", hyp = "spy", chl = "spy", u = "spy"),
      predictorMatrix = predictors_of_u(nhanes_up), ignore = ignored_nhanes
    ),
    # ignore and where: observed where-cells in ignored rows (bmi in row 14,
    # chl in row 17: in wy, not in ry) and in a row that is not ignored (bmi
    # in row 2: in both); hyp left out in the ignored row 6, where bmi is
    # missing: that where-cell of bmi has an incomplete predictor
    chain_case("spy_ignore_where", "spy", nhanes, 1L, 2L, 7438L,
      where = where_cells(nhanes, add = list(bmi = c(2, 14), chl = 17), drop = list(hyp = 6)),
      method = "spy", ignore = ignored_nhanes
    ),
    chain_case("continue_ignore", "scripted", nhanes, 1L, c(2L, 3L), 7439L, method = "norm", ignore = ignored_nhanes),
    # 9 rows left for the fits: R warns and goes on
    chain_case("ignore_fewer_than_10_rows", "deterministic", nhanes, 2L, 2L, 7440L,
      method = "norm.predict", ignore = !(seq_len(n_rows) %in% c(2L, 3L, 5L, 7L, 9L, 10L, 13L, 15L, 20L)),
      warnings = "Fewer than 10 rows for fitting the imputation model. Are you sure?"
    ),
    # one step per iteration (g predicts nothing)
    chain_case("logreg_one_cell_ignored", "scripted", syn_one, 1L, c(1L, 1L, 1L), 7441L,
      method = empty_method(syn_one, g = "logreg"), ignore = ignored_one
    ),
    chain_case("logreg_category_only_ignored", "scripted", syn_only, 1L, c(1L, 1L), 7450L,
      method = empty_method(syn_only, g = "logreg"), ignore = ignored_only
    ),
    chain_case("logreg_clamp_ignored", "scripted", syn_clamp, 1L, c(1L, 1L), 7460L,
      method = empty_method(syn_clamp, g = "logreg"), ignore = ignored_clamp
    ),
    # augment() is skipped: o is complete, its where-cells are observed (in
    # both masks), and the ignored row 3 is the only row outside the fit rows
    chain_case("polr_one_row_outside", "scripted", labels_o, 1L, c(1L, 1L), 7473L,
      where = where_cells(labels_o, add = list(o = c(2, 7, 11, 16, 19))),
      method = c(x = "", o = "polr"), ignore = ignored_lab
    )
  ),
  finally = {
    rm("mice.impute.spy", envir = globalenv())
    invisible(suppressMessages(untrace("mice.impute.logreg", where = asNamespace("mice"))))
    invisible(suppressMessages(untrace("mice.impute.polr", where = asNamespace("mice"))))
  }
)

# Calls that stop in R; what mice-py does there is tested in test_l1_chain
r_errors <- list(
  # polyreg without predictor (augment() skipped), c only in the ignored
  # row 3: multinom() drops the empty level and polyreg then stops
  stop_case("polyreg_no_predictor_label_only_ignored", labels_g_missing, 1L, 1L, 7471L,
    method = c(x = "", g = "polyreg"), predictorMatrix = no_predictor, ignore = ignored_lab
  ),
  # the same with augment() skipped because row 3 is the only row outside
  # the fit rows: g is complete, its where-cells are observed
  stop_case("polyreg_one_row_outside_label_only_ignored", labels_g, 1L, 1L, 7472L,
    where = where_cells(labels_g, add = list(g = c(2, 7, 11, 16))), method = c(x = "", g = "polyreg"),
    ignore = ignored_lab
  ),
  # every observed value of u lies in an ignored row: u starts from rnorm()
  # and its model stops in iteration 1
  stop_case("all_observed_ignored", nhanes_up, 1L, 1L, 7475L,
    method = empty_method(nhanes_up, bmi = "norm", hyp = "norm", chl = "norm", u = "norm"),
    predictorMatrix = predictors_of_u(nhanes_up), ignore = !is.na(nhanes_up$u)
  ),
  stop_case("ignore_na", nhanes, 1L, 1L, 7476L, ignore = replace(ignored_nhanes, 3L, NA)),
  stop_case("ignore_numeric", nhanes, 1L, 1L, 7477L, ignore = as.numeric(ignored_nhanes)),
  stop_case("ignore_short", nhanes, 1L, 1L, 7478L, ignore = ignored_nhanes[-n_rows])
)

write_fixture("l1_chain", list(
  description = paste(
    "Whole chains of mice(data, m, maxit, ..., seed). start: imp of",
    "mice(maxit = 0) with the same seed, per variable with where-cells its",
    "rows (0-based) and per chain the values. kind: 'deterministic' (nothing",
    "drawn after the start), 'scripted' (m = 1; draws: chisq with chisq_df,",
    "normals, uniforms, in the order given as [pool, count] pairs) or 'spy'",
    "(m = 1; calls: what mice.impute.spy received at every call, x after",
    "remove.lindep). args: the arguments of mice() (method a string or one",
    "per column; predictor_matrix, visit_sequence null when not given; where",
    "only when given, a keyword or cols and values, a list of rows; ignore",
    "only when given, one value per row).",
    "maxit: one value per step, the first for mice(), each further one for",
    "mice.mids(). steps: after each step imp, the completed data per chain,",
    "chain_mean and chain_var (numeric variables R imputes, iteration x",
    "chain), logged_events, iteration and the values drawn from each pool so",
    "far (scripted). lost_observed (only in cases that lose observed values):",
    "per variable the number of observed where-cells that are NA in the",
    "completed data of some chain, after the start and after each step.",
    "warnings (only in cases with such warnings): what each call of mice()",
    "warned besides 'Number of logged events'; mice.mids() warns nothing.",
    "r_errors: calls of mice(data, m, maxit, ..., seed) that stop in R, with",
    "args as above, the error message and the warnings before it."
  ),
  cases = cases,
  r_errors = r_errors
))
