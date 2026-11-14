# L1 fixture for whole chains of mice(), compared exactly.
#
#   l1_chain.json   mice(data, m, maxit = 0) for the start of each chain and
#                   the generator state after it; mice(data, m, maxit) (and
#                   mice.mids() for a continued run) for the run: imp, the
#                   completed data, chainMean, chainVar, method,
#                   visitSequence, predictorMatrix, loggedEvents, iteration;
#                   the draws or the imputer calls of the run
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
# - scripted: methods that draw (norm, logreg), m = 1, so that R's order of
#   the draws (iterations outside, imputations inside) is mice-py's (one
#   chain after the other). replay_chain() goes through the chain as
#   sampler() and sampler.univ() do, from start$lastSeedValue: before each
#   imputer call it saves .Random.seed, calls the imputer, saves the state
#   after, restores the state before and draws the expected numbers in the
#   expected order (norm: rchisq(1, df), rnorm(q), rnorm(n0); logreg:
#   rnorm(q), runif(n0)); the same state afterwards proves kind, number and
#   order of the draws. After each step of the run (see below) the replay
#   must have its data and its generator state.
# - spy: the method "spy" (mice.impute.spy below) records at every call what
#   sampler.univ() hands an imputer and imputes as norm.predict; m = 1. As in
#   a deterministic case, nothing is drawn after the start.
#
# A case with several values of maxit runs mice() with the first and
# mice.mids() with each further one, and requires that the result equals
# mice() with their sum. The cases take the arguments of mice() through
# `...` (encode_args() lists the ones it knows).

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
# draws, from a trace() exit hook, so that the replay can require that no
# label depends on the last digits of a probability
logreg_capture <- new.env()

# ---------------------------------------------------------------------------
# Running mice()
# ---------------------------------------------------------------------------

# Evaluate `expr`; the warning "Number of logged events" is expected and
# muffled, any other warning stops the script
quietly <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (!startsWith(conditionMessage(w), "Number of logged events")) {
      stop("unexpected warning: ", conditionMessage(w))
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

# Go through the chain of `start` (m = 1) for sum(maxit) iterations as
# sampler() and sampler.univ() do, from the generator state after the start,
# and draw what each imputer is expected to draw (see the top of the file).
# Returns the draws, their order as [pool, count] pairs, and at the end of
# each step the working data, the generator state and the number of values
# drawn from each pool so far.
replay_chain <- function(start, maxit) {
  stopifnot(start$m == 1L)
  data <- working_data(start)
  r <- !is.na(start$data)
  where <- start$where
  ignore <- start$ignore
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
      } else {
        stopifnot(meth %in% c("norm.predict", "mean"))
      }
      stopifnot(identical(.Random.seed, after))
      # imp of the where-cells: NA where the predictors are incomplete
      new <- data[where[, h], h]
      new[!wy[where[, h]]] <- NA
      new[wy[where[, h]]] <- imputes
      data[fill, h] <- new[fill[where[, h]]]
    }
    if (k %in% cumsum(maxit)) {
      steps[[length(steps) + 1L]] <- list(
        data = data,
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

# The arguments of mice() a case passes, for the Python side
encode_args <- function(args) {
  stopifnot(
    "encode_args() knows the arguments of mice() a case passes" =
      all(names(args) %in% c("method", "predictorMatrix", "visitSequence"))
  )
  method <- args$method
  vis <- args$visitSequence
  keyword <- length(vis) == 1L && vis %in% c("roman", "arabic", "monotone", "revmonotone")
  list(
    method = if (is.null(names(method))) method else as.list(method),
    predictor_matrix = if (is.null(args$predictorMatrix)) NULL else named_matrix(args$predictorMatrix),
    visit_sequence = if (is.null(vis) || keyword) vis else I(vis)
  )
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

chain_case <- function(name, kind, data, m, maxit, seed, ...) {
  args <- list(...)
  stopifnot(kind %in% c("deterministic", "scripted", "spy"), kind == "deterministic" || m == 1L)
  start <- run_mice(data, m, 0, seed, args)
  spy_log$calls <- list()
  runs <- run_steps(data, m, maxit, seed, args)
  calls <- spy_log$calls
  last <- runs[[length(runs)]]
  if (length(maxit) > 1L) {
    stopifnot(same_run(last, run_mice(data, m, sum(maxit), seed, args)))
  }

  draws <- NULL
  used <- rep(list(NULL), length(runs))
  if (kind == "scripted") {
    replay <- replay_chain(start, maxit)
    for (s in seq_along(runs)) {
      stopifnot(
        identical(replay$steps[[s]]$seed, runs[[s]]$lastSeedValue),
        isTRUE(all.equal(replay$steps[[s]]$data, complete(runs[[s]], 1), tolerance = 0))
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

  list(
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
# remove the spy and the trace also when a case fails
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
    )
  ),
  finally = {
    rm("mice.impute.spy", envir = globalenv())
    invisible(suppressMessages(untrace("mice.impute.logreg", where = asNamespace("mice"))))
  }
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
    "per column; predictor_matrix, visit_sequence null when not given).",
    "maxit: one value per step, the first for mice(), each further one for",
    "mice.mids(). steps: after each step imp, the completed data per chain,",
    "chain_mean and chain_var (numeric variables R imputes, iteration x",
    "chain), logged_events, iteration and the values drawn from each pool so",
    "far (scripted)."
  ),
  cases = cases
))
