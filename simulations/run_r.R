# L3 simulation study, R side: impute the datasets written by study.py with
# R mice and record the pooling ingredients per replication.
#
# Run from the repository root, after `study.py generate`:
#   Rscript simulations/run_r.R [--scenarios S1,S4] [--methods pmm,sample]
#                               [--reps 1000] [--rf-reps 500] [--jobs 8]
#
# Reads   simulations/work/data/manifest.json and data/<scenario>/rep_<k>.csv
# Writes  simulations/work/results/r_<scenario>_<method>.csv
#         (scenario, method, rep, term, qbar, ubar, b, m, n, dfcom),
#         r_<scenario>_<method>_status.csv and r_meta.json.
#
# Every incomplete column gets the same method; full predictor matrix; visit
# sequence = incomplete columns in column order; m = 5; maxit from the
# manifest; seed = scenario_index * 100000 + rep. The pseudo-method
# `complete` analyses the data before deletion and must reproduce the Python
# numbers exactly (same data, same OLS).
#
# This script calls mice rather than copying mice source, so that its
# results are those of the installed mice. mice-py, like mice, is licensed
# GPL (>= 2).

suppressPackageStartupMessages({
  library(mice)
  library(jsonlite)
  library(parallel)
})

# One thread per worker: replications run in parallel instead.
options(ranger.num.threads = 1L)

METHODS <- c("pmm", "midastouch", "cart", "rf", "sample")
M_IMPUTATIONS <- 5L
CHUNK_SIZE <- 10L # replications per forked task

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  file.path(getwd(), "simulations")
}

parse_options <- function(args) {
  opts <- list(
    scenarios = NULL, methods = METHODS, reps = 1000L, rf_reps = 500L,
    jobs = max(1L, parallel::detectCores()),
    work_dir = file.path(script_dir(), "work")
  )
  split_list <- function(x) trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  i <- 1L
  while (i <= length(args)) {
    key <- args[i]
    if (grepl("=", key, fixed = TRUE)) {
      value <- sub("^[^=]*=", "", key)
      key <- sub("=.*$", "", key)
      i <- i + 1L
    } else {
      if (i == length(args)) stop("option ", key, " needs a value")
      value <- args[i + 1L]
      i <- i + 2L
    }
    switch(key,
      "--scenarios" = opts$scenarios <- split_list(value),
      "--methods" = opts$methods <- split_list(value),
      "--reps" = opts$reps <- as.integer(value),
      "--rf-reps" = opts$rf_reps <- as.integer(value),
      "--jobs" = opts$jobs <- as.integer(value),
      "--work-dir" = opts$work_dir <- normalizePath(value, mustWork = TRUE),
      stop("unknown option ", key)
    )
  }
  bad <- setdiff(opts$methods, METHODS)
  if (length(bad)) stop("unknown method(s): ", paste(bad, collapse = ", "))
  if (is.na(opts$reps) || opts$reps < 1L) stop("--reps must be positive")
  if (is.na(opts$jobs) || opts$jobs < 1L) stop("--jobs must be positive")
  if (.Platform$OS.type == "windows") opts$jobs <- 1L  # mclapply cannot fork
  opts
}

replication_seed <- function(spec, rep) as.integer(spec$index) * 100000L + rep

# The manifest is read without simplification; turn the vectors into vectors.
normalise_spec <- function(spec) {
  for (key in c("columns", "means", "terms")) spec[[key]] <- as.character(unlist(spec[[key]]))
  spec$regression$predictors <- as.character(unlist(spec$regression$predictors))
  spec$categorical <- lapply(spec$categorical, function(lv) as.character(unlist(lv)))
  spec$maxit <- as.integer(spec$maxit)
  spec$reps <- as.integer(spec$reps)
  spec
}

read_dataset <- function(path, spec) {
  df <- utils::read.csv(path, na.strings = "", stringsAsFactors = FALSE)
  df <- df[spec$columns]
  for (col in spec$columns) {
    if (col %in% names(spec$categorical)) {
      df[[col]] <- factor(df[[col]], levels = spec$categorical[[col]])
    } else {
      df[[col]] <- as.numeric(df[[col]])
    }
  }
  df
}

# Same analysis as scenarios.analyse() in Python.
analyse <- function(df, spec) {
  if (anyNA(df[spec$columns])) {
    stop("completed data still contain missing values")
  }
  n <- nrow(df)
  f <- stats::reformulate(spec$regression$predictors,
    response = spec$regression$response
  )
  fit <- stats::lm(f, data = df)
  est <- stats::coef(fit)
  out <- data.frame(
    term = names(est), estimate = unname(est),
    variance = unname(diag(stats::vcov(fit))),
    dfcom = stats::df.residual(fit), stringsAsFactors = FALSE
  )
  for (col in spec$means) {
    x <- df[[col]]
    out <- rbind(out, data.frame(
      term = sprintf("mean(%s)", col), estimate = mean(x),
      variance = stats::var(x) / n, dfcom = n - 1, stringsAsFactors = FALSE
    ))
  }
  for (term in names(spec$proportions)) {
    pr <- spec$proportions[[term]]
    p <- mean(as.character(df[[pr$column]]) == pr$level)
    out <- rbind(out, data.frame(
      term = term, estimate = p, variance = p * (1 - p) / n, dfcom = Inf,
      stringsAsFactors = FALSE
    ))
  }
  out <- out[match(spec$terms, out$term), ]
  if (anyNA(out$term) || anyNA(out$estimate) || anyNA(out$variance)) {
    stop("analysis did not produce every term (empty factor level?)")
  }
  out
}

impute <- function(data, method, spec, seed) {
  incomplete <- names(data)[colSums(is.na(data)) > 0]
  meth <- stats::setNames(rep("", ncol(data)), names(data))
  meth[incomplete] <- method
  pred <- matrix(1L, ncol(data), ncol(data),
    dimnames = list(names(data), names(data))
  )
  diag(pred) <- 0L
  # Column positions, not names: mice reads a single name as a keyword
  # ("roman", "monotone", ...) and stops.
  visit <- match(incomplete, names(data))
  imp <- mice::mice(data,
    m = M_IMPUTATIONS, maxit = spec$maxit, method = meth,
    predictorMatrix = pred, visitSequence = visit,
    printFlag = FALSE, seed = seed
  )
  if (!identical(unname(imp$visitSequence), incomplete)) {
    stop("unexpected visit sequence: ", paste(imp$visitSequence, collapse = ", "))
  }
  imp
}

one_line <- function(x) substr(gsub("\\s+", " ", paste(x, collapse = " ")), 1L, 300L)

run_one <- function(spec, method, rep, data_dir) {
  t0 <- proc.time()[["elapsed"]]
  status <- list(
    scenario = spec$id, method = method, rep = rep, status = "ok",
    error_class = "", error_message = "", n_events = NA_integer_,
    events = "", n_warnings = 0L, warnings = "", seconds = NA_real_
  )
  warns <- character(0)
  stage <- "imputation"
  result <- tryCatch(
    withCallingHandlers(
      {
        if (method == "complete") {
          df <- read_dataset(
            file.path(data_dir, spec$id, "complete", sprintf("rep_%d.csv", rep)), spec
          )
          stage <- "analysis"
          list(analyses = list(analyse(df, spec)), m = 1L, n = nrow(df))
        } else {
          df <- read_dataset(file.path(data_dir, spec$id, sprintf("rep_%d.csv", rep)), spec)
          imp <- impute(df, method, spec, replication_seed(spec, rep))
          ev <- imp$loggedEvents
          status$n_events <- if (is.null(ev)) 0L else nrow(ev)
          if (!is.null(ev)) {
            status$events <- one_line(unique(paste(ev$dep, ev$meth, ev$out)))
          }
          stage <- "analysis"
          analyses <- lapply(seq_len(imp$m), function(j) analyse(mice::complete(imp, j), spec))
          list(analyses = analyses, m = imp$m, n = nrow(df))
        }
      },
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  status$n_warnings <- length(warns)
  status$warnings <- one_line(unique(warns))
  rows <- NULL
  if (inherits(result, "error")) {
    status$status <- "failed"
    status$error_class <- class(result)[1]
    msg <- one_line(conditionMessage(result))
    status$error_message <- if (stage == "analysis") paste("analysis:", msg) else msg
  } else {
    est <- sapply(result$analyses, function(a) a$estimate)
    var <- sapply(result$analyses, function(a) a$variance)
    est <- matrix(est, ncol = result$m)
    var <- matrix(var, ncol = result$m)
    first <- result$analyses[[1]]
    rows <- data.frame(
      scenario = spec$id, method = method, rep = rep, term = first$term,
      qbar = rowMeans(est), ubar = rowMeans(var),
      b = if (result$m > 1) apply(est, 1, stats::var) else 0,
      m = result$m, n = result$n, dfcom = first$dfcom,
      stringsAsFactors = FALSE
    )
  }
  status$seconds <- proc.time()[["elapsed"]] - t0
  list(rows = rows, status = as.data.frame(status, stringsAsFactors = FALSE))
}

format_numbers <- function(df, cols) {
  for (col in cols) df[[col]] <- sprintf("%.17g", df[[col]])
  df
}

write_outputs <- function(results_dir, spec, method, outs) {
  rows <- do.call(rbind, lapply(outs, `[[`, "rows"))
  status <- do.call(rbind, lapply(outs, `[[`, "status"))
  status <- status[order(status$rep), ]
  if (is.null(rows)) {
    rows <- data.frame(
      scenario = character(0), method = character(0), rep = integer(0),
      term = character(0), qbar = numeric(0), ubar = numeric(0), b = numeric(0),
      m = integer(0), n = integer(0), dfcom = numeric(0)
    )
  } else {
    rows <- rows[order(rows$rep, match(rows$term, spec$terms)), ]
    rows <- format_numbers(rows, c("qbar", "ubar", "b", "dfcom"))
  }
  utils::write.csv(rows,
    file.path(results_dir, sprintf("r_%s_%s.csv", spec$id, method)),
    row.names = FALSE, quote = FALSE
  )
  status$seconds <- sprintf("%.6g", status$seconds)
  utils::write.csv(status,
    file.path(results_dir, sprintf("r_%s_%s_status.csv", spec$id, method)),
    row.names = FALSE, quote = TRUE, na = ""
  )
  n_fail <- sum(status$status != "ok")
  classes <- unique(status$error_class[status$status != "ok"])
  cat(sprintf(
    "  %s %-10s reps=%5d failed=%5d%s  events in %d reps  %.3f s/rep\n",
    spec$id, method, nrow(status), n_fail,
    if (length(classes)) paste0(" (", paste(classes, collapse = ", "), ")") else "",
    sum(status$n_events > 0, na.rm = TRUE), mean(as.numeric(status$seconds))
  ))
}

main <- function() {
  opts <- parse_options(commandArgs(trailingOnly = TRUE))
  data_dir <- file.path(opts$work_dir, "data")
  results_dir <- file.path(opts$work_dir, "results")
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
  manifest_path <- file.path(data_dir, "manifest.json")
  if (!file.exists(manifest_path)) stop(manifest_path, " not found: run study.py generate first")
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  scenarios <- if (is.null(opts$scenarios)) names(manifest$scenarios) else opts$scenarios
  missing_sc <- setdiff(scenarios, names(manifest$scenarios))
  if (length(missing_sc)) stop("not generated: ", paste(missing_sc, collapse = ", "))

  cat(sprintf(
    "R %s, mice %s, %d processes\n", getRversion(), packageVersion("mice"), opts$jobs
  ))
  t_start <- proc.time()[["elapsed"]]
  plan <- list()
  for (sid in scenarios) {
    spec <- normalise_spec(manifest$scenarios[[sid]])
    reps_by_method <- vapply(opts$methods, function(mth) {
      if (mth == "rf") min(opts$rf_reps, opts$reps) else opts$reps
    }, integer(1))
    need <- max(reps_by_method)
    if (spec$reps < need) {
      stop(sid, ": ", spec$reps, " datasets generated, ", need, " needed")
    }
    reps_by_method["complete"] <- need
    # All methods of a scenario in one pool, in chunks of CHUNK_SIZE
    # replications handed out dynamically: cores of different speed stay
    # busy, and results do not depend on scheduling (one seed per rep).
    tasks <- do.call(rbind, lapply(names(reps_by_method), function(mth) {
      data.frame(method = mth, rep = seq_len(reps_by_method[[mth]]), stringsAsFactors = FALSE)
    }))
    chunks <- split(seq_len(nrow(tasks)), ceiling(seq_len(nrow(tasks)) / CHUNK_SIZE))
    outs <- parallel::mclapply(chunks, function(idx) {
      lapply(idx, function(i) run_one(spec, tasks$method[i], tasks$rep[i], data_dir))
    }, mc.cores = opts$jobs, mc.preschedule = FALSE)
    died <- vapply(outs, function(o) inherits(o, "try-error") || is.null(o), logical(1))
    if (any(died)) {
      first <- outs[[which(died)[1]]]
      why <- if (inherits(first, "try-error")) one_line(first) else "no result (process killed?)"
      stop(sum(died), " chunk(s) of ", sid, " died: ", why)
    }
    outs <- unlist(unname(outs), recursive = FALSE)
    for (mth in names(reps_by_method)) {
      write_outputs(results_dir, spec, mth, outs[tasks$method == mth])
      plan[[paste0(sid, "/", mth)]] <- reps_by_method[[mth]]
    }
  }
  wall <- proc.time()[["elapsed"]] - t_start

  meta_path <- file.path(results_dir, "r_meta.json")
  old_plan <- if (file.exists(meta_path)) jsonlite::fromJSON(meta_path)$plan else list()
  for (k in names(plan)) old_plan[[k]] <- plan[[k]]
  meta <- list(
    r = R.version.string,
    mice = as.character(packageVersion("mice")),
    ranger = as.character(packageVersion("ranger")),
    rpart = as.character(packageVersion("rpart")),
    jobs = opts$jobs, cpu_count = parallel::detectCores(),
    wall_seconds = round(wall, 1), plan = old_plan,
    finished = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
  writeLines(
    jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
    meta_path
  )
  cat(sprintf("done in %.1f s\n", wall))
}

main()
