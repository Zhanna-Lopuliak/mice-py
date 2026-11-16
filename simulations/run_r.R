# L3 simulation study, R side: impute the datasets written by study.py with
# R mice and record the pooling ingredients per replication.
#
# Run from the repository root, after `study.py generate`:
#   Rscript simulations/run_r.R [--scenarios S1,S4] [--methods pmm,default]
#                               [--variants none,clip] [--reps N] [--jobs 8]
#
# Reads   simulations/work/data/manifest.json and data/<scenario>/rep_<k>.csv
# Writes  simulations/work/results/r_<scenario>_<label>.csv
#         (scenario, method, rep, term, qbar, ubar, b, m, n, dfcom),
#         r_<scenario>_<label>_status.csv and r_meta.json.
#
# The cells (study method x scenario, optionally with a variant, with their
# replications) and the method of every incomplete column come from the
# manifest that study.py wrote, so both engines run the same cells;
# --scenarios, --methods and --variants (`none` for the cells without one)
# select cells, --reps caps their replications. The label of a cell (the
# `method` column of the results) is its study method, followed by
# `+<variant>` if it has one. A cell without column methods (`default`)
# calls mice() without `method`, so mice chooses by column type
# (make.method). Full predictor matrix; visit sequence = the columns with
# cells to impute, in column order; m = 5; maxit from the manifest; seed =
# scenario_index * 100000 + rep. A variant adds arguments, as _adapter.py
# does for mice-py (variant_arguments() below). The status file records
# imp$method. The pseudo-method `complete` analyses the data before deletion
# and must reproduce the Python numbers exactly (same data, same OLS).
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
    scenarios = NULL, methods = NULL, variants = NULL, reps = NULL,
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
      "--variants" = opts$variants <- split_list(value),
      "--reps" = opts$reps <- as.integer(value),
      "--jobs" = opts$jobs <- as.integer(value),
      "--work-dir" = opts$work_dir <- normalizePath(value, mustWork = TRUE),
      stop("unknown option ", key)
    )
  }
  if (!is.null(opts$reps) && (is.na(opts$reps) || opts$reps < 1L)) stop("--reps must be positive")
  if (is.na(opts$jobs) || opts$jobs < 1L) stop("--jobs must be positive")
  if (.Platform$OS.type == "windows") opts$jobs <- 1L  # mclapply cannot fork
  opts
}

replication_seed <- function(spec, rep) as.integer(spec$index) * 100000L + rep

# The manifest is read without simplification; turn the vectors into vectors.
normalise_spec <- function(spec) {
  for (key in c("columns", "incomplete", "means", "terms")) spec[[key]] <- as.character(unlist(spec[[key]]))
  spec$regression$predictors <- as.character(unlist(spec$regression$predictors))
  spec$categorical <- lapply(spec$categorical, function(f) {
    list(levels = as.character(unlist(f$levels)), ordered = isTRUE(f$ordered))
  })
  spec$maxit <- as.integer(spec$maxit)
  spec$reps <- as.integer(spec$reps)
  spec
}

# A cell of the manifest: `columns` (method of each incomplete column) is
# NULL for `default`, `variant` NULL for a cell without one. Manifests
# written before variants existed have neither `variant` nor `label`.
normalise_cell <- function(cell) {
  cols <- cell$columns
  cell$columns <- if (is.null(cols)) NULL else vapply(cols, as.character, character(1))
  cell$reps <- as.integer(cell$reps)
  if (is.null(cell$label)) {
    cell$label <- if (is.null(cell$variant)) cell$method else paste0(cell$method, "+", cell$variant)
  }
  cell
}

# A factor with the manifest's levels, ordered where the manifest says so,
# so that mice's default for an ordered column is polr.
read_dataset <- function(path, spec) {
  df <- utils::read.csv(path, na.strings = "", stringsAsFactors = FALSE)
  df <- df[spec$columns]
  for (col in spec$columns) {
    f <- spec$categorical[[col]]
    if (!is.null(f)) {
      unknown <- setdiff(unique(df[[col]][!is.na(df[[col]])]), f$levels)
      if (length(unknown)) stop("column ", col, " has values outside its levels: ", paste(unknown, collapse = ", "))
      df[[col]] <- factor(df[[col]], levels = f$levels, ordered = f$ordered)
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
  # Treatment dummies for every factor: lm would give an ordered factor
  # polynomial contrasts (o.L, o.Q, ...), so analyse an unordered copy.
  for (col in names(spec$categorical)) {
    if (is.ordered(df[[col]])) df[[col]] <- factor(df[[col]], levels = levels(df[[col]]), ordered = FALSE)
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

# Arguments of mice() for a cell's variant, besides the methods (study.py
# VARIANTS; _adapter.variant_arguments gives mice-py the same): `where`,
# `ignore`, `post`, the passive formulas and the predictor matrix. NULL
# leaves an argument at mice's default.
variant_arguments <- function(variant, data, pred) {
  out <- list(where = NULL, ignore = NULL, post = NULL, passive = NULL, pred = pred)
  if (is.null(variant)) {
    return(out)
  }
  n <- nrow(data)
  if (variant == "ignore_half") {
    # Rows n/2 + 1 to n
    out$ignore <- seq_len(n) > n %/% 2L
  } else if (variant == "where_all") {
    out$where <- "all"
  } else if (variant == "clip") {
    # squeeze() to the range of the observed values. The bounds are written
    # in hexadecimal, which R parses exactly; a 17-digit decimal is not
    # always parsed back to the same double.
    out$post <- mice::make.post(data)
    for (col in names(data)) {
      if (anyNA(data[[col]]) && is.numeric(data[[col]])) {
        bounds <- range(data[[col]], na.rm = TRUE)
        out$post[col] <- sprintf("imp[[j]][, i] <- squeeze(imp[[j]][, i], c(%a, %a))", bounds[1], bounds[2])
      }
    }
  } else if (variant == "passive_product") {
    out$passive <- c(p = "~ I(x1 * x2)")
    out$pred["x2", "p"] <- 0L
  } else {
    stop("unknown variant ", variant)
  }
  out
}

# `cell$columns`: the method of each incomplete column (names), or NULL for
# no `method` argument (mice's make.method chooses by column type);
# `cell$variant`: see variant_arguments(). Under `where = "all"` a complete
# column gets its default method (make.method); a passive column gets its
# formula, so a `default` cell with a passive column gets make.method's
# methods for the others.
impute <- function(data, cell, spec, seed) {
  columns <- cell$columns
  incomplete <- names(data)[colSums(is.na(data)) > 0]
  pred <- matrix(1L, ncol(data), ncol(data),
    dimnames = list(names(data), names(data))
  )
  diag(pred) <- 0L
  va <- variant_arguments(cell$variant, data, pred)
  where <- if (is.null(va$where)) is.na(data) else mice::make.where(data, va$where)
  # The columns with cells to impute, in column order. Column positions,
  # not names: mice reads a single name as a keyword ("roman", "monotone",
  # ...) and stops.
  imputed <- names(data)[colSums(where) > 0]
  visit <- match(imputed, names(data))
  meth <- NULL
  if (!is.null(columns)) {
    active <- setdiff(incomplete, names(va$passive))
    unassigned <- setdiff(active, names(columns))
    if (length(unassigned)) stop("no method for the incomplete column(s) ", paste(unassigned, collapse = ", "))
    meth <- if (is.null(va$where)) {
      stats::setNames(rep("", ncol(data)), names(data))
    } else {
      mice::make.method(data, where = where)
    }
    meth[active] <- columns[active]
  }
  if (!is.null(va$passive)) {
    if (is.null(meth)) meth <- mice::make.method(data, where = where)
    meth[names(va$passive)] <- va$passive
  }
  # NULL (method, where, ignore, post) is mice's default for each
  imp <- mice::mice(data,
    m = M_IMPUTATIONS, maxit = spec$maxit, method = meth,
    predictorMatrix = va$pred, visitSequence = visit,
    where = va$where, ignore = va$ignore, post = va$post,
    printFlag = FALSE, seed = seed
  )
  if (!identical(unname(imp$visitSequence), imputed)) {
    stop("unexpected visit sequence: ", paste(imp$visitSequence, collapse = ", "))
  }
  imp
}

one_line <- function(x) substr(gsub("\\s+", " ", paste(x, collapse = " ")), 1L, 300L)

# `cell`: list(method, label, columns, variant) from the manifest; method
# and label "complete" for the complete-data analysis.
run_one <- function(spec, cell, rep, data_dir) {
  method <- cell$label
  t0 <- proc.time()[["elapsed"]]
  status <- list(
    scenario = spec$id, method = method, rep = rep, status = "ok",
    error_class = "", error_message = "", n_events = NA_integer_,
    events = "", n_warnings = 0L, warnings = "", methods = "", seconds = NA_real_
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
          imp <- impute(df, cell, spec, replication_seed(spec, rep))
          status$methods <- paste0(names(imp$method), "=", unname(imp$method), collapse = ";")
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
    "  %s %-12s reps=%5d failed=%5d%s  events in %d reps  %.3f s/rep\n",
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
  if (is.null(manifest$cells)) stop(manifest_path, " has no cells: run study.py generate again")
  scenarios <- if (is.null(opts$scenarios)) names(manifest$scenarios) else opts$scenarios
  missing_sc <- setdiff(scenarios, names(manifest$scenarios))
  if (length(missing_sc)) stop("not generated: ", paste(missing_sc, collapse = ", "))

  # The cells of the manifest, selected by --scenarios, --methods and
  # --variants ("none": the cells without a variant), with their
  # replications capped by --reps
  cells <- lapply(manifest$cells, normalise_cell)
  known <- unique(vapply(cells, `[[`, character(1), "method"))
  bad <- setdiff(opts$methods, known)
  if (length(bad)) stop("unknown study method(s): ", paste(bad, collapse = ", "))
  variant_of <- function(cl) if (is.null(cl$variant)) "none" else cl$variant
  known <- unique(c("none", vapply(cells, variant_of, character(1))))
  bad <- setdiff(opts$variants, known)
  if (length(bad)) stop("unknown variant(s): ", paste(bad, collapse = ", "))
  cells <- Filter(function(cl) {
    cl$scenario %in% scenarios &&
      (is.null(opts$methods) || cl$method %in% opts$methods) &&
      (is.null(opts$variants) || variant_of(cl) %in% opts$variants)
  }, cells)
  if (!length(cells)) stop("no cell matches --scenarios, --methods and --variants")
  if (!is.null(opts$reps)) {
    cells <- lapply(cells, function(cl) {
      cl$reps <- min(cl$reps, opts$reps)
      cl
    })
  }

  cat(sprintf(
    "R %s, mice %s, %d processes\n", getRversion(), packageVersion("mice"), opts$jobs
  ))
  t_start <- proc.time()[["elapsed"]]
  plan <- list()
  for (sid in scenarios) {
    own <- Filter(function(cl) cl$scenario == sid, cells)
    if (!length(own)) next
    spec <- normalise_spec(manifest$scenarios[[sid]])
    need <- max(vapply(own, `[[`, integer(1), "reps"))
    if (spec$reps < need) {
      stop(sid, ": ", spec$reps, " datasets generated, ", need, " needed")
    }
    own <- c(own, list(list(method = "complete", label = "complete", columns = NULL, reps = need)))
    # All cells of a scenario in one pool, in chunks of CHUNK_SIZE
    # replications handed out dynamically: cores of different speed stay
    # busy, and results do not depend on scheduling (one seed per rep).
    tasks <- do.call(rbind, lapply(seq_along(own), function(k) {
      data.frame(cell = k, rep = seq_len(own[[k]]$reps))
    }))
    chunks <- split(seq_len(nrow(tasks)), ceiling(seq_len(nrow(tasks)) / CHUNK_SIZE))
    outs <- parallel::mclapply(chunks, function(idx) {
      lapply(idx, function(i) run_one(spec, own[[tasks$cell[i]]], tasks$rep[i], data_dir))
    }, mc.cores = opts$jobs, mc.preschedule = FALSE)
    died <- vapply(outs, function(o) inherits(o, "try-error") || is.null(o), logical(1))
    if (any(died)) {
      first <- outs[[which(died)[1]]]
      why <- if (inherits(first, "try-error")) one_line(first) else "no result (process killed?)"
      stop(sum(died), " chunk(s) of ", sid, " died: ", why)
    }
    outs <- unlist(unname(outs), recursive = FALSE)
    for (k in seq_along(own)) {
      mth <- own[[k]]$label
      write_outputs(results_dir, spec, mth, outs[tasks$cell == k])
      plan[[paste0(sid, "/", mth)]] <- own[[k]]$reps
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
