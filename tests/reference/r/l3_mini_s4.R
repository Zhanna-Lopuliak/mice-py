# L3 in-suite mini simulation, reference side.
#
# Scenario S4 of the L3 study (simulations/scenarios.py): x1, x2 standard
# normal with correlation 0.5, y = 0.5 + x1 + 0.5 x2 + e, n = 50, x1 missing
# completely at random with probability 0.5. Each replication draws a new
# dataset with R's own random numbers and imputes x1 m = 5 times through the
# univariate functions mice.impute.pmm, mice.impute.midastouch and
# mice.impute.norm directly (x1 is the only incomplete column, so this is
# what mice() does with maxit = 1). Analysis: lm(y ~ x1 + x2) on each
# completed dataset.
#
# Stored per method: the mean over replications of the pooled x1 coefficient
# (qbar) and of its between-imputation variance B, each with its Monte Carlo
# standard error sd / sqrt(reps). tests/test_parity/test_l3_mini_simulation.py
# compares 200 Python replications against these numbers.
#
# Reproducibility: the replications run in fixed chunks, each with its own
# L'Ecuyer-CMRG stream derived from one seed, so the fixture does not depend
# on the number of cores. The previous RNG kind is restored at the end,
# because generate_all.R runs every script in the same R session.
#
# Adding a method keeps the numbers of the others. Within a chunk, the first
# pass draws each replication's dataset and imputes it with the FIRST_PASS
# methods (pmm, midastouch: the stream of the original script); every later
# method (norm, ...) then runs in its own pass over the same datasets, in
# the order of METHODS, drawing from the chunk's stream where the previous
# pass stopped. A new method is appended to METHODS, never inserted.
#
# Calls mice rather than copying mice source, as the other scripts do.

source(file.path("tests", "reference", "r", "_common.R"))

local({
  N <- 50L
  M <- 5L
  REPS <- 20000L
  CHUNKS <- 40L
  SEED <- 20261111L
  METHODS <- c("pmm", "midastouch", "norm")
  FIRST_PASS <- c("pmm", "midastouch")
  COLUMNS <- c(paste0(rep(METHODS, each = 2L), c("_qbar", "_b")), "n_obs")

  old_kind <- RNGkind()
  on.exit(RNGkind(old_kind[1], old_kind[2], old_kind[3]), add = TRUE)
  RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
  set.seed(SEED)
  streams <- vector("list", CHUNKS)
  s <- .Random.seed
  for (i in seq_len(CHUNKS)) {
    streams[[i]] <- s
    s <- parallel::nextRNGStream(s)
  }

  impute_fun <- list(
    pmm = mice::mice.impute.pmm,
    midastouch = mice::mice.impute.midastouch,
    norm = mice::mice.impute.norm
  )
  stopifnot(identical(names(impute_fun), METHODS), FIRST_PASS == METHODS[seq_along(FIRST_PASS)])

  draw_dataset <- function() {
    z1 <- rnorm(N)
    z2 <- rnorm(N)
    e <- rnorm(N)
    x1 <- z1
    x2 <- 0.5 * z1 + sqrt(0.75) * z2
    y <- 0.5 + x1 + 0.5 * x2 + e
    ry <- runif(N) >= 0.5 # TRUE = observed
    x1[!ry] <- NA
    list(x1 = x1, x2 = x2, y = y, ry = ry)
  }

  # m imputations of x1 with one method: pooled x1 coefficient and B
  impute_method <- function(d, method) {
    x <- cbind(x2 = d$x2, y = d$y)
    coefs <- numeric(M)
    for (j in seq_len(M)) {
      x1c <- d$x1
      x1c[!d$ry] <- impute_fun[[method]](d$x1, d$ry, x)
      coefs[j] <- stats::.lm.fit(cbind(1, x1c, d$x2), d$y)$coefficients[2]
    }
    c(mean(coefs), stats::var(coefs))
  }

  run_chunk <- function(i) {
    assign(".Random.seed", streams[[i]], envir = globalenv())
    reps <- REPS / CHUNKS
    out <- matrix(NA_real_, reps, length(COLUMNS), dimnames = list(NULL, COLUMNS))
    datasets <- vector("list", reps)
    for (r in seq_len(reps)) {
      datasets[[r]] <- draw_dataset()
      for (method in FIRST_PASS) {
        out[r, paste0(method, c("_qbar", "_b"))] <- impute_method(datasets[[r]], method)
      }
      out[r, "n_obs"] <- sum(datasets[[r]]$ry)
    }
    for (method in setdiff(METHODS, FIRST_PASS)) {
      for (r in seq_len(reps)) {
        out[r, paste0(method, c("_qbar", "_b"))] <- impute_method(datasets[[r]], method)
      }
    }
    out
  }

  cores <- if (.Platform$OS.type == "windows") 1L else max(1L, parallel::detectCores())
  chunks <- parallel::mclapply(seq_len(CHUNKS), run_chunk, mc.cores = cores)
  failed <- vapply(chunks, function(ch) inherits(ch, "try-error"), logical(1))
  if (any(failed)) stop("chunk(s) failed: ", paste(which(failed), collapse = ", "))
  res <- do.call(rbind, chunks)
  stopifnot(nrow(res) == REPS, !anyNA(res))

  summarise <- function(v) list(mean = mean(v), se = stats::sd(v) / sqrt(length(v)), sd = stats::sd(v))
  methods <- lapply(stats::setNames(METHODS, METHODS), function(method) {
    list(
      qbar_x1 = summarise(res[, paste0(method, "_qbar")]),
      b_x1 = summarise(res[, paste0(method, "_b")])
    )
  })

  write_fixture("l3_mini_s4", list(
    description = paste(
      "Scenario S4: n = 50, x1 MCAR 50%, m = 5, univariate mice.impute.* calls,",
      "lm(y ~ x1 + x2); mean over replications of the pooled x1 coefficient",
      "and of its between-imputation variance B, with sd / sqrt(reps)."
    ),
    n = N, m = M, reps = REPS, chunks = CHUNKS, seed = SEED,
    rng = "L'Ecuyer-CMRG, one stream per chunk",
    mean_n_obs = mean(res[, "n_obs"]),
    truth_x1 = 1.0,
    methods = methods
  ))
})
