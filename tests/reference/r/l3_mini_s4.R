# L3 in-suite mini simulation, reference side.
#
# Scenario S4 of the L3 study (simulations/scenarios.py): x1, x2 standard
# normal with correlation 0.5, y = 0.5 + x1 + 0.5 x2 + e, n = 50, x1 missing
# completely at random with probability 0.5. Each replication draws a new
# dataset with R's own random numbers and imputes x1 m = 5 times through the
# univariate functions mice.impute.pmm and mice.impute.midastouch directly
# (x1 is the only incomplete column, so this is what mice() does with
# maxit = 1). Analysis: lm(y ~ x1 + x2) on each completed dataset.
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
# Calls mice rather than copying mice source, as the other scripts do.

source(file.path("tests", "reference", "r", "_common.R"))

local({
  N <- 50L
  M <- 5L
  REPS <- 20000L
  CHUNKS <- 40L
  SEED <- 20261111L
  METHODS <- c("pmm", "midastouch")

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

  impute_fun <- list(pmm = mice::mice.impute.pmm, midastouch = mice::mice.impute.midastouch)

  one_replication <- function() {
    z1 <- rnorm(N)
    z2 <- rnorm(N)
    e <- rnorm(N)
    x1 <- z1
    x2 <- 0.5 * z1 + sqrt(0.75) * z2
    y <- 0.5 + x1 + 0.5 * x2 + e
    ry <- runif(N) >= 0.5 # TRUE = observed
    x1[!ry] <- NA
    x <- cbind(x2 = x2, y = y)
    out <- numeric(0)
    for (method in METHODS) {
      coefs <- numeric(M)
      for (j in seq_len(M)) {
        x1c <- x1
        x1c[!ry] <- impute_fun[[method]](x1, ry, x)
        coefs[j] <- stats::.lm.fit(cbind(1, x1c, x2), y)$coefficients[2]
      }
      out[paste0(method, "_qbar")] <- mean(coefs)
      out[paste0(method, "_b")] <- stats::var(coefs)
    }
    out[["n_obs"]] <- sum(ry)
    out
  }

  run_chunk <- function(i) {
    assign(".Random.seed", streams[[i]], envir = globalenv())
    t(vapply(seq_len(REPS / CHUNKS), function(r) one_replication(), numeric(5)))
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
