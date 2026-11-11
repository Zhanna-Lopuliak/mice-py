# Layer 2 reference at chain level: 200 full mice chains with pmm on a
# synthetic frame of 100 rows and three incomplete numeric columns. Stores,
# per variable, the mean, variance and kurtosis across the 200 chains of the
# last-iteration chainMean and chainVar (chainVar holds variances).
# Writes tests/reference/fixtures/l2_chain.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

set.seed(6001)
n <- 100
sigma <- matrix(c(1, 0.6, 0.4, 0.6, 1, 0.5, 0.4, 0.5, 1), 3)
z <- matrix(rnorm(n * 3), n) %*% chol(sigma)
data <- data.frame(
  y1 = round(z[, 1], L2_DIGITS),
  y2 = round(1 + 0.8 * z[, 2], L2_DIGITS),
  y3 = round(exp(0.5 * z[, 3]), L2_DIGITS)
)
# y1 MCAR, y2 and y3 MAR on y1
miss1 <- runif(n) < 0.2
miss2 <- runif(n) < plogis(-1.2 + 1.0 * data$y1)
miss3 <- runif(n) < plogis(-1.2 - 1.0 * data$y1)
all_missing <- miss1 & miss2 & miss3
miss1[all_missing] <- FALSE # the port drops all-empty rows (finding C4)
data$y1[miss1] <- NA
data$y2[miss2] <- NA
data$y3[miss3] <- NA
stopifnot(all(rowSums(is.na(data)) < 3), all(colSums(is.na(data)) > 0))

m <- 200
maxit <- 5
seed <- 6002
imp <- mice(data, m = m, maxit = maxit, method = "pmm", seed = seed,
            printFlag = FALSE)
stopifnot(is.null(imp$loggedEvents))
stopifnot(identical(unname(imp$visitSequence), names(data)))

chain_stats <- function(arr) {
  last <- arr[, maxit, , drop = TRUE] # variables x chains
  out <- lapply(names(data), function(v) {
    mo <- moments(matrix(last[v, ], ncol = 1))
    list(mean = mo$mean[[1]], var = mo$var[[1]], kurt = mo$kurt[[1]])
  })
  names(out) <- names(data)
  out
}

write_fixture("l2_chain", list(
  layer = "L2",
  call = sprintf(
    "mice(data, m = %d, maxit = %d, method = \"pmm\", seed = %d, printFlag = FALSE)",
    m, maxit, seed
  ),
  data = as.list(data),
  columns = I(names(data)),
  m = m,
  maxit = maxit,
  visit_sequence = I(unname(imp$visitSequence)),
  chain_mean = chain_stats(imp$chainMean),
  chain_var = chain_stats(imp$chainVar)
))
