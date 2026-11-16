# Layer 2 reference at chain level with ignore: 200 full mice chains with
# R's default methods (pmm, logreg, polyreg, polr) on the mixed frame of
# l2_chain_default.R (the same code and seed: x numeric and complete, y
# numeric, b factor no / yes, g factor a / b / c, o ordered factor
# p < q < r < s; 100 rows) with 40 added rows that are ignored. The added
# rows come from a clearly shifted distribution: given the latent z, y is 2
# higher, b is "yes" less often, g is "c" more often and o lower, and z
# itself is higher. They have missing cells too, which are imputed from
# models fitted on the first 100 rows only; fitted on all rows, the
# imputations of both groups of rows would move.
#
# Stored per statistic, over the 200 chains of the last iteration: the mean,
# variance and kurtosis (as in l2_chain_default.R). The statistics are taken
# separately over the imputed cells of the rows that are not ignored
# ("kept") and of the ignored rows ("ignored"): the mean and the variance of
# the imputed values of y, and for each categorical column the share of each
# level but the first among its imputed values.
# Writes tests/reference/fixtures/l2_chain_ignore.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

# The frame of l2_chain_default.R
set.seed(7801)
n <- 100
z <- rnorm(n)
x <- round(z + rnorm(n, sd = 0.6), L2_DIGITS)
y <- round(1 + 0.8 * z + rnorm(n, sd = 0.6), L2_DIGITS)
b <- factor(ifelse(runif(n) < plogis(0.3 + 1.5 * z), "yes", "no"), levels = c("no", "yes"))
eta <- cbind(0, -0.3 + 1.5 * z, -0.3 - 1.5 * z)
p <- exp(eta) / rowSums(exp(eta))
g <- factor(apply(p, 1, function(pr) sample(c("a", "b", "c"), 1, prob = pr)), levels = c("a", "b", "c"))
o <- cut(z + rnorm(n, sd = 0.5), c(-Inf, -0.7, 0, 0.7, Inf), labels = c("p", "q", "r", "s"), ordered_result = TRUE)
data <- data.frame(x = x, y = y, b = b, g = g, o = o)
data$y[runif(n) < plogis(-1.6 + 0.8 * x)] <- NA
data$b[runif(n) < plogis(-1.6 - 0.8 * x)] <- NA
data$g[runif(n) < 0.2] <- NA
data$o[runif(n) < plogis(-1.6 + 0.8 * x)] <- NA
stopifnot(colSums(is.na(data))[["x"]] == 0)

# The 40 shifted rows, about 30% of each incomplete column missing (MCAR)
set.seed(7821)
n_ignored <- 40
zs <- rnorm(n_ignored, mean = 1)
shifted <- data.frame(
  x = round(zs + rnorm(n_ignored, sd = 0.6), L2_DIGITS),
  y = round(3 + 0.8 * zs + rnorm(n_ignored, sd = 0.6), L2_DIGITS),
  b = factor(ifelse(runif(n_ignored) < plogis(-1.5 + 1.5 * zs), "yes", "no"), levels = c("no", "yes"))
)
eta <- cbind(0, -0.3 + 1.5 * zs, 1.5 - 1.5 * zs)
p <- exp(eta) / rowSums(exp(eta))
shifted$g <- factor(apply(p, 1, function(pr) sample(c("a", "b", "c"), 1, prob = pr)), levels = c("a", "b", "c"))
shifted$o <- cut(zs - 1.2 + rnorm(n_ignored, sd = 0.5), c(-Inf, -0.7, 0, 0.7, Inf),
  labels = c("p", "q", "r", "s"), ordered_result = TRUE
)
for (column in c("y", "b", "g", "o")) {
  shifted[[column]][runif(n_ignored) < 0.3] <- NA
}
data <- rbind(data, shifted)
ignore <- rep(c(FALSE, TRUE), c(n, n_ignored))
stopifnot(all(colSums(is.na(data[ignore, c("y", "b", "g", "o")])) >= 8))

m <- 200
maxit <- 5
seed <- 7822
imp <- mice(data, m = m, maxit = maxit, ignore = ignore, seed = seed, printFlag = FALSE)
stopifnot(
  is.null(imp$loggedEvents),
  identical(unname(imp$method), c("", "pmm", "logreg", "polyreg", "polr")),
  identical(unname(imp$visitSequence), names(data))
)

stat_entry <- function(column, statistic, rows, values, level = NULL) {
  mo <- moments(matrix(values, ncol = 1))
  stopifnot(mo$var[[1]] > 0)
  list(
    column = column, statistic = statistic, rows = rows, level = level,
    mean = mo$mean[[1]], var = mo$var[[1]], kurt = mo$kurt[[1]]
  )
}

# The imputations of the last iteration of one column in the rows that are
# not ignored ("kept") or in the ignored rows, one column per chain
imputations <- function(column, rows) {
  ignored <- ignore[imp$where[, column]]
  imp$imp[[column]][ignored == (rows == "ignored"), , drop = FALSE]
}

entries <- function(rows) {
  values <- as.matrix(imputations("y", rows))
  out <- list(
    stat_entry("y", "mean", rows, colMeans(values)),
    stat_entry("y", "var", rows, apply(values, 2, var))
  )
  for (column in c("b", "g", "o")) {
    draws <- imputations(column, rows)
    for (l in levels(data[[column]])[-1]) {
      out[[length(out) + 1L]] <- stat_entry(column, "share", rows, colMeans(draws == l), level = l)
    }
  }
  out
}

statistics <- c(entries("kept"), entries("ignored"))

write_fixture("l2_chain_ignore", list(
  layer = "L2",
  call = sprintf(
    "mice(data, m = %d, maxit = %d, ignore = ignore, seed = %d, printFlag = FALSE)",
    m, maxit, seed
  ),
  data = encode_frame(data),
  ignore = I(ignore),
  m = m,
  maxit = maxit,
  method = as.list(imp$method),
  visit_sequence = I(unname(imp$visitSequence)),
  statistics = statistics
))
