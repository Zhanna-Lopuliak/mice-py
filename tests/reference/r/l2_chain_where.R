# Layer 2 reference at chain level under where = "all": 200 full mice chains
# with R's default methods on the mixed frame of l2_chain_default.R (the
# same code and seed: x numeric and complete, y numeric, b factor no / yes,
# g factor a / b / c, o ordered factor p < q < r < s). Every cell is a
# where-cell, so x gets pmm too, and every observed value is replaced by an
# imputation in the completed data while the chains predict from the
# observed values.
#
# Stored per statistic, over the 200 chains of the last iteration: the mean,
# variance and kurtosis (as in l2_chain_default.R). The statistics are taken
# over the 100 where-cells of each column: the mean and the variance of the
# imputed values of x and y (R's chainMean and chainVar), and for each
# categorical column the share of each level but the first among its
# imputed values.
# Writes tests/reference/fixtures/l2_chain_where.json.
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

m <- 200
maxit <- 5
seed <- 7805
imp <- mice(data, m = m, maxit = maxit, where = "all", seed = seed, printFlag = FALSE)
stopifnot(
  is.null(imp$loggedEvents),
  identical(unname(imp$method), c("pmm", "pmm", "logreg", "polyreg", "polr")),
  identical(unname(imp$visitSequence), names(data)),
  all(vapply(imp$imp, nrow, integer(1)) == n),
  !anyNA(unlist(lapply(imp$imp, unlist)))
)

stat_entry <- function(column, statistic, values, level = NULL) {
  mo <- moments(matrix(values, ncol = 1))
  list(
    column = column, statistic = statistic, level = level,
    mean = mo$mean[[1]], var = mo$var[[1]], kurt = mo$kurt[[1]]
  )
}

# The imputations of the last iteration are imp$imp[[column]], one column
# per chain, one row per where-cell.
share_entries <- function(column) {
  lev <- levels(data[[column]])
  draws <- imp$imp[[column]]
  lapply(lev[-1], function(l) {
    stat_entry(column, "share", colMeans(draws == l), level = l)
  })
}

statistics <- c(
  list(
    stat_entry("x", "chain_mean", imp$chainMean["x", maxit, ]),
    stat_entry("x", "chain_var", imp$chainVar["x", maxit, ]),
    stat_entry("y", "chain_mean", imp$chainMean["y", maxit, ]),
    stat_entry("y", "chain_var", imp$chainVar["y", maxit, ])
  ),
  share_entries("b"), share_entries("g"), share_entries("o")
)

write_fixture("l2_chain_where", list(
  layer = "L2",
  call = sprintf(
    "mice(data, m = %d, maxit = %d, where = \"all\", seed = %d, printFlag = FALSE)",
    m, maxit, seed
  ),
  data = encode_frame(data),
  m = m,
  maxit = maxit,
  where = "all",
  method = as.list(imp$method),
  visit_sequence = I(unname(imp$visitSequence)),
  statistics = statistics
))
