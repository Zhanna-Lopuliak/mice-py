# Layer 2 reference at chain level with R's DEFAULT methods: 200 full mice
# chains on a mixed frame of 100 rows, so that make.method gives pmm, logreg,
# polyreg and polr. Columns: x (numeric, complete), y (numeric), b (factor
# no / yes), g (factor a / b / c), o (ordered factor p < q < r < s), all but
# x incomplete, all driven by one latent variable z. g is nominal: a is the
# level of middling z, b of high z, c of low z.
#
# Stored per statistic, over the 200 chains of the last iteration: the mean,
# variance and kurtosis (as in l2_chain.R). The statistics are the mean and
# the variance of the imputed values of y (R's chainMean and chainVar, which
# holds variances), and for each categorical column the share of each level
# but the first among its imputed values.
# Writes tests/reference/fixtures/l2_chain_default.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

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
# y and o MAR on x, b MAR on x (the other way), g MCAR; about 20% each
data$y[runif(n) < plogis(-1.6 + 0.8 * x)] <- NA
data$b[runif(n) < plogis(-1.6 - 0.8 * x)] <- NA
data$g[runif(n) < 0.2] <- NA
data$o[runif(n) < plogis(-1.6 + 0.8 * x)] <- NA
nmis <- colSums(is.na(data))
observed_levels <- lapply(data[c("b", "g", "o")], function(col) table(col))
stopifnot(
  nmis[["x"]] == 0, all(nmis[c("y", "b", "g", "o")] >= 12),
  all(unlist(observed_levels) >= 8)
)

m <- 200
maxit <- 5
seed <- 7802
imp <- mice(data, m = m, maxit = maxit, seed = seed, printFlag = FALSE)
stopifnot(
  is.null(imp$loggedEvents),
  identical(unname(imp$method), c("", "pmm", "logreg", "polyreg", "polr")),
  identical(unname(imp$visitSequence), names(data))
)

stat_entry <- function(column, statistic, values, level = NULL) {
  mo <- moments(matrix(values, ncol = 1))
  list(
    column = column, statistic = statistic, level = level,
    mean = mo$mean[[1]], var = mo$var[[1]], kurt = mo$kurt[[1]]
  )
}

# The imputations of the last iteration are imp$imp[[column]], one column
# per chain.
share_entries <- function(column) {
  lev <- levels(data[[column]])
  draws <- imp$imp[[column]]
  lapply(lev[-1], function(l) {
    stat_entry(column, "share", colMeans(draws == l), level = l)
  })
}

statistics <- c(
  list(
    stat_entry("y", "chain_mean", imp$chainMean["y", maxit, ]),
    stat_entry("y", "chain_var", imp$chainVar["y", maxit, ])
  ),
  share_entries("b"), share_entries("g"), share_entries("o")
)

write_fixture("l2_chain_default", list(
  layer = "L2",
  call = sprintf(
    "mice(data, m = %d, maxit = %d, seed = %d, printFlag = FALSE)",
    m, maxit, seed
  ),
  data = encode_frame(data),
  m = m,
  maxit = maxit,
  method = as.list(imp$method),
  visit_sequence = I(unname(imp$visitSequence)),
  statistics = statistics
))
