# Layer 2 reference at chain level with a passive variable and post: 200
# full mice chains on a numeric frame of 100 rows. x is complete; y1 and y2
# are incomplete (MAR on x) and imputed by pmm; p = y1 * y2 is passive
# (~ I(y1 * y2), missing where y1 or y2 is) and does not predict y1 and y2;
# y3 is incomplete (MAR on x), depends on p (the product, beyond y1 and y2)
# and is imputed by pmm from x, y1, y2 and p, with bounds by post
# (squeeze(imp, c(2.5, 8))) that bind for a visible share of its
# imputations. The recipes of post and passive are those of _common.R.
#
# Stored per statistic, over the 200 chains of the last iteration: the mean,
# variance and kurtosis (as in l2_chain_default.R). The statistics are R's
# chainMean and chainVar of y1, y2, p and y3 (over their where-cells, after
# post), and the share of the imputations of y3 at the lower bound and at
# the upper bound.
# Writes tests/reference/fixtures/l2_chain_passive.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

set.seed(7841)
n <- 100
x <- round(rnorm(n), L2_DIGITS)
y1 <- round(2 + 0.6 * x + rnorm(n, sd = 0.6), L2_DIGITS)
y2 <- round(1.5 + 0.4 * x + rnorm(n, sd = 0.5), L2_DIGITS)
y3 <- round(1 + 1.2 * y1 * y2 + rnorm(n, sd = 1), L2_DIGITS)
data <- data.frame(x = x, y1 = y1, y2 = y2, p = y1 * y2, y3 = y3)
data$y1[runif(n) < plogis(-1.4 + 0.8 * x)] <- NA
data$y2[runif(n) < plogis(-1.4 - 0.8 * x)] <- NA
data$p[is.na(data$y1) | is.na(data$y2)] <- NA
data$y3[runif(n) < plogis(-1.2 + 0.8 * x)] <- NA
stopifnot(colSums(is.na(data))[["x"]] == 0, all(colSums(is.na(data[c("y1", "y2", "p", "y3")])) >= 12))

passive <- list(p = passive_product("y1", "y2"))
post <- list(y3 = post_bounds(2.5, 8))
pred <- make.predictorMatrix(data)
pred[c("y1", "y2"), "p"] <- 0
args <- with_recipes(list(method = c(x = "", y1 = "pmm", y2 = "pmm", y3 = "pmm"), passive = passive, post = post), data)

m <- 200
maxit <- 5
seed <- 7842
imp <- do.call(mice, c(
  list(data, m = m, maxit = maxit, predictorMatrix = pred, seed = seed, printFlag = FALSE), args
))
stopifnot(
  is.null(imp$loggedEvents),
  identical(unname(imp$method), c("", "pmm", "pmm", "~ I(y1 * y2)", "pmm")),
  identical(unname(imp$visitSequence[imp$method[imp$visitSequence] != ""]), c("y1", "y2", "y3", "p"))
)

stat_entry <- function(column, statistic, values) {
  mo <- moments(matrix(values, ncol = 1))
  stopifnot(mo$var[[1]] > 0)
  list(
    column = column, statistic = statistic,
    mean = mo$mean[[1]], var = mo$var[[1]], kurt = mo$kurt[[1]]
  )
}

# The imputations of the last iteration are imp$imp[[column]], one column
# per chain, one row per where-cell
at_bound <- function(bound) colMeans(imp$imp$y3 == bound)

statistics <- c(
  unlist(lapply(c("y1", "y2", "p", "y3"), function(column) {
    list(
      stat_entry(column, "chain_mean", imp$chainMean[column, maxit, ]),
      stat_entry(column, "chain_var", imp$chainVar[column, maxit, ])
    )
  }), recursive = FALSE),
  list(
    stat_entry("y3", "share_lower", at_bound(post$y3$lower)),
    stat_entry("y3", "share_upper", at_bound(post$y3$upper))
  )
)

write_fixture("l2_chain_passive", list(
  layer = "L2",
  call = sprintf(
    paste(
      "mice(data, m = %d, maxit = %d, method = method, predictorMatrix = pred, post = post,",
      "seed = %d, printFlag = FALSE) with method[\"p\"] = \"%s\" and post[\"y3\"] = \"%s\""
    ),
    m, maxit, seed, passive_formula(passive$p), post_command(post$y3)
  ),
  data = encode_frame(data),
  m = m,
  maxit = maxit,
  passive = passive,
  post = post,
  predictor_matrix = list(rows = I(rownames(pred)), cols = I(colnames(pred)), values = unname(pred)),
  method = as.list(imp$method),
  visit_sequence = I(unname(imp$visitSequence)),
  statistics = statistics
))
