# L1 fixture for the default methods by column type.
#
#   l1_make_method.json   make.method(data) and make.method(data,
#                         defaultMethod = ...) on a frame with one column of
#                         each type that assign.method tells apart
#
# Python target: imputation.make_method (and MICE.impute with method=None).
# R counts the declared levels of a factor (nlevels), unused ones included.
# A character column gets the first default whatever its values; mice-py
# treats it as an unordered factor of its distinct values instead.

source(file.path("tests", "reference", "r", "_common.R"))

n <- 12
na_at <- function(x, rows) {
  x[rows] <- NA
  x
}
lev3 <- c("a", "b", "c")
data <- data.frame(
  num = na_at(seq(0.5, 6, by = 0.5), 2),
  int01 = na_at(rep(0:1, length.out = n), 3),
  lgl = na_at(rep(c(TRUE, FALSE, FALSE), length.out = n), 4),
  f2 = na_at(factor(rep(c("no", "yes"), length.out = n)), 5),
  f3 = na_at(factor(rep(lev3, length.out = n), levels = lev3), 6),
  o2 = na_at(factor(rep(c("lo", "hi"), length.out = n), levels = c("lo", "hi"), ordered = TRUE), 7),
  o4 = na_at(factor(rep(c("p", "q", "r", "s"), length.out = n),
    levels = c("p", "q", "r", "s"), ordered = TRUE
  ), 8),
  f3_unused = na_at(factor(rep(c("a", "c"), length.out = n), levels = lev3), 9),
  f1 = na_at(factor(rep("only", n)), 10),
  chr = na_at(rep(c("u", "v", "w"), length.out = n), 11),
  complete = seq_len(n) / 4,
  stringsAsFactors = FALSE
)
stopifnot(
  is.integer(data$int01), is.logical(data$lgl), is.character(data$chr),
  nlevels(data$f3_unused) == 3, length(unique(na.omit(data$f3_unused))) == 2,
  nlevels(data$f1) == 1, !anyNA(data$complete)
)

other_default <- c("norm", "logreg.boot", "cart", "polyreg")

write_fixture("l1_make_method", list(
  description = paste(
    "make.method(data) and make.method(data, defaultMethod = other_default)",
    "on one column of each type assign.method tells apart. types holds",
    "class(column)[1]; the Python test builds integer columns as Int64,",
    "logical ones as boolean, factors as Categoricals."
  ),
  data = encode_frame(data),
  types = lapply(data, function(col) class(col)[1]),
  default_method = I(c("pmm", "logreg", "polyreg", "polr")),
  method = as.list(make.method(data)),
  other_default_method = I(other_default),
  method_other = as.list(make.method(data, defaultMethod = other_default))
))
