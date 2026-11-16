# Shared helpers for the reference fixture generators.
#
# Run every script from the repository root, for example
#   Rscript tests/reference/r/generate_all.R
#
# The scripts call functions of the R package mice (exported ones, or
# internals through mice:::) rather than copying mice source code, so that
# the fixtures record what the installed mice computes. mice-py, like mice,
# is licensed GPL (>= 2).

suppressPackageStartupMessages({
  library(mice)
  library(jsonlite)
})

MICE_VERSION <- "3.19.0"
stopifnot(
  "run from the repository root" = file.exists("pyproject.toml"),
  "fixtures are pinned to mice 3.19.0" =
    as.character(packageVersion("mice")) == MICE_VERSION
)

# R's default generators, stated explicitly so that set.seed() in a script
# gives the same stream whatever ran before it.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

FIXTURE_DIR <- file.path("tests", "reference", "fixtures")
dir.create(FIXTURE_DIR, showWarnings = FALSE, recursive = TRUE)

# JSON has no literal for Inf or NaN, and jsonlite writes both as null, the
# same as NA. Encode them as the strings "Inf", "-Inf" and "NaN" instead, so
# that a reader can tell them apart from NA. NumPy converts these strings when
# an array is built with dtype=float (see as_array in
# tests/test_parity/_helpers.py). Vectors without such values are untouched.
encode_nonfinite <- function(x) {
  if (is.data.frame(x)) {
    return(x)
  }
  if (is.list(x)) {
    out <- lapply(x, encode_nonfinite)
    attributes(out) <- attributes(x)
    return(out)
  }
  if (!is.double(x) || !any(is.infinite(x) | is.nan(x))) {
    return(x)
  }
  one <- function(v) {
    if (is.nan(v) || is.infinite(v)) as.character(v) else v
  }
  if (is.matrix(x)) {
    return(lapply(seq_len(nrow(x)), function(i) lapply(unname(x[i, ]), one)))
  }
  if (length(x) == 1L) {
    return(one(unname(x)))
  }
  lapply(unname(x), one)
}

# Write one fixture as JSON. Doubles are written with 17 significant digits,
# which reproduces every value bit for bit when read back. Missing values
# become null; Inf, -Inf and NaN become strings (see encode_nonfinite). No
# timestamp is stored, so regenerating gives identical files.
write_fixture <- function(name, payload) {
  payload$meta <- list(
    r = R.version.string,
    mice = as.character(packageVersion("mice"))
  )
  json <- toJSON(
    encode_nonfinite(payload),
    digits = I(17), auto_unbox = TRUE, na = "null", null = "null",
    pretty = TRUE
  )
  writeLines(json, file.path(FIXTURE_DIR, paste0(name, ".json")))
  invisible(name)
}

# Encode a data frame column by column, so that the reader can rebuild it
# exactly: numeric columns as numbers (NA as null), factors as strings plus
# their level order and whether they are ordered. Character columns are kept
# as strings. Read it back with decode_frame() in tests/test_parity/_helpers.py.
encode_frame <- function(df) {
  df <- as.data.frame(df)
  is_fac <- vapply(df, is.factor, logical(1))
  data <- lapply(df, function(col) {
    if (is.factor(col)) I(as.character(col)) else I(as.vector(col))
  })
  levels <- lapply(df[is_fac], function(col) I(levels(col)))
  ordered <- lapply(df[is_fac], is.ordered)
  if (!any(is_fac)) {
    levels <- structure(list(), names = character(0))
    ordered <- structure(list(), names = character(0))
  }
  list(
    columns = I(names(df)),
    nrow = nrow(df),
    data = data,
    levels = levels,
    ordered = ordered
  )
}

# ---------------------------------------------------------------------------
# Recipes for post and passive imputation
# ---------------------------------------------------------------------------
#
# R's post is R code that sampler() evaluates after each univariate
# imputation of a variable j, with the imputations of chain i in
# imp[[j]][, i] (one per where-cell) and the working data in `data`; a
# passive variable has a formula as its method. mice-py takes functions
# instead: post={column: f(imp, data)}, passive={column: f(data)}. A fixture
# cannot hold functions, so a case names a recipe with its parameters, and
# each side builds it: post_command() and passive_formula() here,
# post_function() and passive_function() in tests/test_parity/_helpers.py.
#
#   post_bounds(lower, upper)    squeeze(imp, c(lower, upper))   imp.clip(lower, upper)
#   post_shift(delta)            imp + delta                     imp + delta
#   post_rule(column, below, value)
#                                value where `column` of the data is below `below`, in the rows of the
#                                where-cells                     imp.mask(data[column] < below, value)
#   post_fill_missing(value)     value where imp is NA           imp.fillna(value)
#
#   passive_bmi(weight, height)  ~ I(weight / (height / 100)^2)
#   passive_product(left, right) ~ I(left * right)
#   passive_sum(left, right)     ~ I(left + right)
#   passive_multiple(column, factor)
#                                ~ I(factor * column)
#   passive_centred_square(column)
#                                ~ I((column - mean(column))^2); R evaluates the formula on the rows of
#                                the where-cells only, so its mean is theirs; mice-py passes all rows

recipe <- function(name, ...) list(recipe = name, ...)

post_bounds <- function(lower, upper) recipe("bounds", lower = lower, upper = upper)
post_shift <- function(delta) recipe("shift", delta = delta)
post_rule <- function(column, below, value) recipe("rule", column = column, below = below, value = value)
post_fill_missing <- function(value) recipe("fill_missing", value = value)

passive_bmi <- function(weight, height) recipe("bmi", weight = weight, height = height)
passive_product <- function(left, right) recipe("product", left = left, right = right)
passive_sum <- function(left, right) recipe("sum", left = left, right = right)
passive_multiple <- function(column, factor) recipe("multiple", column = column, factor = factor)
passive_centred_square <- function(column) recipe("centred_square", column = column)

# A number in R code, read back to the same double
r_number <- function(x) sprintf("%.17g", x)

# R's post command for a recipe. It indexes the where-cells of j through
# where[, j] (sampler() has no wy of j at that point)
post_command <- function(r) {
  switch(r$recipe,
    bounds = sprintf("imp[[j]][, i] <- squeeze(imp[[j]][, i], c(%s, %s))", r_number(r$lower), r_number(r$upper)),
    shift = sprintf("imp[[j]][, i] <- imp[[j]][, i] + %s", r_number(r$delta)),
    rule = sprintf("imp[[j]][data$%s[where[, j]] < %s, i] <- %s", r$column, r_number(r$below), r_number(r$value)),
    fill_missing = sprintf("imp[[j]][is.na(imp[[j]][, i]), i] <- %s", r_number(r$value)),
    stop("unknown post recipe: ", r$recipe)
  )
}

# R's passive method for a recipe
passive_formula <- function(r) {
  switch(r$recipe,
    bmi = sprintf("~ I(%s / (%s / 100)^2)", r$weight, r$height),
    product = sprintf("~ I(%s * %s)", r$left, r$right),
    sum = sprintf("~ I(%s + %s)", r$left, r$right),
    multiple = sprintf("~ I(%s * %s)", r_number(r$factor), r$column),
    centred_square = sprintf("~ I((%s - mean(%s))^2)", r$column, r$column),
    stop("unknown passive recipe: ", r$recipe)
  )
}

# The arguments of mice() for `args`, which may hold post and passive as
# recipes (a list per variable): the passive formulas go into `method`, one
# entry per variable in data order (a single method is given to every
# variable first; methods by name must name every variable that is not
# passive, so that mice-py, given them as a dict, has no default to add),
# and the post commands into a post vector (make.post())
with_recipes <- function(args, data) {
  passive <- args$passive
  post <- args$post
  args$passive <- NULL
  args$post <- NULL
  if (!is.null(passive)) {
    method <- args$method
    stopifnot("a case with passive variables names the methods" = !is.null(method))
    if (is.null(names(method))) method <- setNames(rep(method, ncol(data)), names(data))
    stopifnot(
      "the methods and the passive variables name every variable" =
        setequal(union(names(method), names(passive)), names(data))
    )
    method <- setNames(method[names(data)], names(data))
    for (j in names(passive)) method[[j]] <- passive_formula(passive[[j]])
    args$method <- method
  }
  if (!is.null(post)) {
    commands <- make.post(data)
    for (j in names(post)) commands[[j]] <- post_command(post[[j]])
    args$post <- commands
  }
  args
}
