# Shared helpers for the reference fixture generators.
#
# Run every script from the repository root, for example
#   Rscript tests/reference/r/generate_all.R
#
# The scripts call functions of the R package mice (exported ones, or
# internals through mice:::). They must not copy mice source code: mice is
# GPL, this repository is MIT.

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
