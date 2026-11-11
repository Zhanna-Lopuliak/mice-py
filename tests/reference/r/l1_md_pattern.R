# L1 fixture for md.pattern().
#
#   l1_md_pattern.json   md.pattern(data, plot = FALSE) on nhanes, boys and a
#                        synthetic frame
#
# Python target: plotting.utils.md_pattern_like.
# md.pattern() sorts columns by their number of missing values (ties keep
# data order) and rows by the 0/1 missingness pattern as a string, so
# patterns with the same number of missing cells are not grouped together.

source(file.path("tests", "reference", "r", "_common.R"))

# Synthetic frame: several patterns with one and with two missing cells, in
# frequencies that differ from the string order of the patterns.
set.seed(20261119)
n <- 60
synthetic <- data.frame(
  w = rnorm(n), x = rnorm(n), y = rnorm(n), z = rnorm(n)
)
synthetic$w[1:2] <- NA # 2 rows missing w only
synthetic$x[3:7] <- NA # 5 rows missing x only
synthetic$y[8:10] <- NA # 3 rows missing y only
synthetic$z[11:14] <- NA # 4 rows missing z only
synthetic[15:20, c("w", "z")] <- NA # 6 rows missing w and z
synthetic[21, c("x", "y")] <- NA # 1 row missing x and y
synthetic[22:24, c("w", "x", "y")] <- NA # 3 rows missing w, x, y

pattern_case <- function(name, data) {
  r <- md.pattern(data, plot = FALSE)
  list(
    name = name,
    data = encode_frame(data),
    rownames = I(rownames(r)),
    colnames = I(colnames(r)),
    table = unname(r)
  )
}

write_fixture("l1_md_pattern", list(
  description = paste(
    "md.pattern(data, plot = FALSE). table includes the margins: the last",
    "column holds the number of missing cells per pattern, the last row the",
    "number of missing values per column. rownames are the pattern counts;",
    "the last row name and the last column name are empty."
  ),
  cases = list(
    pattern_case("nhanes", nhanes),
    pattern_case("boys", boys),
    pattern_case("synthetic", synthetic)
  )
))
