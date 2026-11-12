# L1 fixture for the augmentation of categorical imputation models.
#
#   l1_augment.json   mice:::augment(y, ry, x, wy), the rows that
#                     White, Daniel and Royston (2010) add before a
#                     logistic or multinomial fit
#
# Python target: imputation.logreg.augment, called with the integer codes of
# y as R sees them (unclass(y): the declared level codes of a factor, the
# values of a numeric or logical target), and imputation.logreg.
# target_categories for the level order R leaves the factor in.
#
# No random numbers are involved. Each case records the inputs, the added
# rows (x, y, weights) and the returned ry and wy; the script checks the
# properties the Python test relies on (clamping binds where it should, the
# skip rules return the input, the level order after augmentation).

source(file.path("tests", "reference", "r", "_common.R"))

augment <- mice:::augment

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

set.seed(7201)
n <- 12
x1 <- rnorm(n)
x2 <- runif(n)
x12 <- cbind(x1, x2)
mis <- c(3L, 6L, 10L)
ry <- !(seq_len(n) %in% mis)
labels_of <- function(k) letters[seq_len(k)]
# every level occurs among the observed rows; the missing cells hold current
# imputations, as inside mice
draw_labels <- function(k) {
  repeat {
    v <- sample(labels_of(k), n, replace = TRUE)
    if (all(labels_of(k) %in% v[ry])) {
      return(v)
    }
  }
}
lab2 <- draw_labels(2)
lab3 <- draw_labels(3)
lab4 <- draw_labels(4)
bin <- as.numeric(lab2 == "b")
# a rare dummy: mean - sd / 2 falls below its minimum 0
dummy <- as.numeric(seq_len(n) %in% c(4L, 9L))

inputs <- list(
  levels2 = list(y = factor(lab2), ry = ry, x = x12),
  levels3 = list(y = factor(lab3), ry = ry, x = x12),
  levels4 = list(y = factor(lab4), ry = ry, x = x12),
  numeric01 = list(y = bin, ry = ry, x = x12),
  logical = list(y = bin == 1, ry = ry, x = x12),
  rare_dummy = list(y = factor(lab2), ry = ry, x = cbind(x1, dummy)),
  no_predictor = list(y = factor(lab2), ry = ry, x = x12[, 0, drop = FALSE]),
  one_missing = list(y = factor(lab2), ry = seq_len(n) != mis[1], x = x12),
  # declared levels in reverse order: augment sorts them
  unsorted_labels = list(y = factor(ifelse(lab2 == "a", "yes", "no"), levels = c("yes", "no")), ry = ry, x = x12),
  unused_unordered = list(y = factor(lab2, levels = c("a", "b", "c")), ry = ry, x = x12),
  unused_ordered = list(
    y = factor(ifelse(lab2 == "a", "lo", "hi"), levels = c("lo", "mid", "hi"), ordered = TRUE), ry = ry, x = x12
  ),
  # "c" only in a cell that is not observed (a current imputation)
  imputed_only = list(y = factor(replace(lab2, mis[2], "c")), ry = ry, x = x12),
  # the cells to impute hold NA (a direct call), and only two of the three are imputed
  na_cells_wy_subset = list(
    y = factor(replace(lab3, mis, NA)), ry = ry, x = x12, wy = seq_len(n) %in% mis[c(1, 3)]
  ),
  too_many = list(y = factor(sprintf("l%02d", seq_len(51))), ry = seq_len(51) > 3, x = cbind(seq_len(51) / 10))
)

# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

kind_of <- function(y) {
  if (is.ordered(y)) "ordered" else if (is.factor(y)) "factor" else if (is.logical(y)) "logical" else "numeric"
}
as_labels <- function(y) if (is.factor(y)) I(as.character(y)) else I(as.vector(y))

augment_case <- function(name) {
  inp <- inputs[[name]]
  y <- inp$y
  x <- inp$x
  ry <- inp$ry
  wy <- if (is.null(inp$wy)) !ry else inp$wy
  p <- ncol(x)
  n_rows <- length(y)
  aug <- tryCatch(augment(y, ry, x, wy), error = function(e) e)
  failed <- inherits(aug, "error")
  out <- if (!failed) {
    k <- length(sort(unique(unclass(y))))
    skipped <- p == 0 || sum(!ry) == 1
    n_added <- if (skipped) 0L else 2L * p * k
    stopifnot(
      length(aug$y) == n_rows + n_added, nrow(aug$x) == n_rows + n_added,
      identical(aug$ry, c(ry, rep(TRUE, n_added))), identical(aug$wy, c(wy, rep(FALSE, n_added))),
      identical(aug$w, c(rep(1, n_rows), rep((p + 1) / max(n_added, 1L), n_added)))
    )
    if (skipped) {
      stopifnot(identical(aug$y, y), identical(aug$x, x))
    }
    # level order after augmentation: declared for an ordered factor, the
    # sorted labels present otherwise
    if (is.ordered(y)) {
      stopifnot(identical(levels(aug$y), levels(y)))
    } else if (is.factor(y) && !skipped) {
      stopifnot(identical(levels(aug$y), sort(unique(as.character(y[!is.na(y)])))))
    }
    list(
      y = as_labels(aug$y),
      levels = if (is.factor(aug$y)) I(levels(aug$y)) else NULL,
      ry = I(aug$ry),
      wy = I(aug$wy),
      x = if (p > 0) unname(as.matrix(aug$x)) else NULL,
      w = I(aug$w)
    )
  }
  list(
    name = name,
    kind = kind_of(y),
    y = as_labels(y),
    levels = if (is.factor(y)) I(levels(y)) else NULL,
    ry = I(ry),
    wy = if (is.null(inp$wy)) NULL else I(inp$wy),
    x = if (p > 0) unname(x) else NULL,
    p = p,
    augmented = out,
    r_error = if (failed) conditionMessage(aug) else NULL
  )
}

cases <- lapply(names(inputs), augment_case)
names(cases) <- names(inputs)

# the clamp binds for the rare dummy: mean - sd / 2 < 0 = its minimum
d_mean <- mean(dummy)
d_sd <- sd(dummy)
stopifnot(d_mean - 0.5 * d_sd < min(dummy), min(cases$rare_dummy$augmented$x[, 2]) == 0)
# and not for the other columns of that case
stopifnot(all(abs(cases$rare_dummy$augmented$x[-seq_len(n), 1] - mean(x1)) <= 0.5 * sd(x1) + 1e-15))
stopifnot(identical(cases$too_many$r_error, "Maximum number of categories (50) exceeded"))
stopifnot(is.null(cases$too_many$augmented))

write_fixture("l1_augment", list(
  description = paste(
    "mice:::augment(y, ry, x, wy) on small inputs. Per case: kind (factor,",
    "ordered, numeric, logical), y (labels; null for NA), levels (declared,",
    "factors only), ry, wy (null means !ry), x without intercept (rows; null",
    "when p = 0), and augmented: the returned y (labels), levels (factors",
    "only), ry, wy, x and w, the input rows first. r_error holds R's error",
    "message where augment stops. The Python side passes the codes R sees",
    "(unclass(y), 0-based; the values of a numeric or logical y)."
  ),
  cases = unname(cases)
))
