# Layer 2 reference for mice.impute.cart: 20,000 calls per input after one
# set.seed, default arguments (minbucket = 5, cp = 1e-4; rpart then uses
# minsplit = 15). Writes tests/reference/fixtures/l2_cart.json.
#
# rpart is deterministic, so for a numeric target only the draw within the
# leaf is random. `reached_2000` records, per missing cell, which donors
# were drawn at least once in the first 2,000 calls; the script checks that
# the full 20,000 calls reach exactly the same donors.
source(file.path("tests", "reference", "r", "_l2_common.R"))

reached_first_2000 <- function(idx, inp) {
  k <- length(donor_values(inp))
  first <- donor_counts(idx[seq_len(2000), , drop = FALSE], k) > 0
  all <- donor_counts(idx, k) > 0
  if (!is.factor(inp$y)) {
    stopifnot("rpart tree is deterministic" = identical(first, all))
  }
  storage.mode(first) <- "integer"
  list(reached_2000 = first)
}

l2_method_fixture(
  "cart",
  input_names = c("F1", "F2", "F3", "F4"),
  seeds = c(3001, 3002, 3003, 3004),
  n_calls = 20000,
  extra = reached_first_2000
)
