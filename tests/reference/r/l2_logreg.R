# Layer 2 reference for mice.impute.logreg: 20,000 calls per input after one
# set.seed, default arguments. logreg draws labels from a fitted model, so
# the fixture is of the category kind: per missing cell the level counts,
# plus the moments of the OLS coefficients of the "yes" indicator of the
# completed target. F5 is a binary target with moderate slopes; on F6 the
# observed rows are separated and the augmentation decides the result.
# Writes tests/reference/fixtures/l2_logreg.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "logreg",
  input_names = c("F5", "F6"),
  seeds = c(7305, 7306),
  n_calls = 20000,
  kind = "category"
)
