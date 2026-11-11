# Layer 2 reference for mice.impute.pmm: 20,000 calls per input after one
# set.seed, default arguments (donors = 5, matchtype = 1, quantify = TRUE).
# Writes tests/reference/fixtures/l2_pmm.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "pmm",
  input_names = c("F1", "F2", "F3", "F4"),
  seeds = c(1001, 1002, 1003, 1004),
  n_calls = 20000
)
