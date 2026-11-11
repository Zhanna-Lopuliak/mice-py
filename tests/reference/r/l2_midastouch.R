# Layer 2 reference for mice.impute.midastouch: 20,000 calls per input after
# one set.seed, default arguments. Writes
# tests/reference/fixtures/l2_midastouch.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "midastouch",
  input_names = c("F1", "F2", "F4"),
  seeds = c(2001, 2002, 2004),
  n_calls = 20000
)
