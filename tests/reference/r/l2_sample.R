# Layer 2 reference for mice.impute.sample: 20,000 calls per input after one
# set.seed. Writes tests/reference/fixtures/l2_sample.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "sample",
  input_names = c("F1", "F3"),
  seeds = c(5001, 5003),
  n_calls = 20000
)
