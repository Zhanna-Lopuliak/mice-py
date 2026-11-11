# Layer 2 reference for mice.impute.rf: 5,000 calls per input after one
# set.seed, default arguments (ranger, ntree = 10). ranger runs on one
# thread so that the file is reproducible. Writes
# tests/reference/fixtures/l2_rf.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "rf",
  input_names = c("F1", "F2", "F3", "F4"),
  seeds = c(4001, 4002, 4003, 4004),
  n_calls = 5000,
  num.threads = 1
)
