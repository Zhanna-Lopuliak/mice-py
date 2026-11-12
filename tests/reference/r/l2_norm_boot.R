# Layer 2 reference for mice.impute.norm.boot: 20,000 calls on input F1
# after one set.seed, default arguments. A fixture of the continuous kind, as
# l2_norm.R. Writes tests/reference/fixtures/l2_norm_boot.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "norm.boot",
  input_names = "F1",
  seeds = 7021,
  n_calls = 20000,
  kind = "continuous"
)
