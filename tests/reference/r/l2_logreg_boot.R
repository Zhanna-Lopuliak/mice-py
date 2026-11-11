# Layer 2 reference for mice.impute.logreg.boot: 20,000 calls on input F5
# after one set.seed, default arguments. A fixture of the category kind, as
# l2_logreg.R. F6 is left out: logreg.boot does not augment, so on separated
# data every bootstrap fit diverges. Writes
# tests/reference/fixtures/l2_logreg_boot.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "logreg.boot",
  input_names = "F5",
  seeds = 7315,
  n_calls = 20000,
  kind = "category"
)
