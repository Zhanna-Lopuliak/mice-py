# Layer 2 reference for mice.impute.norm: 20,000 calls per input after one
# set.seed, default arguments (ls.meth = "qr"). norm imputes new numbers, so
# the fixture is of the continuous kind: per missing cell the moments of the
# imputed value and the decile bins of R's draws, plus the moments of the OLS
# coefficients of the completed y. Writes tests/reference/fixtures/l2_norm.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "norm",
  input_names = c("F1", "F4"),
  seeds = c(7001, 7004),
  n_calls = 20000,
  kind = "continuous"
)
