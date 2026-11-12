# Layer 2 reference for mice.impute.polyreg: 20,000 calls on input F3 (a
# factor target with the levels a, b, c and two numeric predictors) after one
# set.seed, default arguments. polyreg draws labels from a fitted model, so
# the fixture is of the category kind: per missing cell the level counts,
# plus the moments of the OLS coefficients of the "b" and "c" indicators of
# the completed target. Writes tests/reference/fixtures/l2_polyreg.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "polyreg",
  input_names = "F3",
  seeds = 7503,
  n_calls = 20000,
  kind = "category"
)
