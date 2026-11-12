# Layer 2 reference for mice.impute.polr: 20,000 calls on input F7 (an
# ordered target with the levels a < b < c < d and two numeric predictors,
# drawn from a model in which the proportional-odds fit and the multinomial
# fit of polyreg disagree) after one set.seed, default arguments. polr draws
# labels from a fitted model, so the fixture is of the category kind: per
# missing cell the level counts, plus the moments of the OLS coefficients of
# the "b", "c" and "d" indicators of the completed target. Writes
# tests/reference/fixtures/l2_polr.json.
source(file.path("tests", "reference", "r", "_l2_common.R"))

l2_method_fixture(
  "polr",
  input_names = "F7",
  seeds = 7707,
  n_calls = 20000,
  kind = "category"
)
