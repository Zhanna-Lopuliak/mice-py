# Reference fixtures from R mice

This folder holds the reference values that the parity tests in
`tests/test_parity/` compare the Python port against. The R scripts in `r/`
call the R package `mice` and write small JSON files to `fixtures/`. The
fixtures are committed, so the Python tests run without R.

## Pinned versions

| Component | Version |
|---|---|
| R | 4.6.2 |
| mice | 3.19.0 (asserted by `r/_common.R`; the scripts stop on any other version) |
| jsonlite | as installed with R 4.6.2 |
| Random numbers | `RNGkind("Mersenne-Twister", "Inversion", "Rejection")`, R's default, set in `r/_common.R` |

Every fixture records the R and mice versions under `meta`. When reading the
mice source, use the `v3.19.0` tag, not `master`.

## Regenerating

From the repository root:

```bash
Rscript tests/reference/r/generate_all.R
git diff --exit-code tests/reference/fixtures
```

`generate_all.R` sources every `r/l<digit>_*.R` script, each in its own
environment. Each script sets its own seeds, so regenerating reproduces every
file byte for byte; a non-empty diff means R, mice or a script changed. One
script can also be run on its own, for example
`Rscript tests/reference/r/l1_pmm.R`.

Run the tests:

```bash
uv run pytest tests/test_parity -q -p no:cacheprovider -k l1   # layer 1 only
uv run pytest tests/test_parity -m parity                      # all parity tests
```

## Naming

| Kind | Pattern | Example |
|---|---|---|
| Generator | `r/l<layer>_<topic>.R` | `r/l1_pmm.R` |
| Fixture | `fixtures/l<layer>_<topic>[_<part>].json` | `fixtures/l1_pmm_one_donor.json` |
| Python test | `tests/test_parity/test_l<layer>_<topic>.py` | `test_l1_pmm.py` |

Layers: `l1` exact parity on deterministic pieces, `l2` distribution parity,
`l3` the in-suite part of the simulation study. Files starting with `_` are
helpers and are not run by `generate_all.R`.

## Format

Written by `write_fixture(name, payload)` in `r/_common.R` and read by
`load_fixture(name)` in `tests/test_parity/_helpers.py`.

- Doubles are written with 17 significant digits and read back bit for bit.
- `NA` becomes `null`. `Inf`, `-Inf` and `NaN` become the strings `"Inf"`,
  `"-Inf"` and `"NaN"`; `as_array()` turns them back into floats.
- Matrices are lists of rows.
- Length-one vectors are written as scalars unless wrapped in `I()`;
  `as_array()` accepts both.
- Data frames go through `encode_frame()`: one list per column, factors as
  strings plus their level order and an `ordered` flag. `decode_frame()`
  rebuilds them, with factors as pandas Categoricals in R's level order (so
  codes match `data.matrix()` in R).
- Fixtures carry their inputs (data generated in R with `set.seed`), so a test
  never depends on a random stream in Python to rebuild them.
- No timestamps.

## Layer 1 fixtures

| Fixture | R call | Python target |
|---|---|---|
| `l1_estimice` | `mice:::estimice(x, y, ls.meth)`, qr / ridge / svd | `imputation.sampler.estimice` |
| `l1_norm_draw` | `mice:::.norm.draw(y, ry, x)` with `rchisq(1, df)` and `rnorm(p)` captured by re-seeding | `imputation.sampler.norm_draw`, driven by `ScriptedGenerator` |
| `l1_quantify` | `mice:::quantify(y, ry, x)` | `imputation.PMM.quantify` (equal up to sign) |
| `l1_pmm_one_donor` | `mice.impute.pmm(y, ry, x, donors = 1, matchtype)` with captured draws | `imputation.PMM.pmm` |
| `l1_matchindex` | donors reached by 2000 calls of `matchindex(d, t, k)` | `imputation.PMM.matchindex` |
| `l1_setup` | `mice(data, maxit = 0)`: method, predictorMatrix, visitSequence, loggedEvents | `MICE.impute`: `.method`, `.predictor_matrix`, `.visit_sequence` |
| `l1_design` | `mice:::obtain.design`, `mice:::remove.lindep`, `mice:::find.collinear` | `imputation.design` |
| `l1_quickpred` | `quickpred(data, ...)` on nhanes and boys | `imputation.quickpred` |
| `l1_md_pattern` | `md.pattern(data, plot = FALSE)` | `plotting.utils.md_pattern_like` |
| `l1_pooling` | `pool()`, `summary(pool(), type = "all", conf.int = TRUE)`, `pool.table()`, `mice:::barnard.rubin()` | `imputation.pooling.barnard_rubin` / `pool_estimates`; `MICE.fit` / `MICE.pool` on R's completed datasets |
| `l1_midastouch` | `mice.impute.midastouch()` with a `trace()` exit hook capturing `probs`, `midas.kappa`, `omega` | `imputation.midas._donor_probabilities` |
| `l1_norm` | `mice.impute.norm()`, `.norm.nob()`, `.norm.boot()`, `.norm.predict()`, `.mean()` with every draw captured by re-seeding (`rchisq`, `rnorm`, `sample` as 0-based indices) and the expected order of the draws | `imputation.norm.norm` / `norm_nob` / `norm_boot` / `norm_predict`, `imputation.mean.mean`, driven by a strict `ScriptedGenerator` |

Behaviours of R mice 3.19.0 that the fixtures pin down and that are easy to
get wrong:

- `visitSequence` from `mice()` can list variables that are never imputed:
  complete variables for the default order and for `"monotone"` /
  `"revmonotone"`, and variables whose method `mice.edit.setup()` set to `""`
  (its `vis[vis != j]` compares names with an index and removes nothing). The
  fixture stores R's raw sequence and the effective one, restricted to
  variables with a non-empty method.
- A collinear incomplete variable gets method `""` and stays incomplete
  (`l1_setup`, case `synthetic_dup_target`).
- `model.matrix()` codes ordered factors with polynomial contrasts
  (`gen.L`, `gen.Q`, ...), unordered factors with treatment dummies in level
  order (`l1_design`).
- When more than one column has to go, `remove.lindep()` reuses the eigen
  decomposition of all columns after the first drop, so it can keep a
  collinear column or drop an unrelated one. The cases flagged
  `several_dependencies` pin that result.
- `estimice(ls.meth = "svd")` fails in R for a one-column `x`; that case is
  not in the fixture.
- `norm.nob` and `norm.boot` divide the residual sum of squares by
  `n1 - q - 1`, one less than the usual df, and return `Inf` when that is 0
  (case `norm_nob_tiny`); `norm.boot` without predictors stops with
  `'x' must be a matrix` after drawing its bootstrap sample (case
  `norm_boot_no_predictor`, which stores the intercept-only result that
  mice-py computes instead).
- `vcov()` of a `glm` fitted with the default `epsilon = 1e-8` is about 1e-5
  (relative) away from the inverse information at the MLE. The logistic model
  in `l1_pooling` uses `glm.control(epsilon = 1e-14)`.

## Licence

mice is licensed GPL (>= 2), and so is mice-py since 0.2.0 (see
`COPYRIGHT`). The scripts call mice functions (exported ones, internals
through `mice:::`, or a `trace()` hook) and store their output. As a matter
of practice they do not copy mice source code, so that the fixtures record
what the installed mice computes.
