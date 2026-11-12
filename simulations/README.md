# L3 simulation study: mice-py against R mice

This folder holds layer L3 of the validation of mice-py: a simulation study
with known truth. Eight scenarios generate data, delete values, impute them
with mice-py and with R `mice` 3.19.0 **on the same incomplete datasets**,
analyse each completed dataset and pool the results. One Python routine
computes all metrics for both engines, and every Python-minus-R difference is
compared with its Monte Carlo error.

The pass rule is "matches R", not an absolute target: a correct method can
under-cover in a hard scenario, and then R does too.

## Files

| File | What it is |
|---|---|
| `scenarios.py` | Data generators, missingness, the analysis of one completed dataset, the true values |
| `_adapter.py` | The only place that calls the library: `impute(df, methods, m, maxit, seed)` |
| `study.py` | The table of cells; CLI: `generate` writes the datasets and the manifest, `run-python` imputes them with mice-py |
| `run_r.R` | The same with R `mice`, on the same files and the cells of the manifest |
| `metrics.py` | Rubin pooling with Barnard-Rubin df, per-engine metrics, paired differences, identity rule, method records |
| `report.py` | Writes `results/summary.csv` and `results/REPORT.md` |
| `results/` | Committed summary of the last full run |
| `work/` | Datasets and per-replication results (git-ignored; the datasets of all eight scenarios take about 160 MB) |
| `legacy/` | Result files of an earlier study whose code is not in the repository |

The in-suite version of this study (scenario S4, pmm, midas and norm, 200
replications in about two seconds) is
`tests/test_parity/test_l3_mini_simulation.py` with its reference
`tests/reference/r/l3_mini_s4.R`.

## Scenarios

Model M: `x1`, `x2` bivariate normal, means 0, variances 1, correlation 0.5;
`y = 0.5 + 1.0 x1 + 0.5 x2 + e`, `e ~ N(0, 1)`.

"MAR on v with rate r": P(missing) = expit(a + 1.5 z(v)), z(v) standardised
within the sample, `a` solved per dataset so that the mean probability is r.
MCAR: each cell missing independently with probability r.

| ID | Data | n | Missingness | maxit | Analysis and truth |
|---|---|---|---|---|---|
| S1 | M | 200 | `x1` MCAR 30% | 1 | OLS `y ~ x1 + x2` (0.5, 1.0, 0.5); mean(`x1`) = 0 |
| S2 | M | 200 | `x1` MAR on `y`, 30% | 1 | as S1 |
| S3 | M | 200 | `y` and `x1` each 25%, MAR on `x2`, drawn independently | 5 | as S1, plus mean(`y`) = 0.5 |
| S4 | M | 50 | `x1` MCAR 50% | 1 | as S1 |
| S5 | `g` in {a, b, c} (0.5, 0.3, 0.2); `x ~ N(0.5 code, 1)`; `y = 0.5[g=b] + 1.0[g=c] + 0.7x + e` | 200 | `g` MAR on `y` 30%; `x` MCAR 20% | 5 | OLS `y ~ g + x` (0, 0.5, 1.0, 0.7); P(g=c) = 0.2; mean(`x`) = 0.35 |
| S6 | `x1 ~ N(0,1)`; `w = exp(0.5 x1 + u)`, `u ~ N(0, 0.5²)`; `y = 1 + 0.5w + 0.5x1 + e` | 200 | `w` MAR on `y` 30% | 1 | OLS `y ~ w + x1` (1, 0.5, 0.5); mean(`w`) = exp(0.25) |
| S7 | `b` in {no, yes} (0.6, 0.4); `x ~ N(0.5 code, 1)`; `y = 0.5 + 1.0[b=yes] + 0.7x + e` | 200 | `b` MAR on `y` 30%; `x` MCAR 20% | 5 | OLS `y ~ b + x` (0.5, 1.0, 0.7); P(b=yes) = 0.4; mean(`x`) = 0.2 |
| S8 | `o` ordered lo < mid < hi < top (0.2, 0.3, 0.3, 0.2), code 0-3; `x ~ N(0.5 code, 1)`; `y = 0.5 + 0.4 code + 0.7x + e` | 200 | `o` MAR on `y` 30%; `x` MCAR 20% | 5 | OLS `y ~ o + x` with treatment dummies (0.5; 0.4, 0.8, 1.2; 0.7); P(o=top) = 0.2; mean(`x`) = 0.75 |

In S5, S7 and S8 "code" is the level index 0, 1, ...; the category is drawn
first, then `x` given it, then `y` given both (`scenarios.py` documents the
order of the draws). The truths are exact: E(x) = 0.5 E(code), and `y` is
linear in the treatment dummies and `x`.

Dataset `k` of scenario `Si` comes from `numpy.random.default_rng([i, k])`.
Variances: OLS `sigma² (X'X)⁻¹` with `dfcom = n - p`; a mean `s²/n` with
`dfcom = n - 1`; a proportion `p(1-p)/n` with `dfcom = inf`.

Categorical columns: the levels and whether they are ordered belong to the
scenario (`Factor` in `scenarios.py`) and reach R through the manifest.
Python reads an unordered column as object dtype and an ordered one as an
ordered `pd.Categorical` with the declared categories; R reads a factor with
the declared levels, ordered where declared. So the default method of `o` is
`polr` on both sides. The labels of S8 do not sort in level order, so the
order has to come from the declaration. The analysis uses treatment dummies
named `<column><level>` on both sides: R analyses an unordered copy of an
ordered factor, because `lm` would otherwise use polynomial contrasts.

## Cells

The study is a table of cells (`study.CELLS`): a study method on a
scenario, with its number of replications. A study method gives each
incomplete column a method by the column's type (`study.STUDY_METHODS`), or
gives no method at all. m = 5, full predictor matrix, visit sequence =
incomplete columns in column order, seed of replication k in scenario i =
`i * 100000 + k`, the same for every cell of a scenario.

| Study method | Scenarios | Replications | Methods of the incomplete columns | Rule |
|---|---|---|---|---|
| `pmm`, `midastouch` (mice-py: `midas`), `cart`, `sample` | S1-S6 | 1000 | that method on every incomplete column | z |
| `rf` | S1-S6 | 500 | `rf` on every incomplete column | z |
| `norm`, `norm.boot` | S3, S4 | 1000 | that method on every incomplete column | z |
| `norm.nob`¹ | S4 | 1000 | `norm.nob` | z |
| `norm.predict`¹, `mean`¹ | S4 | 1000 | that method | identity |
| `default` | S5, S7, S8 | 1000 | no `method` argument: Python `MICE.impute(method=None)`, R `mice()` without `method`; expected `polyreg` + `pmm` (S5), `logreg` + `pmm` (S7), `polr` + `pmm` (S8) | z |
| `logreg.boot` | S7 | 1000 | `logreg.boot` on `b`, `pmm` on `x` | z |

¹ `norm.nob`, `norm.predict` and `mean` are improper imputation methods:
they ignore the uncertainty of the imputation model (`norm.predict` and
`mean` also the residual noise), so their intervals cover below 95% in both
engines, by design. The pass rule is the same: mice-py matches R.

`study.py generate` writes every cell to `work/data/manifest.json`, with the
method of each incomplete column (`columns`, null for `default`) and, for
`default` cells, the methods R's `make.method` chooses (`expected_methods`).
`run-python` and `run_r.R` both run the cells of the manifest, so the R side
never derives the methods itself; `run-python` stops if the manifest's cells
differ from `study.CELLS` (run `generate` again). Each engine records the
methods it actually used per replication (Python `MICE.method` in R's names,
R `imp$method`), and the report compares them. The pseudo-method `complete`
analyses the data before deletion and runs for every scenario with a
selected cell.

## Running

From the repository root, on a laptop with two worker processes at low
priority:

```bash
nice -n 19 uv run python simulations/study.py generate --jobs 2     # 8000 datasets, about 10 seconds
nice -n 19 uv run python simulations/study.py run-python --jobs 2   # mice-py, every cell
nice -n 19 Rscript simulations/run_r.R --jobs 2                     # R mice, every cell
uv run python simulations/report.py                                 # results/summary.csv, results/REPORT.md
```

`uv run python simulations/study.py` without a command runs `generate` and
then `run-python`. Options of `study.py` and `run_r.R`:

| Option | Default | Meaning |
|---|---|---|
| `--scenarios S1,S4` | all | scenarios whose cells run (`generate`: scenarios to write) |
| `--methods pmm,default` | all | study methods whose cells run (`complete` always runs) |
| `--reps N` | each cell's own | cap on the replications of every cell; `generate` writes as many datasets as the scenario's largest cell needs, capped by N |
| `--jobs N` | all cores | worker processes |
| `--work-dir DIR` | `simulations/work` | where data and results go |

A pilot of the cells added in 0.3, plus pmm on S4 and S5 as a regression
probe:

```bash
nice -n 19 uv run python simulations/study.py run-python --methods norm,norm.boot,norm.nob,norm.predict,mean,default,logreg.boot --reps 40 --jobs 2
nice -n 19 uv run python simulations/study.py run-python --methods pmm --scenarios S4,S5 --reps 40 --jobs 2
# the same two selections with Rscript simulations/run_r.R
uv run python simulations/report.py --out /some/scratch/folder
```

`report.py` takes `--work-dir` and `--out` (default `simulations/results`).
It reports every result file present in `work/results/`, so remove that folder
before a fresh run if earlier runs used other settings.

Requirements: R with `mice` 3.19.0, `ranger`, `rpart`, `nnet`, `MASS` and
`jsonlite`. The R script calls mice functions rather than copying mice
source, so that its results are those of the installed mice.

### Expected runtime

Seconds per replication with two processes on an 8-core Apple machine (4
performance and 4 efficiency cores), from the 0.2 run for the donor methods
and from a 40-replication pilot for the cells of 0.3: most cells take 0.01 to
0.1 s in either engine; `rf` takes 0.25-0.4 s in S3 and S5, R's `cart` and
`midastouch` 0.26-0.41 s in S3 and S5. The whole study takes about 10
minutes for `run-python` and 15 to 30 minutes for `run_r.R` with two
processes. `results/REPORT.md` has the timings of the last run, and
`report.py` prints its own extrapolation from them.

## Outputs

`work/results/<engine>_<scenario>_<method>.csv`, one row per replication and
term: `scenario, method, rep, term, qbar, ubar, b, m, n, dfcom` (the pooling
ingredients, not pooled results). `method` is the study method.

`work/results/<engine>_<scenario>_<method>_status.csv`, one row per
replication: `status` (`ok` or `failed`), `error_class`, `error_message`
(prefixed with the stage that failed), `n_events`/`events` (R `loggedEvents`;
empty for Python), `n_warnings`/`warnings`, `methods` (the method of every
column as the engine ran it, `col=method;...` in data order, R's names,
empty for a column that is not imputed), `seconds`. A failure is recorded,
never fatal.

`results/summary.csv`: per scenario, method, term and metric the Python and R
value with Monte Carlo SE and replication count, the paired difference, its
SE, z and the flag. `results/REPORT.md`: the same as tables, plus the cells,
the harness checks, the methods each engine used, the identity rule, flagged
cells, failures, warnings and timings.

## Metrics and pass rule

`metrics.py` pools each replication itself (Rubin's rules, Barnard-Rubin df,
t interval; checked against R `pool()` to 1e-11) and does not use the
library's pooling, so the study judges imputation and not pooling.

Per scenario, method, term and engine: mean estimate, bias, percent bias,
coverage of the 95% interval, mean width, RMSE, mean B, mean T, each with its
Monte Carlo SE (bias sd/√R, coverage √(CR(1-CR)/R), RMSE by the delta method).
Paired Python-minus-R differences use only replications where both engines
succeeded: coverage SE √(n10 + n01 - (n10 - n01)²/R)/R, others sd(d)/√R.

- **Pass (rule z):** every paired difference of bias, coverage, width, RMSE,
  B and T has |z| ≤ 4.2 (Bonferroni over several hundred comparisons).
- **Pass (identity rule):** `norm.predict` and `mean` on S4 (one incomplete
  column, maxit = 1) draw nothing, so both engines must give the same
  completed data and B = 0; the paired z would be 0/0. These cells get no z
  and no flag. They pass when the largest |qbar Python − qbar R| over
  replications and terms is at most 1e-8, and the largest B of each engine
  is at most 1e-16 (B = 0 to rounding: Python's variance of five identical
  estimates can be about 1e-32, R's is exactly 0).
- **Methods:** in every cell Python's method record equals R's in every
  paired replication; in a `default` cell both engines chose the expected
  methods in every replication.
- **Harness:** the `complete` rows must have coverage within 0.95 ± 3 nominal
  Monte Carlo errors, and Python and R must give the same complete-data
  estimates (they analyse identical data; observed agreement about 1e-14).

z-scores need about 100 or more replications to mean anything.

## Results

`results/` holds the run for mice-py 0.2.0 (the five donor methods on S1-S6)
and `results/baseline/` the run on the code before the fixes (0.1.0 plus the
seed fix, commit `fc1d3e8`); `BASELINE.md` describes that state. On 0.2.0,
18 of 774 paired comparisons are flagged, all for `rf` in S5, where
scikit-learn and ranger grow different trees with categorical predictors; on
the baseline it was 184 of 738. The documentation page "Validation against R
mice" (`docs/source/r_mice/validation.rst`) has the tables and the cause. The
cells added in 0.3 are not in `results/` until the next full run.

midastouch cannot run on S5 in either engine: R's `midastouch` imputes
level codes, which R turns into `NA` ("invalid factor level"), and mice-py's
`midas` raises a `ValueError` for a categorical target. Both are recorded as
failures, not as errors of the study.
