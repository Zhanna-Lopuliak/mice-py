# L3 simulation study: mice-py against R mice

This folder holds layer L3 of the validation of mice-py: a simulation study
with known truth. Six scenarios generate data, delete values, impute them with
mice-py and with R `mice` 3.19.0 **on the same incomplete datasets**, analyse
each completed dataset and pool the results. One Python routine computes all
metrics for both engines, and every Python-minus-R difference is compared with
its Monte Carlo error.

The pass rule is "matches R", not an absolute target: a correct method can
under-cover in a hard scenario, and then R does too.

## Files

| File | What it is |
|---|---|
| `scenarios.py` | Data generators, missingness, the analysis of one completed dataset, the true values |
| `_adapter.py` | The only place that calls the library: `impute(df, method, m, maxit, seed)` |
| `study.py` | CLI: `generate` writes the datasets, `run-python` imputes them with mice-py |
| `run_r.R` | The same with R `mice`, on the same files |
| `metrics.py` | Rubin pooling with Barnard-Rubin df, per-engine metrics, paired differences |
| `report.py` | Writes `results/summary.csv` and `results/REPORT.md` |
| `results/` | Committed summary of the last run |
| `work/` | Datasets and per-replication results (git-ignored; the full study's datasets take about 120 MB) |
| `legacy/` | Result files of an earlier study whose code is not in the repository |

The in-suite version of this study (scenario S4, pmm and midas, 200
replications in about two seconds) is `tests/test_parity/test_l3_mini_simulation.py`
with its reference `tests/reference/r/l3_mini_s4.R`.

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

Dataset `k` of scenario `Si` comes from `numpy.random.default_rng([i, k])`;
the order of the draws is documented in `scenarios.py`. Variances: OLS
`sigma² (X'X)⁻¹` with `dfcom = n - p`; a mean `s²/n` with `dfcom = n - 1`; a
proportion `p(1-p)/n` with `dfcom = inf`.

Methods, forced for every incomplete column on both sides: `pmm`,
`midastouch` (mice-py: `midas`), `cart`, `rf`, `sample`. m = 5, full predictor
matrix, visit sequence = incomplete columns in column order, seed of
replication k in scenario i = `i * 100000 + k`. Default 1000 replications
(rf 500). The pseudo-method `complete` analyses the data before deletion.

## Running

From the repository root:

```bash
uv run python simulations/study.py generate      # 6000 datasets, about 10 seconds
uv run python simulations/study.py run-python    # mice-py, all cores
Rscript simulations/run_r.R                      # R mice, all cores
uv run python simulations/report.py              # results/summary.csv, results/REPORT.md
```

`uv run python simulations/study.py` without a command runs `generate` and
then `run-python`. Options of `study.py` and `run_r.R`:

| Option | Default | Meaning |
|---|---|---|
| `--scenarios S1,S4` | all | scenarios to run |
| `--methods pmm,sample` | all five | methods to run (`complete` always runs) |
| `--reps N` | 1000 | replications per scenario and method |
| `--rf-reps N` | 500 | replications for `rf`, capped at `--reps` |
| `--jobs N` | all cores | worker processes |
| `--work-dir DIR` | `simulations/work` | where data and results go |

`report.py` takes `--work-dir` and `--out` (default `simulations/results`).
It reports every result file present in `work/results/`, so remove that folder
before a fresh run if earlier runs used other settings.

Requirements: R with `mice` 3.19.0, `ranger`, `rpart` and `jsonlite`. The R
script calls mice functions and does not copy mice source (mice is GPL, this
repository is MIT).

### Expected runtime

Measured on an 8-core Apple machine (4 performance and 4 efficiency cores)
with 40 replications of every scenario and method, then extrapolated to the
full plan: about 7 minutes for `run-python` and 7 minutes for `run_r.R`, plus
about 10 seconds for `generate`. Most of the Python time is `rf` in S3 and S5
(1.3-1.4 s per replication against 0.3-0.6 s in R). Two Python cells fail
immediately today (S5 with pmm and midas) and will cost more once they run.
`report.py` prints its own extrapolation from the timings of the last run.

## Outputs

`work/results/<engine>_<scenario>_<method>.csv`, one row per replication and
term: `scenario, method, rep, term, qbar, ubar, b, m, n, dfcom` (the pooling
ingredients, not pooled results).

`work/results/<engine>_<scenario>_<method>_status.csv`, one row per
replication: `status` (`ok` or `failed`), `error_class`, `error_message`
(prefixed with the stage that failed), `n_events`/`events` (R `loggedEvents`;
empty for Python), `n_warnings`/`warnings`, `seconds`. A failure is recorded,
never fatal.

`results/summary.csv`: per scenario, method, term and metric the Python and R
value with Monte Carlo SE and replication count, the paired difference, its
SE, z and the flag. `results/REPORT.md`: the same as tables, plus the harness
checks, flagged cells, failures, warnings and timings.

## Metrics and pass rule

`metrics.py` pools each replication itself (Rubin's rules, Barnard-Rubin df,
t interval; checked against R `pool()` to 1e-11) and does not use the
library's pooling, so the study judges imputation and not pooling.

Per scenario, method, term and engine: mean estimate, bias, percent bias,
coverage of the 95% interval, mean width, RMSE, mean B, mean T, each with its
Monte Carlo SE (bias sd/√R, coverage √(CR(1-CR)/R), RMSE by the delta method).
Paired Python-minus-R differences use only replications where both engines
succeeded: coverage SE √(n10 + n01 - (n10 - n01)²/R)/R, others sd(d)/√R.

- **Pass:** every paired difference of bias, coverage, width, RMSE, B and T
  has |z| ≤ 4.2 (Bonferroni over several hundred comparisons).
- **Harness:** the `complete` rows must have coverage within 0.95 ± 3 nominal
  Monte Carlo errors, and Python and R must give the same complete-data
  estimates (they analyse identical data; observed agreement about 1e-14).

z-scores need about 100 or more replications to mean anything.

## Status before the Phase 2 fixes

- **Seeds (finding C1).** `MICE.impute` has no `seed` argument; chain j is
  always seeded with 42 + j, so all replications reuse the same random
  streams. `_adapter.py` passes the seed as soon as the argument exists and
  until then warns, and the report carries a banner: such runs check the
  pipeline only. The full study runs after the seed fix.
- **midas** draws from NumPy's global generator (finding A5). Workers are
  started with `spawn`, so each has its own fresh global state.
- **Failures today:** pmm on S5 (`AttributeError`, finding B1: categorical
  target; B2 would follow for the categorical predictor), midas on S5
  (`ValueError`, categorical target). R's `midastouch` cannot impute a factor
  either: it imputes level codes, which R turns into `NA` ("invalid factor
  level"), so S5 midastouch fails on both sides.
