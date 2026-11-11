# Baseline: mice-py 0.1.0 against R mice 3.19.0

The state of the library before any fix. Recorded on 2026-11-11 at commit
`af2fd7a`; the library code there is identical to `main` (`c7402d5`).

Versions: R 4.6.2, mice 3.19.0, ranger 0.18.0, rpart 4.1.27; Python 3.13.7,
numpy 2.3.3, pandas 2.3.2, scipy 1.16.2, scikit-learn 1.7.2, statsmodels 0.14.5.

How to read this: every check that fails on this code is marked as an expected
failure (`xfail(strict=True)`) carrying the ID of the defect behind it. A fix
has to make those checks pass, and its marker is then removed. A passing check
is something that already agrees with R.

## Totals

| Layer | What it checks | Passing | Expected failures |
|---|---|---|---|
| L1 | Exact parity with R on deterministic pieces (tolerance 1e-8) | 45 | 102 |
| L2 | Distribution of the imputed values per method, R against Python | 32 | 18 |
| L3 mini | Small simulation inside the test suite | 4 | 0 |
| L4 | Invariants and edge cases | 250 | 114 |
| Old suite | The 105 tests that existed before | 104 | 0 |

The old suite has one flaky test, `test_sample_with_random_state`: it passes a
seed that the function silently ignores (C2).

The full L3 simulation study is not part of this baseline. The library
hard-codes its random seed (C1), so every replication would reuse the same
random streams. The study runs once right after the seed fix; its numbers are
added below at that point.

## What already agrees with R

- `estimice` with `qr` and `ridge`: coefficients, residuals, covariance, df.
- The coefficient draw: `coef` and `sigma` are exact. `beta` is not (A1).
- PMM with one donor and `matchtype=0` gives R's imputations exactly.
- Pooled estimates, standard errors, within and between variance.
- Missing-data pattern: counts, margins and column order.
- `sample`: the distribution of imputed values matches R.
- `midas`: on inputs without tied predictors the distribution matches R. With
  R's bootstrap weights injected, its donor probabilities equal R's to 1e-14
  (a probe, not a committed test).

## Defects and the checks waiting on them

A = wrong result, B = crash, C = design. "Checks" is the number of expected
failures carrying that ID as their first reason. A11, A12 and B6 to B9 were
found by this evaluation; the rest by reading the code against the R source.

| ID | Defect | Checks | Evidence |
|---|---|---|---|
| A1 | Transposed Cholesky factor in the coefficient draw (`sampler.py`) | 15 | With the factor not transposed, `beta` equals R's to 4e-16 and one-donor PMM matches exactly. L2 on input F1: see table below. |
| A2 | Default method is `sample`; R uses `pmm` | 5 | |
| A3 | `quantify_cca` does not reproduce R's `quantify` | 2 | Correlation with R's scores: 4e-7 (3 levels), −0.46 (2 levels) |
| A4 | Pooling: normal reference distribution, no Barnard-Rubin df, `lambda` reported as FMI | 17 | The reported "FMI" equals R's `lambda` exactly |
| A5 | `midas` draws from NumPy's global generator and ignores `rng` | 6 | |
| A6 | CART bootstraps and prunes differently from rpart; RF draw rule and defaults differ from ranger | 7 | See table below |
| A7 | `_quickpred` differs from R `quickpred`; default predictor matrix is not the full matrix | 20 | |
| A8 | `xyplot` colouring; trace plot shows the variance | 2 | |
| A9 | PMM donor ties are resolved deterministically | 8 | F2: see table below. R shuffles the donors once per call |
| A10 | `md_pattern_like` orders rows differently from `md.pattern` | 2 | |
| A11 | `estimice(ls_meth="svd")` multiplies by `Vh` where R multiplies by `V` | 2 | R: 1.047, 0.781, −1.016, 2.045; port: 1.169, 0.586, 1.130, 1.983 |
| A12 | `midas` returns float-converted donors | 1 | |
| B1 | pmm crashes on a categorical target | 7 | `AttributeError` |
| B2 | pmm and midas reject categorical predictors | 24 | |
| B3 | pmm crashes when `donors` exceeds the number of observed values | 8 | |
| B4 | Collinear, constant or too many predictors crash pmm; a redundant column can also change the result without a crash | 25 | |
| B5 | midas crashes when R² is 1 or undefined | 6 | Reproduced with `ridge=0` on linear data and with a constant column |
| B6 | midas crashes when a missing row's predictors equal an observed row's | 6 | 200 of 200 calls on F2 (one binary predictor) |
| B7 | rf crashes when a target has no predictors | 1 | |
| B8 | cart and rf crash on an incomplete boolean column | 2 | |
| B9 | A pandas `string`-dtype predictor crashes cart and rf | 2 | |
| C1 | No `seed` argument; 42 is hard-coded | 8 | |
| C2 | Imputers swallow unknown keyword arguments | 6 | |
| C3 | Defaults differ from R: `maxit`, visit order | 20 | |
| C4 | All-empty rows are dropped; validators print | 10 | |
| C5 | `fit` supports OLS only | 1 | |
| C6 | No setup step: no dummy coding, no removal of constant or collinear variables | 12 | |
| C7 | Plotting: import-time side effects; functions return nothing | 9 | |

## L2 numbers on this code

Each method is called many times on a fixed input in R (20,000 calls; rf
5,000) and in Python (3,000; midas and cart 1,500; rf 300). Pass rules: donor
count test p ≥ 1e-7; |z| ≤ 5.5 for the mean and variance of the imputed value
per cell (T2) and of the regression coefficients of the completed data (T3).

Inputs: F1 has two predictors correlated 0.9 and missing rows away from the
bulk of the data; F2 one binary predictor; F3 a factor target; F4 a lognormal
target.

| Method | Input | Smallest p | Largest \|z\|, T2 | Largest \|z\|, T3 | Verdict |
|---|---|---|---|---|---|
| pmm | F1 | 3e-239 | 9.4 | 13.0 | fails (A1) |
| pmm | F2 | 0 | 76.5 | 57.3 | fails (A9, A1) |
| pmm | F3 | crash | | | fails (B1) |
| pmm | F4 | 0.046 | 2.5 | 1.7 | passes |
| midastouch | F1 | 0.066 | 1.5 | 0.9 | passes |
| midastouch | F2 | crash | | | fails (B6) |
| midastouch | F4 | 0.029 | 2.6 | 1.3 | passes |
| cart | F1 | 0 | 109 | 145 | fails (A6) |
| cart | F2 | 0.057 | 2.4 | 1.4 | agrees |
| cart | F4 | 0 | 31.7 | 15.3 | fails (A6) |
| rf | F1 | 1e-66 | 12.1 | 15.7 | fails (A6) |
| rf | F2 | 0.047 | 2.7 | 0.5 | agrees |
| rf | F4 | 1e-56 | 11.4 | 12.9 | fails (A6) |
| sample | F1 | 0.036 | 1.9 | 1.8 | passes |

pmm passes on F4 because its predictors are nearly uncorrelated, so the
transposed factor (A1) changes almost nothing there.

Three experiments, run in scratch and not committed, say what the fixes should
achieve:

- **A1 is the only defect of pmm on F1 and F4.** With `np.linalg.cholesky`
  patched to undo the transposition, pmm passes all three tests on both (F1:
  p 0.078, |z| 2.1 and 1.8).
- **Exact agreement with rpart is reachable for numeric targets.** A
  scikit-learn tree with `min_samples_split=15`, `min_samples_leaf=5`,
  `max_depth=30` and `ccp_alpha = cp × var(y_obs)` reproduced rpart's leaves
  exactly on F1, F2, F4 and on 12 larger datasets with up to 120 leaves.
- **Random forest can agree with ranger in distribution.** A forest with
  `min_samples_split=5`, `min_samples_leaf=1`, `max_features = floor(sqrt(p))`
  and donors pooled over all trees passed the three tests at 300 calls.

## L3: the simulation study on the baseline code

Run at commit `fc1d3e8`: the code above plus the seed fix (C1, C2, A5), which
the study needs so that replications get independent random streams. No other
defect was fixed. Full output: `simulations/results/baseline/`.

Six scenarios, five methods, 1000 replications (rf 500), m = 5. R mice ran on
the same incomplete datasets. Each cell below shows how many paired
Python-minus-R comparisons (bias, coverage, interval width, RMSE,
between-imputation and total variance, per term) were flagged at |z| > 4.2,
and the largest |z| in the cell.

| Scenario | pmm | midastouch | cart | rf | sample |
|---|---|---|---|---|---|
| S1, normal, MCAR | 0 of 24 (3.4) | 0 of 24 (2.1) | 14 of 24 (15.0) | 11 of 24 (19.3) | 0 of 24 (1.9) |
| S2, normal, MAR | 0 of 24 (1.7) | 0 of 24 (2.1) | 16 of 24 (19.3) | 17 of 24 (22.5) | 0 of 24 (1.1) |
| S3, two incomplete variables | 3 of 30 (5.7) | 0 of 30 (2.2) | 17 of 30 (13.6) | 2 of 30 (22.2) | 0 of 30 (1.9) |
| S4, small sample | 0 of 24 (4.0) | 0 of 24 (1.6) | 15 of 24 (18.3) | 13 of 24 (23.1) | 0 of 24 (1.5) |
| S5, categorical | crashed | not possible | 22 of 36 (8.8) | 22 of 36 (61.6) | 0 of 36 (3.2) |
| S6, skewed | 9 of 24 (8.9) | 0 of 24 (1.3) | 17 of 24 (18.9) | 6 of 24 (12.3) | 0 of 24 (1.0) |

In total 184 of 738 comparisons were flagged. pmm crashed on every S5 dataset
(B1, B2). midastouch cannot run on S5 in either engine: R's own function fails
on a factor target.

As the design predicted, the simulation barely sees the transposed Cholesky
factor (A1): pmm is flagged only in S3 and S6. Layer 2 is what caught it.

The harness itself was checked in the same run: on complete data Python and R
give the same numbers (largest difference in an estimate 6e-15), coverage of
the complete-data analysis is within its Monte Carlo band in every cell, and
the study's own Rubin pooling agrees with R's `pool()` to 1e-11.

## Reproduce

```bash
uv run pytest -p no:cacheprovider -rx                # all layers, with the reason of every expected failure
uv run python -m tests.test_parity._l2_report        # the L2 table above
Rscript tests/reference/r/generate_all.R             # regenerate the R fixtures (about 4 minutes, byte-identical)
```
