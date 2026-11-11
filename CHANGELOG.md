# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.0] - Unreleased

This release aligns mice-py with R `mice` 3.19.0 and adds a validation
against it. **Results change for nearly every user**: the defaults are now
R's, several methods were corrected, and runs are no longer seeded with a
fixed number. The documentation page "Upgrading from 0.1 to 0.2"
(`docs/source/upgrading.rst`) says what to change; "Validation against R mice"
and "Differences from R mice" say what was compared and what still differs.

The IDs in parentheses (A1, B2, C3, ...) are the findings listed in
[`simulations/BASELINE.md`](simulations/BASELINE.md); B10 and B11 were found
after that record was written.

### Fixed

- pmm: the regression coefficients were drawn with a transposed Cholesky
  factor, so imputations with `matchtype` 1 or 2 did not follow R's
  distribution (A1).
- `imputation.sampler.estimice(ls_meth="svd")` returned wrong coefficients
  for three or more predictors (A11). Rank-deficient input no longer fails:
  aliased coefficients are 0, with a ridge fallback for the covariance.
- pmm crashed on a categorical target (B1) and when `donors` exceeded the
  number of observed values (B3, now clamped). The optimal scaling of a
  categorical target now follows R's `quantify` (A3). Ties between donors are
  broken at random, with one shuffle per call as in R 3.19.0 (A9).
- Constant, collinear or too many predictors crashed pmm or changed its result
  silently; linearly dependent predictors are now dropped per call, as R's
  `remove.lindep` does (B4).
- midas drew from NumPy's global generator and ignored `rng` (A5); returned
  its donors converted to float (A12); and crashed when R² was 1 or undefined
  (B5), when a missing row had the same predictor values as an observed row,
  that is with any binary or discrete predictor (B6), with a constant column,
  and with one or two observed values.
- Pooling used the normal distribution for p-values and intervals and
  reported R's `lambda` as the fraction of missing information. It now gives
  the numbers of R's `pool()`: t distribution with Barnard-Rubin degrees of
  freedom, and R's `fmi` (A4).
- cart bootstrapped the observed rows and pruned with an absolute
  `ccp_alpha`; rf drew one donor per tree and then one of those. Both now
  follow R's `mice.impute.cart` and `mice.impute.rf` (A6, see Changed).
- cart and rf crashed on an incomplete boolean column (B8) and on a pandas
  `string` predictor (B9); rf crashed when a target had no predictors (B7).
  cart and rf now raise a clear `ValueError` when the target has no observed
  value (B11); pmm and midas do the same.
- `initial="meanobs"` crashed on object columns (B10).
- `md_pattern_like` ordered the rows differently from R's `md.pattern` (A10).
- `xyplot` drew rows with an imputed `x` in the "Observed" panel; the trace
  plot showed the variance instead of R's standard deviation (A8).
- Importing `plotting` seeded NumPy's global generator and changed pandas
  display options (C7).
- Rows without any observed value were dropped, so the completed datasets
  had fewer rows than the data (C4).
- `impute` and the imputers silently ignored unknown keyword arguments (C2).

### Changed

- **Defaults are R's** (A2, A7, C3): method `"pmm"` for every incomplete
  column (was `"sample"`), also for columns left out of a method dict;
  `maxit=5` (was 10); visit sequence `"roman"`, the data order (was
  `"monotone"`); the full predictor matrix (was the port's own quickpred
  with `min_cor=0.1`).
- **Seeds** (C1): `impute(seed=None)` draws fresh randomness on every call;
  0.1 always seeded with 42. Pass `seed` to reproduce a run. Chain *k* gets
  the same random stream whatever `n_imputations` is. A given seed gives
  different numbers than 0.1 did.
- **Setup step** (C6): as in R, `impute` edits the imputation model before
  the chains. A constant or collinear column stops being a predictor; if it is
  incomplete it is not imputed and keeps its missing values. `impute` then
  warns "Number of logged events: n" and records the changes in the new
  `MICE.logged_events`. Entirely missing columns are kept (they used to be
  dropped).
- **Predictor coding** (B2): predictors are dummy-coded as R's `model.matrix`
  does (treatment dummies in level order, polynomial contrasts for ordered
  categoricals, booleans as `<col>TRUE`), and linearly dependent columns are
  dropped per call. pmm and midas now accept categorical, boolean and string
  predictors; cart and rf results with categorical predictors change.
- **cart**: no bootstrap (`bootstrap=True` restores it); `cp=1e-4`, relative
  to the variance of the observed target as rpart's `cp`, replaces the
  absolute `ccp_alpha=1e-4`; `min_samples_split` defaults to
  3 × `min_samples_leaf` (15) and `max_depth` to 30, as rpart; classification
  trees are not pruned unless `ccp_alpha` is given. On a boolean target cart
  and rf classify and return booleans (they returned floats).
- **rf**: one donor drawn from the observed values sharing the leaf in any
  of the trees; `max_features="sqrt"`; `min_samples_split` 5 for a numeric
  target and 2 otherwise; one job (was `n_jobs=-1`; pass
  `forest_params={"n_jobs": -1}`). Without predictors rf draws from the
  observed values.
- **midas** keeps the target's dtype (integers stay integers, booleans stay
  booleans) and accepts boolean targets; categorical and text targets raise
  `ValueError` (R's midastouch fails on factors as well).
- **Pooling**: the `pool()` dict gains `"table"` (one row per term: `m,
  estimate, ubar, b, t, dfcom, df, riv, lambda, fmi, std_error, statistic,
  p_value, conf_low, conf_high`); `"fraction_missing_info"` holds `fmi`;
  `"pooled_scale"` is a float, NaN for models without a scale.
  `pool(summ=True)` shows `df` and `FMI` columns. `pooled_result.df_resid` is
  the complete-data df (was infinity); new attributes `df`, `dfcom` and
  `table`. With one imputation the between variance and FMI are NaN (were 0)
  and a `UserWarning` is issued.
- `PoolingResult.summary(alpha=0.05)` gives t intervals and new `DF` and
  `Lambda` columns; `FMI` is R's fmi. `pool_descriptive_statistics`
  propagates NaN (it used `nanmean`).
- `fit(formula, model=None, model_kwargs=None, **fit_kwargs)` fits any
  statsmodels formula model, not only OLS (C5).
- `MICE.method` gives `""` for complete columns, and `method=""` is accepted
  for a column that should not be imputed. `MICE.predictor_matrix` is square
  in data order and shows the setup step's edits. `MICE.visit_sequence` and
  the chain statistics cover only the columns that are imputed.
  `"monotone"` breaks ties by data order.
- A predictor matrix may cover only some columns (missing entries count as
  0); an incomplete column without a row raises `ValueError` (was
  `KeyError`).
- `initial="meanobs"` uses the most frequent observed value for non-numeric
  columns.
- Unknown keyword arguments raise `TypeError` in `impute` and in every
  imputer; method parameters take the form `<method>_<parameter>`.
  `True`/`False` are rejected as `n_imputations` or `maxit`. A column whose
  dtype cannot be imputed (datetime, complex, ...) raises `TypeError` in
  `MICE()`. pandas nullable dtypes (`Int64`, `Float64`, `string`, `boolean`)
  are supported.
- Nothing is printed any more: warnings and log records instead.
- Plot functions return the matplotlib `Figure` (they returned `None`);
  `plot_missing_data_pattern` returns the `Figure` (it returned the pattern
  table). With `save_path` the figure is saved and only that figure is
  closed. "Nothing to plot" cases warn and return `None`. Figures are at most
  20 inches a side and wrap after four panels; `bwplot` draws outliers;
  `densityplot` marks values without variance with a dashed line; `xyplot`
  with `merge_imputations=True` overlays all imputations, and its panels share
  their axes. By default `stripplot`, `bwplot` and `densityplot` plot the
  numeric columns with missing values, as R's plots take the variables that
  are not factors (0.1 took every incomplete column, and `bwplot` and
  `densityplot` failed on a categorical one); `bwplot`, `densityplot` and
  `densityplot_split` raise `ValueError` for a categorical column passed
  explicitly.
- Logging: every module logs to a logger named after it (`imputation.MICE`,
  `plotting.diagnostics`, ...); records propagate to the root logger (0.1
  stopped them at the `imputation` logger). `configure_logging` and
  `setup_logging` write a log file only with `file_logging=True` (it was the
  default) and touch only the `imputation` logger; `disable_logging`
  silences by level; `reset_logging` restores the state after import. The
  "No logging configured" message is gone.
- Packaging: `imputation.__version__` and `plotting.__version__` come from
  the installed package metadata; `tqdm` and the direct `patsy` dependency
  are dropped (statsmodels brings patsy); both packages ship `py.typed`;
  Python 3.9 to 3.13. Development uses uv, ruff, mypy and pre-commit; the
  `dev` extra became a `dev` dependency group, and a `notebooks` extra is
  new.
- **Licence**: mice-py is now licensed under the GNU General Public License,
  version 2 or later (`GPL-2.0-or-later`; see `LICENSE` and `COPYRIGHT`),
  instead of the MIT License. mice-py is a port derived from the source of R
  mice, which is licensed GPL (>= 2), so it takes the same licence. Versions
  up to and including 0.1.0 remain available under the MIT License.

### Added

- `imputation.quickpred(data, mincor=0.1, minpuc=0.0, include=None,
  exclude=None, method="pearson")` with R's rule.
- `imputation.design`: `obtain_design`, `remove_lindep` and
  `find_collinear`, ports of the R functions behind the setup step.
- `MICE.impute(seed=...)`, `MICE.logged_events`, visit sequences `"arabic"`
  and `"revmonotone"`.
- `imputation.pooling.pool_estimates` (R's `pool()` followed by
  `summary(type = "all", conf.int = TRUE)`) and
  `imputation.pooling.barnard_rubin`.
- New imputer arguments: `pmm(ls_meth=)`; `cart(cp=, min_samples_split=,
  max_depth=, bootstrap=, ccp_alpha=, tree_params=)`; `rf(max_depth=,
  min_samples_leaf=, min_samples_split=, forest_params=)`; `midas(rng=)`.
- `imputation.PMM.quantify` and `imputation.PMM.matchindex`, ports of R's
  `quantify` and `matchindex`.
- Validation against R mice 3.19.0: parity tests against fixtures generated
  in R (`tests/test_parity`, `tests/reference`), invariant and property tests
  (`tests/test_properties`), and a simulation study run with both engines on
  the same data (`simulations/`).
- Documentation pages "Validation against R mice", "Differences from R
  mice", "Upgrading from 0.1 to 0.2", "Diagnostic plots" and "Logging".
- GitHub Actions workflows: lint, type check, tests on Python 3.9 to 3.13,
  docs build with warnings as errors, example notebooks; a manual workflow
  that regenerates the R fixtures.

### Deprecated

- `impute(min_cor=, min_puc=, include=, exclude=, correlation_method=)`. They
  now follow R's quickpred rule and will be removed in the next version; call
  `imputation.quickpred` and pass its result as `predictor_matrix`.
- `impute(output_dir=)`, which never did anything; it is ignored.

### Removed

- `MICE.result` (and its inner `ImputationResult` class), `MICE.run_output_dir`
  and `MICE._quickpred`.
- `imputation.PMM.quantify_cca` (use `quantify`), `imputation.PMM.matcherid`
  (use `matchindex`) and the `matcher` argument of `pmm`.
- `imputation.midas.compute_beta`. The `midas_kappa` argument of `midas` is
  now `kappa`; through `impute` it is still spelled `midas_kappa`.
- Importing `pool_descriptive_statistics` from `imputation.MICE` (import it
  from `imputation.pooling`).
- `imputation.logging_config._config_logger`.
- Free-form keyword arguments of `cart` and `rf` for the scikit-learn
  estimator; use `tree_params` and `forest_params`.
- `plotting/diagnostics_example.ipynb`, which repeated
  `examples/05_diagnostic_plots.ipynb`. `plotting/visualization_example.ipynb`
  is now `examples/06_missing_data_patterns.ipynb`.
- `LOGGING.md`, `imputation/README_pooling.md` and `plotting/README.md`;
  their content is in the documentation.


## [0.1.0] - 2025-05-10

### Added
- Core MICE implementation with five imputation methods:
  - PMM (Predictive Mean Matching)
  - CART (Classification and Regression Trees)
  - Random Forest
  - MIDAS (R's midastouch: predictive mean matching with distance-aided donor selection)
  - Sample (Random sampling from observed values)
- Automatic predictor matrix estimation using a quickpred algorithm
- Rubin's rules for pooling multiple imputation results
- Formula-based model fitting with statsmodels integration
- Diagnostic plotting tools:
  - Stripplots for observed vs imputed values
  - Box plots for distribution comparison
  - Density plots (KDE)
  - XY scatter plots for bivariate relationships
  - Convergence diagnostics for MICE chains
- Missing data pattern visualization
- Configurable logging
- Input validation for all parameters
- Unit and integration tests
- Sphinx documentation with Read the Docs support
- Jupyter notebook examples for plotting and diagnostics

### Features
- Support for mixed data types (numeric and categorical)
- Flexible visit sequences (monotone, random, custom)
- Method-specific parameter passing
- Chain statistics tracking (mean and variance)
- Fraction of missing information (FMI) calculation
- Multiple initial imputation strategies
- Customizable predictor matrices

### Notes
This version was developed as part of a master's thesis at Ludwig Maximilian
University of Munich, Statistics Department.
