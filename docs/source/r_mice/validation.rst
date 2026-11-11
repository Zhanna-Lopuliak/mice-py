Validation against R mice
=========================

mice-py is a port of the R package `mice <https://cran.r-project.org/package=mice>`_.
Version 0.2.0 was compared with **R mice 3.19.0** under **R 4.6.2**, in four
layers. This page describes what was compared, gives the results, says how to
reproduce them and where the evidence stops.

In short: every deterministic piece that was compared gives R's numbers; every
method gives R's distribution of imputed values on the test inputs; in a
simulation with known truth, 18 of 774 Python-minus-R comparisons differ by
more than Monte Carlo error, all of them in one cell: random forest with a
categorical variable (:ref:`rf-categorical`).

The four layers
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Layer
     - What is compared
     - How
   * - L1 exact parity
     - Deterministic pieces: least squares (``estimice``), the coefficient
       draw (``.norm.draw``, with R's random draws replayed in Python),
       ``quantify``, pmm with one donor, donor matching (``matchindex``), the
       setup step of ``mice()`` (methods, predictor matrix, visit sequence,
       logged events), ``model.matrix`` coding, ``remove.lindep``,
       ``find.collinear``, ``quickpred``, ``md.pattern``, pooling
       (``pool()``, ``pool.table()``, ``barnard.rubin()``), midastouch donor
       probabilities.
     - R output is stored as JSON fixtures together with its inputs. Python
       must match to a relative tolerance of 1e-8 (some checks 1e-12);
       counts, labels and 0/1 matrices must be equal.
   * - L2 distribution parity
     - The imputed values of one call of each method on fixed inputs F1 to
       F4, cell by cell; and 200 full pmm chains.
     - R: 20,000 calls per method (rf 5,000). Python: 3,000 (midas and cart
       1,500; rf 300). Tests below.
   * - L3 simulation with known truth
     - Bias, coverage, interval width, RMSE, between-imputation variance B
       and total variance T of pooled estimates; six scenarios, five methods.
     - Python and R impute the **same** incomplete datasets; one Python
       routine computes the metrics for both; every paired difference is
       divided by its Monte Carlo standard error.
   * - L4 invariants
     - Properties every completed dataset must have (no missing value left in
       an imputed column, observed values unchanged, imputations are
       observed values for donor methods, dtypes kept, ...), seeds, argument
       handling, edge cases, pooling identities, the plotting contract.
     - pytest, partly with ``hypothesis`` on random small data frames.

The test inputs of L2 have 40 observed and 20 missing values of the target
each:

- **F1**: predictors ``x1`` and ``x2`` correlated 0.9 plus an independent
  ``x3``; half of the missing rows lie away from the bulk of the observed
  rows, so the uncertainty of the coefficients matters.
- **F2**: one binary predictor, so pmm has to break many ties between donors.
- **F3**: a factor target with three levels, two numeric predictors.
- **F4**: a lognormal target, two numeric predictors.

Tests of L2, per missing cell:

- **T1**: chi-square test that Python's donor counts follow R's; pass if
  p ≥ 1e-7.
- **T2**: z-scores of the mean and of the log-variance of the imputed values;
  pass if \|z\| ≤ 5.5.
- **T3**: the same z-scores for the regression coefficients of the completed
  target on the predictors.

The thresholds are wide because many cells and statistics are tested at once.
That they detect real defects was checked: tests in
``tests/test_parity/test_l2_mutants.py`` put three old defects back (the
transposed Cholesky factor, deterministic tie-breaking, the bootstrapped CART
tree) and assert that the tests flag each of them.

Versions
--------

.. list-table::
   :header-rows: 1

   * - Side
     - Software
   * - R
     - R 4.6.2, mice 3.19.0, ranger 0.18.0, rpart 4.1.27, R's default random
       number generator (Mersenne-Twister, Inversion, Rejection)
   * - Python
     - mice-py 0.2.0, Python 3.13.7, NumPy 2.3.3, pandas 2.3.2, SciPy 1.16.2,
       scikit-learn 1.7.2, statsmodels 0.14.5

The fixtures record the R and mice versions they were made with; the R
scripts stop on any mice version other than 3.19.0.

Results
-------

L1 and L4
~~~~~~~~~

All checks pass. At this version the test suite has 162 L1 tests, 68 L2 tests
(including the tests of the tests), 4 tests of the in-suite L3 simulation and
474 L4 tests, besides 194 unit and workflow tests; all 902 pass and none is
marked as an expected failure. Before
the fixes, 102 L1 checks and 114 L4 checks failed; their causes are listed in
`simulations/BASELINE.md
<https://github.com/Zhanna-Lopuliak/mice-py/blob/main/simulations/BASELINE.md>`_.

L2: distribution of imputed values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Output of ``python -m tests.test_parity._l2_report`` for mice-py 0.2.0: the
smallest T1 p-value over the missing cells, and the largest \|z\| of T2 and T3.
"-" means the statistic does not apply (a factor target has no mean).

.. list-table::
   :header-rows: 1
   :widths: 14 8 12 10 10 46

   * - Method
     - Input
     - T1 min p
     - T2 max \|z\|
     - T3 max \|z\|
     - Notes
   * - pmm
     - F1
     - 0.045
     - 2.40
     - 0.99
     -
   * - pmm
     - F2
     - 0.0092
     - 2.36
     - 1.50
     -
   * - pmm
     - F3
     - 0.013
     - \-
     - \-
     - factor target
   * - pmm
     - F4
     - 0.017
     - 2.19
     - 1.68
     -
   * - midastouch
     - F1
     - 0.023
     - 2.07
     - 1.67
     -
   * - midastouch
     - F2
     - 0.011
     - 2.57
     - 1.51
     -
   * - midastouch
     - F4
     - 0.12
     - 2.04
     - 1.27
     -
   * - cart
     - F1
     - 0.031
     - 2.76
     - 2.35
     - donor sets equal R's in every cell
   * - cart
     - F2
     - 0.0046
     - 1.76
     - 1.19
     - donor sets equal R's in every cell
   * - cart
     - F3
     - 0.044
     - \-
     - \-
     - level proportions within 0.003 of R's
   * - cart
     - F4
     - 0.030
     - 1.84
     - 1.38
     - donor sets equal R's in every cell
   * - cart, ``bootstrap=True``
     - F1
     - 0
     - 122
     - 150
     - not R's algorithm (the 0.1 tree); differs, as expected
   * - rf
     - F1
     - 0.0021
     - 3.02
     - 2.21
     -
   * - rf
     - F2
     - 0.021
     - 2.26
     - 1.48
     -
   * - rf
     - F3
     - 0.083
     - \-
     - \-
     - level proportions within 0.001 of R's
   * - rf
     - F4
     - 0.0017
     - 2.78
     - 3.35
     -
   * - sample
     - F1
     - 0.036
     - 1.88
     - 1.75
     -
   * - sample
     - F3
     - 0.0041
     - \-
     - \-
     - factor target

Every row with the default settings passes (p ≥ 1e-7, \|z\| ≤ 5.5). For CART
with a numeric target the comparison is stricter than in distribution: for
every missing cell, the set of donors Python can draw equals the set R
reached in 2,000 calls, that is scikit-learn grows the same leaves as rpart on
F1, F2 and F4. midastouch is not run on F3 because neither R's midastouch nor
mice-py's midas imputes a factor. rf with a numeric target also passes the
weaker checks (per-cell means within 0.25 R standard deviations of R's mean,
variance ratio between 0.5 and 2). In the chain-level test (200 full pmm
chains; ``mice(data, m = 200, maxit = 5)`` in R) the distribution of the
last chain means and variances over the 200 chains agrees as well (T2).

For comparison, on 0.1.0 six rows failed (pmm on F1 and F2, cart on F1 and
F4, rf on F1 and F4, with \|z\| up to 145) and two crashed (pmm on F3,
midastouch on F2); see ``simulations/BASELINE.md``.

L3: simulation study
~~~~~~~~~~~~~~~~~~~~

Six scenarios; model M is ``x1``, ``x2`` bivariate normal with correlation 0.5
and ``y = 0.5 + x1 + 0.5 x2 + e``.

.. list-table::
   :header-rows: 1
   :widths: 8 44 8 32 8

   * - ID
     - Data
     - n
     - Missingness
     - maxit
   * - S1
     - model M
     - 200
     - ``x1`` 30% MCAR
     - 1
   * - S2
     - model M
     - 200
     - ``x1`` 30% MAR on ``y``
     - 1
   * - S3
     - model M
     - 200
     - ``y`` and ``x1`` 25% each, MAR on ``x2``
     - 5
   * - S4
     - model M
     - 50
     - ``x1`` 50% MCAR
     - 1
   * - S5
     - group ``g`` in {a, b, c}; ``x`` depends on ``g``; ``y`` on ``g`` and ``x``
     - 200
     - ``g`` 30% MAR on ``y``; ``x`` 20% MCAR
     - 5
   * - S6
     - skewed ``w = exp(0.5 x1 + u)``; ``y = 1 + 0.5 w + 0.5 x1 + e``
     - 200
     - ``w`` 30% MAR on ``y``
     - 1

Each scenario is analyzed with OLS of ``y`` on the other variables, plus the
mean of each incomplete variable (S5: the proportion of ``g = c``). The five
methods are forced on both sides (``midastouch`` is ``midas`` in mice-py),
m = 5, full predictor matrix, 1000 replications (rf 500). Per scenario, method
and term, six paired differences are tested: bias, coverage, interval width,
RMSE, mean B and mean T. A difference is flagged when \|z\| > 4.2, a
Bonferroni bound for several hundred comparisons.

Each cell below gives the number of flagged comparisons, out of all
comparisons in the cell, and the largest \|z\| in brackets. "0.1.0" is the
code before the fixes plus the seed fix, which the study needs so that
replications get independent random streams (commit ``fc1d3e8``).

.. list-table::
   :header-rows: 1
   :widths: 20 8 15 15 15 15 15

   * - Scenario
     - Code
     - pmm
     - midastouch
     - cart
     - rf
     - sample
   * - S1 normal, MCAR
     - 0.1.0
     - 0 of 24 (3.4)
     - 0 of 24 (2.1)
     - 14 of 24 (15.0)
     - 11 of 24 (19.3)
     - 0 of 24 (1.9)
   * -
     - 0.2.0
     - 0 of 24 (2.0)
     - 0 of 24 (2.1)
     - 0 of 24 (1.8)
     - 0 of 24 (2.5)
     - 0 of 24 (1.9)
   * - S2 normal, MAR
     - 0.1.0
     - 0 of 24 (1.7)
     - 0 of 24 (2.1)
     - 16 of 24 (19.3)
     - 17 of 24 (22.5)
     - 0 of 24 (1.1)
   * -
     - 0.2.0
     - 0 of 24 (2.3)
     - 0 of 24 (2.1)
     - 0 of 24 (2.6)
     - 0 of 24 (3.4)
     - 0 of 24 (1.1)
   * - S3 two incomplete variables
     - 0.1.0
     - 3 of 30 (5.7)
     - 0 of 30 (2.2)
     - 17 of 30 (13.6)
     - 2 of 30 (22.2)
     - 0 of 30 (1.9)
   * -
     - 0.2.0
     - 0 of 30 (3.0)
     - 0 of 30 (2.2)
     - 0 of 30 (3.5)
     - 0 of 30 (2.4)
     - 0 of 30 (1.9)
   * - S4 small sample
     - 0.1.0
     - 0 of 24 (4.0)
     - 0 of 24 (1.6)
     - 15 of 24 (18.3)
     - 13 of 24 (23.1)
     - 0 of 24 (1.5)
   * -
     - 0.2.0
     - 0 of 24 (2.1)
     - 0 of 24 (1.6)
     - 0 of 24 (1.6)
     - 0 of 24 (4.0)
     - 0 of 24 (1.5)
   * - S5 categorical
     - 0.1.0
     - crashed
     - not possible
     - 22 of 36 (8.8)
     - 22 of 36 (61.6)
     - 0 of 36 (3.2)
   * -
     - 0.2.0
     - 0 of 36 (2.3)
     - not possible
     - 0 of 36 (2.5)
     - **18 of 36 (65.9)**
     - 0 of 36 (3.2)
   * - S6 skewed
     - 0.1.0
     - 9 of 24 (8.9)
     - 0 of 24 (1.3)
     - 17 of 24 (18.9)
     - 6 of 24 (12.3)
     - 0 of 24 (1.0)
   * -
     - 0.2.0
     - 0 of 24 (2.0)
     - 0 of 24 (1.3)
     - 0 of 24 (1.3)
     - 0 of 24 (2.3)
     - 0 of 24 (1.0)

Totals: 184 of 738 comparisons flagged on 0.1.0, 18 of 774 on 0.2.0. Outside
the S5 rf cell the largest \|z\| per method on 0.2.0 is: pmm 3.0, midastouch
2.2, cart 3.5, rf 4.0, sample 3.2. midastouch and sample were already correct
in 0.1.0 (their numbers are the same in both runs); pmm crashed on every S5
dataset in 0.1.0. midastouch cannot run on S5 in either engine: R's
midastouch imputes factor codes that R turns into missing values, and
mice-py's midas raises an error for a categorical target.

The harness itself was checked in the same run. On the complete data (before
values are deleted) Python and R give the same estimates (largest difference
6e-15), the coverage of the complete-data analysis lies within 0.95 ± 3 Monte
Carlo errors in every scenario, and the study's own pooling agrees with R's
``pool()`` to 1e-11.

The full output, with every metric of every cell, is in
``simulations/results/REPORT.md``; the 0.1.0 run is in
``simulations/results/baseline/``.

.. _rf-categorical:

The cell that does not match: random forest with a categorical variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In S5, rf gives different results in mice-py and in R: 18 of 36 comparisons
are flagged. Both engines are biased, in different places:

.. list-table::
   :header-rows: 1

   * - Term (truth)
     - Bias, mice-py
     - Bias, R
     - Coverage, mice-py
     - Coverage, R
   * - ``x`` (0.7)
     - 0.017
     - -0.099
     - 0.952
     - 0.876
   * - ``gb`` (0.5)
     - -0.033
     - 0.047
     - 0.936
     - 0.940
   * - ``gc`` (1.0)
     - -0.110
     - 0.034
     - 0.928
     - 0.952

R underestimates the coefficient of ``x``; mice-py underestimates the effect
of group c.

**Cause, measured.** R's rf calls ranger, mice-py calls scikit-learn. With
three predictors each split considers one randomly chosen feature
(``floor(sqrt(3)) = 1``). When that feature cannot split a node, ranger makes
the node a leaf; scikit-learn goes on to try the other features (documented
scikit-learn behavior: the search for a split does not stop until a valid
split is found). With dummy-coded categorical predictors, which often cannot
split a node, this makes ranger's trees much shallower. A probe with 300 trees
on the same S5 data measured, on average:

.. list-table::
   :header-rows: 1

   * - Predictors
     - Leaves per tree, ranger
     - Leaves per tree, scikit-learn
   * - two dummies of ``g`` plus one continuous
     - 6.9
     - 53.2
   * - three continuous
     - 66.3
     - 52.8

No combination of scikit-learn settings reproduces ranger's rule, so this
difference stays. It concerns rf with categorical predictors; rf agrees with
R in the five scenarios without categorical data, and in L2 on F1 to F4.

How to reproduce
----------------

From the repository root, with the development environment (``uv sync``):

.. code-block:: bash

   uv run pytest -m parity                        # L1, L2 and the in-suite L3 against the committed fixtures
   uv run pytest tests/test_properties            # L4
   uv run python -m tests.test_parity._l2_report  # the L2 table above, about 30 seconds

The R side needs R with mice 3.19.0, ranger, rpart and jsonlite:

.. code-block:: bash

   Rscript tests/reference/r/generate_all.R       # regenerate the fixtures
   git diff --exit-code tests/reference/fixtures  # they must not change

   uv run python simulations/study.py             # generate the datasets, impute with mice-py
   Rscript simulations/run_r.R                    # impute the same datasets with R mice
   uv run python simulations/report.py            # simulations/results/REPORT.md and summary.csv

The fixtures are committed, so the tests run without R. Details:
``tests/reference/README.md`` (fixtures) and ``simulations/README.md``
(scenarios, metrics, options, runtime).

Limits of the evidence
----------------------

- **Agreement is in distribution, not draw by draw.** mice-py uses NumPy's
  random number generators; no imputation equals R's for the same seed. A
  difference smaller than the tests can detect (about four Monte Carlo
  standard errors in L3) would go unnoticed.
- **The inputs are small.** L2 uses four inputs of a few dozen rows; L3 uses
  n = 50 and 200, at most three variables, one categorical variable with
  three levels, maxit 1 or 5. Many columns, many factor levels, high
  missingness rates, strongly non-linear data and large n were not compared.
- **"Matches R" is not "is correct".** The pass rule of L3 is agreement with
  R. Where R itself under-covers, so does mice-py: in S5 the coverage of
  P(g = c) is 0.89 (mice-py) and 0.90 (R) with pmm, 0.87 and 0.86 with cart,
  against 0.93 for the complete data.
- **Trees.** scikit-learn is neither rpart nor ranger. CART with a numeric
  target grows rpart's leaves on the inputs checked; CART with a categorical
  target and random forests are checked with weaker criteria (level
  proportions, per-cell means within 0.25 R standard deviations, variance
  ratios), and rf differs from R with categorical predictors (above).
- **Not compared with R**: the methods R has and mice-py does not (see
  :doc:`differences`), ``initial="meanobs"`` and ``visit_sequence="random"``
  (not in R), the strip plot, and the plots' appearance in general.
- **Versions.** The numbers above come from the Python versions listed. The
  test suite also passes on Python 3.9 with NumPy 2.0, SciPy 1.13 and
  scikit-learn 1.6; other versions of scikit-learn may grow different trees.
