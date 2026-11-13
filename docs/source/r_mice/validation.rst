Validation against R mice
=========================

mice-py is a port of the R package `mice <https://cran.r-project.org/package=mice>`_.
Version 0.3.0 was compared with **R mice 3.19.0** under **R 4.6.2** (with
nnet 7.3-21 and MASS 7.3-66 for ``polyreg`` and ``polr``), in four layers.
This page describes what was compared, gives the results, says how to
reproduce them and where the evidence stops.

In short: every deterministic piece that was compared gives R's numbers;
the nine model-based methods, given R's random numbers, give R's
imputations; every method gives R's distribution of imputed values on the
test inputs; in a simulation with known truth, 18 of 1044 Python-minus-R
comparisons differ by more than Monte Carlo error, all of them in one cell:
random forest with a categorical variable (:ref:`rf-categorical`). The
agreement is within the tolerances and thresholds the tests enforce, not
identity.

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
       logged events), ``make.method``, ``model.matrix`` coding,
       ``remove.lindep``, ``find.collinear``, ``quickpred``, ``md.pattern``,
       pooling (``pool()``, ``pool.table()``, ``barnard.rubin()``),
       midastouch donor probabilities. The nine model-based methods
       (``norm``, ``norm.nob``, ``norm.boot``, ``norm.predict``, ``mean``,
       ``logreg``, ``logreg.boot``, ``polyreg``, ``polr``) with every random
       number of R replayed in Python, and the pieces behind them:
       ``mice:::augment``, ``glm.fit`` with the logit link, the BFGS of
       ``optim`` (``vmmin``), ``nnet::multinom``, ``MASS::polr``.
     - R output is stored as JSON fixtures together with its inputs. Python
       must match to a relative tolerance of 1e-8 (some checks 1e-12, the
       fits of ``polyreg`` and ``polr`` 1e-6, see below); counts, labels and
       0/1 matrices must be equal.
   * - L2 distribution parity
     - The imputed values of one call of each method on fixed inputs F1 to
       F7, cell by cell; 200 full pmm chains; full chains with R's default
       methods on a mixed frame.
     - R: 20,000 calls per method (rf 5,000). Python: 3,000 (midas and cart
       1,500; polr 2,000; rf 300). Tests below.
   * - L3 simulation with known truth
     - Bias, coverage, interval width, RMSE, between-imputation variance B
       and total variance T of pooled estimates; eight scenarios, 41 cells
       (a method on a scenario).
     - Python and R impute the **same** incomplete datasets; one Python
       routine computes the metrics for both; every paired difference is
       divided by its Monte Carlo standard error.
   * - L4 invariants
     - Properties every completed dataset must have (no missing value left in
       an imputed column, observed values unchanged, imputations are
       observed values for donor methods and labels of the column for the
       categorical methods, finite floats for the norm methods, dtypes kept,
       ...), seeds, argument handling, the default method of each column
       type, edge cases, pooling identities, the plotting contract.
     - pytest, partly with ``hypothesis`` on random small data frames.

L1 replays R's random numbers through a scripted generator: the R script
records the draws of each call (``rchisq``, ``rnorm``, ``runif``, ``sample``)
and proves, by comparing ``.Random.seed``, that it captured all of them and
in the right order; the Python test serves exactly these numbers and refuses
any other draw, then checks that every number was used, in R's order, with
the same chi-square degrees of freedom and bootstrap bounds. The fits are
reproduced step for step, not merely to the same optimum:

- ``glm.fit`` (``logreg``, ``logreg.boot``, the start values of ``polr``):
  the same start, clamps of the logit link, rank rule and convergence rule,
  and ``cov.unscaled`` from the last weighted least squares, as
  ``summary.glm`` returns it. Coefficients, ``cov.unscaled``, drawn
  coefficients and probabilities agree with R to a relative tolerance of
  1e-8; the one fit in the fixtures that does not converge (a separated
  bootstrap sample of ``logreg.boot``) is compared at 1e-6 and without its
  covariance, which ``logreg.boot`` does not use.
- ``optim(method = "BFGS")`` (R's C function ``vmmin``, which ``nnet`` and
  ``MASS::polr`` call): on ten test functions the numbers of function and
  gradient evaluations and the convergence flag equal R's, and the
  parameters and the value agree to 1e-8.
- ``nnet::multinom`` (``polyreg``, the fallback of ``polr``) and
  ``MASS::polr`` (``polr``): they stop where R stops, at R's tolerances and
  iteration limit, not at the exact maximum. Probabilities of the imputed
  cells agree with R's to 1e-6 (absolute), the objective to 1e-8
  (relative), and coefficients and cut points to 1e-6 where the fit
  converged and is well conditioned. Which fit R made (``polr`` or its
  fallback to ``multinom``) and why must match.
- The added rows of ``augment`` agree to 1e-12. Every uniform number in the
  fixtures is at least 1e-4 away from every cumulative probability (checked
  in R), so the imputed labels must be identical.

The L1 fixtures of the new methods (cases): ``l1_norm`` (38), ``l1_augment``
(14), ``l1_logreg`` (24), ``l1_vmmin`` (10), ``l1_polyreg`` (19, plus 3
direct ``multinom`` fits), ``l1_polr`` (23, plus 8 direct ``MASS::polr``
fits), ``l1_make_method`` (one frame with a column of every type). They
include the cases where R stops or imputes unusable values; there the
fixture records R's outcome and the test checks what mice-py does instead
(:doc:`differences`).

The test inputs of L2 have 40 observed and 20 missing values of the target
each:

- **F1**: predictors ``x1`` and ``x2`` correlated 0.9 plus an independent
  ``x3``; half of the missing rows lie away from the bulk of the observed
  rows, so the uncertainty of the coefficients matters.
- **F2**: one binary predictor, so pmm has to break many ties between donors.
- **F3**: a factor target with three levels, two numeric predictors.
- **F4**: a lognormal target, two numeric predictors.
- **F5**: a binary target (``no``, ``yes``) from a logistic model on two
  standard normal predictors; half of the missing rows lie far out, near
  the line where the linear predictor is 0, so the labels depend on the
  poorly determined slopes.
- **F6**: a binary target perfectly separated by ``x1`` in the observed
  rows, so the augmentation of ``logreg`` decides the result.
- **F7**: an ordered target with four levels, drawn from a multinomial model
  that is not a proportional-odds model, so that ``polr`` and ``polyreg``
  predict probabilities that differ by at least 0.15 for the missing rows.

The methods fall into three kinds, which decide what is compared:
**donor** methods (pmm, midastouch, cart, rf, sample) impute observed values;
**continuous** methods (``norm``, ``norm.nob``, ``norm.boot``) impute new
numbers; **category** methods (``logreg``, ``logreg.boot``, ``polyreg``,
``polr``) impute labels drawn from a fitted model. ``norm.predict`` and
``mean`` draw nothing and need no L2. Tests of L2, per missing cell:

- **T1**: chi-square test that Python's counts follow R's: donor counts for
  donor methods, the counts in the ten bins cut at the deciles of R's
  draws for continuous methods, level counts for category methods; pass if
  p ≥ 1e-7.
- **T2**: z-scores of the mean and of the log-variance of the imputed values;
  pass if \|z\| ≤ 5.5. Not for category methods.
- **T3**: the same z-scores for the regression coefficients of the completed
  target on the predictors; for category methods, of the coefficients of
  each level indicator of the completed target regressed on ``[1, x]``.
  T3 sees the dependence between the cells of one call that a shared
  coefficient draw or bootstrap sample creates.

At chain level, ``l2_chain_default`` compares 120 Python chains of
``impute()`` without ``method`` with 200 chains of R's ``mice()`` without
``method``, on a frame of 100 rows with a numeric, a binary, an unordered
three-level and an ordered four-level incomplete column. Both choose pmm,
logreg, polyreg and polr; the mean and variance of the imputed numeric
values and the share of each level among the imputed labels, in the last
iteration, are compared with the z-scores of T2.

The thresholds are wide because many cells and statistics are tested at once.
That they detect real defects was checked by putting defects back on
purpose and asserting that the tests flag each of them:

- ``tests/test_parity/test_l2_mutants.py``: three old defects (the
  transposed Cholesky factor, deterministic tie-breaking, the bootstrapped
  CART tree), and for the new methods ``norm`` without the parameter draw
  (T1, T2 and T3 on F1), ``logreg`` without augmentation (T1 and T3 on F6),
  ``logreg`` without the coefficient draw (T1 and T3 on F5), ``polyreg``
  taking the most probable category instead of a draw (T1 and T3 on F3), and
  ``polyreg`` in place of ``polr`` (T1 and T3 on F7).
- ``tests/test_parity/test_l1_mutants.py``: 30 defects in the new methods,
  29 of them caught by an L1 check, among them a wrong chi-square df, the
  textbook residual df in ``norm.nob``, a bootstrap with wrong bounds or
  without resampling, the augmentation without clamping, with weight 1 or
  applied with one missing value, the covariance of ``logreg`` scaled by
  the dispersion or recomputed at the final coefficients, the declared
  instead of the sorted level order, ``multinom`` run to full convergence
  instead of stopping where nnet stops, the softmax instead of nnet's
  logistic unit for two classes, ``polr`` without its fallback or falling
  back when R does not, other start values for ``polr``, and the uniform
  numbers drawn at the wrong time. The remaining one, ``u >= p`` instead of
  R's ``u > p`` in the category draw, cannot show in any fixture (every
  number is at least 1e-4 from every cumulative probability); a unit test
  of the draw catches it.
- ``tests/test_parity/test_l2_chain_default.py``: the default of 0.2, pmm
  for every column, is caught by the comparison of the methods with R's.
  The distribution statistics of the chain-level test do not flag it
  clearly on that frame (below).

Versions
--------

.. list-table::
   :header-rows: 1

   * - Side
     - Software
   * - R
     - R 4.6.2, mice 3.19.0, nnet 7.3-21, MASS 7.3-66, ranger 0.18.0, rpart
       4.1.27, R's default random number generator (Mersenne-Twister,
       Inversion, Rejection)
   * - Python
     - mice-py 0.3.0, Python 3.13.7, NumPy 2.3.3, pandas 2.3.2, SciPy 1.16.2,
       scikit-learn 1.7.2, statsmodels 0.14.5

The fixtures record the R and mice versions they were made with (those of
``polyreg`` and ``polr`` also the nnet and MASS versions); the R scripts stop
on any mice version other than 3.19.0.

Results
-------

L1 and L4
~~~~~~~~~

All checks pass. At this version the test suite has 763 L1 tests (31 of
them put defects back, see above) and 26 tests of the scripted random
generator that L1 uses, 112 L2 tests (including the tests of the tests), 6
tests of the in-suite L3 simulation and 1304 L4 tests, besides 276 unit and
workflow tests; all 2487 pass and none is marked as an expected failure.
Version 0.2.0 had 917 tests. Before the fixes of 0.2, 102 L1 checks and 114
L4 checks failed; their causes are listed in `simulations/BASELINE.md
<https://github.com/Zhanna-Lopuliak/mice-py/blob/main/simulations/BASELINE.md>`_.

In L1, ``MICE.impute()`` without ``method`` chooses R's methods for every
column of R's ``nhanes2`` (``logreg`` for ``hyp``) and ``boys`` (``polr``
for ``gen`` and ``phb``, ``polyreg`` for ``reg``), and ``make_method`` gives
R's ``make.method`` for a column of every type that R tells apart, with the
default and with other default methods, except the text column: R gives it
the first default method and then does not impute it, mice-py treats it as
an unordered factor (:doc:`differences`).

L2: distribution of imputed values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Output of ``python -m tests.test_parity._l2_report`` for mice-py 0.3.0: the
smallest T1 p-value over the missing cells, and the largest \|z\| of T2 and T3.
"-" means the statistic does not apply (a factor target has no mean, and
category methods have no T2). The rows of the donor methods are those of
0.2.0, unchanged.

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
   * - norm
     - F1
     - 0.025
     - 2.88
     - 1.53
     - continuous; T1 on R's decile bins
   * - norm
     - F4
     - 0.018
     - 2.63
     - 1.96
     - continuous; T1 on R's decile bins
   * - norm.nob
     - F1
     - 0.025
     - 3.18
     - 1.09
     - continuous; T1 on R's decile bins
   * - norm.boot
     - F1
     - 0.034
     - 2.00
     - 0.92
     - continuous; T1 on R's decile bins
   * - logreg
     - F5
     - 0.093
     - \-
     - 1.11
     - category; T1 on level counts
   * - logreg
     - F6
     - 0.037
     - \-
     - 2.82
     - category; observed rows separated
   * - logreg.boot
     - F5
     - 0.13
     - \-
     - 1.22
     - category
   * - polyreg
     - F3
     - 0.011
     - \-
     - 1.49
     - category
   * - polr
     - F7
     - 0.016
     - \-
     - 1.66
     - category; ordered target

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

In the chain-level test with the default methods both engines choose pmm,
logreg, polyreg and polr, neither logs an event, and the 16 z-scores of the
eight statistics (120 Python chains against R's 200) are at most 1.67 in
absolute value. That test cannot tell pmm from the model-based defaults by
distribution: the same 120 chains with ``method="pmm"`` reach a largest
\|z\| of 3.2 to 5.6, depending on the seed, because on that frame pmm draws
from nearly the same conditional distributions. It therefore asserts the
methods R chose, not only the distribution.

For comparison, on 0.1.0 six rows failed (pmm on F1 and F2, cart on F1 and
F4, rf on F1 and F4, with \|z\| up to 145) and two crashed (pmm on F3,
midastouch on F2); see ``simulations/BASELINE.md``.

L3: simulation study
~~~~~~~~~~~~~~~~~~~~

Eight scenarios; model M is ``x1``, ``x2`` bivariate normal with correlation
0.5 and ``y = 0.5 + x1 + 0.5 x2 + e``. In S5, S7 and S8 the categorical
variable is drawn first, then ``x`` given it, then ``y`` given both.

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
   * - S7
     - binary ``b`` in {no, yes} (0.6, 0.4); ``x`` depends on ``b``; ``y``
       on ``b`` and ``x``
     - 200
     - ``b`` 30% MAR on ``y``; ``x`` 20% MCAR
     - 5
   * - S8
     - ordered ``o``, lo < mid < hi < top (0.2, 0.3, 0.3, 0.2); ``x``
       depends on ``o``; ``y`` on ``o`` and ``x``
     - 200
     - ``o`` 30% MAR on ``y``; ``x`` 20% MCAR
     - 5

Each scenario is analyzed with OLS of ``y`` on the other variables (with
treatment dummies for a categorical variable, also the ordered ``o``), plus
the mean of each incomplete variable (S5: the proportion of ``g = c``; S7:
of ``b = yes``; S8: of ``o = top``). The study runs 41 cells, each a study
method on a scenario: the five donor methods on S1 to S6 (the 30 cells of
0.2), and the new cells below. The methods are forced on both sides
(``midastouch`` is ``midas`` in mice-py), except in the ``default`` cells,
where neither engine gets a ``method`` argument. m = 5, full predictor
matrix, 1000 replications (rf 500). Per scenario, method and term, six
paired differences are tested: bias, coverage, interval width, RMSE, mean B
and mean T. A difference is flagged when \|z\| > 4.2, a Bonferroni bound for
several hundred comparisons.

``norm.predict`` and ``mean`` draw no random numbers, so on S4 (one
incomplete column, one iteration) both engines must give the same completed
data, and the paired z would be 0/0. These cells follow an **identity rule**
instead: the largest \|qbar Python − qbar R\| over replications and terms
must be at most 1e-8, and B must be 0 in both engines (at most 1e-16, to
allow for rounding).

Each cell below gives the number of flagged comparisons, out of all
comparisons in the cell, and the largest \|z\| in brackets. "0.1.0" is the
code before the fixes plus the seed fix, which the study needs so that
replications get independent random streams (commit ``fc1d3e8``). The 0.3.0
run reproduced the 30 cells of 0.2.0 exactly: all 1248 rows of the 0.2.0
``summary.csv`` (every metric, Monte Carlo error and z) are identical in
the 0.3.0 one, so the row "0.2.0, 0.3.0" holds for both versions.

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
     - 0.2.0, 0.3.0
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
     - 0.2.0, 0.3.0
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
     - 0.2.0, 0.3.0
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
     - 0.2.0, 0.3.0
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
     - 0.2.0, 0.3.0
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
     - 0.2.0, 0.3.0
     - 0 of 24 (2.0)
     - 0 of 24 (1.3)
     - 0 of 24 (1.3)
     - 0 of 24 (2.3)
     - 0 of 24 (1.0)

The cells added in 0.3.0 (S1, S2 and S6 have no new cell). "default" means
no ``method`` argument in either engine; both chose the methods shown in all
1000 replications.

.. list-table::
   :header-rows: 1
   :widths: 12 16 36 36

   * - Scenario
     - Study method
     - Methods of the incomplete columns
     - Result
   * - S3
     - ``norm``
     - ``x1``, ``y``: norm
     - 0 of 30 (1.8)
   * - S3
     - ``norm.boot``
     - ``x1``, ``y``: norm.boot
     - 0 of 30 (1.5)
   * - S4
     - ``norm``
     - ``x1``: norm
     - 0 of 24 (2.8)
   * - S4
     - ``norm.boot``
     - ``x1``: norm.boot
     - 0 of 24 (1.7)
   * - S4
     - ``norm.nob``
     - ``x1``: norm.nob
     - 0 of 24 (2.0)
   * - S4
     - ``norm.predict``
     - ``x1``: norm.predict
     - identity rule: largest \|qbar Python − qbar R\| 3.3e-15; B = 0
   * - S4
     - ``mean``
     - ``x1``: mean
     - identity rule: largest \|qbar Python − qbar R\| 2.2e-15; B = 0
   * - S5
     - default
     - ``g``: polyreg, ``x``: pmm
     - 0 of 36 (2.1)
   * - S7
     - default
     - ``b``: logreg, ``x``: pmm
     - 0 of 30 (3.1)
   * - S7
     - ``logreg.boot``
     - ``b``: logreg.boot, ``x``: pmm
     - 0 of 30 (1.6)
   * - S8
     - default
     - ``o``: polr, ``x``: pmm
     - 0 of 42 (2.1)

Totals: 184 of 738 comparisons flagged on 0.1.0, 18 of 774 on 0.2.0, and 18
of 1044 on 0.3.0 (the 774 of the cells of 0.2 and 270 in the new cells, of
which none is flagged; the largest \|z\| there is 3.14, for the coverage of
the mean of ``x`` in S7 with the default methods). Outside the S5 rf cell
the largest \|z\| per method on 0.2.0 and 0.3.0 is: pmm 3.0, midastouch 2.2,
cart 3.5, rf 4.0, sample 3.2. midastouch and sample were already correct in
0.1.0 (their numbers are the same in both runs); pmm crashed on every S5
dataset in 0.1.0. midastouch cannot run on S5 in either engine: R's
midastouch imputes factor codes that R turns into missing values, and
mice-py's midas raises an error for a categorical target.

``norm.nob``, ``norm.predict`` and ``mean`` are improper methods: they ignore
the uncertainty of the imputation model (``norm.predict`` and ``mean`` also
the residual noise), so their intervals cover less than 95% in both engines,
by design. On S4 the coverage of the coefficient of ``x1`` under
``norm.predict`` is 0.35, and under ``mean`` the coverage of the coefficient
of ``x2`` is 0.63 and of the mean of ``x1`` 0.66, identical in both engines;
under ``norm.nob`` it lies between 0.93 and 0.95 in mice-py and between 0.92
and 0.94 in R. The pass rule is the same as everywhere: mice-py matches R.

The harness itself was checked in the same run. On the complete data (before
values are deleted) Python and R give the same estimates (largest difference
8.7e-15 over all scenarios), the coverage of the complete-data analysis lies
within 0.95 ± 3 Monte Carlo errors in every scenario, and the study's own
pooling agrees with R's ``pool()`` to 1e-11.

The full output, with every metric of every cell, is in
``simulations/results/REPORT.md``; the 0.1.0 run is in
``simulations/results/baseline/``. A small version runs with the tests
(``tests/test_parity/test_l3_mini_simulation.py``): S4 with pmm, midas and
norm, 200 replications in Python against 20,000 stored R replications.

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

The R side needs R with mice 3.19.0, ranger, rpart, nnet, MASS and jsonlite
(nnet and MASS come with R):

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

- **Agreement of whole runs is in distribution, not draw by draw.** mice-py
  uses NumPy's random number generators; no imputation of a run equals R's
  for the same seed. Draw by draw, with R's random numbers fed to mice-py,
  only single method calls were compared (L1). A difference smaller than
  the tests can detect (about four Monte Carlo standard errors in L3) would
  go unnoticed.
- **The inputs are small.** L2 uses seven inputs of a few dozen rows; L3 uses
  n = 50 and 200, at most three variables, one categorical variable with at
  most four levels, maxit 1 or 5. Many columns, many factor levels, high
  missingness rates, strongly non-linear data and large n were not compared.
- **The chain-level test of the defaults is weak on distribution.** On its
  frame pmm draws from nearly the same conditional distributions as logreg,
  polyreg and polr, so the distribution statistics alone would not flag the
  default of 0.2 (pmm for every column) clearly; the test relies on the
  comparison of the chosen methods with R's. The model-based methods
  themselves are checked by L1 and L2.
- **Fixtures from one platform.** The R fixtures were made on macOS; CI
  also runs the tests on Linux, where they pass for 0.3.0. The optimiser of
  ``polyreg`` and ``polr`` takes up to 100 steps along which rounding can
  differ between R's C code and NumPy and between platforms; hence probabilities at 1e-6, and
  coefficients compared only for converged, well-conditioned fits. The one
  ``glm.fit`` in the fixtures that does not converge (a separated bootstrap
  sample of ``logreg.boot``) is compared at 1e-6 instead of 1e-8, without its
  covariance.
- **"Matches R" is not "is correct".** The pass rule of L3 is agreement with
  R. Where R itself under-covers, so does mice-py: in S5 the coverage of
  P(g = c) is 0.89 (mice-py) and 0.90 (R) with pmm, 0.87 and 0.86 with cart,
  0.90 and 0.90 with the default methods (polyreg), against 0.93 for the
  complete data; in S8 the coverage of P(o = top) with the default methods
  (polr) is 0.91 in both engines, against 0.95 for the complete data. The
  improper methods ``norm.nob``, ``norm.predict`` and ``mean`` under-cover in
  both engines by design.
- **Trees.** scikit-learn is neither rpart nor ranger. CART with a numeric
  target grows rpart's leaves on the inputs checked; CART with a categorical
  target and random forests are checked with weaker criteria (level
  proportions, per-cell means within 0.25 R standard deviations, variance
  ratios), and rf differs from R with categorical predictors (above).
- **Deliberate differences.** Where R stops or imputes unusable values,
  mice-py does something else (:doc:`differences`); the fixtures record R's
  outcome and the tests check mice-py's, which is therefore not R's. The
  one difference in the default methods is a text column, which R does not
  impute and mice-py imputes as an unordered factor; it is pinned in L1
  but not compared in distribution.
- **Not compared with R**: the methods R has and mice-py does not (see
  :doc:`differences`); ``initial="meanobs"`` and ``visit_sequence="random"``
  (not in R); ``norm.predict`` and ``mean`` in L2 (they draw nothing; L1 and
  the identity rule of L3 compare them); ``logreg.boot`` on separated data
  in L2 (L1 has one such bootstrap sample); the imputation of a column with
  a predictor that keeps missing values by the methods that augment, whose
  added rows are computed from fewer rows than R's; the strip plot, and the
  plots' appearance in general.
- **Versions.** The numbers above come from the Python versions listed.
  Version 0.2.0 also passed its test suite on Python 3.9 with NumPy 2.0,
  SciPy 1.13 and scikit-learn 1.6; for 0.3.0 that has not been checked yet.
  Other versions of scikit-learn may grow different trees.
