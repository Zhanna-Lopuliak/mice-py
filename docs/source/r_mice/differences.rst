Differences from R mice
=======================

mice-py follows R mice 3.19.0. This page is for readers who know R mice: it
lists the defaults the two share, how R's names map to mice-py's, and every
known difference, each with the reason for it. How the agreement was checked
is on :doc:`validation`.

Defaults shared with R
----------------------

Since version 0.2.0 an ``impute()`` call without arguments does what
``mice(data)`` does in R, except for the methods of categorical columns.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - mice-py
     - R mice
   * - Number of imputations
     - ``n_imputations=5``
     - ``m = 5``
   * - Iterations
     - ``maxit=5``
     - ``maxit = 5``
   * - Method of a numeric column
     - ``"pmm"``
     - ``"pmm"``
   * - Method of a categorical column
     - ``"pmm"``
     - ``"logreg"``, ``"polyreg"`` or ``"polr"``
   * - Method of a complete column
     - ``""``
     - ``""``
   * - Predictor matrix
     - all ones, zero diagonal
     - all ones, zero diagonal
   * - Visit sequence
     - ``"roman"`` (data order)
     - ``"roman"``
   * - Initial values
     - random draws from the observed values
     - random draws from the observed values
   * - Setup step
     - constant and collinear variables removed, logged events
     - the same (``mice.edit.setup``)
   * - Per call
     - predictors coded as ``model.matrix``; ``remove.lindep``
     - the same
   * - pmm
     - ``donors=5``, ``matchtype=1``, ``quantify=True``, ``ridge=1e-5``,
       ``ls_meth="qr"``
     - ``donors = 5``, ``matchtype = 1``, ``quantify = TRUE``,
       ``ridge = 1e-05``, ``ls.meth = "qr"``
   * - midastouch
     - ``kappa=None``, ``outout=True``, ``ridge=1e-5``
     - ``midas.kappa = NULL``, ``outout = TRUE``, ``ridge = 1e-05``
   * - cart
     - ``min_samples_leaf=5``, ``cp=1e-4``; split size 15, depth 30
     - ``minbucket = 5``, ``cp = 1e-04``; rpart's ``minsplit``, ``maxdepth``
   * - rf
     - ``n_estimators=10``, square root of the number of predictors per split
     - ``ntree = 10``, ranger's ``mtry``
   * - quickpred
     - ``mincor=0.1``, ``minpuc=0.0``, ``method="pearson"``
     - ``mincor = 0.1``, ``minpuc = 0``, ``method = "pearson"``
   * - Pooling
     - Rubin's rules, Barnard-Rubin df, ``dfcom`` from the first fit
     - the same (``pool()``)

Names
-----

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - R mice
     - mice-py
   * - ``imp <- mice(data, m = 5, seed = 1)``
     - ``mice = MICE(data)``; ``mice.impute(n_imputations=5, seed=1)``
   * - ``predictorMatrix``, ``visitSequence``, ``method``, ``maxit``
     - ``predictor_matrix``, ``visit_sequence``, ``method``, ``maxit``
   * - method ``"midastouch"``
     - method ``"midas"``
   * - ``mice(data, donors = 3)`` (passed to every method)
     - ``impute(pmm_donors=3)`` (method name as prefix)
   * - ``complete(imp, k)``
     - ``mice.imputed_datasets[k - 1]``
   * - ``imp$method``, ``imp$predictorMatrix``, ``imp$visitSequence``,
       ``imp$loggedEvents``
     - ``mice.method``, ``mice.predictor_matrix``, ``mice.visit_sequence``,
       ``mice.logged_events``
   * - ``imp$chainMean``, ``imp$chainVar`` (array variable × iteration ×
       imputation)
     - ``mice.chain_mean``, ``mice.chain_var`` (dict of arrays iteration ×
       imputation, numeric columns only)
   * - ``fit <- with(imp, lm(y ~ x))``
     - ``mice.fit("y ~ x")``
   * - ``summary(pool(fit), type = "all", conf.int = TRUE)``
     - ``mice.pool()["table"]``
   * - ``pool.scalar()``
     - ``imputation.pooling.pool_estimates()``
   * - ``quickpred()``
     - ``imputation.quickpred()``
   * - ``md.pattern()``
     - ``plotting.md_pattern_like()``, ``plotting.plot_missing_data_pattern()``
   * - ``stripplot(imp)``, ``bwplot(imp)``, ``densityplot(imp)``,
       ``xyplot(imp, y ~ x)``, ``plot(imp)``
     - ``plotting.stripplot``, ``bwplot``, ``densityplot``, ``xyplot``,
       ``plot_chain_stats`` (see :doc:`../user_guide/diagnostic_plots`)
   * - ``mice.impute.pmm()``, ``mice.impute.midastouch()``,
       ``mice.impute.cart()``, ``mice.impute.rf()``, ``mice.impute.sample()``
     - ``imputation.PMM.pmm()``, ``imputation.midas.midas()``,
       ``imputation.cart.cart()``, ``imputation.rf.rf()``,
       ``imputation.sample.sample()``

Methods and arguments that mice-py does not have
-------------------------------------------------

Version 0.2 ports five methods and the core of ``mice()``; the rest is not
ported yet.

- **Methods**: everything except ``pmm``, ``midastouch``, ``cart``, ``rf`` and
  ``sample``: no ``norm`` and its variants, ``mean``, ``logreg``,
  ``polyreg``, ``polr``, ``lda``, multilevel (``2l.*``) or passive methods.
  Because ``logreg``, ``polyreg`` and ``polr`` are missing, **pmm is the
  default for every column type**; for a categorical column this gives the
  results of R's ``mice(..., method = "pmm")``, not those of R's default.
- **Arguments of mice()**: ``where``, ``blocks``, ``formulas``, ``blots``,
  ``post``, ``ignore``, ``calltype`` and passive imputation (``"~ I(...)"``).
  ``data.init`` (a complete data set of starting values) is replaced by
  ``initial`` (see below).
- **Continuing a run**: there is no ``mice.mids()``; each ``impute()`` call
  starts new chains.
- **Arguments of the methods**: pmm's ``exclude``, ``trim`` and
  ``use.matcher``; midastouch's ``neff`` and ``debug``; rf's ``rfPackage``
  (mice-py always uses scikit-learn).
- **Other functions**: ``ampute``, ``md.pairs``, ``flux``, ``fluxplot``,
  ``pool.r.squared``, and the model comparisons ``D1``, ``D2``, ``D3``.

Arguments that behave differently
---------------------------------

- **maxit must be at least 1** (R allows 0). ``impute()`` always runs the
  chains; the setup it decided on is in ``method``, ``predictor_matrix`` and
  ``logged_events`` after any run.
- **Random numbers never equal R's.** ``seed`` seeds a NumPy
  ``SeedSequence``, from which each chain gets its own stream; R uses
  ``set.seed`` and one stream. The generators differ, so agreement with R is
  checked in distribution, not draw by draw.
- **Method parameters carry the method name** (``pmm_donors=3``,
  ``cart_cp=0.001``) and an unknown keyword argument raises ``TypeError``. R
  passes ``...`` to every method and ignores names a method does not use; the
  prefix makes a misspelled argument an error instead of a silent no-op.
- **initial**: ``"sample"`` is R's initialization; ``"meanobs"`` (the
  observed value closest to the mean, or the most frequent value) exists only
  in mice-py. It was part of mice-py 0.1 and is kept.
- **visit_sequence="random"** exists only in mice-py, kept from 0.1. It is
  drawn once per ``impute()`` call from the seed.
- **visit_sequence after a run** holds the order actually used. R's
  ``visitSequence`` can list variables it never imputes (complete ones, and
  variables whose method the setup step set to ``""``); mice-py lists only
  the variables it visits.
- **A predictor matrix may cover only some columns**; missing entries count
  as 0, and ``predictor_matrix`` after a run is the full square matrix. R
  requires a square matrix. Every incomplete column still needs a row.
- **Logged events** produce one ``UserWarning`` ("Number of logged events:
  n") and log records; R prints a message. A library should not print.
- **Nothing is printed** during the iterations (R's ``printFlag``); use
  :doc:`../user_guide/logging` to follow progress.

Data handling
-------------

- **Object and string columns are treated as categorical variables.** R
  treats a character column as constant, so the setup step removes it. In
  pandas, text columns usually hold categories, so mice-py codes them as
  factors.
- **Level order of object and string columns** is the sorted order of the
  values by code point; R sorts factor levels in the collation order of the
  locale. The two can differ for mixed-case or accented labels (not tested).
  It matters for the dummy coding and for ``quickpred``, which use the codes.
- **No predictor left**: when the setup step removes every predictor, R
  stops (in an order-dependent way); mice-py goes on, imputes each target
  from its observed values and warns. Stopping would lose the run for a
  condition the logged events already report.
- **Exactly duplicated columns**: which copy the setup step drops can differ
  from R; the choice rests on a floating-point tie that the two
  implementations can break differently.
- **Several linear dependencies at once**: ``remove_lindep`` reproduces on
  purpose R's behavior of reusing the decomposition of all columns after the
  first drop, which can keep a collinear column or drop an unrelated one.
  Parity with R was preferred here.
- **Ordered categoricals** are coded with polynomial contrasts as in R;
  checked against R up to six levels.

Methods
-------

midas (R's midastouch)
~~~~~~~~~~~~~~~~~~~~~~

- **Categorical and text targets raise** ``ValueError``. R's midastouch does
  not impute factors either: it imputes level codes, which R turns into
  missing values with a warning.
- **One donor carrying the whole bootstrap sample**: R stops with a LAPACK
  error ("exactly singular"); mice-py draws that donor, the only one with
  positive weight. The result is determined, so stopping gains nothing.

cart
~~~~

- **The trees come from scikit-learn, not rpart.** For a numeric target the
  leaves equal rpart's on the three numeric L2 inputs and on twelve larger
  datasets checked during development (``simulations/BASELINE.md``).
- **Ties between equally good splits** are broken at random by scikit-learn;
  rpart takes the first.
- **Categorical targets are not pruned by** ``cp``; rpart prunes them on the
  misclassification risk, which scikit-learn's cost-complexity pruning does
  not provide. The level proportions were within 0.003 of R's on the L2
  input F3.
- ``bootstrap=True`` fits the tree on a bootstrap sample, as mice-py 0.1 did;
  R has no such option. Off by default.

rf
~~

- **The forests come from scikit-learn, not ranger**; they agree in
  distribution, not tree by tree.
- **Node sizes are counted differently**: ranger counts bootstrap duplicates,
  scikit-learn distinct rows. ``min_samples_split=5`` (numeric target) was
  chosen because it agreed with ranger at 300 calls; 4 also passed, 6
  failed. Whether 4 matches slightly better is open.
- **Categorical predictors**: when the one feature drawn for a split cannot
  split a node, ranger stops and scikit-learn tries the other features, so
  with dummy-coded predictors R's trees are much shallower. This is the one
  cell of the simulation study where mice-py and R disagree; see
  :ref:`rf-categorical`.
- **No predictor**: rf draws from the observed values; R's rf fails.

Analysis and pooling
--------------------

- ``fit()`` accepts statsmodels formula models (``smf.ols`` by default,
  ``smf.logit``, ``smf.glm``, ...); R's ``with()`` evaluates any expression.
  Other models can be pooled with ``imputation.pooling.pool_estimates``.
- **Joint tests**: ``t_test``, ``f_test`` and ``wald_test`` of the pooled
  result raise ``NotImplementedError``; R's ``D1``, ``D2`` and ``D3`` are not
  ported. Per-parameter tests are in the pooled table.
- ``pool()`` has no ``dfcom``, ``rule`` or ``custom.t`` argument: ``dfcom``
  is the residual df of the first fitted model, as R's default.
  ``pool_estimates`` takes ``dfcom`` explicitly.
- **Term names** follow patsy (``Intercept``, ``sex[T.m]``); R writes
  ``(Intercept)``, ``sexm``.
- **GLM standard errors** match R's to 1e-8 only when both fits use a tight
  convergence tolerance. With the default tolerances of statsmodels and R
  they differ by about 4e-6 (relative); R's default fit is itself about 1e-5
  away from the exact maximum.

Plots
-----

- The plot functions take the list of completed datasets and a 0/1 data
  frame of observed cells, not a fitted object.
- ``plot_chain_stats`` draws markers on the lines (R draws lines only) and
  plots every column it is given; R's ``plot.mids`` leaves out variables
  whose chain means contain missing values. Chain statistics exist for
  numeric columns only.
- ``bwplot`` draws outliers in gray; R colors them by group.
- ``stripplot`` was not compared with R.
