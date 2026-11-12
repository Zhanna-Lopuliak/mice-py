Differences from R mice
=======================

mice-py follows R mice 3.19.0. This page is for readers who know R mice: it
lists the defaults the two share, how R's names map to mice-py's, and every
known difference, each with the reason for it. How the agreement was checked
is on :doc:`validation`.

Defaults shared with R
----------------------

Since version 0.3.0 an ``impute()`` call without arguments does what
``mice(data)`` does in R, also in the choice of the methods. The one
difference there is a text column, which R does not impute (see
`Data handling`_).

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
   * - Default methods by column type
     - ``default_method=("pmm", "logreg", "polyreg", "polr")``
     - ``defaultMethod = c("pmm", "logreg", "polyreg", "polr")``
   * - Method of a numeric column
     - ``"pmm"``
     - ``"pmm"``
   * - Method of a boolean column or one with two categories
     - ``"logreg"``
     - ``"logreg"``
   * - Method of an unordered categorical column with more categories
     - ``"polyreg"``
     - ``"polyreg"``
   * - Method of an ordered categorical column with more categories
     - ``"polr"``
     - ``"polr"``
   * - Method of a text (object or string) column
     - ``"logreg"`` or ``"polyreg"``, by its number of distinct values
     - ``"pmm"``; the setup step then finds the column constant, so it is
       not imputed
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
   * - norm, norm.nob, norm.boot, norm.predict
     - ``ls_meth="qr"``, ``ridge=1e-5``
     - ``ls.meth = "qr"``, ``ridge = 1e-05``
   * - polyreg
     - ``maxit=100``, ``max_weights=1500``
     - ``nnet.maxit = 100``, ``nnet.MaxNWts = 1500``
   * - polr
     - ``maxit=100``, ``max_weights=1500``, ``log_fallback=False``
     - ``nnet.maxit = 100``, ``nnet.MaxNWts = 1500``,
       ``polr.to.loggedEvents = FALSE``
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
   * - methods ``"pmm"``, ``"cart"``, ``"rf"``, ``"sample"``, ``"norm"``,
       ``"norm.nob"``, ``"norm.boot"``, ``"norm.predict"``, ``"mean"``,
       ``"logreg"``, ``"logreg.boot"``, ``"polyreg"``, ``"polr"``
     - the same names, dots included
   * - ``mice(data, defaultMethod = c(...))``
     - ``impute(default_method=(...))``
   * - ``make.method(data)``
     - ``imputation.make_method(data)``
   * - ``mice(data, donors = 3)`` (passed to every method)
     - ``impute(pmm_donors=3)`` (method name as prefix)
   * - ``ls.meth``, ``ridge`` of the norm methods
     - ``norm_ls_meth``, ``norm_ridge``; ``norm_boot_ridge`` for
       ``norm.boot`` (the prefix is the method name with every dot replaced
       by an underscore)
   * - ``nnet.maxit``
     - ``polyreg_maxit``, ``polr_maxit``
   * - ``nnet.MaxNWts``
     - ``polyreg_max_weights``, ``polr_max_weights``
   * - ``polr.to.loggedEvents``
     - ``polr_log_fallback``
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
   * - ``mice.impute.norm()``, ``mice.impute.norm.nob()``,
       ``mice.impute.norm.boot()``, ``mice.impute.norm.predict()``,
       ``mice.impute.mean()``
     - ``imputation.norm.norm()``, ``imputation.norm.norm_nob()``,
       ``imputation.norm.norm_boot()``, ``imputation.norm.norm_predict()``,
       ``imputation.mean.mean()``
   * - ``mice.impute.logreg()``, ``mice.impute.logreg.boot()``,
       ``mice.impute.polyreg()``, ``mice.impute.polr()``
     - ``imputation.logreg.logreg()``, ``imputation.logreg.logreg_boot()``,
       ``imputation.polyreg.polyreg()``, ``imputation.polr.polr()``
   * - ``mice:::augment()``
     - ``imputation.logreg.augment()``
   * - ``glm.fit()`` (logit link), ``optim(method = "BFGS")``,
       ``nnet::multinom()``, ``MASS::polr()``
     - ``imputation.fitting.logit_fit()``, ``vmmin()``, ``multinom_fit()``,
       ``polr_fit()`` (only as the methods above use them)

Methods and arguments that mice-py does not have
-------------------------------------------------

Version 0.3 ports fourteen of R's univariate methods (the names above) and
the core of ``mice()``; the rest is not ported yet.

- **Methods**: R's other methods, among them ``lda``, the multilevel methods
  (``2l.*``, ``2lonly.*``), ``quadratic``, ``ri`` and the ``lasso.*``
  methods; and passive imputation (``"~ I(...)"``).
- **Arguments of mice()**: ``where``, ``blocks``, ``formulas``, ``blots``,
  ``post``, ``ignore`` and ``calltype``. ``data.init`` (a complete data set
  of starting values) is replaced by ``initial`` (see below).
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
  ``set.seed`` and one stream. The generators differ, so runs agree with R
  in distribution, not draw by draw; single method calls were also compared
  draw by draw, with R's random numbers fed to mice-py (:doc:`validation`).
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

- **Object and string columns are treated as unordered categorical
  variables.** R's ``make.method`` gives a character column ``"pmm"``
  whatever its values, and the setup step then treats the column as
  constant: it is neither imputed nor used as a predictor. In pandas, text
  columns usually hold categories, so mice-py codes them as factors and, by
  default, imputes them with ``"logreg"`` (two distinct values) or
  ``"polyreg"`` (more). This is the one difference in the default methods;
  a column with more than 50 distinct values, such as names, then needs
  another method (see `Model-based methods`_).
- **Level order of object and string columns** is the sorted order of the
  values by code point; R sorts factor levels in the collation order of the
  locale. The two can differ for mixed-case or accented labels (not tested).
  It matters for the dummy coding and for ``quickpred``, which use the codes,
  and for the order of the categories of the model-based methods (below).
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

Model-based methods
~~~~~~~~~~~~~~~~~~~

``norm``, ``norm.nob``, ``norm.boot``, ``norm.predict``, ``mean``,
``logreg``, ``logreg.boot``, ``polyreg`` and ``polr`` follow R's functions
step for step, with R's random numbers in R's order (:doc:`validation`).
Where R stops with an error or imputes values that cannot be used, mice-py
deliberately does something else:

.. list-table::
   :header-rows: 1
   :widths: 16 30 27 27

   * - Method
     - Case
     - R mice 3.19.0
     - mice-py
   * - ``norm``, ``norm.nob``, ``norm.boot``, ``norm.predict``, ``mean``
     - a boolean, categorical or text column
     - imputes ``NA``, with warnings
     - ``ValueError`` naming the column, before the chains start
   * - ``norm.nob``, ``norm.boot``
     - at most ``q + 1`` observed values, ``q`` the number of coefficients
       (intercept included)
     - imputes ``Inf`` or ``NaN`` (the residual sum of squares is divided
       by ``n_obs - q - 1``)
     - ``ValueError`` naming the column, the method, the iteration and the
       imputation
   * - ``norm.boot``
     - no predictor
     - stops (``'x' must be a matrix``)
     - fits the model with the intercept only
   * - ``norm.nob``, ``norm.boot``, ``norm.predict``
     - a predictor that is a linear combination of earlier ones (in the
       bootstrap sample, for ``norm.boot``)
     - imputes ``NA``
     - its coefficient is 0, as R's ``norm`` does
   * - ``logreg``, ``logreg.boot``
     - more than two categories
     - stops (``invalid 'labels'``)
     - ``ValueError`` naming the column and ``polyreg``, before the chains
       start
   * - ``logreg``, ``logreg.boot``
     - a numeric column with two values other than 0 and 1
     - stops (``y values must be 0 <= y <= 1``)
     - imputes the two values; the smaller one is the first category
   * - ``logreg``, ``logreg.boot``
     - a single category present
     - imputes invented labels such as ``a2``
     - imputes that category, without a fit
   * - ``logreg``
     - a predictor aliased in the fit (for example a constant column, which
       the augmentation keeps constant)
     - stops (non-conformable arguments)
     - its coefficient is 0; the others are drawn
   * - ``logreg``
     - an ordered factor with an unused declared level
     - stops (``invalid 'labels'``)
     - uses the levels present
   * - ``logreg.boot``
     - no predictor and more than one cell to impute
     - stops (non-conformable arguments)
     - fits the model with the intercept only
   * - ``logreg.boot``
     - a predictor aliased in the bootstrap sample
     - imputes ``NA``
     - its coefficient is 0
   * - ``polyreg``
     - two categories and exactly one cell to impute
     - stops (``dim(X) must have a positive length``)
     - imputes the cell
   * - ``polyreg``
     - an ordered factor with an unused declared level
     - stops (``dims ... do not match``)
     - uses the levels present
   * - ``polyreg``
     - a category held only by cells that are not observed, where the
       augmentation is skipped
     - stops (``multinom`` drops the category)
     - fits the categories with observed rows; the other gets probability 0
   * - ``polr``
     - no predictor
     - stops (both ``MASS::polr`` and ``nnet::multinom`` see no response)
     - fits the model with cut points only; where that is not possible, the
       multinomial fallback with the intercept only
   * - ``polr``
     - a single category in the observed rows (the augmented ones
       included)
     - stops (unordered factor) or fits a degenerate model (ordered factor)
     - imputes that category, without a fit
   * - ``polr``
     - the multinomial fallback with two categories and one cell to impute
     - stops (``dim(X) must have a positive length``)
     - imputes the cell
   * - ``polr``
     - the multinomial fallback on an ordered factor with an unused level
     - stops (``dims ... do not match``)
     - the unused level gets probability 0
   * - ``polr``
     - a predictor aliased in the start fit
     - drops it, then its ``predict`` stops (non-conformable arguments)
     - its coefficient is 0
   * - ``polyreg``, ``polr``
     - a uniform number above the last cumulative probability, which is 1
       up to rounding
     - imputes ``NA``
     - the last category with a positive probability
   * - ``logreg``, ``polyreg``, ``polr``
     - more than 50 categories
     - stops inside the chain (``Maximum number of categories (50)
       exceeded``)
     - ``ValueError`` before the chains start, naming the column and what
       to do (a donor method, or removing an identifier column)

The order of the categories decides which label a given random number maps
to. mice-py uses R's order after the augmentation (White, Daniel and
Royston 2010) for every target:

- an ordered ``pd.Categorical`` keeps its declared order. ``polr`` keeps
  every declared level, also those that do not occur, and gives such a
  level a small positive probability, so it can be imputed, as in R;
  ``logreg`` and ``polyreg`` use the levels present;
- any other target (an unordered Categorical, text, booleans, numbers) uses
  its labels present, sorted (``False`` before ``True``, numbers by value,
  text by code point). R sorts the labels of a factor as text, so for a
  Categorical with numeric categories the orders can differ (``10`` before
  ``2`` in R).

R keeps the declared order of an unordered factor (in ``polr`` also its
unused levels) where it skips the augmentation (no predictor, or exactly one
missing value) and in ``logreg.boot``, which does not augment. For ``logreg``,
``logreg.boot`` and ``polyreg`` that changes only which label a random
number maps to, not the distribution of the imputations. For ``polr`` the
order is part of the model: on an **unordered** factor with exactly one
missing value, R fits the declared order and mice-py the sorted labels
present, so the imputation differs in distribution. Declare the column as
an ordered Categorical to choose the order ``polr`` assumes.

Further points:

- **Text columns** are unordered factors for these methods, with their
  sorted labels; R never imputes a character column (see
  `Data handling`_).
- **The limit of 50 categories** is R's (``augment``). The default method
  of a text column with more than two values is ``polyreg``, so a column of
  names or identifiers reaches the limit; mice-py checks it before the
  chains start, on the distinct observed values, and says what to do.
- **The residual standard deviation of** ``norm.nob`` **and**
  ``norm.boot`` is ``sqrt(RSS / (n_obs - q - 1))``, as in R: one degree of
  freedom less than the usual ``n_obs - q``. Kept for parity.
- **Integer columns become float**: a nullable integer column (``Int64``,
  ...) imputed by ``norm``, ``norm.nob``, ``norm.boot``, ``norm.predict`` or
  ``mean`` becomes ``Float64``, as R turns an integer column into double.
- **Rows with an incomplete predictor**: where a predictor keeps missing
  values (a column with method ``""``), mice-py leaves the rows in which it
  is missing out of the data it passes to the method. R's augmentation then
  computes the means, standard deviations and ranges of the predictors over
  more rows than mice-py's, so the added rows, and with them the
  imputations of ``logreg``, ``polyreg`` and ``polr``, can differ slightly
  from R's. Not compared with R.
- **Where the fits stop**: ``polyreg`` and the fallback of ``polr`` stop
  where ``nnet::multinom`` stops (its tolerances ``abstol = 1e-4``,
  ``reltol = 1e-8``, at most ``maxit = 100`` iterations, no error at the
  limit), ``polr`` where ``MASS::polr`` stops (``optim``'s defaults, at
  most 100 iterations). For ``nnet::multinom`` that is often 1e-5 to 1e-4
  away, in the probabilities, from the exact maximum of the likelihood.
  mice-py stops at the same point as R, so its imputations are R's, not
  those of a fully converged fit.
- ``logreg`` draws its coefficients with ``cov.unscaled`` of R's
  quasibinomial ``glm.fit``, not scaled by the dispersion and taken from
  the working weights of the last least-squares step, as R does.
  ``polyreg`` and ``polr`` draw no coefficients; as in R, the augmentation
  is their only allowance for the uncertainty of the fit.

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
