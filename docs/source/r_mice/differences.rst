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
   * - Method of a column without cells to impute (by default a complete
       column)
     - ``""``
     - ``""``
   * - Cells imputed
     - ``where=None``: the missing cells
     - ``where = NULL``: the missing cells (``is.na(data)``)
   * - Rows the models are fitted on
     - ``ignore=None``: every row with an observed target and complete
       predictors
     - ``ignore = NULL``: the same
   * - Changes to the imputations
     - ``post=None``: none
     - ``post = NULL``: none
   * - Predictor matrix
     - all ones, zero diagonal
     - all ones, zero diagonal
   * - Visit sequence
     - ``visit_sequence=None``: data order, passive columns last
     - ``visitSequence = NULL``: the same
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
   * - ``make.method(data)``, ``make.method(data, where = w)``
     - ``imputation.make_method(data)``, ``imputation.make_method(data,
       where=w)``
   * - ``mice(data, where = w)``
     - ``impute(where=w)``: a keyword, a boolean DataFrame or a boolean
       array
   * - ``make.where(data, "missing")``, ``"all"``, ``"observed"``,
       ``"none"``
     - the keywords themselves, ``where="missing"`` (the default),
       ``"all"``, ``"observed"``, ``"none"``; as a frame to edit,
       ``data.isna()``, ``data.notna()``
   * - ``mice(data, ignore = v)``
     - ``impute(ignore=v)``
   * - ``post["tv"] <- "imp[[j]][, i] <- squeeze(imp[[j]][, i], c(1, 25))"``
     - ``impute(post={"tv": lambda imp, data: imp.clip(1, 25)})``
   * - ``squeeze(x, c(1, 25))``
     - ``Series.clip(1, 25)``
   * - passive imputation, ``meth["bmi"] <- "~ I(wgt / (hgt / 100)^2)"``
     - ``impute(passive={"bmi": lambda d: d["wgt"] / (d["hgt"] / 100) **
       2})``; ``mice.method["bmi"]`` is ``"passive"``
   * - ``imp <- mice(data, maxit = 2)``; ``imp <- mice.mids(imp, maxit =
       3)``
     - ``mice.impute(maxit=2)``; ``mice.continue_chains(3)``
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
   * - ``imp$where``, ``imp$ignore``, ``imp$post``, ``imp$iteration``
     - ``mice.where``, ``mice.ignore``, ``mice.post``, ``mice.iteration``;
       the functions of the passive columns are in ``mice.passive``
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

mice-py ports fourteen of R's univariate methods (the names above), passive
imputation, and of ``mice()`` the arguments in the tables above, among them
``where``, ``ignore`` and ``post``; ``mice.mids()`` is
``continue_chains()``. Not ported:

- **Methods**: R's other methods, among them ``lda``, the multilevel methods
  (``2l.*``, ``2lonly.*``), ``quadratic``, ``ri`` and the ``lasso.*``
  methods.
- **Arguments of mice()**: ``blocks``, ``formulas``, ``blots`` and
  ``calltype``. ``data.init`` (a complete data set of starting values) is
  replaced by ``initial`` (see below).
- **New data for a finished run**: ``mice.mids(newdata = )``.
  ``continue_chains()`` has no ``newdata``; to impute new rows with models
  fitted on the old ones only, impute both together and ignore the new rows
  (:ref:`the recipe <train-and-test>`).
- **Arguments of the methods**: pmm's ``exclude``, ``trim`` and
  ``use.matcher``; midastouch's ``neff`` and ``debug``; rf's ``rfPackage``
  (mice-py always uses scikit-learn).
- **Other functions**: ``ampute``, ``md.pairs``, ``flux``, ``fluxplot``,
  ``pool.r.squared``, and the model comparisons ``D1``, ``D2``, ``D3``.
  ``make.where()`` and ``squeeze()`` are not needed: ``data.isna()`` and the
  keywords of ``where`` give the frames of ``make.where()``, and
  ``Series.clip`` does what ``squeeze()`` does.

Arguments that behave differently
---------------------------------

- **Random numbers never equal R's.** ``seed`` seeds a NumPy
  ``SeedSequence``, from which each chain gets its own stream; R uses
  ``set.seed`` and one stream. The generators differ, so runs agree with R
  in distribution, not draw by draw; single method calls, and whole chains
  from R's start values, were also compared draw by draw, with R's random
  numbers fed to mice-py (:doc:`validation`).
- **Method parameters carry the method name** (``pmm_donors=3``,
  ``cart_cp=0.001``) and an unknown keyword argument raises ``TypeError``. R
  passes ``...`` to every method and ignores names a method does not use; the
  prefix makes a misspelled argument an error instead of a silent no-op.
- **initial**: ``"sample"`` is R's initialization; ``"meanobs"`` (the
  observed value closest to the mean, or the most frequent value) exists only
  in mice-py. It was part of mice-py 0.1 and is kept.
- **visit_sequence="random"** exists only in mice-py, kept from 0.1. It is
  drawn once per ``impute()`` call from the seed; ``continue_chains()``
  keeps it.
- **visit_sequence after a run** holds the order actually used, a column
  named twice included twice. R's ``visitSequence`` can list variables it
  never imputes (complete ones, and variables whose method the setup step
  set to ``""``); mice-py lists only the variables it visits.
- **A predictor matrix may cover only some columns**; missing entries count
  as 0, and ``predictor_matrix`` after a run is the full square matrix. R
  requires a square matrix. Every column that is imputed still needs a row;
  a passive column needs none.
- **Logged events** produce one ``UserWarning`` ("Number of logged events:
  n") and log records; R prints a message. A library should not print.
  ``continue_chains()`` warns when its iterations logged an event; R's
  ``mice.mids()`` prints nothing.
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

where, ignore, post and passive columns
---------------------------------------

The chain loop follows R's ``sampler`` and ``sampler.univ``. For a column,
the model is fitted on the rows where the column is observed, its
predictors are complete and ``ignore`` keeps the row; the method imputes the
where-cells whose predictors are complete. As in R, the method receives
**all rows**, also those in neither set (an ignored row with an observed
cell, a missing cell that ``where`` leaves out, a row with a missing
predictor value); only ``logreg``, ``polyreg`` and ``polr`` use such rows,
in their augmentation. The imputations are kept per where-cell; the data the
chains work on hold them only in the missing where-cells, so an observed
where-cell keeps its observed value there, and the completed datasets get
the imputation in every where-cell. ``post`` runs right after each
univariate imputation, a passive column is computed at each of its visits,
and the chain statistics are taken over all where-cells at the end of each
iteration. Whole chains with these arguments were compared with R's, from
R's start values and with R's draws (:doc:`validation`).

mice-py deliberately differs from R in these cases:

.. list-table::
   :header-rows: 1
   :widths: 36 32 32

   * - Case
     - R mice 3.19.0
     - mice-py
   * - A where-cell that cannot be imputed: its predictors are incomplete in
       its row, or its column has no method
     - ``NA`` in the completed data, also over an observed value, silently
     - the same, and a ``UserWarning`` with the number of observed values
       lost per column and the reason
   * - ``where`` as a data frame with the columns in another order
     - takes the columns by position, silently misaligned
     - aligns the columns by label; other labels raise ``ValueError``
   * - ``where`` as a data frame with another index or its rows in
       another order
     - takes the rows by position
     - ``ValueError``: the index must be that of the data, the same labels
       in the same order (rows are not reordered)
   * - ``where`` of another shape, with missing values, or with the numbers
       0 and 1
     - stops; stops; accepted
     - ``ValueError`` for all three: booleans only
   * - ``ignore`` not logical (also 0 and 1), of another length, or with a
       missing value
     - stops ("Argument ignore not a logical.", or a missing value in an
       ``if``)
     - ``ValueError`` for all three; also for a Series with another index
       than the data
   * - Every observed value of a column that is imputed lies in an ignored
       row
     - starts the column from ``rnorm()``, then stops in iteration 1 ("0
       (non-NA) cases")
     - ``ValueError`` before the chains start, naming the column
   * - ``post`` for a passive column
     - kept in ``imp$post`` and never run, silently
     - ``ValueError`` before the chains start
   * - A ``post`` command or a passive formula that fails
     - stops with R's error, a failing formula in iteration 1
     - a passive function is called once on the data before the chains
       start, and its error raised as ``ValueError`` naming the column; an
       error inside a chain is raised as ``ValueError`` naming the column,
       ``post`` or ``passive``, the iteration and the imputation (the
       original exception is its cause)
   * - A passive formula with an aggregate, such as ``~ I((chl -
       mean(chl))^2)``
     - evaluated on the rows of the where-cells only, so the mean is taken
       over them (197.56 in the fixture case)
     - the function gets all rows, so the mean is that of the whole column
       (193.86 in the same case)
   * - A passive formula with two terms
     - uses the first, with a warning
     - ``ValueError``: the function must return one value per row
   * - A passive column without any observed value (in the rows that
       ``ignore`` keeps)
     - starts from ``rnorm()``, then holds the formula
     - the same for a function that returns numbers; otherwise
       ``ValueError`` before the chains start
   * - A column named twice in a visit sequence vector
     - ``initialize.imp()`` draws its start values twice and keeps the
       second draw
     - draws them once (the same distribution)
   * - A visit sequence vector that leaves out a column that is imputed
     - the column is never imputed and stays ``NA``
     - ``ValueError`` naming the column
   * - ``logreg.boot`` with a row that is both fitted on and imputed (an
       observed where-cell)
     - predicts the row from the predictors of the bootstrap row that
       replaced it
     - predicts the row from its own predictors (7 of 30 labels differ in
       the fixture case)
   * - ``polyreg`` where the augmentation is skipped (no predictor, or
       exactly one row outside the fit rows) and a label occurs only
       outside the fit rows, for example in an ignored row
     - ``nnet::multinom`` drops the label and ``polyreg`` stops (``dims
       ... do not match``)
     - imputes; the label gets probability 0. ``polr`` matches R here
   * - ``logreg``, ``logreg.boot`` with a third label in a row outside the
       fit rows, for example an ignored row
     - stops in ``glm.fit``
     - ``ValueError`` for more than two categories, before the chains start
   * - A run that stops with an error
     - ``mice()`` returns nothing; after a failing ``mice.mids()`` the
       earlier ``mids`` object is unchanged and can be continued
     - the ``MICE`` object is changed in place: its chains are dropped, and
       ``continue_chains()`` raises ``ValueError``; call ``impute()``
       again
   * - Logged events in a continued run
     - ``mice.mids()`` prints nothing
     - ``continue_chains()`` warns "Number of logged events" when its
       iterations logged an event
   * - The diagnostic plots under ``where``
     - group the cells by whether they are missing in the data
       (``is.na(data)``)
     - group them by the frame passed; with ``~mice.where``, by whether
       they were imputed (see `Plots`_)

Further points:

- **Dtypes are fixed before the chains start.** A NumPy integer or boolean
  column with where-cells (possible only under ``where``, since it is
  complete) becomes ``Int64`` or ``boolean`` in the completed datasets,
  because a where-cell can end up missing; an integer column imputed by a
  norm method or ``mean`` becomes ``Float64``, as R turns it into double. A
  passive column takes its dtype from what its function returns on the
  data.
- **post and passive functions are checked.** A function that returns
  another number of values, a Series with another index, or values the
  column cannot hold (a number that is not whole in an integer column, a
  label that is not a category) raises ``ValueError`` naming the column.
- **post gets no random generator.** R's ``post`` code draws from R's one
  stream; mice-py gives a function no generator, so a function that draws
  random numbers must own a seeded one, and then a run split between
  ``impute()`` and ``continue_chains()`` equals one run only if the
  function draws the same numbers each time.
- **continue_chains() has no newdata** (see above). Like R's
  ``mice.mids()``, it goes on from the current imputations and the
  generator state of each chain, keeps the visit sequence, the methods, the
  predictor matrix, ``where``, ``ignore``, ``post`` and the passive
  functions, checks "df set to 1" only in the first iteration of the run,
  and does not check ``ignore`` again, so it does not repeat the warning
  about fewer than 10 rows.
- **quickpred** gives a complete column no predictors, as R's does, so under
  ``where="all"`` its matrix imputes the complete columns without
  predictors; edit the matrix for them.

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
       to do (removing an identifier column, or else a donor method)

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
row outside the fit rows, by default exactly one missing value) and in
``logreg.boot``, which does not augment. For ``logreg``, ``logreg.boot`` and
``polyreg`` that changes only which label a random number maps to, not the
distribution of the imputations. For ``polr`` the order is part of the
model: on an **unordered** factor with exactly one row outside the fit rows,
R fits the declared order and mice-py the sorted labels present, so the
imputation differs in distribution. Declare the column as
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
- **Rows outside the masks**: as in R, the methods receive every row, also
  rows in neither the fit rows nor the rows to impute (see
  `where, ignore, post and passive columns`_). The augmentation of
  ``logreg``, ``polyreg`` and ``polr`` takes the means, standard deviations
  and ranges of the predictors over all rows, column by column without
  missing values, and the categories from all of the target, as R's
  ``augment`` does.
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

- The plot functions take the list of completed datasets and a data frame
  that marks the cells that were not imputed (``~mice.where``, or a 0/1
  frame), not a fitted object. R's plots group the cells by whether they
  are missing in the data (``is.na(data)``); mice-py's group them by the
  frame passed, so with ``~mice.where`` an observed cell that was imputed
  under ``where`` is drawn as imputed.
- ``plot_chain_stats`` draws markers on the lines (R draws lines only) and
  plots every column it is given; R's ``plot.mids`` leaves out variables
  whose chain means contain missing values. Chain statistics exist for
  numeric columns only.
- ``bwplot`` draws outliers in gray; R colors them by group.
- ``stripplot`` was not compared with R.
