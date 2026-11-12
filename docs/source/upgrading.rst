Upgrading
=========

This page lists what an existing user has to know when moving from one
version to the next; the complete list is in the `changelog
<https://github.com/Zhanna-Lopuliak/mice-py/blob/main/CHANGELOG.md>`_. To
move from 0.1 to 0.3, read both sections.

Upgrading from 0.2 to 0.3
-------------------------

Version 0.3.0 adds nine of R mice's model-based methods (``norm``,
``norm.nob``, ``norm.boot``, ``norm.predict``, ``mean``, ``logreg``,
``logreg.boot``, ``polyreg``, ``polr``; see
:doc:`user_guide/imputation_methods`) and, as R does, chooses the default
method of a column by its type. Code written for 0.2 still runs, but **its
results change** when the data have incomplete boolean, categorical or text
columns whose method the call does not name. Data whose incomplete columns
are all numeric give the same results as in 0.2.

Default methods by column type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 46 18 36

   * - Incomplete column
     - 0.2
     - 0.3 (R's default)
   * - numeric (also ``Int64``, ``Float64``)
     - ``"pmm"``
     - ``"pmm"``
   * - boolean, or with two categories
     - ``"pmm"``
     - ``"logreg"``
   * - unordered categorical, object or string, more than two categories
     - ``"pmm"``
     - ``"polyreg"``
   * - ordered categorical, more than two categories
     - ``"pmm"``
     - ``"polr"``
   * - categorical with a single category
     - ``"pmm"``
     - ``"pmm"``

The categories of a ``category`` column are its declared categories, unused
ones included, as R counts the levels of a factor; those of an object or
string column are its distinct observed values. ``imputation.make_method(df)``
shows the method each column gets, and ``mice.method`` the methods a run used.

The defaults apply to ``impute()`` without ``method``, **and also to the
columns a method dict leaves out**. A ``method`` string (``method="cart"``) still
applies to every column. With the :ref:`example data <example-data>` the
text column ``sex`` (``"f"``, ``"m"``) is now imputed with ``logreg``.

To keep the methods of 0.2:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Call in 0.2
     - Call in 0.3 with the methods of 0.2
   * - ``impute()``
     - ``impute(method="pmm")``
   * - ``impute(method={"chl": "midas"})``
     - ``impute(method={"chl": "midas"}, default_method=("pmm",) * 4)``

``default_method`` is R's ``defaultMethod``: the methods of numeric columns,
of columns with two categories, of unordered and of ordered categorical
columns with more categories, in this order. With the same seed these calls
give the imputations of 0.2 (checked on the example data).

Other changes that can affect a 0.2 script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **More than 50 categories.** A text or categorical column with more than
  50 distinct observed values, for example a column of names or
  identifiers, now gets ``polyreg`` (``polr`` if it is ordered) by default,
  and R limits these methods to 50 categories. ``impute`` raises
  ``ValueError`` naming the column before the chains start (0.2 imputed it
  with ``pmm``). Give the column a donor method,
  ``method={"name": "pmm"}``, or remove it from the data if it only
  identifies the rows.
- **Text columns are unordered categorical variables**, as everywhere in
  mice-py, so an incomplete text column gets ``logreg`` or ``polyreg`` by its
  number of distinct values. R gives a character column ``pmm`` and then
  does not impute it (its setup step finds the column constant); see
  :doc:`r_mice/differences`.
- **Nullable integer columns.** A column of dtype ``Int64`` (or another
  nullable integer dtype) imputed with ``norm``, ``norm.nob``, ``norm.boot``,
  ``norm.predict`` or ``mean`` comes back as ``Float64``, because these
  methods impute new values, not observed ones; R turns an integer column
  into double as well. Float columns keep their dtype, and the default
  ``pmm`` keeps integers.
- **polr can impute a level that never occurs.** Under ``polr`` an ordered
  column may receive any of its declared categories, also one that has no
  observed value: the model gives such a level a small positive probability,
  as R's ``MASS::polr`` does. Remove unused categories
  (``Series.cat.remove_unused_categories()``) if that is not wanted.
- **Checks before the chains start.** ``norm``, ``norm.nob``, ``norm.boot``,
  ``norm.predict`` or ``mean`` on a boolean, categorical or text column,
  and ``logreg`` or ``logreg.boot`` on a column with more than two distinct
  observed values, raise ``ValueError`` with the column named.
- **Errors inside a chain name the column.** A ``ValueError`` raised by a
  method starts with "Column 'c', method 'm' (iteration i, imputation k):".
  Code that matches the text of such an error has to allow for the prefix.

New warning: ``logreg``, ``logreg.boot``, ``polyreg`` or ``polr`` on a
numeric column is allowed, with R's ``UserWarning`` "Type mismatch for
variable(s): <column>", followed on a second line by "Imputation method
<method> is for categorical data.". The imputations are the column's own
values.

Method parameters of the new methods carry the method name with every dot
replaced by an underscore: ``norm_boot_ridge=1e-4``, ``polyreg_maxit=200``,
``polr_log_fallback=True``. The deprecated arguments ``min_cor``,
``min_puc``, ``include``, ``exclude`` and ``correlation_method`` of
``impute`` still work in 0.3 and will be removed in version 0.4.

Upgrading from 0.1 to 0.2
-------------------------

Version 0.2.0 aligns mice-py with R mice 3.19.0. Code written for 0.1 mostly
still runs, but **its results change**, a few names moved, and some arguments
that were silently ignored now raise an error.

Results change
~~~~~~~~~~~~~~

Expect different imputations, and therefore different pooled estimates, for
the same code. Three things cause it.

**1. The defaults are now R's.**

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Argument of ``impute()``
     - 0.1
     - 0.2
     - To get the 0.1 setting
   * - ``method``
     - ``"sample"``
     - ``"pmm"`` (also for columns left out of a method dict)
     - ``method="sample"``
   * - ``maxit``
     - 10
     - 5
     - ``maxit=10``
   * - ``visit_sequence``
     - ``"monotone"``
     - ``"roman"`` (data order)
     - ``visit_sequence="monotone"`` (ties are now in data order)
   * - ``predictor_matrix``
     - mice-py's own quickpred with ``min_cor=0.1``
     - all ones, zero diagonal
     - not reproducible; ``imputation.quickpred(data)`` gives R's rule

**2. Runs are no longer seeded with a fixed number.** 0.1 always seeded the
chains with 42, so every run gave the same numbers. In 0.2 ``seed=None``
draws fresh randomness on every call. Pass a seed to make a run
reproducible (``df`` is any incomplete data frame, for example the
:ref:`example data <example-data>`):

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(n_imputations=5, seed=2024)

A given seed gives different numbers than 0.1 did, and the numbers never
equal R's.

**3. Methods were corrected.**

- **pmm** drew its regression coefficients with the wrong covariance (a
  transposed Cholesky factor), resolved ties between donors
  deterministically, and crashed on categorical targets. With the default
  ``matchtype=1`` every pmm imputation changes.
- **cart** no longer fits the tree on a bootstrap sample and prunes as rpart
  does (``cp`` relative to the variance of the target, minimum split 15,
  depth 30). ``cart_bootstrap=True`` brings back the resampling, not the old
  pruning.
- **rf** draws one donor from the observed values that share the leaf in any
  tree (0.1 drew one per tree, then one of those) and uses the square root of
  the number of predictors per split.
- **midas** uses the chain's random generator (0.1 used NumPy's global one)
  and keeps the target's dtype.
- **Categorical predictors** are coded as R does (dummies in level order,
  polynomial contrasts for ordered categoricals), which changes cart and rf
  results when such predictors are present.
- **Setup step**: a constant or collinear incomplete column is no longer
  imputed and keeps its missing values; ``impute`` warns "Number of logged
  events". Rows that are entirely missing are kept and imputed (0.1 dropped
  them, so the datasets had fewer rows); columns that are entirely missing
  are kept and stay missing (0.1 dropped them).

**Pooled results.** p-values and confidence intervals now use the t
distribution with Barnard-Rubin degrees of freedom: intervals are wider and
p-values larger than in 0.1, which used the normal distribution. The
fraction of missing information is now R's ``fmi``; the value 0.1 reported
under that name is R's ``lambda``, now in the ``"table"`` under that name.

Names that moved or were removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.1
     - 0.2
   * - ``MICE.result``, ``ImputationResult``
     - removed; the datasets are in ``MICE.imputed_datasets``, the pooled
       result in ``MICE.pooled_result``
   * - ``MICE.run_output_dir``
     - removed
   * - ``MICE._quickpred``; ``impute(min_cor=..., min_puc=..., include=...,
       exclude=..., correlation_method=...)``
     - ``imputation.quickpred(data, mincor=..., minpuc=..., include=...,
       exclude=..., method=...)``, passed as ``predictor_matrix``; the
       ``impute`` arguments still work in 0.2 with a ``DeprecationWarning``
   * - ``imputation.PMM.quantify_cca``
     - ``imputation.PMM.quantify``
   * - ``imputation.PMM.matcherid``; ``pmm(..., matcher=...)``
     - ``imputation.PMM.matchindex(d, t, k=5, rng=None)``; no ``matcher``
   * - ``midas(..., midas_kappa=3)``
     - ``midas(..., kappa=3)``; through ``impute`` still ``midas_kappa=3``
   * - ``imputation.midas.compute_beta``
     - removed
   * - ``cart(..., <scikit-learn argument>=...)``, ``rf(..., <argument>=...)``
     - ``cart(..., tree_params={...})``, ``rf(..., forest_params={...})``;
       through ``impute``: ``cart_tree_params={...}``,
       ``rf_forest_params={...}``
   * - ``cart(..., ccp_alpha=1e-4)`` as the default pruning
     - ``cp=1e-4``, relative; ``ccp_alpha`` (absolute) only if given
   * - ``from imputation.MICE import pool_descriptive_statistics``
     - ``from imputation.pooling import pool_descriptive_statistics``
   * - ``impute(output_dir=...)``
     - ignored with a ``DeprecationWarning`` (it never did anything)
   * - ``plot_missing_data_pattern(...)`` returned the pattern table
     - returns the ``Figure``; the table comes from ``md_pattern_like``
   * - plot functions returned ``None``
     - return the ``Figure``
   * - ``pip install -e ".[dev]"``
     - ``uv sync`` (the ``dev`` extra is now a dependency group)

Arguments that now raise
~~~~~~~~~~~~~~~~~~~~~~~~

Calls that 0.1 accepted, and often ignored, now fail:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Call
     - Error
   * - An unknown keyword argument of ``impute`` or of an imputer, such as
       ``impute(donors=3)`` or ``impute(pmm_donor=3)``
     - ``TypeError``, listing the valid names (``pmm_donors=3``)
   * - A method parameter without the method prefix, such as
       ``impute(n_estimators=50)``
     - ``TypeError``; write ``rf_n_estimators=50``
   * - ``cart_tree_params`` or ``rf_forest_params`` setting an argument that
       has its own parameter (``max_depth``, ``min_samples_split``, ...)
     - ``TypeError``
   * - ``n_imputations=True`` or ``maxit=False``
     - ``ValueError``
   * - A column of an unsupported dtype (datetime, timedelta, period,
       complex) in ``MICE(data)``
     - ``TypeError`` naming the column
   * - ``method="midas"`` on a categorical or text column
     - ``ValueError``
   * - A target without any observed value in the rows of its model (pmm,
       midas, cart, rf)
     - ``ValueError``
   * - A predictor matrix without a row for an incomplete column
     - ``ValueError`` (was ``KeyError``)
   * - ``estimice(..., ls_meth=<unknown>)``
     - ``ValueError``
   * - ``pooled_result.t_test(...)``, ``f_test``, ``wald_test``
     - ``NotImplementedError``

New warnings: "Number of logged events" (``UserWarning``) when the setup
step changed the model; a ``UserWarning`` for rows without any observed
value and for ``pool()`` with a single imputation; ``DeprecationWarning``
for the deprecated arguments above.

Logging and plots
~~~~~~~~~~~~~~~~~

- The package's log records now propagate to the root logger (0.1 stopped
  them at the ``imputation`` logger). An application that configures the root
  logger at level INFO will see mice-py's INFO messages; raise the level of
  the ``imputation`` logger to hide them, for example
  ``logging.getLogger("imputation").setLevel(logging.WARNING)``.
- Logger names follow the modules (``imputation.MICE``, was
  ``imputation.mice``; ``plotting.diagnostics``, was
  ``imputation.plotting.diagnostics``).
- ``configure_logging()`` no longer writes ``logs/mice_<date>.log``; pass
  ``file_logging=True`` for a log file.
- ``plot_chain_stats`` shows the standard deviation of the imputed values,
  not the variance, as R does. ``MICE.chain_var`` still holds variances.
- By default ``stripplot``, ``bwplot`` and ``densityplot`` plot the numeric
  columns with missing values, as R's plots do; 0.1 took every incomplete
  column. Name a categorical column in ``columns`` to see it in
  ``stripplot``.
- In a notebook, a plot call on the last line of a cell shows the figure
  twice (once drawn, once as the returned value); end the line with ``;``.

Licence
~~~~~~~

mice-py 0.2 is licensed under the GNU General Public License, version 2 or
later (``GPL-2.0-or-later``), the licence of R mice, from which it is ported;
see the files ``LICENSE`` and ``COPYRIGHT`` of the repository. Versions up to
and including 0.1.0 were released under the MIT License and remain available
under it. This matters if you distribute mice-py, or software that includes
it, under a licence that is not compatible with the GPL: from 0.2 the
conditions of the GPL apply to that distribution.

A 0.1 script in 0.2
~~~~~~~~~~~~~~~~~~~

A typical 0.1 call:

.. code-block:: python

   # mice-py 0.1; in 0.2 this raises TypeError because of donors=3
   mice = MICE(df)
   mice.impute(n_imputations=5, min_cor=0.2, donors=3)

In 0.1 ``donors=3`` was silently ignored and the default method was
``sample``, so this call did not use pmm at all. The 0.2 equivalent of what
was intended:

.. code-block:: python

   import imputation
   from imputation import MICE

   pred = imputation.quickpred(df, mincor=0.2)
   mice = MICE(df)
   mice.impute(n_imputations=5, predictor_matrix=pred, pmm_donors=3, seed=1)

To reproduce the old method instead, pass ``method="sample"``.
