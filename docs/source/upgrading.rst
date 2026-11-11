Upgrading from 0.1 to 0.2
=========================

Version 0.2.0 aligns mice-py with R mice 3.19.0. Code written for 0.1 mostly
still runs, but **its results change**, a few names moved, and some arguments
that were silently ignored now raise an error. This page lists what an
existing user has to know; the complete list is in the `changelog
<https://github.com/Zhanna-Lopuliak/mice-py/blob/main/CHANGELOG.md>`_.

Results change
--------------

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
--------------------------------

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
------------------------

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
-----------------

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
- In a notebook, a plot call on the last line of a cell shows the figure
  twice (once drawn, once as the returned value); end the line with ``;``.

A 0.1 script in 0.2
-------------------

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
