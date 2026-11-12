MICE Overview
=============

This page describes what ``MICE.impute()`` does and what its arguments mean.
The examples use ``df`` from the :ref:`example data <example-data>`.

The algorithm
-------------

MICE (multiple imputation by chained equations, van Buuren and
Groothuis-Oudshoorn 2011) imputes one incomplete variable at a time, each
from the others, and cycles through the variables several times. mice-py
follows R's ``mice()``:

1. **Setup.** Variables that are constant or collinear are removed from the
   imputation model (details below). Each incomplete column gets a method,
   by default the one R chooses for its type (``"pmm"``, ``"logreg"``,
   ``"polyreg"`` or ``"polr"``); complete columns get ``""``.
2. **Initial values.** Each chain fills the missing values of every column it
   imputes with random draws from that column's observed values
   (``initial="sample"``).
3. **Iterations.** In each of ``maxit`` iterations, the chain visits the
   incomplete columns in the visit sequence. For each one it builds the
   design matrix of its predictors from the current completed data (dummy
   coding categorical predictors as R's ``model.matrix``), drops predictors
   that are constant or linearly dependent, and calls the method, which
   fits its model to the observed rows and draws new values for the missing
   cells.
4. **Several chains.** Steps 2 and 3 run ``n_imputations`` times, each chain
   with its own random stream. The final state of each chain is one
   completed dataset.

The last-iteration datasets are the result; earlier iterations are only
recorded through their chain statistics (``chain_mean``, ``chain_var``).

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   datasets = mice.impute(n_imputations=5, maxit=5, seed=1)

   print(len(datasets), datasets[0].isna().sum().sum())   # 5 datasets, nothing missing

Arguments of ``impute``
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Argument
     - Default
     - Meaning
   * - ``n_imputations``
     - 5
     - Number of completed datasets (R's ``m``).
   * - ``maxit``
     - 5
     - Iterations per chain; at least 1.
   * - ``method``
     - ``None``
     - A method name (``"pmm"``, ``"norm"``, ``"logreg"``, ``"polyreg"``,
       ...; R's names, dots included) or ``""`` (do not impute); a string
       for all columns or a dict per column. ``None`` and columns left out
       of a dict get the default of their type: ``"pmm"`` for numeric,
       ``"logreg"`` for boolean or two categories, ``"polyreg"`` for
       unordered categorical or text, ``"polr"`` for ordered categorical
       columns. See :doc:`imputation_methods`.
   * - ``predictor_matrix``
     - ``None``
     - 0/1 data frame, rows are targets, columns predictors. ``None`` means
       every other column predicts every incomplete column. See
       :doc:`predictor_matrices`.
   * - ``initial``
     - ``"sample"``
     - Starting values: random draws from the observed values (as R), or
       ``"meanobs"``: the observed value closest to the mean of a numeric
       column, the most frequent observed value of any other (not in R).
   * - ``visit_sequence``
     - ``"roman"``
     - ``"roman"`` (data order), ``"arabic"`` (reverse), ``"monotone"``
       (fewest missing values first, ties in data order), ``"revmonotone"``,
       ``"random"`` (drawn once per call), or a list of column labels.
   * - ``seed``
     - ``None``
     - Seed of the random number generators. ``None``: different results on
       every call. With a seed the call is reproducible, and chain *k* gets
       the same stream whatever ``n_imputations`` is.
   * - ``default_method``
     - ``("pmm", "logreg", "polyreg", "polr")``
     - The defaults by column type, in the order above (R's
       ``defaultMethod``).
   * - ``<method>_<parameter>``
     -
     - A parameter of a method, for example ``pmm_donors=3``,
       ``rf_n_estimators=50`` or ``norm_boot_ridge=1e-4`` (a dot in the
       method name becomes an underscore). An unknown name raises
       ``TypeError``.

.. code-block:: python

   mice.impute(
       n_imputations=10,
       maxit=10,
       method={"bmi": "pmm", "chl": "midas", "sex": "cart"},
       visit_sequence="monotone",
       pmm_donors=3,
       seed=2024,
   )
   print(mice.method, mice.visit_sequence)

After a run
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Content
   * - ``imputed_datasets``
     - The completed datasets, a list of data frames with the original
       columns, dtypes and index.
   * - ``method``
     - The method of each column (``""`` for columns not imputed), in R's
       names.
   * - ``predictor_matrix``
     - The predictor matrix after the setup step: square, in data order.
   * - ``visit_sequence``
     - The columns in the order they were visited.
   * - ``logged_events``
     - What the setup step and the iterations changed (R's
       ``loggedEvents``); empty if nothing.
   * - ``chain_mean``, ``chain_var``
     - For each imputed numeric column, the mean and variance of its imputed
       values, arrays of shape ``(maxit, n_imputations)``.

The setup step and logged events
--------------------------------

Before the chains start, ``impute`` edits the imputation model as R's
``mice`` does:

- a **constant** column (variance below 1000 times machine epsilon, or fewer
  than two observed values) is no longer used as a predictor;
- a column that (nearly) **duplicates** another (correlation of at least
  0.999) is no longer used as a predictor;
- if such a column is incomplete, it is **not imputed** and keeps its
  missing values.

In every univariate imputation, predictors that are constant or linearly
dependent among the rows used for the model are dropped from that call.

Each change is a row of ``logged_events`` (columns ``it``, ``im``, ``dep``,
``meth``, ``out`` as in R; ``it`` 0 is the setup step), and ``impute`` issues
one ``UserWarning`` with their number:

.. code-block:: python

   import warnings

   df_extra = df.assign(site=1, weight_kg=3.1 * df["bmi"])   # a constant and a copy of bmi

   mice_extra = MICE(df_extra)
   with warnings.catch_warnings():
       warnings.simplefilter("ignore", UserWarning)   # "Number of logged events: 2. ..."
       mice_extra.impute(seed=1)

   print(mice_extra.logged_events)
   print(mice_extra.imputed_datasets[0]["weight_kg"].isna().sum())   # still missing

Here ``site`` is dropped as a predictor and ``weight_kg``, a multiple of
``bmi``, is neither used nor imputed. Remove or recode such columns before
imputing if you need them completed.

Rows in which every value is missing are kept and imputed (with a
``UserWarning``); their imputations carry no information from the row itself.
Columns in which every value is missing are constant and stay missing.

Supported column types
----------------------

Numeric (including pandas ``Int64`` and ``Float64``), boolean, categorical
(ordered or not), object and string columns. Object and string columns are
treated as unordered categorical, with their sorted values as levels.
Other dtypes (dates, durations, complex numbers) raise ``TypeError`` in
``MICE()``; convert them first. Imputed columns keep their dtype, except
that a nullable integer column (``Int64``, ...) imputed with ``"norm"``,
``"norm.nob"``, ``"norm.boot"``, ``"norm.predict"`` or ``"mean"`` becomes
``Float64``: these methods impute new values, not observed ones.

Imputation model and analysis model
-----------------------------------

The imputation model should include every variable of the analysis model,
including the outcome, and variables that predict the missingness or the
incomplete variables. With the default full predictor matrix this holds for
all columns of ``df``; leave out columns that should not be used, such as
identifiers.

Next steps
----------

- :doc:`imputation_methods`
- :doc:`predictor_matrices`
- :doc:`convergence_diagnostics`
