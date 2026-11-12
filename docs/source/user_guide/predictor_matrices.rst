Predictor Matrices
==================

The predictor matrix says which variables are used to impute which. The
examples use ``df`` from the :ref:`example data <example-data>`.

What it is
----------

A data frame of 0s and 1s with column labels as index and as columns:
**rows are targets, columns are predictors**. A 1 in row ``chl``, column
``age`` means that ``age`` is a predictor in the imputation model of ``chl``.
The diagonal is 0. Rows of complete columns are not used.

The default
-----------

Without ``predictor_matrix``, every other column predicts every incomplete
column, as in R (all ones, zero diagonal). The setup step then removes
constant and collinear variables (see :doc:`mice_overview`), and
``mice.predictor_matrix`` shows the matrix actually used:

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(seed=1)
   print(mice.predictor_matrix)

Your own matrix
---------------

Start from the full matrix and set entries to 0:

.. code-block:: python

   import numpy as np
   import pandas as pd

   pred = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(pred.values, 0)
   pred.loc["bmi", "chl"] = 0      # do not use chl to impute bmi

   mice.impute(predictor_matrix=pred, seed=1)
   print(mice.predictor_matrix.loc["bmi"])

The matrix may also list only some rows and columns; missing entries count
as 0. Every incomplete column that is imputed needs a row, otherwise
``impute`` raises ``ValueError``:

.. code-block:: python

   pred_small = pd.DataFrame(
       {"age": [1, 1, 1], "sex": [0, 1, 1], "bmi": [1, 0, 1], "chl": [0, 1, 0]},
       index=["sex", "bmi", "chl"],
   )
   mice.impute(predictor_matrix=pred_small, seed=1)
   print(mice.predictor_matrix)

An incomplete predictor is used with its current imputed values, so it has
to be imputed itself. To use a variable as a predictor without imputing it,
give it method ``""``: rows where it is missing are then left out of the
models that use it, and their missing cells in the target are not imputed.

quickpred
---------

For data with many columns, ``imputation.quickpred`` selects predictors by
simple statistics, with the rule of R's ``quickpred``: a column predicts a
target when its absolute correlation with the target, or with the target's
missingness indicator, exceeds ``mincor``.

.. code-block:: python

   import imputation

   pred = imputation.quickpred(df, mincor=0.2, include=["age"])
   print(pred)

   mice.impute(predictor_matrix=pred, seed=1)

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Argument
     - Default
     - Meaning
   * - ``mincor``
     - 0.1
     - minimum absolute correlation (strictly exceeded); a number, one value
       per target, or a matrix
   * - ``minpuc``
     - 0.0
     - minimum proportion of usable cases: of the rows where the target is
       missing, the share where the predictor is observed
   * - ``include``
     - ``None``
     - columns used as predictors of every target
   * - ``exclude``
     - ``None``
     - columns never used as predictors (``include`` wins)
   * - ``method``
     - ``"pearson"``
     - ``"pearson"``, ``"kendall"`` or ``"spearman"``

Categorical columns enter the correlations through their level codes, as in
R. Unlike R, ``include`` and ``exclude`` take exact labels, and an unknown
label raises ``ValueError``.

The arguments ``min_cor``, ``min_puc``, ``include``, ``exclude`` and
``correlation_method`` of ``impute`` are deprecated: they call ``quickpred``
and will be removed in version 0.4.

Guidelines
----------

These follow van Buuren (2018, chapter 6):

- Include every variable of the analysis model, including its outcome.
- Include variables related to the missingness, and variables strongly
  correlated with the incomplete variables.
- Leave out identifiers and variables that are mostly missing where the
  target is missing (``minpuc``).
- With many variables, a smaller set (for example 15 to 25 predictors per
  target) is usually enough; ``quickpred`` helps to find it.

Next steps
----------

- :doc:`convergence_diagnostics`: check the chains after changing the model.
