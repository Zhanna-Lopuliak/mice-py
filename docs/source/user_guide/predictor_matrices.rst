Predictor Matrices
==================

The predictor matrix says which variables are used to impute which. The
examples use ``df`` from the :ref:`example data <example-data>`.

What it is
----------

A data frame of 0s and 1s with column labels as index and as columns:
**rows are targets, columns are predictors**. A 1 in row ``chl``, column
``age`` means that ``age`` is a predictor in the imputation model of ``chl``.
The diagonal is 0. Only the rows of the columns that are imputed are used:
by default the incomplete columns; under ``where`` (see
:doc:`controlling_imputation`) the columns with where-cells, complete ones
included.

The default
-----------

Without ``predictor_matrix``, every other column predicts every column, as
in R (all ones, zero diagonal). The setup step then removes
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
as 0. Every column that is imputed needs a row, otherwise ``impute`` raises
``ValueError``; a column that is not imputed (method ``""``) and a passive
column need none:

.. code-block:: python

   pred_small = pd.DataFrame(
       {"age": [1, 1, 1], "sex": [0, 1, 1], "bmi": [1, 0, 1], "chl": [0, 1, 0]},
       index=["sex", "bmi", "chl"],
   )
   mice.impute(predictor_matrix=pred_small, seed=1)
   print(mice.predictor_matrix)

An incomplete predictor is used with its current imputed values, so it has
to be imputed itself. To use a variable as a predictor without imputing it,
give it method ``""``: the rows where it is missing are then not used to fit
the models that use it, and the missing cells of the target in those rows
are not imputed. As in R, the methods still receive those rows; only
``logreg``, ``polyreg`` and ``polr`` use them, in the rows they add to the
data (R's ``augment``).

Passive columns
---------------

A passive column (see :doc:`controlling_imputation`) is computed by its
function, not imputed, so its row is not used. Its column says which
targets it predicts. Set its entries to 0 in the rows of the columns it is
computed from, so that a column is not predicted from a function of itself:

.. code-block:: python

   df_int = df.assign(age_bmi=df["age"] * df["bmi"])

   pred = pd.DataFrame(1, index=df_int.columns, columns=df_int.columns)
   np.fill_diagonal(pred.values, 0)
   pred.loc["bmi", "age_bmi"] = 0

   mice_int = MICE(df_int)
   mice_int.impute(predictor_matrix=pred, passive={"age_bmi": lambda d: d["age"] * d["bmi"]}, seed=1)
   print(mice_int.predictor_matrix)

One iteration of a monotone pattern
-----------------------------------

With ``visit_sequence="monotone"`` and ``maxit=1``, ``impute`` edits the
predictor matrix as R does: in the order of all columns by their number of
where-cells (columns without any first), each column keeps as predictors
only the columns before it. If the missing values follow a monotone
pattern, one iteration then imputes it without using start values.
``mice.predictor_matrix`` shows the edited matrix:

.. code-block:: python

   mice.impute(visit_sequence="monotone", maxit=1, seed=1)
   print(mice.visit_sequence)      # ['sex', 'bmi', 'chl']
   print(mice.predictor_matrix)    # sex: age; bmi: age, sex; chl: age, sex, bmi

Other ``maxit``, the other keywords and a list visit sequence leave the
matrix as it is.

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
label raises ``ValueError``. As R's, ``quickpred`` gives the rows of
complete columns no predictors; under ``where`` such a column can be
imputed, so set the predictors of its row yourself.

The arguments ``min_cor``, ``min_puc``, ``include``, ``exclude`` and
``correlation_method`` of ``impute`` were removed in version 0.4: call
``quickpred`` and pass its result as ``predictor_matrix``, as above.

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
