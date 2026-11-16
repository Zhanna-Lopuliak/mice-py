Controlling the Imputation
==========================

Four arguments of ``impute()`` change what the chains do. ``where``,
``ignore`` and ``post`` have the names of the arguments of R's ``mice()``;
``passive`` is R's passive imputation, which R writes as a formula in
``method``:

- ``where``: which cells get an imputation;
- ``ignore``: which rows the imputation models leave out;
- ``post``: a function that changes the imputations of a column right after
  they are drawn;
- ``passive``: columns that are computed from other columns instead of
  imputed.

The examples use ``df`` from the :ref:`example data <example-data>`.

Which cells: ``where``
----------------------

A cell that gets an imputation is a **where-cell**. By default the
where-cells are the missing cells, as in R. ``where`` takes R's keywords:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``where``
     - Where-cells
   * - ``None``, ``"missing"``
     - the missing cells (the default)
   * - ``"all"``
     - every cell, the observed ones too
   * - ``"observed"``
     - the observed cells
   * - ``"none"``
     - no cell: nothing is imputed, and the completed datasets are copies of
       the data

or a boolean DataFrame with the index and the column labels of the data
(its columns may be in any order; they are aligned by label), or a boolean
NumPy array of the shape of the data. Numbers, also 0 and 1, and missing
values are not accepted. After a run, ``mice.where`` is the frame used.

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   datasets = mice.impute(where="all", seed=1)

   print(mice.method)           # age, complete, gets a method too
   print(mice.where.sum())      # 300 where-cells in every column

A column without where-cells gets the method ``""`` and a column with
where-cells the default method of its type, also a complete one; under
``where="all"`` the complete ``age`` gets ``pmm``.
``imputation.make_method(df, where="all")`` shows the methods before a run.
Imputing observed cells is a way to check the imputation model: the
imputations of observed cells can be compared with the values observed
there. A complete integer column with where-cells, such as ``age``, becomes
``Int64`` in the completed datasets, because a where-cell can end up
missing (below).

To change single cells, start from ``df.isna()``, the default, and edit it.
Here the missing ``chl`` of the participants aged 70 or more are left out:

.. code-block:: python

   where = df.isna()
   where.loc[df["age"] >= 70, "chl"] = False   # leave these cells missing

   datasets = mice.impute(where=where, seed=1)
   print(datasets[0].isna().sum())   # chl 19, sex 2, bmi 2

A missing cell that ``where`` leaves out stays missing, and it is missing
where ``chl`` predicts other columns too: a where-cell is imputed only where
all its predictors are complete in its row, so a missing ``sex`` or ``bmi``
in a row with a left-out ``chl`` stays missing as well (two cells of each
here). R does the same.

**Observed cells.** An observed where-cell gets an imputation in the
completed datasets. In the data the chains work on it keeps its observed
value, so the model of its own column is fitted on that value and the other
columns are predicted from it, as in R. Where such a cell cannot be imputed
(its column has no method, or a predictor is missing in its row), it is
missing in the completed datasets, and its observed value is lost; R does
this silently, mice-py warns:

.. code-block:: python

   where = df.isna()
   where.loc[df["age"] >= 70, "chl"] = False
   where["bmi"] = True                         # impute every bmi, the observed ones too

   datasets = mice.impute(where=where, seed=1)
   # UserWarning: Observed values replaced by missing values in the completed
   # data, because where marks them and they could not be imputed (as in R);
   # per column: 'bmi': 17 (predictors incomplete in these rows). See MICE.where.

The same holds for ``where="observed"``: missing cells stay missing, so an
observed cell is imputed only in a row without missing predictors.

**Plots.** The plots of :doc:`diagnostic_plots` take a frame of the cells
that were not imputed; after a run with ``where`` that is ``~mice.where``:

.. code-block:: python

   from plotting import densityplot

   densityplot(mice.imputed_datasets, ~mice.where, save_path="density_where.png")

Which rows the models use: ``ignore``
-------------------------------------

``ignore`` takes one boolean per row (a list, a tuple, a NumPy or pandas
array, or a Series with the index of the data). An ignored row is left out
of the model of every column in every iteration, and its observed values
are not drawn as start values; its where-cells are imputed like any other,
from models fitted on the other rows. After a run, ``mice.ignore`` is the
array used.

``ignore`` does not remove the rows: their observed values stay in the
completed datasets, their values are the predictors of their own
imputations, and the steps before the chains (the methods, the setup step,
the visit sequence) use all rows. As in R, ``logreg``, ``polyreg`` and
``polr`` take the ignored rows into their augmentation. When fewer than 10
rows are left for the models, ``impute`` warns with R's text "Fewer than 10
rows for fitting the imputation model. Are you sure?"; a column that is
imputed and has no observed value outside the ignored rows raises
``ValueError``.

.. _train-and-test:

**New rows imputed with the models of the old ones.** The typical use is a
training set and a test set: the test rows must be imputed, but must not
influence the imputation models. R does this with ``mice.mids(newdata = )``
or with ``ignore``; mice-py with ``ignore``. Put the two sets together,
ignore the test rows, and take their completed rows:

.. code-block:: python

   import numpy as np
   import pandas as pd

   train, test = df.iloc[:200], df.iloc[200:]

   both = pd.concat([train, test], ignore_index=True)
   ignore = np.arange(len(both)) >= len(train)        # True for the rows of test

   mice = MICE(both)
   datasets = mice.impute(ignore=ignore, seed=1)
   test_completed = [data.iloc[len(train):] for data in datasets]

   print(len(test_completed), test_completed[0].isna().sum().sum())   # 5 datasets, nothing missing

The training rows are imputed in the same run, from the same models.

Changing the imputations: ``post``
----------------------------------

``post`` maps a column to a function ``f(imp, data)``. Right after each
univariate imputation of the column, in every iteration and chain, ``f``
gets

- ``imp``: the current imputations of all where-cells of the column, a
  Series indexed by their row labels (missing where a cell could not be
  imputed);
- ``data``: the data the chains work on, in the same rows, all columns;

both as copies, and returns the values to keep: a Series with the index of
``imp``, or a sequence of its length, of values the column can hold. They
replace the imputations; the next columns are imputed from them, and the
chain statistics are taken after them.

.. code-block:: python

   def show(imp, data):
       print(type(imp).__name__, len(imp), data.shape)
       return imp

   MICE(df).impute(n_imputations=1, maxit=1, post={"bmi": show}, seed=1)   # Series 40 (40, 4)

**Bounds.** ``norm`` imputes new values, also outside the observed range.
``Series.clip`` keeps them within bounds, as R's ``squeeze()``:

.. code-block:: python

   low, high = df["chl"].min(), df["chl"].max()

   mice = MICE(df)
   datasets = mice.impute(
       method={"chl": "norm"},
       post={"chl": lambda imp, data: imp.clip(low, high)},
       seed=1,
   )
   print(all(d["chl"].between(low, high).all() for d in datasets))   # True

**A shift, for a sensitivity analysis.** Under missing at random the
imputations of ``chl`` follow the observed relations. To see how a result
changes if the missing values were systematically lower, shift the
imputations by ``delta`` and repeat the analysis for several values (a
delta adjustment):

.. code-block:: python

   def shift(delta):
       return lambda imp, data: imp + delta

   means = {}
   for delta in (0, -10, -20):
       mice = MICE(df)
       mice.impute(post={"chl": shift(delta)}, seed=1)
       mice.fit("chl ~ 1")                       # the mean of chl
       means[delta] = mice.pool()["table"].loc["Intercept", ["estimate", "conf_low", "conf_high"]]

   print(pd.DataFrame(means).T.round(1))         # estimates 224.4, 221.5, 218.5

**A rule that reads another column.** ``data`` holds the other columns of
the same rows, with their current imputations. Here the imputed ``bmi`` of
participants under 30 is capped at 30:

.. code-block:: python

   datasets = mice.impute(
       method={"bmi": "norm"},
       post={"bmi": lambda imp, data: imp.where(data["age"] >= 30, imp.clip(upper=30))},
       seed=1,
   )

As in R, ``f`` is not called for a column that is not imputed (no
where-cells, method ``""``, or removed by the setup step), and it is called
also when none of the column's where-cells could be imputed. A function
that raises, or returns another number of values, a Series with another
index or values the column cannot hold, stops the run with a
``ValueError`` that names the column, the iteration and the imputation.
``f`` gets no random generator: a function that draws random numbers must
own a seeded one. After a run, ``mice.post`` holds the functions in effect.

Derived columns: ``passive``
----------------------------

A column that is a function of other columns, such as a body mass index
from weight and height or an interaction for the analysis model, should
stay that function of the imputed values. Imputed on its own it would not;
computed after the imputation it would not take part in the imputation
model. A **passive** column is computed at each of its visits from the
current data, so it is both. ``passive`` maps the column to a function
``f(data)``, which gets a copy of the data the chains work on (all rows,
with the current imputations) and returns one value per row; the
where-cells of the column take the values of their rows.

The example adds the interaction of ``age`` and ``bmi``, for the analysis
model ``chl ~ age + bmi + age_bmi + sex``:

.. code-block:: python

   df_int = df.assign(age_bmi=df["age"] * df["bmi"])

   pred = pd.DataFrame(1, index=df_int.columns, columns=df_int.columns)
   np.fill_diagonal(pred.values, 0)
   pred.loc["bmi", "age_bmi"] = 0          # age_bmi is computed from bmi: it does not predict bmi

   mice = MICE(df_int)
   datasets = mice.impute(
       passive={"age_bmi": lambda d: d["age"] * d["bmi"]},
       predictor_matrix=pred,
       seed=1,
   )
   print(mice.method["age_bmi"], mice.visit_sequence)   # passive ['sex', 'bmi', 'chl', 'age_bmi']
   print(all((d["age_bmi"] == d["age"] * d["bmi"]).all() for d in datasets))   # True

   mice.fit("chl ~ age + bmi + age_bmi + sex")
   print(mice.pool()["table"][["estimate", "std_error"]].round(3))

What to know:

- **The predictor matrix.** A passive column has no model, so it needs no
  row. Whether it predicts its inputs is set in its column: set the entries
  of its inputs to 0, as above, so that ``bmi`` is not predicted from a
  function of itself. This is R's practice too.
- **The visit order.** By default (``visit_sequence=None``) the passive
  columns are visited last, after their inputs, so they are computed from
  the imputations of the same iteration; a column visited before them is
  predicted from their values of the previous iteration. A list places a
  passive column anywhere, for example right after its input,
  ``visit_sequence=["sex", "bmi", "age_bmi", "chl"]``, and may name it
  twice. The keywords (``"roman"``, ...) place it like any other column.
- **The start.** As in R, a passive column starts from sampled observed
  values, like an imputed column, and gets its function's values at its
  first visit. Each function is also called once on the data before the
  chains start, so that one that fails, or does not return one value per
  row, stops the run before any imputation; the dtype of the column is
  fixed from what it returns.
- **A function that is not row-wise** sees all rows, with the current
  imputations: in ``lambda d: (d["bmi"] - d["bmi"].mean()) ** 2`` the mean
  is that of the whole completed column. R evaluates its formula on the
  rows being imputed only (see :doc:`../r_mice/differences`). A passive
  column that is a linear function of one other column, such as a centred
  ``bmi``, is collinear with it, and the setup step removes it, as in R.
- **The name** ``"passive"``. ``mice.method`` shows ``"passive"`` for a
  passive column. It is not a method: ``method`` does not take it, a
  ``method`` dict must not name a passive column, and ``post`` must not
  either (put the change into the function).

After a run, ``mice.passive`` holds the functions in effect.

Continuing a run
----------------

``continue_chains()`` runs more iterations with the same ``where``,
``ignore``, ``post`` and ``passive``; ``impute(maxit=0)`` gives the start of
the chains. See :doc:`convergence_diagnostics`.

Next steps
----------

- :doc:`convergence_diagnostics`
- :doc:`predictor_matrices`
- :doc:`../r_mice/differences`: where these arguments differ from R's.
