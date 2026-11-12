Quick start
===========

The workflow has three steps, as in R mice:

1. **Impute**: ``MICE(data).impute()`` creates several completed datasets.
2. **Analyze**: ``fit()`` fits the same model to each of them.
3. **Pool**: ``pool()`` combines the fits with Rubin's rules.

.. _example-data:

Example data
------------

The examples in this documentation use a small simulated dataset: ``age``
and ``sex`` explain ``bmi``, and all three explain the cholesterol level
``chl``. ``bmi``, ``chl`` and ``sex`` have missing values; ``age`` is
complete. The other pages of the user guide start from this data frame
``df``.

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(1)
   n = 300
   df = pd.DataFrame({"age": rng.integers(20, 80, n), "sex": rng.choice(["f", "m"], n)})
   df["bmi"] = 18 + 0.1 * df["age"] + rng.normal(0, 3, n)
   df["chl"] = 120 + 1.2 * df["age"] + 2.0 * df["bmi"] + rng.normal(0, 20, n)
   df.loc[rng.random(n) < 0.2, "bmi"] = np.nan
   df.loc[rng.random(n) < 0.3, "chl"] = np.nan
   df.loc[rng.random(n) < 0.1, "sex"] = np.nan

   print(df.isna().sum())

Impute, analyze, pool
---------------------

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   datasets = mice.impute(n_imputations=5, seed=1)   # methods by column type, 5 iterations: as R's mice()

   mice.fit("chl ~ age + bmi + sex")                 # one OLS fit per completed dataset
   result = mice.pool()                              # Rubin's rules, Barnard-Rubin df

   print(result["table"][["estimate", "std_error", "df", "fmi", "conf_low", "conf_high"]])

``impute`` returns the completed datasets (also kept in
``mice.imputed_datasets``). Without ``seed`` every call gives different
imputations; with the same seed it gives the same ones.

``pool()`` returns a dictionary. Its ``"table"`` has one row per model term
with R's columns: the pooled ``estimate``, the within (``ubar``), between
(``b``) and total (``t``) variance, the degrees of freedom ``df``, ``riv``,
``lambda``, the fraction of missing information ``fmi``, the standard error,
t statistic, p-value and 95% interval. ``mice.pool(summ=True)`` returns the
same results as a printable statsmodels summary.

What ``impute`` decided
-----------------------

.. code-block:: python

   print(mice.method)            # {'age': '', 'sex': 'logreg', 'bmi': 'pmm', 'chl': 'pmm'}
   print(mice.visit_sequence)    # ['sex', 'bmi', 'chl']
   print(mice.predictor_matrix)  # rows: targets, columns: predictors
   print(mice.logged_events)     # changes the setup step made; empty here

As in R, each incomplete column gets the method of its type: ``pmm`` for
the numeric ``bmi`` and ``chl``, ``logreg`` (logistic regression) for
``sex``, which has two values. A complete column gets the method ``""`` and
is not imputed. If the setup step removes a constant or collinear variable,
``impute`` warns and the change appears in ``mice.logged_events``; see
:doc:`user_guide/mice_overview`.

Check the imputations
---------------------

.. code-block:: python

   from plotting import densityplot, plot_chain_stats

   plot_chain_stats(mice.chain_mean, mice.chain_var, save_path="trace.png")
   densityplot(mice.imputed_datasets, df.notna().astype(int), columns=["bmi", "chl"], save_path="density.png")

The trace plot shows the mean and standard deviation of the imputed values
per iteration and chain; the chains should mix and show no trend. The density
plot compares the observed values (blue) with the imputed ones (red). In a
notebook, leave out ``save_path`` to show the figures. See
:doc:`user_guide/convergence_diagnostics` and
:doc:`user_guide/diagnostic_plots`.

Next steps
----------

- :doc:`user_guide/imputation_methods`: the fourteen methods, the defaults
  by column type, and the parameters.
- :doc:`user_guide/predictor_matrices`: choosing the predictors.
- :doc:`user_guide/pooling_analysis`: other models, the pooled table, pooling
  without ``MICE``.
- :doc:`r_mice/differences`: for users of R mice.
- :doc:`upgrading`: for users of mice-py 0.1 and 0.2.
