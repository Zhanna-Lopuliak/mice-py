Pooling Analysis
================

After imputation, the analysis model is fitted to every completed dataset and
the results are combined with Rubin's rules (Rubin 1987), using the
small-sample degrees of freedom of Barnard and Rubin (1999), as R's
``pool()`` does. The formulas are on :doc:`../theory/rubins_rules`. The
examples use ``df`` from the :ref:`example data <example-data>`.

Fit and pool
------------

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(n_imputations=10, seed=1)

   fits = mice.fit("chl ~ age + bmi + sex")   # one statsmodels result per dataset
   result = mice.pool()

``fit`` is R's ``with(imp, lm(...))``: it fits the formula to each completed
dataset with ``statsmodels.formula.api.ols`` and keeps the fits in
``mice.model_results``. ``pool`` combines the latest fits.

The pooled table
~~~~~~~~~~~~~~~~

``result["table"]`` is R's ``summary(pool(fit), type = "all", conf.int =
TRUE)``: one row per term with these columns.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``m``
     - number of imputations
   * - ``estimate``
     - pooled estimate: the mean of the m estimates
   * - ``ubar``
     - within-imputation variance: the mean of the squared standard errors
   * - ``b``
     - between-imputation variance of the m estimates
   * - ``t``
     - total variance ``ubar + (1 + 1/m) b``
   * - ``dfcom``
     - degrees of freedom of the complete-data analysis (residual df of the
       first fit)
   * - ``df``
     - Barnard-Rubin degrees of freedom
   * - ``riv``
     - relative increase in variance due to missingness, ``(1 + 1/m) b / ubar``
   * - ``lambda``
     - proportion of the total variance due to missingness,
       ``(1 + 1/m) b / t``
   * - ``fmi``
     - fraction of missing information, ``(riv + 2/(df + 3)) / (riv + 1)``
   * - ``std_error``, ``statistic``, ``p_value``
     - ``sqrt(t)``, ``estimate / std_error``, two-sided p-value from the t
       distribution with ``df`` degrees of freedom
   * - ``conf_low``, ``conf_high``
     - 95% interval from the same t distribution

.. code-block:: python

   table = result["table"]
   print(table[["estimate", "std_error", "df", "p_value", "conf_low", "conf_high", "fmi"]].round(3))

The rest of the dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Content
   * - ``"pooled_result"``
     - a ``MICEresult``, a statsmodels results object (see below)
   * - ``"pooled_params"``
     - the pooled estimates as an array
   * - ``"pooled_covariance"``, ``"within_covariance"``,
       ``"between_covariance"``
     - the total, within and between covariance matrices
   * - ``"fraction_missing_info"``
     - ``fmi`` per term, as in the table
   * - ``"pooled_scale"``
     - mean residual scale of the fits (NaN for models without one)
   * - ``"n_imputations"``, ``"parameter_names"``, ``"formula"``
     - as the names say

``mice.pool(summ=True)`` returns a printable statsmodels summary instead,
with the pooled estimates, t-based p-values and intervals, and a ``df`` and
an ``FMI`` column:

.. code-block:: python

   print(mice.pool(summ=True))

The pooled result
~~~~~~~~~~~~~~~~~

``result["pooled_result"]`` (also ``mice.pooled_result``) behaves like a
statsmodels result: ``params``, ``bse``, ``tvalues``, ``pvalues`` and
``conf_int()`` use each term's own degrees of freedom (attribute ``df``);
``frac_miss_info`` holds ``fmi``. Tests of several coefficients at once
(``t_test``, ``f_test``, ``wald_test``) raise ``NotImplementedError``: they
need their own pooling rule (R's ``D1``), which is not implemented.

.. code-block:: python

   pooled = result["pooled_result"]
   print(pooled.params)
   print(pooled.conf_int(alpha=0.10))   # 90% intervals

With ``n_imputations=1`` nothing can be pooled: ``pool`` warns, reports the
single analysis with ``dfcom`` degrees of freedom, and ``b``, ``fmi`` and
related columns are NaN.

Other models
------------

``fit`` takes any statsmodels formula model as ``model``; ``model_kwargs``
go to the model and further keyword arguments to its ``fit()``:

.. code-block:: python

   import statsmodels.api as sm
   import statsmodels.formula.api as smf

   # logistic regression of high cholesterol
   mice.fit("I((chl > 230) * 1) ~ age + bmi", model=smf.logit, disp=False)
   print(mice.pool()["table"][["estimate", "std_error", "df", "fmi"]].round(3))

   # gamma GLM with a log link
   mice.fit("chl ~ age + bmi", model=smf.glm, model_kwargs={"family": sm.families.Gamma(sm.families.links.Log())})
   print(mice.pool()["table"][["estimate", "std_error", "df", "fmi"]].round(4))

Formulas use patsy syntax: ``C(x)`` for a categorical term, ``x1:x2`` and
``x1 * x2`` for interactions, ``I(x**2)`` and ``np.log(x)`` for
transformations. Term names follow patsy (``Intercept``, ``sex[T.m]``).
Every variable in the formula must be a column of the data.

Any model with an estimate and a standard error
-----------------------------------------------

For any other model or statistic, compute the estimate and its standard
error on each completed dataset yourself and pool them with
``imputation.pooling.pool_estimates``, which returns the same table. Its
arguments are the m × p arrays of estimates and of squared standard errors
(1-D arrays for a single quantity), and the complete-data degrees of freedom
``dfcom``, for example ``n - p`` for a regression; ``np.inf`` (the default)
assumes a large sample.

A correlation is not approximately normal, so it is pooled on Fisher's z
scale, with variance ``1 / (n - 3)``, and transformed back:

.. code-block:: python

   import numpy as np
   from imputation.pooling import pool_estimates

   z = [np.arctanh(data["bmi"].corr(data["chl"])) for data in mice.imputed_datasets]
   variance = [1 / (len(df) - 3)] * len(z)

   table = pool_estimates(np.array(z), np.array(variance), param_names=["bmi, chl"])
   print(np.tanh(table[["estimate", "conf_low", "conf_high"]]).round(3))   # pooled correlation
   print(table["fmi"].round(3))

``imputation.pooling.barnard_rubin(m, b, t, dfcom)`` gives the degrees of
freedom alone.

Pooling descriptive statistics
------------------------------

``imputation.pooling`` also pools means and proportions from any list of
completed data frames, without the ``MICE`` class:

.. code-block:: python

   from imputation.pooling import pool_descriptive_statistics, pool_subset

   stats = pool_descriptive_statistics(mice.imputed_datasets)
   print(stats.summary().round(3))

   print(pool_subset(mice.imputed_datasets, columns=["bmi", "chl"]).table().round(3))

What is pooled:

- **numeric columns**: the mean, with within-imputation variance
  ``s² / n``;
- **categorical, object and boolean columns**: the proportion of each level,
  named ``column[level]``, with within-imputation variance ``p (1 - p) / n``.

Pooling uses ``pool_estimates`` with ``dfcom = n - 1``. The result is a
``PoolingResult`` with the arrays ``estimates``, ``variances`` (total),
``within_variance``, ``between_variance`` and ``frac_miss_info`` (``fmi``),
plus ``param_names``, ``n_imputations`` and ``sample_size``. Its
``summary(alpha=0.05)`` returns a data frame with ``Estimate``,
``Std_Error``, ``DF``, the interval, the three variances, ``Lambda`` and
``FMI``; ``table(alpha=0.05)`` returns the full table of ``pool_estimates``.

Both functions take ``include_numeric`` and ``include_categorical`` to skip
column types. ``pool_from_files(paths, read_kwargs=None)`` reads the datasets
from CSV files first. The datasets must have the same shape and columns; a
dataset that still has missing values is pooled with a logged warning.

Common mistakes
---------------

- **Analyzing one completed dataset.** Its standard errors ignore the
  uncertainty about the missing values and are too small.
- **Averaging the completed datasets** and analyzing the average. This is a
  single imputation in disguise, with the same problem, and it shrinks the
  variability of the imputed variables.
- **Pooling quantities that are not approximately normal**, such as
  correlations or variances, directly. Pool a transformed version (for
  example Fisher's z for a correlation) and transform back (van Buuren 2018,
  section 5.2).

How many imputations
--------------------

The default of 5 is often enough for point estimates. Standard errors,
p-values and intervals are more stable with more imputations; a common rule
of thumb is to use at least as many imputations as the percentage of
incomplete cases (White, Royston and Wood 2011). A large ``fmi`` is another
sign that more imputations help.
