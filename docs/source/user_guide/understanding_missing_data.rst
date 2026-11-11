Understanding Missing Data
==========================

The examples use ``df`` from the :ref:`example data <example-data>`.

How much is missing
-------------------

.. code-block:: python

   print(df.isna().sum())            # missing values per column
   print(df.isna().mean().round(2))  # share missing per column
   print(df.isna().any(axis=1).sum(), "incomplete rows of", len(df))

Missing-data patterns
---------------------

``md_pattern_like`` gives the table of R's ``md.pattern``: one row per
pattern, 1 for observed and 0 for missing cells, the number of rows with
that pattern as the index, the number of missing cells per pattern in the
last column and per variable in the last row. Columns are sorted by their
number of missing values; rows in R's order.

.. code-block:: python

   from plotting import md_pattern_like, plot_missing_data_pattern

   pattern = md_pattern_like(df)
   print(pattern)

   plot_missing_data_pattern(pattern, save_path="pattern.png")

The pattern shows which variables are missing together. It cannot tell the
mechanisms below apart: the same pattern can arise under each of them.

Mechanisms
----------

**MCAR (missing completely at random)**
   Missingness does not depend on any data. Complete-case analysis is
   unbiased, but wastes information.

**MAR (missing at random)**
   Missingness depends only on observed data. Multiple imputation with an
   imputation model that includes the variables related to the missingness
   gives valid inference. This is the assumption of MICE.

**MNAR (missing not at random)**
   Missingness depends on the missing values themselves. MICE alone does not
   remove the bias; sensitivity analyses are needed.

In the example data the values are deleted completely at random, so MCAR
holds. In real data MAR cannot be tested from the data; it becomes more
plausible when the imputation model includes the variables that predict
missingness. See :doc:`../theory/missing_data_mechanisms` for the
definitions.

Next steps
----------

- :doc:`mice_overview`: how MICE imputes.
- :doc:`imputation_methods`: which method to use.
