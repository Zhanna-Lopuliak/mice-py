Diagnostic Plots
================

The ``plotting`` package draws R mice's diagnostic plots with matplotlib and
seaborn. The examples use ``df`` from the :ref:`example data <example-data>`.

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Function
     - R equivalent
     - Shows
   * - ``md_pattern_like(df)``
     - ``md.pattern(df, plot = FALSE)``
     - the missing-data pattern as a table
   * - ``plot_missing_data_pattern(pattern)``
     - ``md.pattern(df)``
     - that table as a figure
   * - ``stripplot``
     - ``stripplot(imp)``
     - observed and imputed values as points
   * - ``bwplot``
     - ``bwplot(imp)``
     - box plots of observed and imputed values
   * - ``densityplot``
     - ``densityplot(imp)``
     - kernel densities of observed and imputed values
   * - ``densityplot_split``
     - (none)
     - one density panel per imputation, for one column
   * - ``xyplot``
     - ``xyplot(imp, y ~ x)``
     - scatter plot, colored by whether ``y`` was imputed
   * - ``plot_chain_stats``
     - ``plot(imp)``
     - trace plot of the chains (see :doc:`convergence_diagnostics`)

All functions can be imported from ``plotting`` directly.

Common arguments and return value
---------------------------------

The plots of imputed data take the list of completed datasets and a data
frame of the same shape with **1 where a value was observed and 0 where it
was missing** in the original data:

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(seed=1)

   datasets = mice.imputed_datasets
   observed = df.notna().astype(int)    # 1 observed, 0 missing

Further arguments:

- ``columns``: the columns to plot; by default every numeric column with
  missing values, as R's plots take the variables that are not factors.
  (``densityplot_split`` and ``xyplot`` take single columns.)
- ``merge_imputations`` (``stripplot``, ``bwplot``, ``xyplot``): one panel
  for all imputations instead of one per imputation.
- ``observed_color``, ``imputed_color``: default blue and red, as in R.
- ``save_path``: save the figure there (as PNG at 300 dpi, creating missing
  directories) and close it, instead of showing it.

Every function returns its matplotlib ``Figure``, so it can be changed or
saved again. When there is nothing to plot (no numeric column with missing
values, an unknown column) the function warns and returns ``None``.
Figures are at most 20 inches a side; per-variable panels wrap after four.

In a Jupyter notebook the figure is shown when it is drawn, and the returned
``Figure`` is shown a second time if the call is the last line of a cell; end
that line with ``;`` or assign the result.

Missing-data pattern
--------------------

.. code-block:: python

   from plotting import md_pattern_like, plot_missing_data_pattern

   pattern = md_pattern_like(df)
   print(pattern)
   plot_missing_data_pattern(pattern, title="Example data", save_path="plots/pattern.png")

In the table, columns are sorted by their number of missing values and rows
follow R's order. In the figure, blue cells are observed and pink cells
missing; the counts of rows are on the left, the missing cells per pattern
on the right and per column at the bottom.

Observed and imputed values
---------------------------

.. code-block:: python

   from plotting import bwplot, densityplot, densityplot_split, stripplot

   stripplot(datasets, observed, save_path="plots/strip.png")
   stripplot(datasets, observed, merge_imputations=True, save_path="plots/strip_merged.png")
   bwplot(datasets, observed, save_path="plots/box.png")
   densityplot(datasets, observed, save_path="plots/density.png")
   densityplot_split(datasets, observed, column="chl", save_path="plots/density_chl.png")

The imputed values (red) should be plausible values of the variable. They
need not have the same distribution as the observed values (blue): under MAR
the missing values can differ systematically from the observed ones. Large
differences are worth explaining, though.

By default ``stripplot``, ``bwplot`` and ``densityplot`` plot the numeric
columns with missing values, here ``bmi`` and ``chl`` but not ``sex``.
``bwplot``, ``densityplot`` and ``densityplot_split`` need numeric columns
and raise ``ValueError`` for a categorical one passed explicitly;
``stripplot`` and ``xyplot`` also draw categorical columns. In
``densityplot``, a set of imputed values without variance (for example a
single imputed cell) is drawn as a dashed vertical line.

Relations between two variables
-------------------------------

.. code-block:: python

   from plotting import xyplot

   xyplot(datasets, observed, x="bmi", y="chl", save_path="plots/xy.png")
   xyplot(datasets, observed, x="bmi", y="chl", merge_imputations=True, save_path="plots/xy_merged.png")

As in R's ``xyplot``, points are colored by whether ``y`` was imputed. The
first panel shows the incomplete data (rows with both ``x`` and ``y``
observed); each further panel one completed dataset, where a row with an
observed ``y`` and an imputed ``x`` is drawn in the observed color at its
imputed ``x``.

Trace plot
----------

.. code-block:: python

   from plotting import plot_chain_stats

   plot_chain_stats(mice.chain_mean, mice.chain_var, save_path="plots/trace.png")

It shows the mean and the standard deviation of the imputed values per
iteration and chain; see :doc:`convergence_diagnostics`. Where the plots
differ from R's is listed in :doc:`../r_mice/differences`.
