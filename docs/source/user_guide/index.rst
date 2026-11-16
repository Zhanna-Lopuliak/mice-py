User Guide
==========

How to use mice-py, from looking at the missing data to the pooled result.
The examples continue from the :ref:`example data <example-data>` of the
quick start: run that block first, then the blocks of a page in order.

.. toctree::
   :maxdepth: 2

   understanding_missing_data
   mice_overview
   imputation_methods
   predictor_matrices
   controlling_imputation
   convergence_diagnostics
   diagnostic_plots
   pooling_analysis
   logging
   best_practices

**Understanding missing data**
   Missing-data patterns and mechanisms.

**MICE overview**
   What ``impute()`` does, step by step, and its arguments.

**Imputation methods**
   The fourteen methods, the defaults by column type, algorithms and
   parameters.

**Predictor matrices**
   Which variables predict which, ``quickpred``, and the setup step.

**Controlling the imputation**
   Which cells are imputed (``where``), which rows the models leave out
   (``ignore``), changes to the imputations (``post``), and derived columns
   (``passive``).

**Convergence diagnostics**
   Chain statistics, trace plots, and more iterations with
   ``continue_chains``.

**Diagnostic plots**
   All plot functions of the ``plotting`` package.

**Pooling analysis**
   Fitting models, Rubin's rules, and pooling without the ``MICE`` class.

**Logging**
   How to see what the package does.

**Best practices**
   A short checklist.

If something is unclear, see :doc:`../examples/index` for notebooks,
:doc:`../api/index` for every function, or open an issue on
`GitHub <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_.
