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
   pmm, midas, cart, rf and sample: algorithms and parameters.

**Predictor matrices**
   Which variables predict which, ``quickpred``, and the setup step.

**Convergence diagnostics**
   Chain statistics and trace plots.

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
