Convergence Diagnostics
=======================

MICE is an iterative algorithm: each chain starts from random draws and
updates the imputations ``maxit`` times. The chains have converged when the
imputations no longer drift from one iteration to the next and the chains
mix with each other. The examples use ``df`` from the
:ref:`example data <example-data>`.

Chain statistics
----------------

After ``impute``, ``chain_mean`` and ``chain_var`` hold, for each imputed
numeric column, the mean and the variance of its imputed values in every
iteration and chain, as arrays of shape ``(maxit, n_imputations)``. They are
R's ``chainMean`` and ``chainVar``. Categorical columns have no chain
statistics.

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=20, seed=1)

   print(mice.chain_mean["chl"].shape)      # (20, 5)
   print(mice.chain_mean["chl"][-3:].round(1))

Trace plot
----------

``plot_chain_stats`` draws, per column, the chain means and the standard
deviations of the imputed values (the square root of ``chain_var``) against
the iteration, one line per chain, as R's ``plot(imp)``.

.. code-block:: python

   from plotting import plot_chain_stats

   plot_chain_stats(mice.chain_mean, mice.chain_var, columns=["bmi", "chl"], save_path="trace.png")

What to look for:

- **No trend**: after the first few iterations the lines move around a
  stable level.
- **Mixing**: the lines of the different chains overlap; no chain stays
  apart from the others.

A trend or chains that stay apart mean more iterations are needed, or that
the imputation model has a problem, for example an incomplete variable that
is (nearly) a function of another one.

How many iterations
-------------------

The default is ``maxit=5``, as in R. Van Buuren (2018) notes that a small
number of iterations, often 5 to 20, is usually enough, and that more are
needed when the incomplete variables are strongly correlated or much is
missing. Run
more iterations, for example ``maxit=20`` or ``50``, and check the trace plot
whenever you are unsure; each ``impute`` call starts the chains anew.

What else can help
------------------

- Remove variables that duplicate others, or derived variables such as sums
  and ratios of incomplete variables; they slow convergence.
- Reduce the predictors of each target, for example with
  ``imputation.quickpred`` (see :doc:`predictor_matrices`).
- Check ``mice.logged_events`` for predictors the setup step removed.

The trace plot checks the algorithm, not the imputation model. Compare the
imputed with the observed values as well; see :doc:`diagnostic_plots`.
