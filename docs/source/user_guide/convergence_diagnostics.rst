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
numeric column, the mean and the variance of its imputations (over all its
where-cells, see :doc:`controlling_imputation`) at the end of every
iteration, per chain, as arrays of shape ``(iteration, n_imputations)``.
They are R's ``chainMean`` and ``chainVar``. Categorical columns have no
chain statistics. ``mice.iteration`` is the number of iterations the chains
have run (R's ``imp$iteration``).

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
missing. Run more iterations, for example 20 or 50, and check the trace plot
whenever you are unsure.

More iterations of the same chains
----------------------------------

``continue_chains(maxit)`` runs ``maxit`` more iterations of the chains of
the last ``impute``, as R's ``mice.mids()``: each chain goes on from its
current imputations, with its own random generator where it stopped, and
the methods, the predictor matrix, the visit sequence, ``where``,
``ignore``, ``post`` and ``passive`` stay those of the run. ``chain_mean``
and ``chain_var`` get one row per new iteration, so the trace plot shows the
whole run:

.. code-block:: python

   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=5, seed=1)
   plot_chain_stats(mice.chain_mean, mice.chain_var, save_path="trace_5.png")

   mice.continue_chains(15)                  # not converged yet? 15 more
   print(mice.iteration, mice.chain_mean["chl"].shape)   # 20 (20, 5)
   plot_chain_stats(mice.chain_mean, mice.chain_var, save_path="trace_20.png")

The result is the same as one run with all iterations:
``impute(maxit=5, seed=1)`` followed by ``continue_chains(15)`` gives the
completed datasets, chain statistics and logged events of
``impute(maxit=20, seed=1)``. ``continue_chains`` returns the new completed
datasets and replaces ``mice.imputed_datasets``; fits of the earlier
datasets are removed, so call ``fit`` again (see :doc:`pooling_analysis`).

``impute(maxit=0)`` runs no iteration, as in R: the completed datasets hold
the start of each chain (random draws from the observed values), and
``chain_mean`` and ``chain_var`` have no rows yet. ``continue_chains``
goes on from there.

.. code-block:: python

   start = MICE(df)
   start.impute(maxit=0, seed=1)
   print(start.iteration, start.chain_mean["chl"].shape)   # 0 (0, 5)
   start.continue_chains(5)

``continue_chains`` raises ``ValueError`` before ``impute``, and after an
``impute`` or ``continue_chains`` that stopped with an error: call
``impute`` again then.

What else can help
------------------

- Remove variables that duplicate others. Derived variables such as sums
  and ratios of incomplete variables slow convergence when they are imputed
  as variables of their own; make them passive instead (see
  :doc:`controlling_imputation`), or compute them after the imputation.
- Reduce the predictors of each target, for example with
  ``imputation.quickpred`` (see :doc:`predictor_matrices`).
- Check ``mice.logged_events`` for predictors the setup step removed.

The trace plot checks the algorithm, not the imputation model. Compare the
imputed with the observed values as well; see :doc:`diagnostic_plots`.
