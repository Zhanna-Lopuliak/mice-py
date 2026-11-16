Best Practices
==============

A checklist for an imputation with mice-py. Each point links to the page
with the details. Most points follow van Buuren (2018) and White, Royston and
Wood (2011).

Before imputing
---------------

- Look at the missing-data pattern and the share missing per variable
  (:doc:`understanding_missing_data`).
- Think about the mechanism: which observed variables could explain the
  missingness? Include them in the data.
- Drop identifiers and columns that should not predict anything; convert
  dates to numbers; give categorical variables a categorical dtype, and
  declare ordered ones as ordered (``pd.Categorical(..., ordered=True)``),
  so that they get R's default method for their type. A text column with
  more than 50 distinct values, such as names, cannot be imputed with its
  default ``polyreg``.
- Do not impute derived variables (sums, ratios, transformations,
  interactions) as columns of their own: a derived column and its sources
  would be imputed inconsistently, or the setup step removes the derived
  column as collinear. Compute them **after** imputation from the imputed
  variables, or, if they belong in the imputation model, make them passive
  (``passive``, :doc:`controlling_imputation`).
- Rows that must not influence the imputation models (a test set, for
  example) can still be imputed: leave them out of the models with
  ``ignore`` (:doc:`controlling_imputation`).

Imputing
--------

- Start with the defaults, which are R's: the method by column type (pmm
  for numeric, logreg, polyreg or polr for categorical columns), full
  predictor matrix, ``maxit=5`` (:doc:`mice_overview`).
- Pass a ``seed`` so that the analysis can be reproduced.
- Read the warnings. "Number of logged events" means a variable was removed
  from the model; see ``mice.logged_events``. "Type mismatch" means a
  categorical method imputes a numeric column. "Observed values replaced by
  missing values" means that ``where`` marks observed cells that could not
  be imputed; see ``mice.where``.
- Choose methods per column where the default is not suitable
  (:doc:`imputation_methods`); midas and the norm methods do not take
  categorical targets. Avoid the improper methods ``norm.nob``,
  ``norm.predict`` and ``mean`` for analyses.
- With many variables, restrict the predictors (:doc:`predictor_matrices`).

Checking
--------

- Check convergence with the trace plot; run more iterations with
  ``continue_chains`` if the chains trend or do not mix
  (:doc:`convergence_diagnostics`).
- Compare imputed and observed values with the strip, box, density and
  scatter plots; imputed values should be plausible
  (:doc:`diagnostic_plots`). Where they are not (for example outside the
  possible range), bound them with ``post``
  (:doc:`controlling_imputation`).

Analyzing
---------

- Fit the analysis model to every completed dataset and pool with Rubin's
  rules (``fit`` and ``pool``); never analyze a single completed dataset or
  the average of the datasets (:doc:`pooling_analysis`). Fit again after
  every ``impute`` or ``continue_chains``, which remove earlier fits.
- Report the number of imputations, the methods, and the fraction of missing
  information of the main estimates.
- Use more imputations when standard errors and p-values matter and much is
  missing; at least as many as the percentage of incomplete cases is a
  common rule.
