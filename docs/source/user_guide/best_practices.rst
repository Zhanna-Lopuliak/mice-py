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
  dates to numbers; give categorical variables a categorical dtype.
- Create derived variables (sums, ratios, transformations) **after**
  imputation from the imputed variables, rather than imputing them as
  separate columns: a derived column and its sources would be imputed
  inconsistently, or the setup step removes the derived column as collinear.

Imputing
--------

- Start with the defaults, which are R's: pmm, full predictor matrix,
  ``maxit=5`` (:doc:`mice_overview`).
- Pass a ``seed`` so that the analysis can be reproduced.
- Read the warnings. "Number of logged events" means a variable was removed
  from the model; see ``mice.logged_events``.
- Choose methods per column where pmm is not suitable
  (:doc:`imputation_methods`); midas does not take categorical targets.
- With many variables, restrict the predictors (:doc:`predictor_matrices`).

Checking
--------

- Check convergence with the trace plot; increase ``maxit`` if the chains
  trend or do not mix (:doc:`convergence_diagnostics`).
- Compare imputed and observed values with the strip, box, density and
  scatter plots; imputed values should be plausible
  (:doc:`diagnostic_plots`).

Analyzing
---------

- Fit the analysis model to every completed dataset and pool with Rubin's
  rules (``fit`` and ``pool``); never analyze a single completed dataset or
  the average of the datasets (:doc:`pooling_analysis`).
- Report the number of imputations, the methods, and the fraction of missing
  information of the main estimates.
- Use more imputations when standard errors and p-values matter and much is
  missing; at least as many as the percentage of incomplete cases is a
  common rule.
