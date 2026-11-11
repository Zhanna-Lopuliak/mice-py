Multiple Imputation Theory
==========================

Why multiple imputation?
------------------------

**Complete-case analysis** discards every row with a missing value. It wastes
information and is biased unless the data are MCAR (or, for some analyses,
under specific forms of MAR).

**Single imputation** fills each missing value once and then treats it as
observed. Standard errors are too small and p-values too optimistic, because
the uncertainty about the missing values is ignored.

**Multiple imputation** (Rubin 1987) creates *m* completed datasets whose
imputations differ by a random amount that reflects this uncertainty. The
analysis is run on each, and the *m* results are combined so that the
variance between them enters the standard errors. Under MAR and with a
suitable imputation model, the inference is valid.

The three steps
---------------

1. **Imputation**: create *m* completed datasets (``MICE.impute``).
2. **Analysis**: analyze each dataset separately (``MICE.fit``).
3. **Pooling**: combine the results with Rubin's rules (``MICE.pool``).

Chained equations
-----------------

MICE, also called fully conditional specification (van Buuren and
Groothuis-Oudshoorn 2011), does not need a joint model for all variables.
It specifies one conditional model per incomplete variable and iterates:

1. **Initialize**: fill the missing values with random draws from the
   observed values of each variable.
2. **Iterate**: for each incomplete variable in turn, fit its model on the
   rows where it is observed, using the current values of the other variables,
   and draw new imputations for its missing values.
3. **Repeat** step 2 for ``maxit`` iterations, until the chains are stable.
4. Run steps 1 to 3 *m* times, each with its own random stream, for *m*
   completed datasets.

Assumptions and limits
----------------------

- The data are MAR given the variables in the imputation model.
- The conditional models fit the data well enough; in particular they
  include the relations that the analysis model will look at.
- The chains have converged.
- The conditional models need not be compatible with any joint distribution.
  This is rarely a problem in practice, but it means that convergence is a
  practical check, not a guarantee.

See :doc:`../user_guide/mice_overview` for how mice-py implements the
algorithm.
