Method Details
==============

Brief technical overview of the five imputation methods in mice-py.

PMM: Predictive Mean Matching
------------------------------

**Algorithm**:

1. Fit Bayesian linear regression on observed cases
2. Draw parameters from posterior
3. Generate predictions for observed and missing cases
4. For each missing value, find k nearest observed cases (by predicted value)
5. Randomly select one donor and use its observed value

**Key feature**: Imputed values come from observed data (prevents impossible values).

**Best for**: Numeric variables, preserving distributions, data with outliers.

**Parameters**: ``pmm_donors`` (default: 5), ``pmm_matchtype``, ``pmm_ridge``

CART: Classification and Regression Trees
------------------------------------------

**Algorithm**:

1. Build decision tree on complete observations
2. Use tree to predict missing values
3. Add random variation to predictions

**Key feature**: Automatically captures interactions and non-linear patterns.

**Best for**: Non-linear relationships, interactions, categorical variables.

**Parameters**: ``cart_max_depth``, ``cart_min_samples_split``, ``cart_min_samples_leaf``

Random Forest
-------------

**Algorithm**:

1. Build multiple decision trees on bootstrap samples
2. Average predictions across trees
3. Add random variation to predictions

**Key feature**: More stable than single tree, handles complexity well.

**Best for**: Complex patterns, high-dimensional data, many interactions.

**Parameters**: ``rf_n_estimators`` (default: 100), ``rf_max_depth``, ``rf_max_features``

MIDAS: Distance Aided Substitution
-----------------------------------

**Algorithm**:

1. Calculate distances between cases in predictor space
2. Weight observed cases by inverse distance
3. Select k donors with highest weights
4. Use weighted average plus noise

**Key feature**: Uses local structure of data, good for skewed distributions.

**Best for**: Small samples, skewed distributions, when PMM struggles.

**Parameters**: ``midas_donors`` (default: 5), ``midas_ridge``

Sample: Random Sampling
-----------------------

**Algorithm**:

1. Pool all observed values of the variable
2. Randomly sample one value for each missing case

**Key feature**: Simplest method, preserves marginal distribution exactly.

**Best for**: Initial imputation, categorical variables with many levels, quick exploration.

**Parameters**: None

Comparison Summary
------------------

.. list-table::
   :header-rows: 1

   * - Method
     - Best For
     - Speed
   * - PMM
     - General purpose numeric
     - Fast
   * - CART
     - Non-linear, interactions
     - Fast
   * - RF
     - Complex patterns
     - Slow
   * - MIDAS
     - Skewed, small samples
     - Fast
   * - Sample
     - Quick/simple
     - Very fast

See :doc:`../user_guide/imputation_methods` for practical selection guidance.
