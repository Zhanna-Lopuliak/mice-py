Imputation Methods
==================

mice-py provides five different imputation methods. This guide helps you choose and 
configure the right method for your data.

Overview of Methods
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Method
     - Best For
     - Data Type
     - Preserves
     - Complexity
   * - PMM
     - General purpose
     - Numeric
     - Distribution
     - Low
   * - CART
     - Non-linear
     - Both
     - Interactions
     - Medium
   * - Random Forest
     - Complex patterns
     - Both
     - Interactions
     - High
   * - MIDAS
     - Small samples
     - Numeric
     - Local patterns
     - Low
   * - Sample
     - Quick & simple
     - Both
     - Observed values
     - Very Low

PMM: Predictive Mean Matching
------------------------------

**When to use**: Default choice for numeric data, especially when preserving the 
original distribution is important.

How It Works
~~~~~~~~~~~~

1. Fit a Bayesian linear regression on observed values
2. Generate predictions for both observed and missing values
3. For each missing value, find the *k* closest observed values (donors) based on 
   predicted values
4. Randomly select one donor and use its observed value as the imputed value

**Key advantage**: Imputed values are always from the observed data, so impossible 
values cannot be generated.

Usage
~~~~~

.. code-block:: python

   # Basic usage
   mice.impute(method='pmm')
   
   # With custom parameters
   mice.impute(
       method='pmm',
       pmm_donors=5,         # Number of donor candidates (default: 5)
       pmm_matchtype=1,      # Matching type (0, 1, or 2)
       pmm_ridge=1e-5        # Ridge regularization parameter
   )

Parameters
~~~~~~~~~~

**donors** (default: 5)
   Number of closest donors to consider. Larger values increase variability; smaller 
   values make imputations more deterministic.

**matchtype** (default: 1)
   - 0: Match predicted values (no randomness)
   - 1: Match using drawn parameter values (default, adds uncertainty)
   - 2: Maximum randomness

**ridge** (default: 1e-5)
   Regularization to stabilize estimation with collinear predictors.

When PMM Works Best
~~~~~~~~~~~~~~~~~~~

✓ Numeric data with moderate to large sample size
✓ Preserving distribution properties is important
✓ Data has outliers that should be preserved
✓ MAR mechanism with linear relationships

Limitations
~~~~~~~~~~~

✗ Only generates values already in the data
✗ Assumes approximately linear relationships
✗ May struggle with highly skewed data
✗ Not suitable for categorical variables

CART: Classification and Regression Trees
------------------------------------------

**When to use**: Data with non-linear relationships or interactions between variables.

How It Works
~~~~~~~~~~~~

1. Build a decision tree using complete observations
2. For classification (categorical): predict class probabilities
3. For regression (numeric): predict values
4. Add appropriate random variation to predictions

**Key advantage**: Automatically captures interactions and non-linear patterns without 
needing to specify them.

Usage
~~~~~

.. code-block:: python

   # Basic usage
   mice.impute(method='cart')
   
   # With custom parameters
   mice.impute(
       method='cart',
       cart_max_depth=None,        # Maximum tree depth
       cart_min_samples_split=2,   # Min samples to split
       cart_min_samples_leaf=1     # Min samples in leaf
   )

Parameters
~~~~~~~~~~

**max_depth** (default: None)
   Maximum depth of the tree. ``None`` allows unlimited depth. Use smaller values 
   (e.g., 10-20) to prevent overfitting.

**min_samples_split** (default: 2)
   Minimum samples required to split an internal node.

**min_samples_leaf** (default: 1)
   Minimum samples required at a leaf node.

When CART Works Best
~~~~~~~~~~~~~~~~~~~~

✓ Non-linear relationships
✓ Interaction effects between variables
✓ Mixed data types (numeric and categorical)
✓ Robust to outliers
✓ Categorical variables with many levels

Limitations
~~~~~~~~~~~

✗ Can overfit with small samples
✗ May not preserve distribution as well as PMM
✗ Less stable than other methods (high variance)

Random Forest
-------------

**When to use**: Complex data with many interactions and non-linear relationships.

How It Works
~~~~~~~~~~~~

1. Build an ensemble of decision trees using bootstrap samples
2. Each tree uses a random subset of predictors
3. Average predictions across all trees
4. Add random variation appropriate for the data type

**Key advantage**: More stable and accurate than CART, especially with complex patterns.

Usage
~~~~~

.. code-block:: python

   # Basic usage
   mice.impute(method='rf')
   
   # With custom parameters
   mice.impute(
       method='rf',
       rf_n_estimators=100,     # Number of trees
       rf_max_depth=None,       # Maximum depth
       rf_min_samples_split=2,  # Min samples to split
       rf_max_features='sqrt'   # Features per split
   )

Parameters
~~~~~~~~~~

**n_estimators** (default: 100)
   Number of trees in the forest. More trees = more stable but slower.

**max_depth** (default: None)
   Maximum depth of each tree.

**max_features** (default: 'sqrt')
   Number of features to consider for each split. Options: 'sqrt', 'log2', or an integer.

When Random Forest Works Best
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ Complex, non-linear relationships
✓ Many interaction effects
✓ Large datasets
✓ High-dimensional data
✓ Mixed data types
✓ When accuracy is more important than interpretability

Limitations
~~~~~~~~~~~

✗ Computationally expensive
✗ Slower than other methods
✗ Less interpretable than simpler methods
✗ May not preserve marginal distributions as well as PMM

MIDAS: Multiple Imputation with Distant Average Substitution
-------------------------------------------------------------

**When to use**: Numeric data, especially with small samples or skewed distributions.

How It Works
~~~~~~~~~~~~

1. For each missing value, identify nearby observed values using distance metrics
2. Use a weighted average of distant donors (farther donors get less weight)
3. Add random variation

**Key advantage**: Often performs well with small samples and skewed distributions 
where PMM struggles.

Usage
~~~~~

.. code-block:: python

   # Basic usage
   mice.impute(method='midas')
   
   # With custom parameters
   mice.impute(
       method='midas',
       midas_donors=5,      # Number of donors
       midas_ridge=1e-5     # Ridge parameter
   )

When MIDAS Works Best
~~~~~~~~~~~~~~~~~~~~~

✓ Small sample sizes
✓ Skewed distributions
✓ Numeric data
✓ When PMM struggles with distribution

Limitations
~~~~~~~~~~~

✗ Only for numeric variables
✗ Less commonly used (less validated than PMM/CART/RF)
✗ May require parameter tuning

Sample: Random Sampling
-----------------------

**When to use**: Quick imputations, initial values, or when other methods aren't suitable.

How It Works
~~~~~~~~~~~~

Simply draws random values from the observed values of each variable.

**Key advantage**: Very fast, simple, preserves observed distribution exactly.

Usage
~~~~~

.. code-block:: python

   mice.impute(method='sample')

When Sample Works Best
~~~~~~~~~~~~~~~~~~~~~~

✓ Initial imputation (before MICE iterations)
✓ Categorical variables with many levels
✓ Quick exploratory analysis
✓ When no predictive relationship exists

Limitations
~~~~~~~~~~~

✗ Ignores relationships between variables
✗ No predictive component
✗ Only useful for simple cases or initialization

Choosing a Method
-----------------

Decision Tree
~~~~~~~~~~~~~

.. code-block:: text

   Is your data numeric or categorical?
   │
   ├── Mostly numeric
   │   │
   │   ├── Linear relationships? → PMM
   │   │
   │   ├── Non-linear? → CART or RF
   │   │
   │   └── Small sample or skewed? → MIDAS or PMM
   │
   └── Mixed or mostly categorical
       │
       ├── Simple relationships? → CART
       │
       └── Complex interactions? → RF

General Guidelines
~~~~~~~~~~~~~~~~~~

**Start with PMM**
   It's the most well-studied method and works well in most cases.

**Use CART for interactions**
   If you know or suspect important interactions between variables.

**Use RF for complexity**
   When you have complex patterns and computational resources.

**Use MIDAS when PMM fails**
   Particularly with small samples or skewed data.

**Use Sample for initialization**
   Or for very simple cases.

Using Different Methods for Different Variables
------------------------------------------------

You can use different methods for different variables:

.. code-block:: python

   method_dict = {
       'age': 'pmm',           # Numeric, approximately normal
       'income': 'midas',      # Numeric, highly skewed
       'education': 'sample',  # Categorical with few levels
       'job_type': 'cart',     # Categorical with many levels
       'health_score': 'rf'    # Numeric with complex patterns
   }
   
   mice.impute(n_imputations=5, method=method_dict)

Comparing Methods
-----------------

To compare methods empirically:

.. code-block:: python

   from plotting.diagnostics import densityplot, stripplot
   
   # Try PMM
   mice_pmm = MICE(df)
   mice_pmm.impute(method='pmm')
   
   # Try CART
   mice_cart = MICE(df)
   mice_cart.impute(method='cart')
   
   # Compare distributions
   missing_pattern = df.notna().astype(int)
   densityplot(mice_pmm.imputed_datasets, missing_pattern, 
               save_path='pmm_density.png')
   densityplot(mice_cart.imputed_datasets, missing_pattern,
               save_path='cart_density.png')

Look for:
   - How well imputed values match observed distribution
   - Whether extreme values are reasonable
   - Smooth transitions between observed and imputed

Research Findings
-----------------

Based on simulation studies in the thesis:

- **PMM** performs reliably under MCAR and mild MAR with symmetric distributions
- **MIDAS** consistently matches or outperforms PMM with skewness or small samples
- **CART/RF** handle non-linear relationships effectively but may not preserve 
  marginal distributions as well
- Method choice should consider data characteristics, missingness patterns, and sample size

Next Steps
----------

- Learn about :doc:`predictor_matrices` to control which variables predict which
- Check :doc:`convergence_diagnostics` after imputation
- See practical examples in :doc:`../examples/index`

