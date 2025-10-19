Method Details
==============

This page provides brief technical overviews of the imputation methods available 
in mice-py.

Overview
--------

mice-py implements five imputation methods:

1. **PMM** - Predictive Mean Matching
2. **CART** - Classification and Regression Trees  
3. **Random Forest** - Ensemble tree method
4. **MIDAS** - Multiple Imputation with Distance Aided Substitution
5. **Sample** - Simple random sampling

PMM: Predictive Mean Matching
------------------------------

Algorithm
~~~~~~~~~

For each missing value in variable :math:`Y`:

1. **Fit model**: Bayesian linear regression on observed cases:

   .. math::

      Y_{obs} = X_{obs}\beta + \epsilon

   where :math:`\epsilon \sim N(0, \sigma^2)`

2. **Draw parameters**: Sample :math:`\beta^*` and :math:`\sigma^{*2}` from 
   posterior distribution

3. **Predict**: Calculate fitted values for observed and missing cases:

   .. math::

      \hat{Y}_{obs} &= X_{obs}\beta^* \\
      \hat{Y}_{mis} &= X_{mis}\beta^*

4. **Match**: For each missing case, find *k* observed cases with closest fitted values

5. **Impute**: Randomly select one of the *k* donors and use its observed value

Key Features
~~~~~~~~~~~~

- **Preserves distribution**: Imputed values come from observed values
- **Handles non-normality**: No parametric assumptions about :math:`Y`
- **Prevents impossible values**: Can't generate values outside observed range
- **Includes uncertainty**: Via Bayesian parameter draws and random donor selection

Parameters
~~~~~~~~~~

- **donors** (k): Number of candidate donors (typical: 5-10)
- **matchtype**: How to match (0=deterministic, 1=default, 2=maximum randomness)
- **ridge**: Regularization parameter for stability

When to Use
~~~~~~~~~~~

✓ Numeric continuous variables
✓ Preserving distribution important
✓ Data may have outliers
✓ Relationships approximately linear

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

The Bayesian bootstrap draws:

.. math::

   \beta^* &\sim N(\hat{\beta}, V_\beta) \\
   \sigma^{*2} &\sim \text{Inv-}\chi^2(n-p, \hat{\sigma}^2)

where :math:`\hat{\beta}` and :math:`\hat{\sigma}^2` are ML estimates, and 
:math:`V_\beta` is the covariance matrix (with ridge regularization).

Distance metric:

.. math::

   d(i,j) = |\hat{Y}_i - \hat{Y}_j|

The *k* donors are the observed cases with smallest distances to the missing case.

CART: Classification and Regression Trees
------------------------------------------

Algorithm
~~~~~~~~~

For each variable with missing data:

1. **Build tree**: Fit decision tree on complete observations
2. **Predict**: Use tree to predict missing values
3. **Add noise**: Add appropriate random variation:
   
   - Regression (numeric): :math:`\hat{Y} + N(0, \hat{\sigma}^2)`
   - Classification (categorical): Sample from predicted probabilities

Key Features
~~~~~~~~~~~~

- **Non-parametric**: No distributional assumptions
- **Interactions**: Automatically captures interaction effects
- **Non-linear**: Handles complex non-linear relationships
- **Mixed types**: Works for both numeric and categorical

Parameters
~~~~~~~~~~

- **max_depth**: Maximum tree depth (controls complexity)
- **min_samples_split**: Minimum samples to split node
- **min_samples_leaf**: Minimum samples in leaf node

When to Use
~~~~~~~~~~~

✓ Non-linear relationships
✓ Known or suspected interactions
✓ Categorical variables
✓ Robustness to outliers desired

Tree Structure
~~~~~~~~~~~~~~

A regression tree partitions the predictor space:

.. math::

   \hat{Y}(x) = \sum_{m=1}^{M} c_m \cdot I(x \in R_m)

where :math:`R_m` are regions, :math:`c_m` are constants (means), and :math:`M` 
is the number of terminal nodes.

For classification trees, :math:`c_m` represents class probabilities.

Random Forest
-------------

Algorithm
~~~~~~~~~

For each variable with missing data:

1. **Bootstrap**: Draw B bootstrap samples
2. **Build trees**: Fit tree on each sample, using random predictor subset at each split
3. **Predict**: Average predictions across all trees
4. **Add noise**: Add random variation to predictions

Key Features
~~~~~~~~~~~~

- **Ensemble method**: More stable than single tree
- **Handles complexity**: Can capture very complex patterns
- **Variable importance**: Can assess predictor importance
- **Reduced overfitting**: Compared to single CART

Parameters
~~~~~~~~~~

- **n_estimators**: Number of trees (typical: 100)
- **max_features**: Predictors per split (typical: sqrt(p))
- **max_depth**: Maximum tree depth
- **min_samples_split**: Minimum samples to split

When to Use
~~~~~~~~~~~

✓ Complex, high-dimensional data
✓ Many interactions
✓ High accuracy important
✓ Computational resources available

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

For regression:

.. math::

   \hat{Y}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)

where :math:`T_b` is the b-th tree.

Random variation added:

.. math::

   Y^{imp} \sim N(\hat{Y}, \hat{\sigma}^2)

where :math:`\hat{\sigma}^2` is estimated from residuals.

MIDAS: Distance Aided Substitution
-----------------------------------

Algorithm
~~~~~~~~~

For each missing value in variable :math:`Y`:

1. **Standardize**: Standardize predictors
2. **Distance**: Calculate distances to all observed cases
3. **Weight**: Weight observed cases by inverse distance
4. **Select**: Choose *k* donors with highest weights
5. **Impute**: Weighted average of donor values plus noise

Key Features
~~~~~~~~~~~~

- **Distance-based**: Uses multivariate distance in predictor space
- **Weighted**: Distant donors contribute less
- **Continuous**: Works for numeric variables
- **Local**: Uses local structure of data

Parameters
~~~~~~~~~~

- **donors** (k): Number of donors
- **distance_metric**: How to calculate distances

When to Use
~~~~~~~~~~~

✓ Numeric variables
✓ Small sample sizes
✓ Skewed distributions
✓ When PMM struggles

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

Mahalanobis distance:

.. math::

   d(i,j) = \sqrt{(X_i - X_j)^T \Sigma^{-1} (X_i - X_j)}

where :math:`\Sigma` is the covariance matrix of predictors.

Imputed value:

.. math::

   Y^{imp} = \sum_{j \in \text{donors}} w_j Y_j + \epsilon

where weights :math:`w_j \propto 1/d(i,j)` and :math:`\epsilon \sim N(0, \sigma^2)`.

Sample: Random Sampling
-----------------------

Algorithm
~~~~~~~~~

For each missing value:

1. **Observed pool**: Get all observed values of the variable
2. **Random draw**: Randomly sample one observed value
3. **Impute**: Use sampled value

Key Features
~~~~~~~~~~~~

- **Simplest method**: No model fitting
- **Preserves distribution**: Matches observed distribution exactly
- **No relationships**: Ignores relationships with other variables
- **Fast**: Very computationally efficient

When to Use
~~~~~~~~~~~

✓ Initial imputation (before MICE iterations)
✓ Categorical variables with many levels
✓ No predictive relationships
✓ Quick exploratory analysis

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

If :math:`Y_{obs} = \{y_1, y_2, ..., y_n\}`, then:

.. math::

   Y^{imp} \sim \text{Uniform}(Y_{obs})

Each observed value has equal probability :math:`1/n` of being selected.

Comparing Methods
-----------------

Distributional Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Method
     - Preserves Mean
     - Preserves Variance
     - Preserves Distribution
   * - PMM
     - Yes
     - Yes
     - Yes
   * - CART
     - Yes
     - Approximate
     - Approximate
   * - RF
     - Yes
     - Approximate
     - Approximate
   * - MIDAS
     - Yes
     - Yes
     - Yes
   * - Sample
     - Yes
     - Yes
     - Yes

Relationship Modeling
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Method
     - Linear
     - Non-linear
     - Interactions
   * - PMM
     - Excellent
     - Poor
     - No (unless specified)
   * - CART
     - Good
     - Excellent
     - Automatic
   * - RF
     - Good
     - Excellent
     - Automatic
   * - MIDAS
     - Good
     - Moderate
     - No
   * - Sample
     - None
     - None
     - None

Computational Cost
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Method
     - Speed
     - Memory
   * - PMM
     - Fast
     - Low
   * - CART
     - Fast
     - Low
   * - RF
     - Slow
     - High
   * - MIDAS
     - Fast
     - Low
   * - Sample
     - Very fast
     - Very low

Adding Random Variation
-----------------------

Why Add Noise?
~~~~~~~~~~~~~~

Without random variation, imputations would be:

- Too certain (underestimate uncertainty)
- Identical across imputations (no between-imputation variance)
- Invalid for inference

All methods except Sample explicitly add appropriate random variation.

How Much Noise?
~~~~~~~~~~~~~~~

The variance of added noise should reflect:

1. **Model uncertainty**: Uncertainty in parameter estimates
2. **Prediction uncertainty**: Residual variance

For linear models:

.. math::

   \text{Var}(Y^{imp}) = X^T V_\beta X + \sigma^2

For tree-based methods, typically:

.. math::

   \text{Var}(Y^{imp}) = \hat{\sigma}^2

where :math:`\hat{\sigma}^2` is estimated from residuals.

Method Selection Guidelines
----------------------------

Decision Flow
~~~~~~~~~~~~~

1. **Data type**: Numeric or categorical?
   
   - Categorical with many levels → CART or Sample
   - Numeric → Continue

2. **Relationship type**: Linear or non-linear?
   
   - Linear, approximately normal → PMM
   - Non-linear or interactions → CART or RF

3. **Sample size and computational resources**:
   
   - Small sample or limited time → PMM or MIDAS
   - Large sample and resources available → RF

4. **Distribution characteristics**:
   
   - Skewed or small sample → MIDAS
   - Outliers to preserve → PMM

Combinations
~~~~~~~~~~~~

You can use different methods for different variables:

.. code-block:: python

   method_dict = {
       'age': 'pmm',        # Numeric, normal
       'income': 'midas',   # Numeric, skewed
       'education': 'cart', # Ordered categorical
       'city': 'sample'     # Categorical, many levels
   }

Research Findings
-----------------

Based on simulation studies:

- **PMM** performs reliably under MCAR and mild MAR with symmetric distributions
- **MIDAS** matches or outperforms PMM with skewness or small samples
- **CART/RF** handle non-linear relationships and interactions effectively
- **Choice matters** more when relationships are complex or distributions non-normal

See Also
--------

- :doc:`../user_guide/imputation_methods` for practical usage
- :doc:`rubins_rules` for pooling methodology
- :doc:`../references` for detailed research papers

