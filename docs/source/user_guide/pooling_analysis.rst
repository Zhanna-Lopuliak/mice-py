Pooling Analysis
================

After creating multiple imputed datasets, you need to analyze them and combine the 
results. This guide explains how to pool results using Rubin's rules.

Why Pool Results?
-----------------

You have multiple complete datasets (e.g., 5 or 10), each with different imputed 
values. To get final estimates, you must:

1. **Fit your model** on each imputed dataset
2. **Pool the results** to get single estimates
3. **Account for uncertainty** from both within and between imputations

Simply averaging or using one dataset would underestimate uncertainty and produce 
invalid inferences.

Rubin's Rules
-------------

Rubin's rules provide a principled way to combine estimates from multiple imputed 
datasets, properly accounting for the uncertainty introduced by imputation.

Basic Concept
~~~~~~~~~~~~~

For each parameter (e.g., regression coefficient):

1. Fit the model on each imputed dataset → get *m* estimates and standard errors
2. Calculate **within-imputation variance** (average of squared SEs)
3. Calculate **between-imputation variance** (variance of estimates across imputations)
4. **Total variance** = within + between + correction term
5. Use these to construct confidence intervals and perform inference

Using mice-py for Pooling
--------------------------

Simple Workflow
~~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE
   
   # 1. Perform imputation
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10)
   
   # 2. Fit a model using formula syntax
   mice.fit('outcome ~ predictor1 + predictor2 + predictor3')
   
   # 3. Pool results
   pooled = mice.pool(summ=True)
   print(pooled)

The ``fit()`` method uses formula syntax (like R or statsmodels) and fits the model 
on all imputed datasets.

The ``pool()`` method combines results using Rubin's rules.

Understanding the Output
-------------------------

The pooled results include:

**Estimate**
   The pooled coefficient (average across imputed datasets)

**Std.Error**
   The pooled standard error (accounting for both within and between variance)

**t-statistic**
   The test statistic for the coefficient

**df**
   Degrees of freedom (adjusted for imputation)

**p-value**
   Statistical significance

**95% CI Lower/Upper**
   Confidence interval bounds

**FMI**
   Fraction of Missing Information (see below)

Example Output
~~~~~~~~~~~~~~

.. code-block:: text

                    Estimate  Std.Error  t-statistic     df   P>|t|  [0.025  0.975]    FMI
   Intercept         45.234      5.321        8.502  42.15  <0.001  34.449  56.019  0.156
   predictor1         0.823      0.142        5.796  38.27  <0.001   0.535   1.111  0.198
   predictor2        -1.234      0.387       -3.189  51.83   0.002  -2.012  -0.456  0.089
   predictor3         2.156      0.921        2.341  45.67   0.024   0.301   4.011  0.132

Fraction of Missing Information (FMI)
--------------------------------------

FMI indicates how much the uncertainty in your estimate is due to missing data:

- **FMI = 0**: No missing information, equivalent to complete data analysis
- **FMI = 0.1**: 10% of uncertainty is due to missingness
- **FMI = 0.5**: Half the uncertainty is from missingness
- **FMI = 1**: Complete uncertainty from missingness (rare)

**Interpretation**:
   - Low FMI (<0.1): Missingness has little impact
   - Moderate FMI (0.1-0.3): Some impact, multiple imputation important
   - High FMI (>0.3): Substantial impact, consider implications

**Rule of thumb**: Higher FMI suggests you need more imputations (m) for stable results.

Formula Syntax
--------------

The ``fit()`` method uses Patsy formula syntax:

Basic Formulas
~~~~~~~~~~~~~~

.. code-block:: python

   # Simple linear regression
   mice.fit('y ~ x')
   
   # Multiple predictors
   mice.fit('y ~ x1 + x2 + x3')
   
   # With interaction
   mice.fit('y ~ x1 + x2 + x1:x2')
   
   # Or equivalently
   mice.fit('y ~ x1 * x2')  # Includes x1, x2, and x1:x2
   
   # Polynomial terms
   mice.fit('y ~ x + I(x**2)')
   
   # No intercept
   mice.fit('y ~ x - 1')

Categorical Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Categorical predictor (automatically creates dummies)
   mice.fit('income ~ age + C(education)')
   
   # Change reference category
   mice.fit('income ~ age + C(education, Treatment("High School"))')

Transformations
~~~~~~~~~~~~~~~

.. code-block:: python

   # Log transformation
   mice.fit('log_y ~ x1 + x2')
   
   # Use numpy functions
   mice.fit('y ~ np.log(x1) + np.sqrt(x2)')

Advanced Pooling
----------------

Pool Without Summary
~~~~~~~~~~~~~~~~~~~~

Get detailed results for each imputation:

.. code-block:: python

   # Get individual results and pooled results
   pooled_detailed = mice.pool(summ=False)
   
   # Access individual imputation results
   individual_results = pooled_detailed['individual']
   
   # Access pooled results
   pooled_results = pooled_detailed['pooled']

Custom Analysis
~~~~~~~~~~~~~~~

For models not supported by ``fit()``, manually fit and pool:

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LogisticRegression
   
   # Fit on each imputed dataset
   coefficients = []
   std_errors = []
   
   for dataset in mice.imputed_datasets:
       X = dataset[['predictor1', 'predictor2']]
       y = dataset['outcome']
       
       model = LogisticRegression()
       model.fit(X, y)
       
       coefficients.append(model.coef_[0])
       # Calculate std errors (simplified)
       # In practice, use proper methods for your model
   
   # Pool manually using Rubin's rules
   from imputation.pooling import pool_estimates
   pooled = pool_estimates(coefficients, std_errors)

Interpreting Pooled Results
----------------------------

Statistical Significance
~~~~~~~~~~~~~~~~~~~~~~~~

Use the pooled p-values and confidence intervals for inference:

.. code-block:: python

   pooled = mice.pool(summ=True)
   
   # Check significance
   significant = pooled[pooled['P>|t|'] < 0.05]
   print("Significant predictors:")
   print(significant)

The pooled standard errors are larger than those from a single dataset (accounting 
for imputation uncertainty), so some predictors significant in a single imputation 
might not be significant when properly pooled.

Effect Sizes
~~~~~~~~~~~~

The pooled estimates are your best point estimates:

.. code-block:: python

   # Extract coefficient for predictor1
   coef = pooled.loc['predictor1', 'Estimate']
   ci_lower = pooled.loc['predictor1', '[0.025']
   ci_upper = pooled.loc['predictor1', '0.975]']
   
   print(f"predictor1: {coef:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

Model Comparison
~~~~~~~~~~~~~~~~

When comparing models, use pooled results:

.. code-block:: python

   # Fit two models
   mice.fit('y ~ x1')
   results_simple = mice.pool(summ=True)
   
   mice.fit('y ~ x1 + x2 + x3')
   results_complex = mice.pool(summ=True)
   
   # Compare based on pooled coefficients and FMI

How Many Imputations?
---------------------

General Guidelines
~~~~~~~~~~~~~~~~~~

**Minimum**: 5 imputations
   Acceptable for low missingness (<10%)

**Recommended**: 10-20 imputations
   Good balance between computation and precision

**High missingness**: 20-100 imputations
   When missingness >30% or FMI >0.3

**Rule of thumb**: Number of imputations ≈ percentage of missing cases

Von Hippel (2020) suggests: m = # of missing cases / # of complete cases × 100

Checking If You Have Enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If FMI is high (>0.3) and results are unstable across repeated analyses, you may 
need more imputations:

.. code-block:: python

   # Check FMI
   pooled = mice.pool(summ=True)
   max_fmi = pooled['FMI'].max()
   
   if max_fmi > 0.3:
       print(f"High FMI ({max_fmi:.2f}). Consider more imputations.")

Common Pitfalls
---------------

Don't Use Single Imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

❌ **Wrong**:

.. code-block:: python

   # Using only the first imputed dataset
   dataset = mice.imputed_datasets[0]
   model = smf.ols('y ~ x1 + x2', data=dataset).fit()
   print(model.summary())

✓ **Correct**:

.. code-block:: python

   # Fit on all and pool
   mice.fit('y ~ x1 + x2')
   pooled = mice.pool(summ=True)
   print(pooled)

Don't Average Imputed Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

❌ **Wrong**:

.. code-block:: python

   # Averaging imputed datasets
   averaged = pd.concat(mice.imputed_datasets).groupby(level=0).mean()
   model = smf.ols('y ~ x1 + x2', data=averaged).fit()

This is actually single imputation and underestimates uncertainty!

✓ **Correct**: Use proper pooling with Rubin's rules

Don't Ignore Imputation Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard errors from a single imputed dataset are too small. Always pool!

Reporting Results
-----------------

When publishing, report:

1. **Number of imputations** (m)
2. **Number of iterations**
3. **Imputation method(s)** used
4. **Pooled estimates** with standard errors or confidence intervals
5. **FMI** for key parameters
6. **Convergence** assessment

Example Text
~~~~~~~~~~~~

.. code-block:: text

   Missing data were handled using multiple imputation by chained equations 
   (MICE) with m=20 imputations. Variables were imputed using predictive mean 
   matching. The algorithm ran for 20 iterations and convergence was confirmed 
   by visual inspection of trace plots. Results were pooled using Rubin's rules. 
   The fraction of missing information ranged from 0.08 to 0.25 across parameters.

Example Results Table
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Table 1: Pooled regression results (n=500, m=10 imputations)
   
   Variable       Estimate    SE      95% CI            p      FMI
   ─────────────────────────────────────────────────────────────
   Intercept      45.23      5.32   [34.45, 56.02]  <0.001  0.16
   Age             0.82      0.14   [ 0.54,  1.11]  <0.001  0.20
   Gender(F)      -1.23      0.39   [-2.01, -0.46]   0.002  0.09
   Education       2.16      0.92   [ 0.30,  4.01]   0.024  0.13

Diagnostic Statistics
~~~~~~~~~~~~~~~~~~~~~

After pooling, check:

.. code-block:: python

   pooled = mice.pool(summ=True)
   
   # Summary statistics
   print(f"Mean FMI: {pooled['FMI'].mean():.3f}")
   print(f"Max FMI: {pooled['FMI'].max():.3f}")
   print(f"Mean df: {pooled['df'].mean():.1f}")

Tips for Better Pooling
------------------------

1. **More imputations**: When in doubt, use more (20-50)
2. **Check FMI**: High values suggest need for more imputations
3. **Complete convergence**: Ensure MICE converged before pooling
4. **Include all relevant variables**: In both imputation and analysis models
5. **Be cautious with transformations**: Pool on the analysis scale
6. **Report thoroughly**: Include all relevant details in your methods

Next Steps
----------

- Read :doc:`best_practices` for overall guidance
- Review :doc:`../theory/rubins_rules` for mathematical details
- See complete examples in :doc:`../examples/index`

