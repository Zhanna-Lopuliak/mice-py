Best Practices
==============

This guide provides practical recommendations for using MICE effectively and avoiding 
common mistakes.

Before Imputation
-----------------

Understand Your Data
~~~~~~~~~~~~~~~~~~~~

Before imputing, thoroughly explore your data:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from plotting.utils import md_pattern_like, plot_missing_data_pattern
   
   # 1. Check amount of missingness
   missing_pct = df.isnull().mean() * 100
   print("Missing percentages:")
   print(missing_pct.sort_values(ascending=False))
   
   # 2. Visualize missing patterns
   pattern = md_pattern_like(df)
   plot_missing_data_pattern(pattern)
   
   # 3. Check for systematic missingness
   # Are certain groups more likely to have missing data?
   print(df.groupby('group').apply(lambda x: x.isnull().mean()))

**Questions to ask**:
   - How much data is missing overall?
   - Which variables have the most missingness?
   - Are there patterns to the missingness?
   - Why might the data be missing?

Consider the Missing Data Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Think about *why* data is missing:

- **MCAR**: Truly random (rare in practice)
- **MAR**: Related to observed variables (MICE assumption)
- **MNAR**: Related to unobserved values (problematic)

If you suspect MNAR, consider:
   - Sensitivity analyses
   - Collecting more information
   - Using specialized methods
   - Being cautious about conclusions

Include Relevant Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Your imputation model should include:

✓ All variables in your analysis model
✓ Variables that predict missingness
✓ Variables correlated with incomplete variables
✓ Auxiliary variables that help prediction

.. code-block:: python

   # Include auxiliary variables even if not in final model
   # For example, if analyzing 'income', include:
   # - Variables you'll use in analysis
   # - Other income-related variables
   # - Variables that predict who's missing income data

Don't include:
   ✗ Variables with >50% missingness (impute them last or separately)
   ✗ Completely redundant variables
   ✗ Variables measured after your outcome (in causal analyses)

During Imputation
-----------------

Choose Appropriate Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Start with PMM**: It's well-tested and works for most cases

.. code-block:: python

   mice = MICE(df)
   mice.impute(n_imputations=20, maxit=20, method='pmm')

**Use method-specific approaches** when needed:

.. code-block:: python

   method_dict = {
       'age': 'pmm',           # Numeric, normal-ish
       'income': 'midas',      # Numeric, skewed
       'education': 'cart',    # Categorical, ordered
       'city': 'sample',       # Categorical, many levels
       'health_score': 'rf'    # Numeric, complex relationships
   }
   mice.impute(method=method_dict)

Run Sufficient Iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum**: 10 iterations
**Recommended**: 20 iterations
**Complex data**: 30-50+ iterations

Always check convergence:

.. code-block:: python

   from plotting.diagnostics import plot_chain_stats
   plot_chain_stats(mice.chain_mean, mice.chain_var)

Use Enough Imputations
~~~~~~~~~~~~~~~~~~~~~~

**Minimum**: 5 imputations (only for very low missingness)
**Recommended**: 20 imputations
**High missingness**: 50-100 imputations

**Rule of thumb**: 
   - m ≈ percentage of incomplete cases
   - If FMI > 0.3, increase m

.. code-block:: python

   # Check if you need more imputations
   pooled = mice.pool(summ=True)
   max_fmi = pooled['FMI'].max()
   
   if max_fmi > 0.3:
       print(f"Consider more imputations (current max FMI: {max_fmi:.2f})")

Enable Logging for Transparency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track what's happening:

.. code-block:: python

   from imputation import configure_logging
   
   configure_logging(
       level='INFO',           # INFO for progress, DEBUG for details
       log_dir='logs',         # Save to file
       console_level='INFO'    # Also print to console
   )

Set Random Seed for Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure your results are reproducible:

.. code-block:: python

   import numpy as np
   
   np.random.seed(42)  # Set seed before imputation
   mice = MICE(df)
   mice.impute(n_imputations=20, maxit=20)

Checking Quality
----------------

Always Assess Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~

Never skip this step:

.. code-block:: python

   # Visual check (primary method)
   from plotting.diagnostics import plot_chain_stats
   plot_chain_stats(mice.chain_mean, mice.chain_var, 
                    save_path='convergence.png')

Look for stable, flat lines in later iterations.

Compare Observed vs Imputed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check that imputed values are reasonable:

.. code-block:: python

   from plotting.diagnostics import stripplot, densityplot, boxplot
   
   missing_pattern = df.notna().astype(int)
   
   # Visual comparison
   stripplot(mice.imputed_datasets, missing_pattern,
             save_path='stripplot.png')
   densityplot(mice.imputed_datasets, missing_pattern,
               save_path='density.png')
   boxplot(mice.imputed_datasets, missing_pattern,
           save_path='boxplot.png')

**Look for**:
   - Imputed values within range of observed
   - Similar distributions
   - No impossible values (e.g., negative ages)

Check for Outliers
~~~~~~~~~~~~~~~~~~

Examine extreme imputed values:

.. code-block:: python

   # Check each imputed dataset
   for i, dataset in enumerate(mice.imputed_datasets):
       print(f"\nImputation {i+1}:")
       print(dataset.describe())
       
       # Check for outliers in key variables
       for col in ['income', 'age']:
           outliers = dataset[col] > dataset[col].quantile(0.99)
           if outliers.sum() > 0:
               print(f"  {col}: {outliers.sum()} extreme values")

Analysis
--------

Use Proper Pooling
~~~~~~~~~~~~~~~~~~

Always pool results using Rubin's rules:

✓ **Correct**:

.. code-block:: python

   mice.fit('outcome ~ predictor1 + predictor2')
   pooled = mice.pool(summ=True)
   print(pooled)

✗ **Wrong** (using single imputation):

.. code-block:: python

   # DON'T DO THIS
   single_dataset = mice.imputed_datasets[0]
   # Fit model on single dataset...

✗ **Wrong** (averaging imputations):

.. code-block:: python

   # DON'T DO THIS
   averaged = pd.concat(mice.imputed_datasets).groupby(level=0).mean()
   # Fit model on averaged dataset...

Include All Analysis Variables in Imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The imputation model should be at least as complex as your analysis model:

.. code-block:: python

   # If you plan to analyze:
   # outcome ~ age + gender + age:gender
   
   # Make sure age, gender, and outcome are all used in imputation
   # The interaction will be preserved

Report FMI
~~~~~~~~~~

Report the Fraction of Missing Information:

.. code-block:: python

   pooled = mice.pool(summ=True)
   print(f"FMI range: {pooled['FMI'].min():.2f} to {pooled['FMI'].max():.2f}")

This helps readers understand the impact of missingness on your results.

Common Mistakes to Avoid
------------------------

Mistake 1: Imputing After Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✗ **Wrong**:

.. code-block:: python

   # Creating derived variables before imputation
   df['log_income'] = np.log(df['income'])
   df['age_squared'] = df['age'] ** 2
   # Now impute...

This can lead to incompatible imputations (e.g., income ≠ exp(log_income)).

✓ **Correct**:

.. code-block:: python

   # Impute original variables
   mice = MICE(df[['income', 'age', 'other_vars']])
   mice.impute(n_imputations=20)
   
   # Create transformations after imputation
   imputed_with_transforms = []
   for dataset in mice.imputed_datasets:
       dataset['log_income'] = np.log(dataset['income'])
       dataset['age_squared'] = dataset['age'] ** 2
       imputed_with_transforms.append(dataset)

Mistake 2: Not Checking Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always check convergence! Non-converged imputations produce unreliable results.

Mistake 3: Too Few Imputations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5 imputations is often not enough. Use 20+ for most applications.

Mistake 4: Ignoring Imputation in Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When writing papers, document your imputation procedure thoroughly.

Mistake 5: Imputing Outcomes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Be cautious about imputing outcome variables:

.. code-block:: python

   # If outcome is missing, consider:
   # 1. Is it Missing At Random?
   # 2. Should these cases be excluded?
   # 3. Does imputation make sense for your research question?

For predictive modeling, imputing outcomes may be fine. For causal inference, 
be more careful.

Mistake 6: Perfect Separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With categorical outcomes and predictors, check for perfect separation after 
imputation:

.. code-block:: python

   # Check cross-tabulation
   for dataset in mice.imputed_datasets:
       print(pd.crosstab(dataset['outcome'], dataset['predictor']))

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use faster methods
   mice.impute(method='cart')  # Faster than RF
   
   # Use quickpred to reduce predictors
   from imputation.utils import quickpred
   pred_matrix = quickpred(df, mincor=0.3)
   mice.impute(predictor_matrix=pred_matrix)
   
   # Reduce number of imputations initially
   mice.impute(n_imputations=5)  # Then increase if needed

For Many Variables
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Select predictors automatically
   from imputation.utils import quickpred
   pred_matrix = quickpred(df, mincor=0.2, minpuc=0.1)
   
   # Use simpler methods
   mice.impute(method='pmm')  # PMM faster than RF

Parallel Processing
~~~~~~~~~~~~~~~~~~~

For very large analyses, consider parallelizing:

.. code-block:: python

   # Imputation itself is not parallelized, but you can
   # run multiple imputation runs in parallel (advanced)
   
   # Or parallelize your analysis across imputed datasets
   from joblib import Parallel, delayed
   
   def analyze_dataset(dataset):
       # Your analysis here
       return results
   
   results = Parallel(n_jobs=4)(
       delayed(analyze_dataset)(ds) for ds in mice.imputed_datasets
   )

Documentation and Reporting
----------------------------

What to Report
~~~~~~~~~~~~~~

In your methods section, include:

1. **Software**: "Imputation performed using mice-py version X.X.X"
2. **Method**: "Predictive Mean Matching (PMM)"
3. **Number of imputations**: "m = 20"
4. **Number of iterations**: "maxit = 20"
5. **Convergence**: "Convergence confirmed by trace plots"
6. **Variables**: "All variables in the analysis model plus [auxiliary vars]"
7. **Assumption**: "Assuming missing at random (MAR)"

In your results, include:

1. **Pooled estimates** with confidence intervals
2. **FMI** for key parameters
3. **Sample sizes**: Original n, n with complete data, n after imputation

Example Template
~~~~~~~~~~~~~~~~

.. code-block:: text

   Missing data were handled using multiple imputation by chained equations 
   (MICE; van Buuren & Groothuis-Oudshoorn, 2011) implemented in mice-py 
   (version 0.1.0). We created m=20 imputed datasets using predictive mean 
   matching for continuous variables and classification trees for categorical 
   variables. The imputation model included all variables in the analysis 
   model plus [list auxiliary variables]. The algorithm ran for 20 iterations; 
   convergence was confirmed through visual inspection of trace plots. Results 
   were combined using Rubin's rules. The fraction of missing information 
   ranged from X to Y across parameters.

Sensitivity Analyses
--------------------

Test Robustness
~~~~~~~~~~~~~~~

.. code-block:: python

   # Try different methods
   mice_pmm = MICE(df)
   mice_pmm.impute(method='pmm')
   mice_pmm.fit('outcome ~ predictors')
   results_pmm = mice_pmm.pool(summ=True)
   
   mice_cart = MICE(df)
   mice_cart.impute(method='cart')
   mice_cart.fit('outcome ~ predictors')
   results_cart = mice_cart.pool(summ=True)
   
   # Compare results
   print("PMM results:", results_pmm.loc['predictor1', 'Estimate'])
   print("CART results:", results_cart.loc['predictor1', 'Estimate'])

Vary Number of Imputations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   for m in [5, 10, 20, 50]:
       mice = MICE(df)
       mice.impute(n_imputations=m)
       mice.fit('outcome ~ predictor')
       results = mice.pool(summ=True)
       print(f"m={m}: {results.loc['predictor', 'Estimate']:.3f}")

If results change substantially, you need more imputations.

Complete Case Analysis
~~~~~~~~~~~~~~~~~~~~~~

Compare with complete case analysis:

.. code-block:: python

   import statsmodels.formula.api as smf
   
   # Complete cases only
   df_complete = df.dropna()
   model_cc = smf.ols('outcome ~ predictor1 + predictor2', 
                      data=df_complete).fit()
   print("Complete cases:", model_cc.params)
   
   # Multiple imputation
   mice = MICE(df)
   mice.impute(n_imputations=20)
   mice.fit('outcome ~ predictor1 + predictor2')
   results_mi = mice.pool(summ=True)
   print("MI:", results_mi['Estimate'])

Large differences suggest missingness is not MCAR.

Checklist
---------

Before finalizing your analysis:

☐ Explored missing data patterns and mechanisms
☐ Included all relevant variables in imputation model
☐ Chose appropriate imputation method(s)
☐ Ran sufficient iterations (≥20)
☐ Created enough imputations (≥20)
☐ Checked convergence diagnostics
☐ Compared observed vs imputed distributions
☐ Used proper pooling (Rubin's rules)
☐ Reported FMI
☐ Documented procedure thoroughly
☐ Performed sensitivity analyses
☐ Set random seed for reproducibility

Further Reading
---------------

- :doc:`../theory/index` for theoretical background
- :doc:`../examples/index` for practical examples
- Van Buuren, S. (2018). *Flexible Imputation of Missing Data*. 2nd edition. 
  Chapman & Hall/CRC Press.
- See :doc:`../references` for more resources

