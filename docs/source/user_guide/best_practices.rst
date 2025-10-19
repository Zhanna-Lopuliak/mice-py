Best Practices
==============

Quick reference for using MICE correctly.

Basic Setup
-----------

.. code-block:: python

   from imputation import MICE, configure_logging
   from plotting.diagnostics import plot_chain_stats
   from plotting.utils import md_pattern_like
   
   # Enable logging
   configure_logging(level='INFO')
   
   # Check missing patterns
   pattern = md_pattern_like(df)
   print(pattern)
   
   # Set random seed for reproducibility
   import numpy as np
   np.random.seed(42)

Recommended Parameters
----------------------

.. code-block:: python

   mice = MICE(df)
   mice.impute(
       n_imputations=20,      # Use 20+, not just 5
       maxit=20,              # Ensure convergence
       method='pmm'           # Start with PMM
   )

**Number of imputations**:
   - Minimum: 5 (only for low missingness)
   - Recommended: 20
   - High missingness (>30%): 50-100

**Number of iterations**:
   - Minimum: 10
   - Recommended: 20
   - Check convergence to confirm sufficiency

Method Selection
----------------

.. code-block:: python

   # Use different methods for different variables
   method_dict = {
       'numeric_normal': 'pmm',
       'numeric_skewed': 'midas',
       'categorical': 'cart',
       'complex': 'rf'
   }
   mice.impute(method=method_dict)

**Quick guide**:
   - PMM: General numeric data
   - CART: Categorical or non-linear
   - RF: Complex patterns
   - MIDAS: Skewed or small samples
   - Sample: Initial imputation or simple cases

Always Check Convergence
-------------------------

.. code-block:: python

   from plotting.diagnostics import plot_chain_stats
   
   plot_chain_stats(mice.chain_mean, mice.chain_var, 
                    save_path='convergence.png')

Look for flat, stable chains in later iterations.

Compare Distributions
---------------------

.. code-block:: python

   from plotting.diagnostics import stripplot, densityplot
   
   missing_pattern = df.notna().astype(int)
   
   stripplot(mice.imputed_datasets, missing_pattern)
   densityplot(mice.imputed_datasets, missing_pattern)

Ensure imputed values are within reasonable range.

Proper Pooling
--------------

Always use Rubin's rules:

.. code-block:: python

   # Correct
   mice.fit('outcome ~ predictor')
   pooled = mice.pool(summ=True)
   print(pooled)
   
   # Check FMI
   print(f"Max FMI: {pooled['FMI'].max():.3f}")

Never:
   - Use only one imputed dataset
   - Average the imputed datasets
   - Use standard analysis on single imputation

Common Mistakes
---------------

**Mistake 1: Too few imputations**

.. code-block:: python

   # Don't do this
   mice.impute(n_imputations=5)  # Often not enough
   
   # Do this
   mice.impute(n_imputations=20)  # Better

**Mistake 2: Not checking convergence**

.. code-block:: python

   # Always check
   plot_chain_stats(mice.chain_mean, mice.chain_var)

**Mistake 3: Using single imputation**

.. code-block:: python

   # Don't do this
   single_dataset = mice.imputed_datasets[0]
   model = fit_model(single_dataset)  # Wrong!
   
   # Do this
   mice.fit('y ~ x')
   pooled = mice.pool(summ=True)  # Correct

**Mistake 4: Imputing after transformations**

.. code-block:: python

   # Don't do this
   df['log_income'] = np.log(df['income'])
   mice = MICE(df)  # Imputes both income and log_income separately!
   
   # Do this
   mice = MICE(df[['income', 'other_vars']])
   mice.impute(n_imputations=20)
   # Then create transformations after imputation
   for dataset in mice.imputed_datasets:
       dataset['log_income'] = np.log(dataset['income'])

Variable Selection
------------------

Include in imputation:
   - All variables in your analysis model
   - Variables that predict missingness
   - Variables correlated with incomplete variables

.. code-block:: python

   # If analyzing: income ~ age + education
   # Impute: income, age, education, plus auxiliary variables
   
   mice = MICE(df[['income', 'age', 'education', 
                    'occupation', 'zip_code']])  # auxiliaries

Predictor Matrix
----------------

For most cases, use default (all variables predict each other):

.. code-block:: python

   mice.impute(n_imputations=20)  # Uses default predictor matrix

For custom control:

.. code-block:: python

   import numpy as np
   
   predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(predictor_matrix.values, 0)
   
   # Customize as needed
   predictor_matrix.loc['var1', 'var2'] = 0
   
   mice.impute(predictor_matrix=predictor_matrix)

See :doc:`predictor_matrices` for details.

Performance Tips
----------------

**For large datasets**:

.. code-block:: python

   # Use faster methods
   mice.impute(method='cart')  # Faster than RF
   
   # Use quickpred to reduce predictors
   from imputation.utils import quickpred
   pred_matrix = quickpred(df, mincor=0.3)
   mice.impute(predictor_matrix=pred_matrix)

**For many variables**:

.. code-block:: python

   # Automatic predictor selection
   pred_matrix = quickpred(df, mincor=0.2, minpuc=0.1)
   mice.impute(predictor_matrix=pred_matrix)

Essential Checklist
-------------------

Before finalizing:

☐ Checked missing patterns
☐ Used appropriate methods
☐ Ran sufficient iterations (≥20)
☐ Created enough imputations (≥20)
☐ Checked convergence
☐ Compared observed vs imputed distributions
☐ Used proper pooling (Rubin's rules)
☐ Set random seed for reproducibility

Documentation
-------------

**In methods section, report**:
   - Software and version
   - Imputation method(s)
   - Number of imputations (m)
   - Number of iterations
   - Convergence assessment
   - Variables included

**In results, report**:
   - Pooled estimates with CI
   - FMI for key parameters
   - Sample sizes

Example:

.. code-block:: text

   Missing data were handled using MICE (mice-py v0.1.0) with m=20 
   imputations using predictive mean matching. The algorithm ran for 
   20 iterations; convergence was confirmed by visual inspection of 
   trace plots. Results were pooled using Rubin's rules.

See Also
--------

- :doc:`mice_overview` - How MICE works
- :doc:`imputation_methods` - Method details
- :doc:`convergence_diagnostics` - Checking convergence
- :doc:`pooling_analysis` - Analyzing results
