MICE Overview
=============

This page explains how the MICE (Multiple Imputation by Chained Equations) algorithm 
works and how to use it in mice-py.

What is MICE?
-------------

MICE is an iterative algorithm for imputing missing data that:

1. Creates **multiple** imputed datasets (not just one)
2. Uses **chained equations** (imputes one variable at a time)
3. Accounts for **uncertainty** in the missing values
4. Enables **valid statistical inference** under MAR

The algorithm was developed by van Buuren and Groothuis-Oudshoorn (2011) and is 
implemented in the widely-used R package ``mice``.

How MICE Works
--------------

The MICE Algorithm
~~~~~~~~~~~~~~~~~~

**Step 1: Initialization**
   Fill in missing values using simple imputation (e.g., random sampling from observed 
   values or means).

**Step 2: Iteration**
   For each variable with missing data (in a specified order):
   
   a. Set the variable's imputed values back to missing
   b. Use observed values as the target and other variables as predictors
   c. Fit a model and predict the missing values
   d. Replace missing values with predictions (plus random variation)

**Step 3: Repeat**
   Cycle through all incomplete variables multiple times until convergence.

**Step 4: Multiple Imputations**
   Repeat the entire process to create multiple different completed datasets.

Visual Example
~~~~~~~~~~~~~~

Suppose you have three variables (Age, Income, Education) with missing values:

.. code-block:: text

   Iteration 1:
   1. Impute Age using Income + Education
   2. Impute Income using Age + Education  
   3. Impute Education using Age + Income
   
   Iteration 2:
   1. Re-impute Age using updated Income + Education
   2. Re-impute Income using updated Age + Education
   3. Re-impute Education using updated Age + Income
   
   ... continue until convergence

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE
   import pandas as pd
   
   # Your data with missing values
   df = pd.read_csv('data.csv')
   
   # Initialize MICE
   mice = MICE(df)
   
   # Run imputation
   mice.impute(
       n_imputations=5,   # Create 5 complete datasets
       maxit=10,          # Run 10 iterations
       method='pmm'       # Use Predictive Mean Matching
   )
   
   # Access results
   imputed_datasets = mice.imputed_datasets

Key Parameters
~~~~~~~~~~~~~~

**n_imputations**
   Number of imputed datasets to create. Common choices: 5-10 for moderate missingness, 
   20-100 for high missingness or specific analyses.

**maxit**
   Number of iterations through all variables. Usually 10-20 is sufficient. Check 
   convergence diagnostics to determine if more are needed.

**method**
   Imputation method(s) to use. Can be:
   
   - A string (same method for all variables): ``'pmm'``, ``'cart'``, ``'rf'``, 
     ``'midas'``, ``'sample'``
   - A dictionary mapping column names to methods

**initial**
   Method for initial imputation before iterations begin:
   
   - ``'sample'`` (default): Random sampling from observed values
   - ``'mean'``: Use mean for numeric, mode for categorical

**visit_sequence**
   Order to visit variables during each iteration:
   
   - ``'monotone'`` (default): Ordered by amount of missingness
   - ``'random'``: Random order in each iteration
   - A list of column names for custom order

Controlling the Imputation Process
-----------------------------------

Predictor Matrix
~~~~~~~~~~~~~~~~

By default, each variable is predicted using all other variables. You can customize 
this with a predictor matrix:

.. code-block:: python

   import numpy as np
   
   # Create custom predictor matrix
   predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(predictor_matrix.values, 0)
   
   # Exclude certain predictors
   predictor_matrix.loc['income', 'education'] = 0  # Don't use education to predict income
   
   mice.impute(predictor_matrix=predictor_matrix)

See :doc:`predictor_matrices` for more details.

Method-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different methods have specific parameters you can tune:

.. code-block:: python

   # PMM with custom number of donors
   mice.impute(method='pmm', pmm_donors=3)
   
   # CART with maximum depth
   mice.impute(method='cart', cart_max_depth=10)
   
   # Random Forest with number of trees
   mice.impute(method='rf', rf_n_estimators=50)

Accessing Results
-----------------

Imputed Datasets
~~~~~~~~~~~~~~~~

.. code-block:: python

   # List of pandas DataFrames
   imputed_datasets = mice.imputed_datasets
   
   # Access individual datasets
   dataset_1 = imputed_datasets[0]
   dataset_2 = imputed_datasets[1]

Convergence Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mean and variance chains for each variable
   chain_mean = mice.chain_mean
   chain_var = mice.chain_var
   
   # Visualize
   from plotting.diagnostics import plot_chain_stats
   plot_chain_stats(chain_mean, chain_var)

See :doc:`convergence_diagnostics` for details on interpreting these.

Model Fitting and Pooling
~~~~~~~~~~~~~~~~~~~~~~~~~~

After imputation, fit models and pool results:

.. code-block:: python

   # Fit a model on all imputed datasets
   mice.fit('outcome ~ predictor1 + predictor2')
   
   # Pool using Rubin's rules
   results = mice.pool(summ=True)
   print(results)

See :doc:`pooling_analysis` for more on analyzing imputed data.

When MICE Works Well
--------------------

MICE is effective when:

✓ **Data is MAR**: Missingness can be predicted from observed variables
✓ **Relationships are clear**: Variables have predictable relationships
✓ **Sufficient data**: Enough observed cases to model relationships
✓ **Multiple variables**: Missing data across several variables
✓ **Complex patterns**: Non-monotone missingness patterns

Limitations of MICE
-------------------

Be aware of potential issues:

✗ **MNAR data**: MICE assumes MAR; with MNAR, results may be biased
✗ **High missingness**: If >50% missing in key variables, predictions may be unstable
✗ **Small samples**: Need sufficient data to estimate relationships
✗ **Incompatible models**: The separate models for each variable may be theoretically 
  inconsistent (though this rarely causes problems in practice)
✗ **Perfect collinearity**: Variables with perfect relationships may cause issues

The Imputation Model vs Analysis Model
---------------------------------------

An important concept: the **imputation model** (used to fill in missing values) should 
be at least as complex as your **analysis model** (the model you'll fit to the data).

**Imputation model**: The set of all univariate models used to impute each variable

**Analysis model**: The model you fit to the completed data (e.g., regression)

.. tip::
   Include all variables that:
   
   - Are in your analysis model
   - Predict missingness
   - Are correlated with incomplete variables
   
   This ensures the MAR assumption is more plausible and improves imputation quality.

Typical Workflow
----------------

1. **Explore your data**: Understand patterns and mechanisms of missingness
2. **Configure MICE**: Choose methods, predictor matrix, and parameters
3. **Run imputation**: Create multiple complete datasets
4. **Check convergence**: Ensure the algorithm has stabilized
5. **Diagnose quality**: Compare observed vs imputed distributions
6. **Analyze**: Fit your statistical model(s)
7. **Pool results**: Combine estimates using Rubin's rules

Example Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE, configure_logging
   from plotting.diagnostics import plot_chain_stats, stripplot
   from plotting.utils import md_pattern_like
   
   # Enable logging
   configure_logging(level='INFO')
   
   # 1. Explore
   pattern = md_pattern_like(df)
   print(pattern)
   
   # 2. Configure and run
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=15, method='pmm')
   
   # 3. Check convergence
   plot_chain_stats(mice.chain_mean, mice.chain_var)
   
   # 4. Diagnose
   missing_pattern = df.notna().astype(int)
   stripplot(mice.imputed_datasets, missing_pattern)
   
   # 5-7. Analyze and pool
   mice.fit('outcome ~ predictor1 + predictor2')
   results = mice.pool(summ=True)
   print(results)

Next Steps
----------

- Learn about different :doc:`imputation_methods` and when to use each
- Understand :doc:`predictor_matrices` for fine control
- Read about :doc:`convergence_diagnostics` to ensure quality
- See :doc:`pooling_analysis` for analyzing imputed data

