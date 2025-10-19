Quickstart Guide
================

This guide will get you started with mice-py in just a few minutes.

Basic Workflow
--------------

The typical MICE workflow consists of three main steps:

1. **Initialize** a MICE object with your data
2. **Impute** missing values multiple times
3. **Analyze** the imputed datasets and pool results

Minimal Example
---------------

Here's a complete example using the NHANES dataset:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from imputation import MICE
   
   # 1. Load data with missing values
   df = pd.DataFrame({
       'age': [25, 30, np.nan, 45, 50, np.nan, 35, 40],
       'income': [50000, np.nan, 60000, np.nan, 80000, 70000, np.nan, 75000],
       'education': ['Bachelor', 'Master', 'Bachelor', np.nan, 
                     'PhD', 'Master', 'Bachelor', np.nan],
       'employed': [1, 1, 0, 1, 1, np.nan, 1, 0]
   })
   
   # 2. Initialize MICE object
   mice = MICE(df)
   
   # 3. Perform imputation
   mice.impute(
       n_imputations=5,    # Create 5 imputed datasets
       maxit=10,           # Run 10 iterations
       method='pmm'        # Use Predictive Mean Matching
   )
   
   # 4. Access imputed datasets
   imputed_datasets = mice.imputed_datasets
   print(f"Created {len(imputed_datasets)} complete datasets")
   
   # 5. Fit a statistical model
   mice.fit('income ~ age + education + employed')
   
   # 6. Pool results using Rubin's rules
   pooled_results = mice.pool(summ=True)
   print(pooled_results)

Understanding the Output
------------------------

After imputation, you'll have:

**Multiple Complete Datasets**
   The ``mice.imputed_datasets`` attribute contains a list of pandas DataFrames, 
   each with all missing values filled in differently.

**Convergence Diagnostics**
   - ``mice.chain_mean``: Mean of each variable across iterations
   - ``mice.chain_var``: Variance of each variable across iterations

**Pooled Results**
   When you call ``mice.pool()``, you get combined estimates from all imputed datasets
   using Rubin's rules, including:
   
   - Pooled coefficients
   - Standard errors
   - Confidence intervals
   - Fraction of missing information (FMI)

Checking for Convergence
-------------------------

Before analyzing results, check if the imputation converged:

.. code-block:: python

   from plotting.diagnostics import plot_chain_stats
   
   # Visualize convergence
   plot_chain_stats(
       chain_mean=mice.chain_mean,
       chain_var=mice.chain_var,
       save_path='convergence.png'
   )

The chains should stabilize after a few iterations. If they haven't, increase ``maxit``.

Visualizing Imputations
-----------------------

Compare observed and imputed values:

.. code-block:: python

   from plotting.diagnostics import stripplot, densityplot
   
   # Create missing pattern indicator
   missing_pattern = df.notna().astype(int)
   
   # Stripplot: points for observed (blue) and imputed (red) values
   stripplot(mice.imputed_datasets, missing_pattern, 
             save_path='stripplot.png')
   
   # Density plot: distribution comparison
   densityplot(mice.imputed_datasets, missing_pattern,
               save_path='density.png')

Using Different Methods
-----------------------

PMM (Default)
~~~~~~~~~~~~~

Predictive Mean Matching is the default method and works well for most numeric data:

.. code-block:: python

   mice.impute(n_imputations=5, method='pmm')

CART
~~~~

Classification and Regression Trees handle non-linear relationships:

.. code-block:: python

   mice.impute(n_imputations=5, method='cart')

Random Forest
~~~~~~~~~~~~~

Random Forest captures complex interactions:

.. code-block:: python

   mice.impute(n_imputations=5, method='rf')

Method Per Variable
~~~~~~~~~~~~~~~~~~~

Use different methods for different variables:

.. code-block:: python

   method_dict = {
       'age': 'pmm',
       'income': 'cart',
       'education': 'sample',
       'employed': 'rf'
   }
   mice.impute(n_imputations=5, method=method_dict)

Logging
-------

Enable logging to track progress:

.. code-block:: python

   from imputation import configure_logging
   
   # Enable INFO level logging
   configure_logging(level='INFO')
   
   # Now run MICE - you'll see progress messages
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10)

Common Parameters
-----------------

Here are the most commonly used parameters:

**n_imputations** (default: 5)
   Number of imputed datasets to create. More datasets provide more accurate 
   pooled estimates but take longer to compute.

**maxit** (default: 10)
   Number of MICE iterations. Check convergence diagnostics to determine if 
   more iterations are needed.

**method** (default: 'pmm')
   Imputation method. Can be a string (same method for all variables) or 
   a dictionary mapping column names to methods.

**initial** (default: 'sample')
   Method for initial imputation before MICE iterations. Options: 'sample' 
   or 'mean'.

**visit_sequence** (default: 'monotone')
   Order in which variables are imputed. Options: 'monotone', 'random', or 
   a custom list.

Next Steps
----------

Now that you understand the basics:

- **Explore methods**: Read :doc:`user_guide/imputation_methods` to choose 
  the best method for your data
  
- **Advanced parameters**: Learn about predictor matrices and visit sequences 
  in :doc:`user_guide/predictor_matrices`
  
- **Theory**: Understand the theory behind MICE in :doc:`theory/index`

- **Examples**: See complete workflows in :doc:`examples/index`

