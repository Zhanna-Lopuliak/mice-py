Predictor Matrices
==================

The predictor matrix controls which variables are used to predict (impute) each 
incomplete variable. This guide explains how predictor matrices work and when to 
customize them.

What is a Predictor Matrix?
----------------------------

A predictor matrix is a square matrix where:

- **Rows** represent variables to be imputed (target variables)
- **Columns** represent predictor variables
- A value of **1** means "use this column to predict this row"
- A value of **0** means "don't use this column to predict this row"
- The **diagonal** is always 0 (a variable doesn't predict itself)

Example
~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Sample predictor matrix for variables: age, income, education
   predictor_matrix = pd.DataFrame(
       [[0, 1, 1],   # To impute age, use income and education
        [1, 0, 1],   # To impute income, use age and education
        [1, 1, 0]],  # To impute education, use age and income
       index=['age', 'income', 'education'],
       columns=['age', 'income', 'education']
   )
   
   print(predictor_matrix)

Output:

.. code-block:: text

            age  income  education
   age        0       1          1
   income     1       0          1
   education  1       1          0

Default Behavior
----------------

If you don't specify a predictor matrix, MICE uses **all other variables** as 
predictors for each incomplete variable:

.. code-block:: python

   mice = MICE(df)
   mice.impute(n_imputations=5)  # Uses default predictor matrix

This is equivalent to:

.. code-block:: python

   # Create full predictor matrix
   predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(predictor_matrix.values, 0)
   
   mice.impute(predictor_matrix=predictor_matrix)

When to Customize
-----------------

You should customize the predictor matrix when:

1. **Variables shouldn't predict each other** (logical constraints)
2. **Too many predictors** cause computational issues
3. **Known causal relationships** suggest specific prediction structures
4. **Auxiliary variables** should be used for prediction but not imputed
5. **Multicollinearity** between predictors causes problems

Creating Custom Predictor Matrices
-----------------------------------

Start with Default
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Start with all-ones matrix
   predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(predictor_matrix.values, 0)

Then modify as needed.

Exclude Specific Predictors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prevent one variable from predicting another:

.. code-block:: python

   # Don't use education to predict income
   predictor_matrix.loc['income', 'education'] = 0

Use Only Specific Predictors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use only age and income to predict education
   predictor_matrix.loc['education', :] = 0  # First, exclude all
   predictor_matrix.loc['education', ['age', 'income']] = 1  # Then include specific

Block of Variables
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Don't use any demographic variables to predict health outcomes
   demographic_vars = ['age', 'gender', 'ethnicity']
   health_vars = ['blood_pressure', 'cholesterol', 'bmi']
   
   for health_var in health_vars:
       predictor_matrix.loc[health_var, demographic_vars] = 0

Common Patterns
---------------

Include Complete Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete variables (no missing values) can be used as predictors but don't need to 
be imputed:

.. code-block:: python

   # Identify complete variables
   complete_vars = df.columns[df.isnull().sum() == 0].tolist()
   incomplete_vars = df.columns[df.isnull().sum() > 0].tolist()
   
   # Create predictor matrix only for incomplete variables
   predictor_matrix = pd.DataFrame(
       1, 
       index=incomplete_vars,
       columns=df.columns  # Use all variables as predictors
   )
   
   # Set diagonal to 0
   for var in incomplete_vars:
       predictor_matrix.loc[var, var] = 0

Auxiliary Variables
~~~~~~~~~~~~~~~~~~~

Variables that help prediction but aren't part of your analysis model:

.. code-block:: python

   # Suppose 'auxiliary_score' helps predict income but won't be in final model
   # Include it as predictor for income
   predictor_matrix.loc['income', 'auxiliary_score'] = 1
   
   # But don't impute it if missing
   # (Remove from rows if it has missing values you don't care about)

Quickpred: Automatic Predictor Selection
-----------------------------------------

For datasets with many variables, the ``quickpred`` algorithm automatically selects 
predictors based on correlations:

.. code-block:: python

   from imputation.utils import quickpred
   
   # Automatically select predictors
   predictor_matrix = quickpred(
       df,
       mincor=0.1,    # Minimum correlation
       minpuc=0.0,    # Minimum proportion of usable cases
       include=None,  # Variables to always include
       exclude=None   # Variables to always exclude
   )
   
   mice.impute(predictor_matrix=predictor_matrix)

Parameters:

- **mincor**: Only use predictors with absolute correlation >= this threshold
- **minpuc**: Require minimum proportion of usable complete cases
- **include**: List of variables to always include as predictors
- **exclude**: List of variables to never use as predictors

Monotone Patterns
-----------------

If your data has a monotone missing pattern, you can use a block structure:

.. code-block:: python

   # Variables ordered by missingness: time1, time2, time3, time4
   # time2 can only use time1; time3 can use time1-2; etc.
   
   predictor_matrix = pd.DataFrame(0, index=df.columns, columns=df.columns)
   
   predictor_matrix.loc['time2', 'time1'] = 1
   predictor_matrix.loc['time3', ['time1', 'time2']] = 1
   predictor_matrix.loc['time4', ['time1', 'time2', 'time3']] = 1

This respects the temporal structure of the data.

Practical Examples
------------------

Example 1: Exclude Future Predictors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In longitudinal data, future values shouldn't predict past values:

.. code-block:: python

   # Time-ordered variables
   time_vars = ['baseline', 'month3', 'month6', 'month12']
   
   predictor_matrix = pd.DataFrame(1, index=time_vars, columns=time_vars)
   np.fill_diagonal(predictor_matrix.values, 0)
   
   # Exclude future predictors
   for i, target in enumerate(time_vars):
       for predictor in time_vars[i+1:]:
           predictor_matrix.loc[target, predictor] = 0

Example 2: Separate Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variables from different domains might not predict each other well:

.. code-block:: python

   physical_health = ['height', 'weight', 'blood_pressure']
   mental_health = ['depression_score', 'anxiety_score']
   demographics = ['age', 'gender', 'education']
   
   # Demographics predict everything
   # But physical and mental health don't predict each other
   
   predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(predictor_matrix.values, 0)
   
   # Physical doesn't predict mental
   for phys in physical_health:
       for mental in mental_health:
           predictor_matrix.loc[mental, phys] = 0
   
   # Mental doesn't predict physical
   for mental in mental_health:
       for phys in physical_health:
           predictor_matrix.loc[phys, mental] = 0

Example 3: High-Dimensional Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With many variables, use only the most correlated:

.. code-block:: python

   from imputation.utils import quickpred
   
   # Select predictors with correlation >= 0.3
   predictor_matrix = quickpred(df, mincor=0.3)
   
   # Always include key variables
   key_vars = ['age', 'treatment_group']
   for var in df.columns:
       if var not in key_vars:
           for key_var in key_vars:
               predictor_matrix.loc[var, key_var] = 1

Checking Your Predictor Matrix
-------------------------------

Before running MICE, verify your predictor matrix:

.. code-block:: python

   # Check dimensions
   print(f"Shape: {predictor_matrix.shape}")
   
   # Check diagonal is zero
   assert (np.diag(predictor_matrix.values) == 0).all(), "Diagonal should be 0"
   
   # Check each incomplete variable has at least one predictor
   print("Number of predictors per variable:")
   print(predictor_matrix.sum(axis=1))
   
   # Visualize
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 8))
   sns.heatmap(predictor_matrix, cmap='RdYlGn', center=0.5, 
               cbar_kws={'label': 'Use as predictor'})
   plt.title('Predictor Matrix')
   plt.tight_layout()
   plt.savefig('predictor_matrix.png')

Common Issues
-------------

Too Few Predictors
~~~~~~~~~~~~~~~~~~

**Problem**: Variables have insufficient predictors, leading to poor imputations.

**Solution**: 
   - Ensure each variable has at least 2-3 relevant predictors
   - Use the default full matrix if unsure

Too Many Predictors
~~~~~~~~~~~~~~~~~~~~

**Problem**: Model fitting is slow or fails due to multicollinearity.

**Solution**:
   - Use ``quickpred()`` to select based on correlations
   - Manually remove redundant predictors
   - Increase ridge parameter in PMM

Circular Dependencies
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Worried about A predicting B and B predicting A.

**Solution**: 
   - This is actually **fine** in MICE! The algorithm handles it iteratively.
   - Only exclude if there's a logical reason (e.g., temporal ordering)

Tips and Best Practices
------------------------

1. **Start simple**: Use the default full matrix first
2. **Be conservative**: Only exclude predictors if you have good reason
3. **Include analysis model variables**: All variables in your final analysis model 
   should be used as predictors
4. **Use auxiliary variables**: Variables that help prediction even if not in your 
   analysis model
5. **Respect time order**: In longitudinal data, don't let future predict past
6. **Check convergence**: Restrictive predictor matrices may slow convergence

Testing Your Choices
---------------------

Compare imputation quality with different predictor matrices:

.. code-block:: python

   # Default: all predictors
   mice_full = MICE(df)
   mice_full.impute(n_imputations=5)
   
   # Custom: restricted predictors
   mice_custom = MICE(df)
   mice_custom.impute(n_imputations=5, predictor_matrix=custom_matrix)
   
   # Compare convergence
   from plotting.diagnostics import plot_chain_stats
   plot_chain_stats(mice_full.chain_mean, mice_full.chain_var, 
                    save_path='full_convergence.png')
   plot_chain_stats(mice_custom.chain_mean, mice_custom.chain_var,
                    save_path='custom_convergence.png')

Next Steps
----------

- Learn about :doc:`convergence_diagnostics` to check if your predictor matrix works well
- See :doc:`best_practices` for overall guidance
- Try examples in :doc:`../examples/index`

