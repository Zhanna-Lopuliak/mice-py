MICE Class
==========

The main class for performing multiple imputation by chained equations.

.. currentmodule:: imputation

.. autoclass:: MICE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Overview
--------

The ``MICE`` class is the primary interface for multiple imputation in mice-py.
It handles the entire imputation process, from initialization through to analysis
and pooling.

Basic Usage
-----------

.. code-block:: python

   from imputation import MICE
   import pandas as pd
   
   # Load data with missing values
   df = pd.read_csv('data.csv')
   
   # Initialize MICE object
   mice = MICE(df)
   
   # Perform imputation
   mice.impute(
       n_imputations=5,
       maxit=10,
       method='pmm'
   )
   
   # Access imputed datasets
   imputed_datasets = mice.imputed_datasets
   
   # Fit a statistical model
   mice.fit('outcome ~ predictor1 + predictor2')
   
   # Pool results
   pooled = mice.pool(summ=True)
   print(pooled)

Main Methods
------------

__init__(data)
~~~~~~~~~~~~~~

Initialize a MICE object with your data.

**Parameters**:
   - **data** (pandas.DataFrame): Input data with missing values

**Raises**:
   - ValueError: If data is not a DataFrame or has duplicate column names

impute()
~~~~~~~~

Perform multiple imputation.

**Parameters**:
   - **n_imputations** (int): Number of imputed datasets (default: 5)
   - **maxit** (int): Number of iterations (default: 10)
   - **method** (str or dict): Imputation method(s) (default: 'pmm')
   - **initial** (str): Initial imputation method (default: 'sample')
   - **predictor_matrix** (DataFrame, optional): Custom predictor matrix
   - **visit_sequence** (str or list): Variable visit order (default: 'monotone')
   - **seed** (int, optional): Random seed for reproducibility
   - Additional method-specific parameters (see below)

**Method-specific parameters**:
   - PMM: ``pmm_donors``, ``pmm_matchtype``, ``pmm_ridge``
   - CART: ``cart_max_depth``, ``cart_min_samples_split``, ``cart_min_samples_leaf``
   - RF: ``rf_n_estimators``, ``rf_max_depth``, ``rf_max_features``
   - MIDAS: ``midas_donors``, ``midas_ridge``

**Returns**:
   - None (modifies object in-place)

**Raises**:
   - ValueError: If parameters are invalid

fit(formula)
~~~~~~~~~~~~

Fit a statistical model on all imputed datasets.

**Parameters**:
   - **formula** (str): Model formula in Patsy syntax (e.g., 'y ~ x1 + x2')

**Returns**:
   - None (stores results internally)

**Example**:

.. code-block:: python

   # Simple regression
   mice.fit('income ~ age + education')
   
   # With interaction
   mice.fit('income ~ age * education')
   
   # Multiple predictors
   mice.fit('outcome ~ x1 + x2 + x3 + C(categorical_var)')

pool(summ=True)
~~~~~~~~~~~~~~~

Pool results from multiple imputed datasets using Rubin's rules.

**Parameters**:
   - **summ** (bool): Return summary (True) or detailed results (False)

**Returns**:
   - pandas.DataFrame: Pooled results with columns:
     - Estimate: Pooled coefficient
     - Std.Error: Pooled standard error
     - t-statistic: Test statistic
     - df: Degrees of freedom
     - P>|t|: p-value
     - [0.025]: Lower 95% CI bound
     - 0.975]: Upper 95% CI bound
     - FMI: Fraction of missing information

**Example**:

.. code-block:: python

   results = mice.pool(summ=True)
   print(results)
   
   # Access specific values
   coef = results.loc['age', 'Estimate']
   pval = results.loc['age', 'P>|t|']
   fmi = results.loc['age', 'FMI']

Attributes
----------

data
~~~~

The original input data (pandas.DataFrame).

imputed_datasets
~~~~~~~~~~~~~~~~

List of imputed datasets (list of pandas.DataFrames). Available after calling ``impute()``.

chain_mean
~~~~~~~~~~

Dictionary mapping variable names to mean chains across iterations. Used for 
convergence diagnostics.

chain_var
~~~~~~~~~

Dictionary mapping variable names to variance chains across iterations. Used for
convergence diagnostics.

id_obs
~~~~~~

Dictionary mapping variable names to boolean arrays indicating observed values.

id_mis
~~~~~~

Dictionary mapping variable names to boolean arrays indicating missing values.

Examples
--------

Basic Imputation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE
   import pandas as pd
   import numpy as np
   
   # Create sample data
   df = pd.DataFrame({
       'age': [25, 30, np.nan, 45, 50],
       'income': [50000, np.nan, 60000, 75000, np.nan],
       'education': ['HS', 'BS', 'MS', np.nan, 'PhD']
   })
   
   # Impute
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10, method='pmm')
   
   # Check results
   print(f"Created {len(mice.imputed_datasets)} complete datasets")

Custom Methods Per Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   method_dict = {
       'age': 'pmm',
       'income': 'cart',
       'education': 'sample'
   }
   
   mice.impute(n_imputations=10, method=method_dict)

Custom Predictor Matrix
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Create predictor matrix
   pred_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
   np.fill_diagonal(pred_matrix.values, 0)
   
   # Don't use education to predict income
   pred_matrix.loc['income', 'education'] = 0
   
   mice.impute(predictor_matrix=pred_matrix)

With Method-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PMM with more donors
   mice.impute(method='pmm', pmm_donors=10)
   
   # CART with depth limit
   mice.impute(method='cart', cart_max_depth=15)
   
   # Random Forest with more trees
   mice.impute(method='rf', rf_n_estimators=200)

Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE, configure_logging
   from plotting.diagnostics import plot_chain_stats
   
   # Enable logging
   configure_logging(level='INFO')
   
   # Load data
   df = pd.read_csv('data.csv')
   
   # Impute
   mice = MICE(df)
   mice.impute(n_imputations=20, maxit=20, method='pmm')
   
   # Check convergence
   plot_chain_stats(mice.chain_mean, mice.chain_var, 
                    save_path='convergence.png')
   
   # Fit model
   mice.fit('outcome ~ age + gender + treatment')
   
   # Pool results
   results = mice.pool(summ=True)
   print(results)
   
   # Check FMI
   print(f"\nMax FMI: {results['FMI'].max():.3f}")

See Also
--------

- :doc:`methods` for imputation method details
- :doc:`pooling` for pooling functions
- :doc:`../user_guide/mice_overview` for conceptual overview
- :doc:`../examples/index` for more examples

