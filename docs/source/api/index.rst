API Reference
=============

Complete API documentation for all modules, classes, and functions in mice-py.

.. toctree::
   :maxdepth: 2

   mice
   methods
   pooling
   plotting
   utilities

Overview
--------

The mice-py package is organized into two main modules:

**imputation**
   Core MICE implementation, imputation methods, and analysis tools

**plotting**
   Diagnostic and visualization functions

Main Components
---------------

MICE Class
~~~~~~~~~~

The primary interface for multiple imputation:

.. code-block:: python

   from imputation import MICE
   
   mice = MICE(data)
   mice.impute(n_imputations=5, maxit=10)
   mice.fit('y ~ x1 + x2')
   results = mice.pool(summ=True)

See :doc:`mice` for complete documentation.

Imputation Methods
~~~~~~~~~~~~~~~~~~

Five imputation methods are available:

- **PMM**: Predictive Mean Matching
- **CART**: Classification and Regression Trees
- **Random Forest**: Ensemble tree method
- **MIDAS**: Distance-aided substitution
- **Sample**: Simple random sampling

See :doc:`methods` for detailed API documentation.

Pooling Functions
~~~~~~~~~~~~~~~~~

Functions for combining results using Rubin's rules:

- ``pool()``: Main pooling method (via MICE class)
- ``pool_descriptive_statistics()``: Pool descriptive statistics

See :doc:`pooling` for detailed documentation.

Plotting Functions
~~~~~~~~~~~~~~~~~~

Diagnostic and visualization tools:

- ``plot_chain_stats()``: Convergence diagnostics
- ``stripplot()``: Compare observed vs imputed values
- ``densityplot()``: Distribution comparison
- ``boxplot()``: Box plot comparison
- ``md_pattern_like()``: Missing data pattern summary
- ``plot_missing_data_pattern()``: Visualize patterns

See :doc:`plotting` for complete API.

Utility Functions
~~~~~~~~~~~~~~~~~

Helper functions for configuration and validation:

- ``configure_logging()``: Set up logging
- ``quickpred()``: Automatic predictor selection
- Various validators

See :doc:`utilities` for detailed documentation.

Quick Reference
---------------

Common Imports
~~~~~~~~~~~~~~

.. code-block:: python

   # Core functionality
   from imputation import MICE, configure_logging
   
   # Plotting
   from plotting.diagnostics import (
       plot_chain_stats, stripplot, densityplot, boxplot
   )
   from plotting.utils import md_pattern_like, plot_missing_data_pattern
   
   # Utilities
   from imputation.utils import quickpred

Typical Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from imputation import MICE, configure_logging
   from plotting.diagnostics import plot_chain_stats
   
   # Optional: enable logging
   configure_logging(level='INFO')
   
   # Load data
   df = pd.read_csv('data.csv')
   
   # Initialize and impute
   mice = MICE(df)
   mice.impute(
       n_imputations=20,
       maxit=20,
       method='pmm'
   )
   
   # Check convergence
   plot_chain_stats(mice.chain_mean, mice.chain_var)
   
   # Fit model and pool
   mice.fit('outcome ~ predictor1 + predictor2')
   results = mice.pool(summ=True)
   print(results)

Parameter Reference
-------------------

MICE.impute() Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Description
     - Default
   * - n_imputations
     - Number of imputed datasets to create
     - 5
   * - maxit
     - Number of MICE iterations
     - 10
   * - method
     - Imputation method(s)
     - 'pmm'
   * - initial
     - Initial imputation method
     - 'sample'
   * - predictor_matrix
     - Matrix controlling predictors
     - Auto-generated
   * - visit_sequence
     - Order to visit variables
     - 'monotone'

Method-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PMM**:
   - ``pmm_donors``: Number of donors (default: 5)
   - ``pmm_matchtype``: Matching type (default: 1)
   - ``pmm_ridge``: Ridge parameter (default: 1e-5)

**CART**:
   - ``cart_max_depth``: Maximum tree depth (default: None)
   - ``cart_min_samples_split``: Min samples to split (default: 2)
   - ``cart_min_samples_leaf``: Min samples in leaf (default: 1)

**Random Forest**:
   - ``rf_n_estimators``: Number of trees (default: 100)
   - ``rf_max_depth``: Maximum depth (default: None)
   - ``rf_max_features``: Features per split (default: 'sqrt')

**MIDAS**:
   - ``midas_donors``: Number of donors (default: 5)
   - ``midas_ridge``: Ridge parameter (default: 1e-5)

Return Values
-------------

MICE.imputed_datasets
~~~~~~~~~~~~~~~~~~~~~

List of pandas DataFrames, each containing a complete imputed dataset.

MICE.chain_mean, MICE.chain_var
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dictionaries mapping variable names to arrays of chain statistics across iterations.

MICE.pool()
~~~~~~~~~~~

Returns pandas DataFrame with columns:

- **Estimate**: Pooled coefficient
- **Std.Error**: Pooled standard error
- **t-statistic**: Test statistic
- **df**: Degrees of freedom
- **P>|t|**: p-value
- **[0.025, 0.975]**: 95% confidence interval
- **FMI**: Fraction of missing information

See Also
--------

- :doc:`../user_guide/index` for usage guidance
- :doc:`../examples/index` for practical examples
- :doc:`../theory/index` for theoretical background

