mice-py: Multiple Imputation by Chained Equations in Python
============================================================

A comprehensive Python implementation of **Multiple Imputation by Chained Equations (MICE)** for handling missing data in statistical analysis and machine learning workflows.

Developed as part of a master's thesis at Ludwig Maximilian University of Munich, Statistics Department.

Overview
--------

The framework provides:

- **Five imputation methods**: PMM, CART, Random Forest, MIDAS, and Sample imputation
- **Comprehensive diagnostics**: Visualization tools for analyzing imputation quality and missing data patterns
- **Statistical pooling**: Rubin's rules for combining estimates with formula-based model fitting
- **Professional logging**: Configurable logging system following Python best practices
- **Production-ready**: Extensive validation, testing, and documentation

Key Features
------------

- **Multiple Imputation by Chained Equations (MICE)**: Full implementation with customizable parameters
- **Automatic predictor matrix estimation**: Using quickpred algorithm or custom matrices
- **Flexible visit sequences**: Monotone, random, or custom imputation order
- **Mixed data types**: Support for numeric and categorical variables
- **Diagnostic Tools**: Stripplots, density plots, convergence diagnostics, and missing data patterns

Getting Started
---------------

Installation
~~~~~~~~~~~~

Install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

Or clone and install in development mode:

.. code-block:: bash

   git clone https://github.com/Zhanna-Lopuliak/mice-py.git
   cd mice-py
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE
   import pandas as pd
   
   # Load your data with missing values
   data = pd.read_csv('your_data.csv')
   
   # Configure logging (optional)
   from imputation import configure_logging
   configure_logging()
   
   # Create MICE object and perform imputation
   mice = MICE(data)
   mice.impute(n_imputations=5, maxit=10, method='pmm')
   
   # Access imputed datasets
   imputed_datasets = mice.imputed_datasets
   
   # Fit a model and pool results
   mice.fit('outcome ~ predictor1 + predictor2')
   results = mice.pool(summ=True)
   print(results)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: mice-py package

   imputation/index
   plotting/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
