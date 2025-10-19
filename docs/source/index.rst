mice-py: Multiple Imputation by Chained Equations in Python
============================================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://readthedocs.org/projects/mice-py/badge/?version=latest
   :target: https://mice-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

A comprehensive Python implementation of **Multiple Imputation by Chained Equations (MICE)** for handling missing data in statistical analysis and machine learning workflows.

Key Features
------------

**Multiple Imputation Methods**
   Choose from five robust imputation strategies:
   
   - **PMM** (Predictive Mean Matching) - Maintains distributional properties
   - **CART** (Classification and Regression Trees) - Handles non-linear relationships
   - **Random Forest** - Captures complex interactions
   - **MIDAS** (Multiple Imputation with Distant Average Substitution) - Efficient for small samples
   - **Sample** - Simple random sampling from observed values

**Flexible Configuration**
   - Automatic predictor matrix estimation
   - Custom visit sequences for imputation order
   - Method-specific parameter control
   - Mixed data types (numeric and categorical)

**Statistical Pooling**
   - Rubin's rules for combining estimates
   - Fraction of missing information (FMI)
   - Confidence intervals and standard errors
   - Formula-based model fitting with statsmodels integration

**Diagnostic Tools**
   - Convergence diagnostics (chain statistics)
   - Stripplots, box plots, and density plots
   - Missing data pattern visualization
   - XY plots for bivariate relationships

Quick Example
-------------

.. code-block:: python

   import pandas as pd
   from imputation import MICE
   
   # Load your data with missing values
   df = pd.read_csv("your_data.csv")
   
   # Initialize and run MICE
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10, method='pmm')
   
   # Fit a model and pool results
   mice.fit('outcome ~ predictor1 + predictor2')
   results = mice.pool(summ=True)
   print(results)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   references

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
