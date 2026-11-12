mice-py: Multiple Imputation by Chained Equations in Python
============================================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg
   :target: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
   :alt: License: GPL v2+

.. image:: https://readthedocs.org/projects/mice-py/badge/?version=latest
   :target: https://mice-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

mice-py is a Python port of the R package `mice
<https://cran.r-project.org/package=mice>`_ (van Buuren and
Groothuis-Oudshoorn 2011) for multiple imputation of missing data by chained
equations. It works on pandas data frames and fits analysis models with
statsmodels.

What it has:

- **Imputation** with fourteen of R mice's methods, and R's default method
  for each column type: predictive mean matching (``pmm``, the default for
  numeric columns), ``midas`` (R's ``midastouch``), classification and
  regression trees (``cart``), random forests (``rf``), random draws
  (``sample``), linear regression (``norm``, ``norm.boot``, ``norm.nob``,
  ``norm.predict``), the mean (``mean``), logistic regression (``logreg``,
  ``logreg.boot``; the default for two categories), multinomial logistic
  regression (``polyreg``; the default for unordered categories) and the
  proportional-odds model (``polr``; the default for ordered categories);
  numeric, boolean, categorical and text columns.
- **R's model set-up**: full predictor matrix by default, ``quickpred``,
  removal of constant and collinear variables with logged events, R's dummy
  coding of categorical predictors.
- **Pooling** with Rubin's rules and Barnard-Rubin degrees of freedom, for
  any statsmodels formula model, or for estimates you computed yourself.
- **Diagnostic plots** after R mice: missing-data pattern, strip, box,
  density and scatter plots of observed and imputed values, trace plots.

**Validated against R mice 3.19.0**: exact comparisons of the deterministic
parts and, with R's random numbers replayed, of the model-based methods; the
distribution of imputed values per method; and a simulation study in which
both engines impute the same datasets. Every comparison agrees with R (within
the tolerances of the tests, the random ones within Monte Carlo error)
except random forests with categorical predictors; see
:doc:`r_mice/validation`. Version 0.3 changes results for data with
categorical columns, version 0.2 for users of 0.1; see :doc:`upgrading`.

A first example is on :doc:`quickstart`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   upgrading

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: R mice

   r_mice/validation
   r_mice/differences

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

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
