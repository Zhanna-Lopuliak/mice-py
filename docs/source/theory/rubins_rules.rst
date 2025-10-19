Rubin's Rules
=============

How to combine results from multiple imputed datasets.

The Problem
-----------

After imputation, you have *m* estimates: :math:`\hat{\theta}_1, ..., \hat{\theta}_m`

Simple averaging ignores imputation uncertainty. Rubin's rules provide the correct solution.

Basic Formulas
--------------

**Pooled Estimate**

.. math::

   \bar{\theta} = \frac{1}{m}\sum_{i=1}^{m} \hat{\theta}_i

**Within-Imputation Variance** (average sampling variance)

.. math::

   \bar{U} = \frac{1}{m}\sum_{i=1}^{m} SE_i^2

**Between-Imputation Variance** (variance due to missing data)

.. math::

   B = \frac{1}{m-1}\sum_{i=1}^{m} (\hat{\theta}_i - \bar{\theta})^2

**Total Variance**

.. math::

   T = \bar{U} + B + \frac{B}{m}

**Standard Error**

.. math::

   SE = \sqrt{T}

**Confidence Interval**

.. math::

   \bar{\theta} \pm t_{df} \times SE

where :math:`t_{df}` is from t-distribution with adjusted degrees of freedom.

Fraction of Missing Information (FMI)
--------------------------------------

.. math::

   FMI = \frac{(1 + 1/m)B}{T}

**Interpretation**:
   - FMI = 0: No impact from missing data
   - FMI = 0.3: 30% of uncertainty due to missingness
   - FMI > 0.3: Consider more imputations

How Many Imputations?
----------------------

**Old rule**: m = 5
**Modern recommendation**: m = 20+
**High missingness**: m = 50-100

**Rule of thumb**: m ≈ percentage of incomplete cases

Usage in mice-py
----------------

.. code-block:: python

   mice.fit('outcome ~ predictor')
   pooled = mice.pool(summ=True)

Output includes:
   - ``Estimate``: Pooled coefficient
   - ``Std.Error``: Pooled SE
   - ``P>|t|``: p-value
   - ``FMI``: Fraction of missing information

See :doc:`../user_guide/pooling_analysis` for practical usage.
